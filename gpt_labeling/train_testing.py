import os
import json
import time
from collections import deque  # kept in case you want to reuse later, not required

import numpy as np
import pyarrow.parquet as pq
from numpy import dot

import openai
import backoff
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
# ==============================
# CONFIG
# ==============================

# Models
EMBEDDING_MODEL = "text-embedding-ada-002"
GPT_MODEL = "GPT5-nano"
# GPT_MODEL = "llama-3.3-70b-versatile"

# Directory containing Parquet shards like train-00000-of-00002.parquet
PARQUET_DIR = "mimic_cxr_data"

# Max total number of OpenAI API calls (Chat + Embedding)
API_CALL_BUDGET = 20  # <-- change this to test with more/less calls


# ==============================
# API KEY SETUP
# ==============================

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# if not GROQ_API_KEY:
#     raise RuntimeError(
#         "GROQ_API_KEY environment variable is not set. "
#         "Set it in your shell or a .env file before running."
#     )



# openai.api_key = GROQ_API_KEY
# openai.api_base = "https://api.groq.com/openai/v1"

openai.api_key = DATABRICKS_TOKEN
openai.api_base = "https://dbc-0005fb62-4075.cloud.databricks.com/serving-endpoints"

try:
    model = SentenceTransformer('BAAI/bge-m3')
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'sentence-transformers' is installed (`pip install sentence-transformers`)")
    exit()

IS_LOCAL_EMBEDDING = True  # Set to True to use local embeddings

# ==============================
# CALL BUDGET TRACKING
# ==============================

class APIBudgetExceeded(Exception):
    """Raised when we exceed the configured API call budget."""
    pass


api_call_count = 0


def increment_api_calls(n: int = 1):
    """Increment the global API call counter and enforce the budget."""
    global api_call_count
    if api_call_count + n > API_CALL_BUDGET:
        raise APIBudgetExceeded(
            f"API call budget exceeded: attempted {api_call_count + n}, "
            f"limit is {API_CALL_BUDGET}"
        )
    api_call_count += n


# ==============================
# OPENAI HELPERS
# ==============================

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def json_gpt(input: str):
    # Count this call against the budget
    increment_api_calls(1)

    completion = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You will be provided with a emr sentence."},
            {"role": "user", "content": input},
        ],
        temperature=0.8,
    )
    text = completion.choices[0].message.content
    parsed = json.loads(text)
    return parsed


@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def get_embedding(text, model=EMBEDDING_MODEL):
    # Count this call against the budget
    increment_api_calls(1)

    text = text.replace("\n", " ")
    return openai.Embedding.create(
        input=[text],
        model=model
    )["data"][0]["embedding"]

def get_embedding_local(text, model=EMBEDDING_MODEL):
    # Count this call against the budget

    text = text.replace("\n", " ")
    return model.encode(text)


# ==============================
# TEXT UTIL
# ==============================

def get_report_text(row) -> str:
    """
    Extract the report text from a Parquet row.
    For your schema, we use 'findings' and 'impression'.
    """
    findings = row.get("findings")
    impression = row.get("impression")

    parts = []
    if isinstance(findings, str) and findings.strip():
        parts.append(findings.strip())
    if isinstance(impression, str) and impression.strip():
        parts.append(impression.strip())

    return ". ".join(parts)



# ==============================
# MAIN
# ==============================

if __name__ == "__main__":
    start = time.time()

    # Datasets for the 3 passes
    dataset = []
    dataset1 = []
    dataset2 = []

    trouble_rows = []  # track (parquet_file, row_index) where errors exceeded threshold

    # Discover Parquet files in PARQUET_DIR
    parquet_files = sorted(
        f for f in os.listdir(PARQUET_DIR)
        if f.endswith(".parquet")
    )

    if not parquet_files:
        raise RuntimeError(
            f"No .parquet files found in {PARQUET_DIR}. "
            f"Add train-00000-of-00002.parquet etc. to that directory."
        )

    print("Found Parquet files:", parquet_files)
    print(f"API call budget: {API_CALL_BUDGET}")

    try:
        for pf_idx, pf in enumerate(parquet_files):
            parquet_path = os.path.join(PARQUET_DIR, pf)
            print(f"\n=== Processing Parquet file {pf_idx + 1}/{len(parquet_files)}: {pf} ===")

            table = pq.read_table(parquet_path)
            df = table.to_pydict()  # columns -> list

            num_rows = len(next(iter(df.values()))) if df else 0
            print(f"Rows in {pf}: {num_rows}")

            # iterate by index to access dict-of-lists
            for row_idx in range(num_rows):
                # Build a "row" dict: column -> value
                row = {col: df[col][row_idx] for col in df}

                report_text = get_report_text(row)
                if not report_text:
                    continue  # skip rows without valid text

                # Split report into sentences
                raw_sentences = report_text.split(".")
                sentences = [s.strip() for s in raw_sentences if s.strip()]

                for sent_idx, sentence in enumerate(sentences):
                    query = f"""Use the below sentence to answer the subsequent question.
                    Emr_report:
                    \"\"\" 
                    {sentence}
                    \"\"\" 
                    Question: Does the patient have the specific disease in the chest based on the provied EMR report's sentence? 
                    Answer form should be JSON object like following script. The JSON object has two key, "Result", and "Explanation".
                    For [Result], if the sentence doesn't have enough information or evidence to classify, you should return "Uncertain". 
                    If the sentence has the clear evidence that indicates absence of any abnormalities in chest, you should answer "No". 
                    If the sentence has the clear observational evidence that indicates presence of any abnormalities in chest (only for present), you should answer "Yes". 

                    For [Explanation], you should give a sentence more than 40 letters and less than 60 letters which explain the reason about why you choose those answers. You should elucidating the rationale behind your choice, not a direct repetition, of the input text.
                    [Result] : Uncertain / No / Yes
                    """

                    error_count = 0

                    # ---------- PASS 1 ----------
                    success_flag = 0
                    while success_flag == 0 and error_count <= 50:
                        try:
                            response = json_gpt(query)
                            response["Context"] = sentence
                            if IS_LOCAL_EMBEDDING:
                                context_emb = get_embedding_local(response["Context"])
                                explanation_emb = get_embedding_local(response["Explanation"])
                            else:
                                context_emb = get_embedding(response["Context"])
                                explanation_emb = get_embedding(response["Explanation"])
                            cosine_sim = (dot(context_emb, explanation_emb) + 1) / 2
                            response["confidence"] = cosine_sim
                            response["parquet_file"] = pf
                            response["row_index"] = int(row_idx)
                            response["sentence_index"] = int(sent_idx)
                            dataset.append(response)
                            success_flag = 1

                        except APIBudgetExceeded as e:
                            print(str(e))
                            raise  # bubble up to stop everything

                        except Exception:
                            time.sleep(10)
                            print("error occurred in chatgpt server-train (pass 1)")
                            error_count += 1
                            continue

                    if error_count > 50:
                        trouble_rows.append((pf, int(row_idx)))
                        break  # stop this report

                    # ---------- PASS 2 ----------
                    success_flag1 = 0
                    while success_flag1 == 0 and error_count <= 50:
                        try:
                            response = json_gpt(query)
                            response["Context"] = sentence
                            if IS_LOCAL_EMBEDDING:
                                context_emb = get_embedding_local(response["Context"])
                                explanation_emb = get_embedding_local(response["Explanation"])
                            else:
                                context_emb = get_embedding(response["Context"])
                                explanation_emb = get_embedding(response["Explanation"])
                            cosine_sim = (dot(context_emb, explanation_emb) + 1) / 2
                            response["confidence"] = cosine_sim
                            response["parquet_file"] = pf
                            response["row_index"] = int(row_idx)
                            response["sentence_index"] = int(sent_idx)
                            dataset1.append(response)
                            success_flag1 = 1

                        except APIBudgetExceeded as e:
                            print(str(e))
                            raise

                        except Exception:
                            time.sleep(10)
                            print("error occurred in chatgpt server-train (pass 2)")
                            error_count += 1
                            continue

                    if error_count > 50:
                        trouble_rows.append((pf, int(row_idx)))
                        break

                    # ---------- PASS 3 ----------
                    success_flag2 = 0
                    while success_flag2 == 0 and error_count <= 50:
                        try:
                            response = json_gpt(query)
                            response["Context"] = sentence
                            if IS_LOCAL_EMBEDDING:
                                context_emb = get_embedding_local(response["Context"])
                                explanation_emb = get_embedding_local(response["Explanation"])
                            else:
                                context_emb = get_embedding(response["Context"])
                                explanation_emb = get_embedding(response["Explanation"])
                            cosine_sim = (dot(context_emb, explanation_emb) + 1) / 2
                            response["confidence"] = cosine_sim
                            response["parquet_file"] = pf
                            response["row_index"] = int(row_idx)
                            response["sentence_index"] = int(sent_idx)
                            dataset2.append(response)
                            success_flag2 = 1

                        except APIBudgetExceeded as e:
                            print(str(e))
                            raise

                        except Exception:
                            time.sleep(10)
                            print("error occurred in chatgpt server-train (pass 3)")
                            error_count += 1
                            continue

                    if error_count > 50:
                        trouble_rows.append((pf, int(row_idx)))
                        break

                # end sentence loop

    except APIBudgetExceeded as e:
        print("\n=== API BUDGET REACHED ===")
        print(str(e))
        print(f"Total API calls made: {api_call_count}")
        print("Stopping early and saving partial results.\n")

    end = time.time()

    print("total_time:", end - start)
    print("Total API calls used:", api_call_count)

    # Save whatever we have so far
    with open("multi_label_dataset_final_v1.json", "w") as f:
        json.dump(dataset, f)

    with open("multi_label_dataset_final_v2.json", "w") as f:
        json.dump(dataset1, f)

    with open("multi_label_dataset_final_v3.json", "w") as f:
        json.dump(dataset2, f)

    np.save("trouble_rows.npy", np.array(trouble_rows, dtype=object))

    print("Saved datasets and trouble_rows.npy to disk.")
