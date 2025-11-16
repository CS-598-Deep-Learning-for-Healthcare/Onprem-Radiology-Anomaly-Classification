import os
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from numpy import dot

from openai import OpenAI  # OpenAI-compatible client for Databricks
import backoff
from dotenv import load_dotenv


# =============================================================================
# ENV + CLIENT SETUP
# =============================================================================

# Force-load .env from project root (one level up from this file)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "").rstrip("/")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
<<<<<<< HEAD
DATABRICKS_CHAT_ENDPOINT = "databricks-qwen3-next-80b-a3b-instruct"
=======
DATABRICKS_CHAT_ENDPOINT = "databricks-gpt-oss-120b"
>>>>>>> 28dc61d2e2b0494ba9964b4a4b1c16e1738405e0
DATABRICKS_EMBEDDING_ENDPOINT = os.getenv(
    "DATABRICKS_EMBEDDING_ENDPOINT",
    "databricks-gte-large-en"  # default; override in .env if needed
)

if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
    raise RuntimeError(
        "DATABRICKS_HOST and/or DATABRICKS_TOKEN are not set. "
        "Please add them to your .env or environment."
    )

# OpenAI-compatible client pointing at Databricks Serving Endpoints
client = OpenAI(
    api_key=DATABRICKS_TOKEN,
    base_url=f"{DATABRICKS_HOST}/serving-endpoints",
    timeout=60.0,  # seconds per request
)


# =============================================================================
# CONFIG
# =============================================================================

# PARQUET_DIR = "mimic_cxr_data"
PARQUET_DIR = "G:\Jupyterstuff\CS598DLH\Onprem-Radiology-Anomaly-Classification\mimic_cxr_data"

# Safety / debugging knobs
API_CALL_BUDGET = 5          # total Chat + Embedding calls
MAX_FILES = 1                 # how many parquet shards to process (for testing)
MAX_ROWS_PER_FILE = 5         # max reports per file (for testing)
MAX_SENTENCES_PER_REPORT = 3  # max sentences per report (for testing)


# =============================================================================
# API CALL BUDGET TRACKING
# =============================================================================

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


# =============================================================================
# DATABRICKS LLM / EMBEDDING HELPERS
# =============================================================================

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def json_gpt(input: str):
    """
    Call the Databricks LLM endpoint and parse JSON output.
    """
    increment_api_calls(1)
    print("[json_gpt] Calling LLM...", flush=True)

    completion = client.chat.completions.create(
        model=DATABRICKS_CHAT_ENDPOINT,
        messages=[
            {"role": "system", "content": "You will be provided with a emr sentecne."},
            {"role": "user", "content": input},
        ],
        temperature=0.8,
    )
    print("[json_gpt] LLM call complete.")
<<<<<<< HEAD
    text = completion.choices[0].message.content
    print("text complete, text is of type :", type(text))
    print("text: ", text)
    print("[json_gpt] Raw LLM output:", text[:120].replace("\n", " "), "...", flush=True)
=======
    # print(completion)
    raw_content = completion.choices[0].message.content
    #different versions of libraries may have this as a string or list
    text = ""
    if isinstance(raw_content, list):
        for item in raw_content:
            print("item:", item)
            if isinstance(item, dict) and item.get('type') == 'text' and 'text' in item:
                text = item['text']
                break
    elif isinstance(raw_content, str):
        text = raw_content
    print("[json_gpt] Raw LLM output:", text[:120].replace("\n", " "), "...")
>>>>>>> 28dc61d2e2b0494ba9964b4a4b1c16e1738405e0

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        print("[json_gpt] JSON decode error, text was:", text, flush=True)
        raise e
    return parsed


@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def get_embedding(text, model: str = DATABRICKS_EMBEDDING_ENDPOINT):
    """
    Get text embedding from a Databricks embedding endpoint.
    """
    increment_api_calls(1)
    print(f"[get_embedding] Calling embedding endpoint '{model}'", flush=True)

    text = text.replace("\n", " ")
    resp = client.embeddings.create(
        model=model,
        input=[text]
    )
    return resp.data[0].embedding


# =============================================================================
# TEXT UTIL
# =============================================================================

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


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    start = time.time()

    dataset = []
    dataset1 = []
    dataset2 = []
    trouble_rows = []  # (parquet_file, row_index)

    if not os.path.isdir(PARQUET_DIR):
        raise RuntimeError(
            f"Parquet directory '{PARQUET_DIR}' does not exist. "
            f"Create it and put your MIMIC-CXR parquet shards there."
        )

    parquet_files = sorted(
        f for f in os.listdir(PARQUET_DIR)
        if f.endswith(".parquet")
    )

    if not parquet_files:
        raise RuntimeError(
            f"No .parquet files found in {PARQUET_DIR}. "
            f"Add train-00000-of-00002.parquet etc. to that directory."
        )

    print("Found Parquet files:", parquet_files, flush=True)
    print(f"API call budget: {API_CALL_BUDGET}", flush=True)

    try:
        for pf_idx, pf in enumerate(parquet_files):
            if pf_idx >= MAX_FILES:
                print(f"Reached MAX_FILES={MAX_FILES}, stopping.", flush=True)
                break

            parquet_path = os.path.join(PARQUET_DIR, pf)
            print(f"\n=== Processing Parquet file {pf_idx + 1}/{len(parquet_files)}: {pf} ===", flush=True)

            table = pq.read_table(parquet_path)
            df = table.to_pydict()
            num_rows = len(next(iter(df.values()))) if df else 0
            print(f"Rows in {pf}: {num_rows}", flush=True)

            for row_idx in range(num_rows):
                if row_idx >= MAX_ROWS_PER_FILE:
                    print(f"Reached MAX_ROWS_PER_FILE={MAX_ROWS_PER_FILE} in {pf}, moving to next file.", flush=True)
                    break

                if row_idx % 1 == 0:
                    print(f"[{pf}] Row {row_idx + 1}/{num_rows}", flush=True)

                row = {col: df[col][row_idx] for col in df}
                report_text = get_report_text(row)
                if not report_text:
                    print(f"[{pf}] Row {row_idx} has no findings/impression text, skipping.", flush=True)
                    continue

                raw_sentences = report_text.split(".")
                sentences = [s.strip() for s in raw_sentences if s.strip()]

                print(f"[{pf}] Row {row_idx}: {len(sentences)} sentences found.", flush=True)

                for sent_idx, sentence in enumerate(sentences):
                    if sent_idx >= MAX_SENTENCES_PER_REPORT:
                        print(
                            f"[{pf}] Row {row_idx}: Reached MAX_SENTENCES_PER_REPORT="
                            f"{MAX_SENTENCES_PER_REPORT}, moving to next row.",
                            flush=True,
                        )
                        break

                    print(
                        f"[{pf}] Row {row_idx}, sentence {sent_idx}: '{sentence[:80]}...'",
                        flush=True,
                    )

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
                            print(str(e), flush=True)
                            raise

                        except Exception as e:
                            error_count += 1
                            print(f"[PASS 1] Error (attempt {error_count}): {e}", flush=True)
                            #print stacktrace
                            import traceback
                            traceback.print_exc()
                            time.sleep(2)
                            continue

                    if error_count > 50:
                        trouble_rows.append((pf, int(row_idx)))
                        break

                    # ---------- PASS 2 ----------
                    success_flag1 = 0
                    while success_flag1 == 0 and error_count <= 50:
                        try:
                            response = json_gpt(query)
                            response["Context"] = sentence
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
                            print(str(e), flush=True)
                            raise

                        except Exception as e:
                            error_count += 1
                            print(f"[PASS 2] Error (attempt {error_count}): {e}", flush=True)
                            time.sleep(2)
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
                            print(str(e), flush=True)
                            raise

                        except Exception as e:
                            error_count += 1
                            print(f"[PASS 3] Error (attempt {error_count}): {e}", flush=True)
                            time.sleep(2)
                            continue

                    if error_count > 50:
                        trouble_rows.append((pf, int(row_idx)))
                        break

    except APIBudgetExceeded as e:
        print("\n=== API BUDGET REACHED ===", flush=True)
        print(str(e), flush=True)
        print(f"Total API calls made: {api_call_count}", flush=True)
        print("Stopping early and saving partial results.\n", flush=True)

    end = time.time()

    print("total_time:", end - start, flush=True)
    print("Total API calls used:", api_call_count, flush=True)

    # Save whatever we have so far
    with open("multi_label_dataset_final_v1.json", "w") as f:
        json.dump(dataset, f)

    with open("multi_label_dataset_final_v2.json", "w") as f:
        json.dump(dataset1, f)

    with open("multi_label_dataset_final_v3.json", "w") as f:
        json.dump(dataset2, f)

    np.save("trouble_rows.npy", np.array(trouble_rows, dtype=object))

    print("Saved datasets and trouble_rows.npy to disk.", flush=True)
