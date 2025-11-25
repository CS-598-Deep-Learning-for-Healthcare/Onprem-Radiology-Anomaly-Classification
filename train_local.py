import os
import pandas as pd
import torch
from dotenv import load_dotenv
from databricks import sql

epoch_count = 20

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("No GPU visible to PyTorch")

# -----------------------------
# 1. Load credentials
# -----------------------------
load_dotenv()

DATABRICKS_HOST  = os.getenv("DATABRICKS_HOST")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
    raise ValueError("Missing Databricks credentials in .env")

# -----------------------------
# 2. Connect to Databricks
# -----------------------------
connection = sql.connect(
    server_hostname=DATABRICKS_HOST.replace("https://", "").replace("http://", ""),
    http_path="/sql/1.0/warehouses/fe659a9780b351a1",
    access_token=DATABRICKS_TOKEN,
)

# -----------------------------
# 3. Query your consensus table
# -----------------------------
query = """
SELECT
    subject_id,
    study_id,
    findings,
    impression,
    label,
    confidence
FROM workspace.default.mimic_cxr_train_set_label_explanation_consensus_v1
WHERE findings IS NOT NULL
  AND impression IS NOT NULL
  AND label IN ('Normal', 'Abnormal')
"""

print("Querying Databricks...")
df = pd.read_sql(query, connection)
print(df.head())
print("Loaded rows:", len(df))

connection.close()

# -----------------------------
# 4. Clean data for training
# -----------------------------
df["Context"] = (df["findings"].fillna("") + " " +
                 df["impression"].fillna("")).str.strip()

df = df[df["Context"].str.len() > 0]

label_map = {"Normal": 0, "Abnormal": 1}
df["Result"] = df["label"].map(label_map)

texts  = df["Context"].tolist()
labels = df["Result"].tolist()

print("Processed dataset size:", len(texts))

# -----------------------------
# 5. Train RadBERT locally
# -----------------------------
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm.auto import tqdm

# model_name = "zzxslp/RadBERT-RoBERTa-4m"
model_names = ["emilyalsentzer/Bio_ClinicalBERT", "microsoft/deberta-v3-base", "zzxslp/RadBERT-RoBERTa-4m"]

for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=120,
        return_tensors="pt"
    )

    class CXDataset(Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {k:v[idx] for k,v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx]).long()
            return item

        def __len__(self):
            return len(self.labels)

    dataset = CXDataset(encodings, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        output_hidden_states=True
    )

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=4e-5)

    # -----------------------------
    # 6. Simple training loop
    # -----------------------------
    for epoch in range(epoch_count):
        total_loss = 0
        model.train()

        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for batch in loop:
            batch = {k:v.to(device) for k,v in batch.items()}
            output = model(**batch)

            loss = output.loss
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} loss: {total_loss/len(dataloader):.4f}")

    # -----------------------------
    # 7. Save model locally
    # -----------------------------
    model.save_pretrained(f"./{model_name}_local")
    tokenizer.save_pretrained(f"./{model_name}_local")

    print(f"Model saved to ./{model_name}_local")
