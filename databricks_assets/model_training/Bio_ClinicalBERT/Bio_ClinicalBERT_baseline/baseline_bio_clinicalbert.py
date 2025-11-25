import os
from datetime import datetime

import pandas as pd
import torch
from databricks import sql
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, BertForSequenceClassification

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
else:
    print("No GPU visible to PyTorch")

# -----------------------------
# 1. Load credentials
# -----------------------------
load_dotenv()

DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
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
# 3. Query train and test data
# -----------------------------
train_query = """
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

test_query = """
SELECT
    subject_id,
    study_id,
    findings,
    impression,
    label,
    confidence
FROM workspace.default.mimic_cxr_test_set_label_explanation_consensus_v1
WHERE findings IS NOT NULL
  AND impression IS NOT NULL
  AND label IN ('Normal', 'Abnormal')
"""

print("Querying Databricks for train set...")
df_train = pd.read_sql(train_query, connection)
print(df_train.head())
print("Loaded train rows:", len(df_train))

print("Querying Databricks for test set...")
df_test = pd.read_sql(test_query, connection)
print(df_test.head())
print("Loaded test rows:", len(df_test))

connection.close()

# -----------------------------
# 4. Clean data for training / testing
# -----------------------------
def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Context"] = (
        df["findings"].fillna("") + " " + df["impression"].fillna("")
    ).str.strip()
    df = df[df["Context"].str.len() > 0]

    label_map = {"Normal": 0, "Abnormal": 1}
    df["Result"] = df["label"].map(label_map)
    return df


df_train = prepare_dataframe(df_train)
df_test = prepare_dataframe(df_test)

train_texts = df_train["Context"].tolist()
train_labels = df_train["Result"].tolist()
test_texts = df_test["Context"].tolist()
test_labels = df_test["Result"].tolist()

print("Processed train dataset size:", len(train_texts))
print("Processed test dataset size:", len(test_texts))

# -----------------------------
# 5. Prepare model and data
# -----------------------------
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding="max_length",
    max_length=120,
    return_tensors="pt",
)

test_encodings = tokenizer(
    test_texts,
    truncation=True,
    padding="max_length",
    max_length=120,
    return_tensors="pt",
)


class CXDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = CXDataset(train_encodings, train_labels)
test_dataset = CXDataset(test_encodings, test_labels)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    output_hidden_states=True,
    problem_type="single_label_classification",
)

optimizer = AdamW(model.parameters(), lr=4e-5)

# -----------------------------
# 6. Training loop
# -----------------------------
def train(num_epochs, model, dataloader, optimizer, device):
    model.to(device)

    for e in range(1, num_epochs + 1):
        total_loss = 0.0
        preds = []
        labels_list = []
        model.train()
        count = 0

        progress_bar = tqdm(dataloader, desc=f"TRAIN - EPOCH {e} |")

        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(**batch)
            current_loss = output.loss
            total_loss += current_loss

            ground_truth = batch["labels"].detach().cpu().numpy()
            batch_preds = output.logits.argmax(-1).detach().cpu().numpy()
            preds += list(batch_preds)
            labels_list += list(ground_truth)

            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()

            current_np_loss = float(current_loss.detach().cpu().numpy())
            if count % 100 == 0:
                print(f"TRAIN - EPOCH {e} | current-loss: {current_np_loss:.3f}")
            count += 1

            batch = {k: v.detach().cpu() for k, v in batch.items()}

        avg_loss = total_loss / len(dataloader)
        print("=" * 64)
        print(f"TRAIN - EPOCH {e} | LOSS: {avg_loss:.4f}")

        cm = confusion_matrix(labels_list, preds)
        print("Confusion matrix (train set, epoch-level):")
        print(cm)
        print("=" * 64)

    return model


model = train(20, model, train_dataloader, optimizer, device)

# -----------------------------
# 7. Evaluation on train & test
# -----------------------------
def evaluate(model, dataloader, device, split_name: str):
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc=f"EVAL ({split_name})")
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits

            batch_preds = logits.argmax(-1).detach().cpu().numpy()
            batch_labels = batch["labels"].detach().cpu().numpy()

            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

    cm = confusion_matrix(all_labels, all_preds)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )

    print("=" * 64)
    print(f"{split_name.upper()} CONFUSION MATRIX:")
    print(cm)
    print(f"{split_name.upper()} ACCURACY:  {acc:.4f}")
    print(f"{split_name.upper()} PRECISION: {precision:.4f}")
    print(f"{split_name.upper()} RECALL:    {recall:.4f}")
    print(f"{split_name.upper()} F1:        {f1:.4f}")
    print("=" * 64)

    tn, fp, fn, tp = cm.ravel()

    return {
        "split": split_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


train_metrics = evaluate(model, train_dataloader, device, split_name="train")
test_metrics = evaluate(model, test_dataloader, device, split_name="test")

# -----------------------------
# 8. Save metrics in this model folder
# -----------------------------
def save_metrics(metrics_list, model_name: str, num_epochs: int, lr: float):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_path = os.path.join(base_dir, "bio_clinicalbert_metrics.csv")

    rows = []
    run_timestamp = datetime.utcnow().isoformat()
    for m in metrics_list:
        row = {
            "run_timestamp_utc": run_timestamp,
            "model_name": model_name,
            "num_epochs": num_epochs,
            "learning_rate": lr,
        }
        row.update(m)
        rows.append(row)

    metrics_df = pd.DataFrame(rows)

    if os.path.exists(metrics_path):
        metrics_df.to_csv(metrics_path, index=False)

    print(f"Metrics saved to {metrics_path}")


save_metrics([train_metrics, test_metrics], model_name=model_name, num_epochs=20, lr=4e-5)

# -----------------------------
# 9. Save model locally
# -----------------------------
model.save_pretrained("./bio_clinicalbert_local")
tokenizer.save_pretrained("./bio_clinicalbert_local")

print("Model saved to ./bio_clinicalbert_local")
