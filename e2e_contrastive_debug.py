import os
from datetime import datetime

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from databricks import sql
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tqdm.auto import tqdm

# If you have this module already, use it:
from losses import SupConLoss
# otherwise you can inline SupConLoss here.


# ==========================
# Config
# ==========================
MODEL_NAME = "zzxslp/RadBERT-RoBERTa-4m"   # change to Bio_ClinicalBERT or DeBERTa later if you want
ENCODER_SAVE_DIR = "./trained_models/zzxslp_RadBERT-RoBERTa-4m_contrastive_encoder_debug"

EPOCHS_ENCODER = 5
EPOCHS_HEAD = 5
BATCH_SIZE = 32
LR_ENCODER = 4e-5
LR_HEAD = 4e-5
MAX_LEN = 120

# Optional: for quick debugging, you can subsample
MAX_TRAIN_SAMPLES = None   # e.g. 500 or None for all
MAX_TEST_SAMPLES = None    # e.g. 500 or None for all


# ==========================
# Data loading from Databricks
# ==========================
def load_data_from_databricks():
    load_dotenv()
    host = os.getenv("DATABRICKS_HOST")
    token = os.getenv("DATABRICKS_TOKEN")
    if not host or not token:
        raise ValueError("Missing Databricks credentials in .env")

    print("Connecting to Databricks...")
    conn = sql.connect(
        server_hostname=host.replace("https://", "").replace("http://", ""),
        http_path="/sql/1.0/warehouses/fe659a9780b351a1",
        access_token=token,
    )

    train_query = """
    SELECT subject_id, study_id, findings, impression, label, confidence
    FROM workspace.default.mimic_cxr_train_set_label_explanation_consensus_v1
    WHERE findings IS NOT NULL AND impression IS NOT NULL
      AND label IN ('Normal', 'Abnormal')
    """

    test_query = """
    SELECT subject_id, study_id, findings, impression, label, confidence
    FROM workspace.default.mimic_cxr_test_set_label_explanation_consensus_v1
    WHERE findings IS NOT NULL AND impression IS NOT NULL
      AND label IN ('Normal', 'Abnormal')
    """

    print("Querying train data...")
    df_train = pd.read_sql(train_query, conn)
    print("Querying test data...")
    df_test = pd.read_sql(test_query, conn)
    conn.close()

    def process(df):
        df["Context"] = (df["findings"].fillna("") + " " + df["impression"].fillna("")).str.strip()
        df = df[df["Context"].str.len() > 0]
        label_map = {"Normal": 0, "Abnormal": 1}
        df["Result"] = df["label"].map(label_map)
        return df["Context"].tolist(), df["Result"].tolist()

    train_texts, train_labels = process(df_train)
    test_texts, test_labels = process(df_test)

    print(f"Total train samples: {len(train_texts)}")
    print(f"Total test samples: {len(test_texts)}")

    # Optional subsampling for speed
    if MAX_TRAIN_SAMPLES is not None:
        train_texts = train_texts[:MAX_TRAIN_SAMPLES]
        train_labels = train_labels[:MAX_TRAIN_SAMPLES]
        print(f"Using first {len(train_texts)} train samples")

    if MAX_TEST_SAMPLES is not None:
        test_texts = test_texts[:MAX_TEST_SAMPLES]
        test_labels = test_labels[:MAX_TEST_SAMPLES]
        print(f"Using first {len(test_texts)} test samples")

    return train_texts, train_labels, test_texts, test_labels


# ==========================
# Dataset & model helpers
# ==========================
class EMRDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx]).long()
        return item


class MLP(nn.Module):
    def __init__(self, input_size=768, num_classes=2):
        super().__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


# ==========================
# Evaluation helper
# ==========================
def evaluate_epoch(encoder, classifier, dataloader, device, split_name):
    encoder.eval()
    classifier.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            outputs = encoder(**batch)
            features = torch.mean(outputs.last_hidden_state, dim=1)
            features = F.normalize(features, p=2, dim=1)  # MUST match training

            logits = classifier(features)
            preds = logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n[{split_name}] Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    print(f"[{split_name}] Confusion matrix:\n{cm}")

    return acc, prec, rec, f1, cm


# ==========================
# Contrastive encoder training
# ==========================
def train_contrastive_encoder(encoder, train_loader, device):
    print("\n=== Phase 1: Contrastive encoder training ===")
    encoder.to(device)
    optimizer = AdamW(encoder.parameters(), lr=LR_ENCODER)
    criterion = SupConLoss(temperature=0.07)

    for epoch in range(1, EPOCHS_ENCODER + 1):
        encoder.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Encoder Epoch {epoch}/{EPOCHS_ENCODER}")

        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            outputs = encoder(**batch)
            # Mean-pool token embeddings
            embeddings = torch.mean(outputs.last_hidden_state, dim=1)  # [B, H]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            features = embeddings.unsqueeze(1)  # [B, 1, H]

            loss = criterion(features, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Encoder epoch {epoch} - avg contrastive loss: {avg_loss:.4f}")

    return encoder


# ==========================
# MLP head training with metrics each epoch
# ==========================
def train_head_with_metrics(encoder, train_loader, test_loader, device):
    print("\n=== Phase 2: Linear head training on frozen encoder ===")

    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    encoder.to(device)

    hidden_size = encoder.config.hidden_size
    classifier = MLP(input_size=hidden_size, num_classes=2).to(device)

    optimizer = AdamW(classifier.parameters(), lr=LR_HEAD)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS_HEAD + 1):
        print("\n" + "=" * 60)
        print(f"Head Epoch {epoch}/{EPOCHS_HEAD}")
        print("=" * 60)

        classifier.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Head Train Epoch {epoch}")

        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            with torch.no_grad():
                outputs = encoder(**batch)
                features = torch.mean(outputs.last_hidden_state, dim=1)
                features = F.normalize(features, p=2, dim=1)

            logits = classifier(features)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / len(train_loader)
        print(f"\n[Head] Train loss (epoch {epoch}): {avg_loss:.4f}")

        # Per-epoch metrics on train + test
        evaluate_epoch(encoder, classifier, train_loader, device, split_name=f"Train epoch {epoch}")
        evaluate_epoch(encoder, classifier, test_loader, device, split_name=f"Test epoch {epoch}")

    return classifier


# ==========================
# Main
# ==========================
def main():
    print("CUDA available:", torch.cuda.is_available())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_texts, train_labels, test_texts, test_labels = load_data_from_databricks()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(texts):
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt",
        )

    train_enc = tokenize(train_texts)
    test_enc = tokenize(test_texts)

    train_ds = EMRDataset(train_enc, train_labels)
    test_ds = EMRDataset(test_enc, test_labels)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 1) Initialize encoder from HF
    encoder = AutoModel.from_pretrained(MODEL_NAME)

    # 2) Train encoder contrastively
    encoder = train_contrastive_encoder(encoder, train_loader, device)

    # (Optional) save encoder
    os.makedirs(ENCODER_SAVE_DIR, exist_ok=True)
    encoder.save_pretrained(ENCODER_SAVE_DIR)
    print(f"Saved contrastive encoder to: {ENCODER_SAVE_DIR}")

    # 3) Train linear head with metrics each epoch
    classifier = train_head_with_metrics(encoder, train_loader, test_loader, device)

    print("\nDone.")


if __name__ == "__main__":
    main()
