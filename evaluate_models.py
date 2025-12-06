import os
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from tqdm.auto import tqdm
from dotenv import load_dotenv
from databricks import sql
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
BATCH_SIZE = 32
SAVE_DIR = "./trained_models"
OUTPUT_CSV = "./evaluation_results.csv"

# -----------------------------
# 1. Database & Data Loading
# -----------------------------
def load_data_from_databricks():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    
    load_dotenv()
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")

    if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
        raise ValueError("Missing Databricks credentials in .env")

    print("Connecting to Databricks...")
    connection = sql.connect(
        server_hostname=DATABRICKS_HOST.replace("https://", "").replace("http://", ""),
        http_path="/sql/1.0/warehouses/fe659a9780b351a1",
        access_token=DATABRICKS_TOKEN,
    )

    # --- Test Query ---
    test_query = """
    SELECT subject_id, study_id, findings, impression, label, confidence
    FROM workspace.default.mimic_cxr_test_set_label_explanation_consensus_v1
    WHERE findings IS NOT NULL AND impression IS NOT NULL
      AND label IN ('Normal', 'Abnormal')
    """

    print("Querying Test data...")
    df_test = pd.read_sql(test_query, connection)
    connection.close()
    
    # Helper to clean and map
    def process_df(df):
        df["Context"] = (df["findings"].fillna("") + " " + df["impression"].fillna("")).str.strip()
        df = df[df["Context"].str.len() > 0]
        label_map = {"Normal": 0, "Abnormal": 1}
        df["Result"] = df["label"].map(label_map)
        return df["Context"].tolist(), df["Result"].tolist()

    test_texts, test_labels = process_df(df_test)
    
    print(f"Data loaded. Test: {len(test_texts)}")
    return test_texts, test_labels


# -----------------------------
# 2. Shared Utilities & Classes
# -----------------------------

class EMRDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).long()
        return item

    def __len__(self):
        return len(self.labels)


class MLP(nn.Module):
    """Simple Classifier for the Last Layer Step"""
    def __init__(self, target_size, input_size=768):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, target_size)

    def forward(self, x):
        return self.fc1(x)


def get_clean_name(model_path):
    """Sanitize model path for folder creation"""
    return model_path.replace("/", "_")


def evaluate_baseline(model, dataloader, device):
    """Evaluate the baseline (full fine-tuned) model."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Baseline"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            outputs = model(**batch)
            logits = outputs.logits

            batch_preds = logits.argmax(-1).detach().cpu().numpy()
            batch_labels = labels.detach().cpu().numpy()

            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

    return compute_metrics(all_labels, all_preds)


def evaluate_contrastive_mlp(encoder, classifier, dataloader, device):
    """Evaluate the contrastive encoder + MLP classifier."""
    encoder.eval()
    classifier.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Contrastive MLP"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            # Get encoder features
            outputs = encoder(**batch)
            features = torch.mean(outputs.last_hidden_state, dim=1)
            
            # CRITICAL FIX: Normalize features (must match training)
            features = F.normalize(features, p=2, dim=1)
            
            logits = classifier(features)

            batch_preds = logits.argmax(-1).detach().cpu().numpy()
            batch_labels = labels.detach().cpu().numpy()

            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

    return compute_metrics(all_labels, all_preds)


def compute_metrics(all_labels, all_preds):
    """Compute and return metrics dictionary."""
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  Confusion Matrix:\n{cm}\n")

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def save_results(results_list, output_path):
    """Save all results to CSV."""
    df = pd.DataFrame(results_list)
    df.to_csv(output_path, index=False)
    print(f"\nAll results saved to {output_path}")


# -----------------------------
# 3. Main Evaluation
# -----------------------------
def main():
    # Load test data
    test_texts, test_labels = load_data_from_databricks()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_names = [
        "zzxslp/RadBERT-RoBERTa-4m",
        "emilyalsentzer/Bio_ClinicalBERT", 
        "microsoft/deberta-v3-base", 
    ]

    all_results = []

    for model_name in model_names:
        print("=" * 60)
        print(f"EVALUATING MODEL: {model_name}")
        print("=" * 60)

        clean_name = get_clean_name(model_name)
        
        # Load tokenizer
        tokenizer_path = os.path.join(SAVE_DIR, f"{clean_name}_tokenizer")
        if not os.path.exists(tokenizer_path):
            print(f"  [WARNING] Tokenizer not found at {tokenizer_path}, using original model tokenizer")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Tokenize test data
        test_enc = tokenizer(
            test_texts, truncation=True, padding="max_length", max_length=120, return_tensors="pt"
        )
        test_dataset = EMRDataset(test_enc, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # ----- Evaluate Baseline -----
        baseline_path = os.path.join(SAVE_DIR, f"{clean_name}_baseline")
        if os.path.exists(baseline_path):
            print(f"\n[BASELINE] Loading from {baseline_path}")
            baseline_model = AutoModelForSequenceClassification.from_pretrained(baseline_path)
            baseline_model.to(device)
            
            metrics = evaluate_baseline(baseline_model, test_loader, device)
            metrics["model_name"] = model_name
            metrics["method"] = "Baseline"
            metrics["timestamp"] = datetime.utcnow().isoformat()
            all_results.append(metrics)
            
            # Free memory
            del baseline_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            print(f"  [SKIP] Baseline model not found at {baseline_path}")

        # ----- Evaluate Contrastive MLP -----
        encoder_path = os.path.join(SAVE_DIR, f"{clean_name}_contrastive_encoder")
        mlp_path = os.path.join(SAVE_DIR, f"{clean_name}_contrastive_classifier_mlp.pth")
        
        if os.path.exists(encoder_path) and os.path.exists(mlp_path):
            print(f"\n[CONTRASTIVE MLP] Loading encoder from {encoder_path}")
            print(f"[CONTRASTIVE MLP] Loading MLP from {mlp_path}")
            
            # Load encoder
            encoder = AutoModel.from_pretrained(encoder_path)
            encoder.to(device)
            
            # Load MLP
            hidden_size = encoder.config.hidden_size
            classifier = MLP(target_size=2, input_size=hidden_size)
            classifier.load_state_dict(torch.load(mlp_path, map_location=device))
            classifier.to(device)
            
            metrics = evaluate_contrastive_mlp(encoder, classifier, test_loader, device)
            metrics["model_name"] = model_name
            metrics["method"] = "Contrastive_Last_Layer"
            metrics["timestamp"] = datetime.utcnow().isoformat()
            all_results.append(metrics)
            
            # Free memory
            del encoder, classifier
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        else:
            print(f"  [SKIP] Contrastive encoder or MLP not found")
            if not os.path.exists(encoder_path):
                print(f"    Missing: {encoder_path}")
            if not os.path.exists(mlp_path):
                print(f"    Missing: {mlp_path}")

    # Save all results
    if all_results:
        save_results(all_results, OUTPUT_CSV)
        
        # Print summary table
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        df = pd.DataFrame(all_results)
        summary_cols = ["model_name", "method", "accuracy", "precision", "recall", "f1"]
        print(df[summary_cols].to_string(index=False))
    else:
        print("\nNo models were evaluated. Check that trained models exist in:", SAVE_DIR)


if __name__ == "__main__":
    main()