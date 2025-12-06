import os
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from tqdm.auto import tqdm
from dotenv import load_dotenv
from databricks import sql
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from plyer import notification

# Import custom loss
from losses import SupConLoss

# -----------------------------
# Configuration
# -----------------------------
EPOCH_COUNT = 20
BATCH_SIZE = 32
LEARNING_RATE = 4e-5
SAVE_DIR = "./trained_models"

# Ensure output directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

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

    # --- Train Query ---
    train_query = """
    SELECT subject_id, study_id, findings, impression, label, confidence
    FROM workspace.default.mimic_cxr_train_set_label_explanation_consensus_v1
    WHERE findings IS NOT NULL AND impression IS NOT NULL
      AND label IN ('Normal', 'Abnormal')
    """

    # --- Test Query ---
    test_query = """
    SELECT subject_id, study_id, findings, impression, label, confidence
    FROM workspace.default.mimic_cxr_test_set_label_explanation_consensus_v1
    WHERE findings IS NOT NULL AND impression IS NOT NULL
      AND label IN ('Normal', 'Abnormal')
    """

    print("Querying Train data...")
    df_train = pd.read_sql(train_query, connection)
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

    train_texts, train_labels = process_df(df_train)
    test_texts, test_labels = process_df(df_test)
    
    print(f"Data loaded. Train: {len(train_texts)}, Test: {len(test_texts)}")
    return train_texts, train_labels, test_texts, test_labels

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

def get_classification_status(true_label, pred_label):
    """Helper to return text status"""
    if true_label == 1 and pred_label == 1: return "TP"
    if true_label == 0 and pred_label == 1: return "FP"
    if true_label == 1 and pred_label == 0: return "FN"
    if true_label == 0 and pred_label == 0: return "TN"
    return "Unknown"

def evaluate_model(model, dataloader, device, split_name, model_name, method_tag, raw_texts=None, is_mlp=False, encoder=None):
    """
    Evaluation loop. 
    Now generates a detailed predictions CSV containing the text and TP/FP status.
    """
    model.eval()
    if encoder: 
        encoder.eval()
    
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"EVAL ({split_name})"):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            if is_mlp and encoder is not None:
                # --- Pooler Output, No Normalization (Stage 2) ---
                outputs = encoder(**batch)
                
                # Check for pooler_output (BERT/RoBERTa), fall back to CLS if missing
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state[:, 0, :]
                
                logits = model(features)
            else:
                outputs = model(**batch)
                logits = outputs.logits

            batch_preds = logits.argmax(-1).detach().cpu().numpy()
            batch_labels = labels.detach().cpu().numpy()

            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

    # 1. Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n{split_name.upper()} RESULTS:")
    print(f"Acc: {acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")

    # 2. Generate Detailed Predictions CSV (if raw texts are provided)
    if raw_texts and len(raw_texts) == len(all_preds):
        detailed_rows = []
        for text, true, pred in zip(raw_texts, all_labels, all_preds):
            status = get_classification_status(true, pred)
            detailed_rows.append({
                "text": text,
                "prediction": pred,
                "true_label": true,
                "status": status  # TP, FP, TN, FN
            })
        
        clean_name = get_clean_name(model_name)
        pred_filename = f"{clean_name}_{method_tag}_predictions.csv"
        pred_path = os.path.join(SAVE_DIR, pred_filename)
        pd.DataFrame(detailed_rows).to_csv(pred_path, index=False)
        print(f"Detailed predictions saved to {pred_path}")
    else:
        print("Warning: Raw texts not provided or length mismatch, skipping predictions CSV.")
    
    return {
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp
    }

def save_metrics_to_csv(metrics, model_name, method_tag):
    """Saves metrics to a CSV file."""
    clean_name = get_clean_name(model_name)
    filename = f"{clean_name}_metrics.csv"
    filepath = os.path.join(SAVE_DIR, filename)
    
    # Define exact column order requested
    cols = [
        "run_timestamp_utc", "model_name", "method", "num_epochs", "learning_rate",
        "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"
    ]
    
    row = {
        "run_timestamp_utc": datetime.utcnow().isoformat(),
        "model_name": model_name,
        "method": method_tag, 
        "num_epochs": EPOCH_COUNT,
        "learning_rate": LEARNING_RATE
    }
    row.update(metrics)
    
    # Create DF with specific column order
    df = pd.DataFrame([row], columns=cols)
    
    # Append if exists, else create new
    if os.path.exists(filepath):
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)
    
    print(f"Metrics saved to {filepath}")

# -----------------------------
# 3. Training Loops
# -----------------------------

def train_baseline(model_name, train_loader, test_loader, device, test_texts):
    """Standard Supervised Fine-Tuning"""
    print(f"\n[BASELINE] Starting training for {model_name}...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(EPOCH_COUNT):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Base Epoch {epoch+1}")
        
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    # Evaluate with raw texts passed for CSV generation
    metrics = evaluate_model(
        model, test_loader, device, "Test (Baseline)", 
        model_name, "Baseline", raw_texts=test_texts
    )
    save_metrics_to_csv(metrics, model_name, "Baseline")

    # Save
    clean_name = get_clean_name(model_name)
    save_path = os.path.join(SAVE_DIR, f"{clean_name}_baseline")
    model.save_pretrained(save_path)
    print(f"[BASELINE] Saved to {save_path}")


def train_contrastive_encoder(model_name, train_loader, device):
    """Supervised Contrastive Learning (Encoder only)"""
    print(f"\n[CONTRASTIVE ENCODER] Starting training for {model_name}...")
    
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = SupConLoss(temperature=0.07)
    
    for epoch in range(EPOCH_COUNT):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Enc Epoch {epoch+1}")
        
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            outputs = model(**batch)
            
            # Robust Mean Pooling
            embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            
            # Normalize before unsqueezing for SupConLoss
            embeddings = F.normalize(embeddings, p=2, dim=1)
            features = embeddings.unsqueeze(1) # [bsz, 1, dim]
            
            loss = criterion(features, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Clip grad norm (standard practice)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    # Save
    clean_name = get_clean_name(model_name)
    save_path = os.path.join(SAVE_DIR, f"{clean_name}_contrastive_encoder")
    model.save_pretrained(save_path)
    print(f"[CONTRASTIVE ENCODER] Saved to {save_path}")
    return save_path


def train_contrastive_last_layer(original_model_name, encoder_path, train_loader, test_loader, device, test_texts):
    print(f"\n[LAST LAYER] Training MLP on top of {encoder_path}...")
    
    encoder = AutoModel.from_pretrained(encoder_path)
    encoder.to(device)
    
    # Freeze encoder
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    
    # Initialize MLP
    hidden_size = encoder.config.hidden_size
    classifier = MLP(target_size=2, input_size=hidden_size)
    classifier.to(device)
    
    optimizer = AdamW(classifier.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCH_COUNT):
        classifier.train()
        loop = tqdm(train_loader, desc=f"MLP Epoch {epoch+1}")
        
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            with torch.no_grad():
                outputs = encoder(**batch)
                
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    features = outputs.pooler_output
                else:
                    features = outputs.last_hidden_state[:, 0, :]
                
            logits = classifier(features)
            loss = loss_func(logits, labels)
            
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())

    # Evaluate
    metrics = evaluate_model(
        classifier, test_loader, device, "Test (Contrastive MLP)", 
        original_model_name, "Contrastive_Last_Layer",
        raw_texts=test_texts, is_mlp=True, encoder=encoder
    )
    save_metrics_to_csv(metrics, original_model_name, "Contrastive_Last_Layer")

# -----------------------------
# 4. Main Execution
# -----------------------------
def main():
    train_texts, train_labels, test_texts, test_labels = load_data_from_databricks()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_names = [
        "zzxslp/RadBERT-RoBERTa-4m",
        "emilyalsentzer/Bio_ClinicalBERT", 
        "microsoft/deberta-v3-base", 
    ]

    for model_name in model_names:
        print("="*60)
        print(f"PROCESSING MODEL: {model_name}")
        print("="*60)

        # Tokenize
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        def tokenize_data(texts):
            return tokenizer(
                texts, truncation=True, padding="max_length", max_length=120, return_tensors="pt"
            )

        train_enc = tokenize_data(train_texts)
        test_enc = tokenize_data(test_texts)
        
        # Save Tokenizer
        clean_name = get_clean_name(model_name)
        tokenizer.save_pretrained(os.path.join(SAVE_DIR, f"{clean_name}_tokenizer"))

        train_dataset = EMRDataset(train_enc, train_labels)
        test_dataset = EMRDataset(test_enc, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        # Note: shuffle=False is critical here so predictions match the list order of test_texts
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 1. Run Baseline
        train_baseline(model_name, train_loader, test_loader, device, test_texts)
        
        # 2. Run Contrastive Encoder
        encoder_path = train_contrastive_encoder(model_name, train_loader, device)
        
        # 3. Run Contrastive Last Layer
        train_contrastive_last_layer(model_name, encoder_path, train_loader, test_loader, device, test_texts)
        
        notification.notify(
            title='Training Complete for model '+model_name,
            message='The training process for '+model_name+' has finished successfully.',
            app_name='DL4H Training Script',
            timeout=10  # Notification will disappear after 10 seconds
        )

    print("\nAll training runs completed.")
    notification.notify(
        title='Training Complete',
        message='All model training processes have finished successfully.',
        app_name='DL4H Training Script',
        timeout=10
    )

if __name__ == "__main__":
    main()