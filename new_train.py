import os
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel, AutoConfig
from tqdm.auto import tqdm
from dotenv import load_dotenv
from databricks import sql
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# Import your custom loss
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
    
    # Print label distribution
    train_label_counts = pd.Series(train_labels).value_counts()
    test_label_counts = pd.Series(test_labels).value_counts()
    print(f"Train label distribution: {dict(train_label_counts)}")
    print(f"Test label distribution: {dict(test_label_counts)}")
    
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


def evaluate_model_detailed(model, dataloader, device, split_name, is_mlp=False, encoder=None, verbose=True):
    """
    Detailed evaluation with diagnostics for debugging.
    Returns metrics dict and additional diagnostic info.
    """
    model.eval()
    if encoder: 
        encoder.eval()
    
    all_preds = []
    all_labels = []
    all_logits = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            if is_mlp and encoder is not None:
                outputs = encoder(**batch)
                features = torch.mean(outputs.last_hidden_state, dim=1)
                features = F.normalize(features, p=2, dim=1)
                logits = model(features)
            else:
                outputs = model(**batch)
                logits = outputs.logits

            all_logits.append(logits.detach().cpu())
            batch_preds = logits.argmax(-1).detach().cpu().numpy()
            batch_labels = labels.detach().cpu().numpy()

            all_preds.extend(batch_preds)
            all_labels.extend(batch_labels)

    # Concatenate all logits for analysis
    all_logits = torch.cat(all_logits, dim=0).numpy()
    
    # Compute metrics
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    
    # Handle case where not all classes are predicted
    unique_preds = np.unique(all_preds)
    unique_labels = np.unique(all_labels)
    
    if len(unique_preds) < 2 or len(unique_labels) < 2:
        # Can't compute full confusion matrix
        cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    else:
        cm = confusion_matrix(all_labels, all_preds)
    
    # Safe extraction of confusion matrix values
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        if 0 in unique_preds and 0 in unique_labels:
            tn = cm[0, 0] if cm.shape[0] > 0 else 0

    # Diagnostic info
    pred_counts = pd.Series(all_preds).value_counts().to_dict()
    label_counts = pd.Series(all_labels).value_counts().to_dict()
    
    # Logit statistics
    logit_stats = {
        "logit_class0_mean": float(np.mean(all_logits[:, 0])),
        "logit_class0_std": float(np.std(all_logits[:, 0])),
        "logit_class1_mean": float(np.mean(all_logits[:, 1])),
        "logit_class1_std": float(np.std(all_logits[:, 1])),
        "logit_diff_mean": float(np.mean(all_logits[:, 1] - all_logits[:, 0])),  # positive means predicting class 1
    }

    if verbose:
        print(f"\n  {split_name} Results:")
        print(f"    Acc: {acc:.4f}, Prec: {precision:.4f}, Rec: {recall:.4f}, F1: {f1:.4f}")
        print(f"    Predictions: {pred_counts} | Actuals: {label_counts}")
        print(f"    Logits - Class0: {logit_stats['logit_class0_mean']:.3f}±{logit_stats['logit_class0_std']:.3f}, "
              f"Class1: {logit_stats['logit_class1_mean']:.3f}±{logit_stats['logit_class1_std']:.3f}")
        print(f"    Logit diff (class1-class0) mean: {logit_stats['logit_diff_mean']:.3f}")

    metrics = {
        "accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "pred_class0": pred_counts.get(0, 0),
        "pred_class1": pred_counts.get(1, 0),
    }
    metrics.update(logit_stats)

    return metrics


def save_epoch_metrics_to_csv(epoch_metrics_list, model_name, method_tag):
    """Saves per-epoch metrics to a CSV file."""
    clean_name = get_clean_name(model_name)
    filename = f"{clean_name}_{method_tag}_epoch_metrics.csv"
    filepath = os.path.join(SAVE_DIR, filename)
    
    df = pd.DataFrame(epoch_metrics_list)
    df.to_csv(filepath, index=False)
    print(f"Epoch metrics saved to {filepath}")


# -----------------------------
# 3. Training Loops
# -----------------------------

def train_baseline(model_name, train_loader, test_loader, device):
    """Standard Supervised Fine-Tuning with per-epoch metrics"""
    print(f"\n[BASELINE] Starting training for {model_name}...")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, ignore_mismatched_sizes=True
    )
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    
    epoch_metrics = []
    
    for epoch in range(EPOCH_COUNT):
        model.train()
        total_loss = 0
        batch_count = 0
        loop = tqdm(train_loader, desc=f"Base Epoch {epoch+1}")
        
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / batch_count
        
        # Evaluate on both train and test
        print(f"\n--- Epoch {epoch+1}/{EPOCH_COUNT} (Avg Loss: {avg_loss:.4f}) ---")
        train_metrics = evaluate_model_detailed(model, train_loader, device, "Train", verbose=True)
        test_metrics = evaluate_model_detailed(model, test_loader, device, "Test", verbose=True)
        
        # Store metrics
        row = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_pred_class0": train_metrics["pred_class0"],
            "train_pred_class1": train_metrics["pred_class1"],
            "test_acc": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_pred_class0": test_metrics["pred_class0"],
            "test_pred_class1": test_metrics["pred_class1"],
        }
        epoch_metrics.append(row)
    
    # Save epoch metrics
    save_epoch_metrics_to_csv(epoch_metrics, model_name, "baseline")

    # Save model
    clean_name = get_clean_name(model_name)
    save_path = os.path.join(SAVE_DIR, f"{clean_name}_baseline")
    model.save_pretrained(save_path)
    print(f"[BASELINE] Saved to {save_path}")


def train_contrastive_encoder(model_name, train_loader, device):
    """Supervised Contrastive Learning (Encoder only) with per-epoch loss tracking"""
    print(f"\n[CONTRASTIVE ENCODER] Starting training for {model_name}...")
    
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = SupConLoss(temperature=0.07)
    
    epoch_losses = []
    
    for epoch in range(EPOCH_COUNT):
        model.train()
        total_loss = 0
        batch_count = 0
        loop = tqdm(train_loader, desc=f"Enc Epoch {epoch+1}")
        
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            outputs = model(**batch)
            embeddings = torch.mean(outputs.last_hidden_state, dim=1)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            features = embeddings.unsqueeze(1)
            
            loss = criterion(features, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / batch_count
        epoch_losses.append({"epoch": epoch + 1, "contrastive_loss": avg_loss})
        print(f"  Epoch {epoch+1} Avg Contrastive Loss: {avg_loss:.4f}")

    # Save epoch losses
    clean_name = get_clean_name(model_name)
    loss_df = pd.DataFrame(epoch_losses)
    loss_df.to_csv(os.path.join(SAVE_DIR, f"{clean_name}_contrastive_encoder_losses.csv"), index=False)

    # Save model
    save_path = os.path.join(SAVE_DIR, f"{clean_name}_contrastive_encoder")
    model.save_pretrained(save_path)
    print(f"[CONTRASTIVE ENCODER] Saved to {save_path}")
    return save_path


def train_contrastive_last_layer(original_model_name, encoder_path, train_loader, test_loader, device):
    """Train MLP Classifier with per-epoch metrics on train and test"""
    print(f"\n[LAST LAYER] Training MLP on top of {encoder_path}...")
    
    # Load Encoder
    encoder = AutoModel.from_pretrained(encoder_path)
    encoder.to(device)
    encoder.eval()  # Keep frozen
    
    # Initialize MLP
    hidden_size = encoder.config.hidden_size
    print(f"  Encoder hidden size: {hidden_size}")
    
    classifier = MLP(target_size=2, input_size=hidden_size)
    classifier.to(device)
    
    # Print initial weights
    print(f"  Initial MLP weights - shape: {classifier.fc1.weight.shape}")
    print(f"  Initial MLP weight stats: mean={classifier.fc1.weight.mean().item():.4f}, std={classifier.fc1.weight.std().item():.4f}")
    print(f"  Initial MLP bias: {classifier.fc1.bias.data}")
    
    optimizer = AdamW(classifier.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()
    
    epoch_metrics = []
    
    # Pre-training evaluation
    print("\n--- Pre-training Evaluation ---")
    train_metrics = evaluate_model_detailed(classifier, train_loader, device, "Train (Pre)", is_mlp=True, encoder=encoder)
    test_metrics = evaluate_model_detailed(classifier, test_loader, device, "Test (Pre)", is_mlp=True, encoder=encoder)
    
    for epoch in range(EPOCH_COUNT):
        classifier.train()
        total_loss = 0
        batch_count = 0
        loop = tqdm(train_loader, desc=f"MLP Epoch {epoch+1}")
        
        # Track gradient stats
        grad_norms = []
        
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            with torch.no_grad():
                outputs = encoder(**batch)
                features = torch.mean(outputs.last_hidden_state, dim=1)
                features = F.normalize(features, p=2, dim=1)

            logits = classifier(features)
            loss = loss_func(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Track gradient norm
            grad_norm = classifier.fc1.weight.grad.norm().item()
            grad_norms.append(grad_norm)
            
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            loop.set_postfix(loss=loss.item())
        
        avg_loss = total_loss / batch_count
        avg_grad_norm = np.mean(grad_norms)
        
        # Evaluate on both train and test
        print(f"\n--- Epoch {epoch+1}/{EPOCH_COUNT} (Avg Loss: {avg_loss:.4f}, Avg Grad Norm: {avg_grad_norm:.4f}) ---")
        print(f"  MLP weight stats: mean={classifier.fc1.weight.mean().item():.4f}, std={classifier.fc1.weight.std().item():.4f}")
        print(f"  MLP bias: {classifier.fc1.bias.data}")
        
        train_metrics = evaluate_model_detailed(classifier, train_loader, device, "Train", is_mlp=True, encoder=encoder)
        test_metrics = evaluate_model_detailed(classifier, test_loader, device, "Test", is_mlp=True, encoder=encoder)
        
        # Store metrics
        row = {
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            "avg_grad_norm": avg_grad_norm,
            "mlp_weight_mean": classifier.fc1.weight.mean().item(),
            "mlp_weight_std": classifier.fc1.weight.std().item(),
            "mlp_bias_0": classifier.fc1.bias[0].item(),
            "mlp_bias_1": classifier.fc1.bias[1].item(),
            "train_acc": train_metrics["accuracy"],
            "train_f1": train_metrics["f1"],
            "train_precision": train_metrics["precision"],
            "train_recall": train_metrics["recall"],
            "train_pred_class0": train_metrics["pred_class0"],
            "train_pred_class1": train_metrics["pred_class1"],
            "train_logit_class0_mean": train_metrics["logit_class0_mean"],
            "train_logit_class1_mean": train_metrics["logit_class1_mean"],
            "train_logit_diff_mean": train_metrics["logit_diff_mean"],
            "test_acc": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_pred_class0": test_metrics["pred_class0"],
            "test_pred_class1": test_metrics["pred_class1"],
            "test_logit_class0_mean": test_metrics["logit_class0_mean"],
            "test_logit_class1_mean": test_metrics["logit_class1_mean"],
            "test_logit_diff_mean": test_metrics["logit_diff_mean"],
        }
        epoch_metrics.append(row)
    
    # Save epoch metrics
    save_epoch_metrics_to_csv(epoch_metrics, original_model_name, "contrastive_mlp")

    # Save classifier
    clean_name = get_clean_name(original_model_name)
    save_path = os.path.join(SAVE_DIR, f"{clean_name}_contrastive_classifier_mlp.pth")
    torch.save(classifier.state_dict(), save_path)
    print(f"[LAST LAYER] MLP Saved to {save_path}")


# -----------------------------
# 4. Main Execution
# -----------------------------
def main():
    train_texts, train_labels, test_texts, test_labels = load_data_from_databricks()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model_names = [
        "zzxslp/RadBERT-RoBERTa-4m",  # Fixed: added comma
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
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # 1. Run Baseline
        train_baseline(model_name, train_loader, test_loader, device)
        
        # 2. Run Contrastive Encoder
        encoder_path = train_contrastive_encoder(model_name, train_loader, device)
        
        # 3. Run Contrastive Last Layer
        train_contrastive_last_layer(model_name, encoder_path, train_loader, test_loader, device)

    print("\nAll training runs completed.")

if __name__ == "__main__":
    main()