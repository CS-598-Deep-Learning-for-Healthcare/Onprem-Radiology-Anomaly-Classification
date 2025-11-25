import os
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, BertForSequenceClassification, AutoModel, AutoConfig
from tqdm.auto import tqdm
from dotenv import load_dotenv
from databricks import sql

# Import your custom loss
from document_level_kd.losses import SupConLoss

# -----------------------------
# Configuration
# -----------------------------
EPOCH_COUNT = 20
BATCH_SIZE = 16
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

    query = """
    SELECT
        subject_id, study_id, findings, impression, label, confidence
    FROM workspace.default.mimic_cxr_train_set_label_explanation_consensus_v1
    WHERE findings IS NOT NULL
      AND impression IS NOT NULL
      AND label IN ('Normal', 'Abnormal')
    """

    print("Querying data...")
    df = pd.read_sql(query, connection)
    connection.close()
    
    # Preprocessing
    df["Context"] = (df["findings"].fillna("") + " " + df["impression"].fillna("")).str.strip()
    df = df[df["Context"].str.len() > 0]
    
    label_map = {"Normal": 0, "Abnormal": 1}
    df["Result"] = df["label"].map(label_map)
    
    texts = df["Context"].tolist()
    labels = df["Result"].tolist()
    
    print(f"Data loaded. Rows: {len(df)}")
    return texts, labels

# -----------------------------
# 2. Shared Utilities & Classes
# -----------------------------

class EMRDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Allow accessing keys regardless of tokenizer output format
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
    """Sanitize model path for folder creation (e.g. 'org/model' -> 'org_model')"""
    return model_path.replace("/", "_")

# -----------------------------
# 3. Training Loops
# -----------------------------

def train_baseline(model_name, train_loader, device):
    """
    Standard Supervised Fine-Tuning
    """
    print(f"\n[BASELINE] Starting training for {model_name}...")
    
    # Initialize Model for Classification
    model = BertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2, 
        output_hidden_states=True,
        ignore_mismatched_sizes=True 
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
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}")

    # Save
    clean_name = get_clean_name(model_name)
    save_path = os.path.join(SAVE_DIR, f"{clean_name}_baseline")
    model.save_pretrained(save_path)
    print(f"[BASELINE] Saved to {save_path}")


def train_contrastive_encoder(model_name, train_loader, device):
    """
    Supervised Contrastive Learning (Training the Encoder only)
    """
    print(f"\n[CONTRASTIVE ENCODER] Starting training for {model_name}...")
    
    # Initialize Base Model (Not Classifier)
    config = AutoConfig.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, config=config)
    model.to(device)
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = SupConLoss(temperature=0.07) # From losses.py
    
    for epoch in range(EPOCH_COUNT):
        model.train()
        total_loss = 0
        loop = tqdm(train_loader, desc=f"Enc Epoch {epoch+1}")
        
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels') # Remove labels from input, keep for loss
            
            outputs = model(**batch)
            
            # Mean Pooling strategy (matches your script)
            # outputs[0] is last_hidden_state
            # shape: [bsz, seq_len, hidden_dim] -> mean -> [bsz, hidden_dim]
            embeddings = torch.mean(outputs[0], dim=1)
            
            # Normalize
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # SupConLoss expects [bsz, n_views, dim]. Since n_views=1:
            features = embeddings.unsqueeze(1)
            
            loss = criterion(features, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}")

    # Save
    clean_name = get_clean_name(model_name)
    save_path = os.path.join(SAVE_DIR, f"{clean_name}_contrastive_encoder")
    model.save_pretrained(save_path)
    print(f"[CONTRASTIVE ENCODER] Saved to {save_path}")
    return save_path


def train_contrastive_last_layer(original_model_name, encoder_path, train_loader, device):
    """
    Train MLP Classifier on top of Frozen Contrastive Encoder
    """
    print(f"\n[LAST LAYER] Training MLP on top of {encoder_path}...")
    
    # Load the Encoder trained in the previous step
    encoder = AutoModel.from_pretrained(encoder_path)
    encoder.to(device)
    
    # Initialize MLP
    # Auto-detect hidden size (usually 768, but 1024 for large models)
    hidden_size = encoder.config.hidden_size
    classifier = MLP(target_size=2, input_size=hidden_size)
    classifier.to(device)
    
    # OPTIMIZER: Only pass classifier parameters!
    # This implicitly "freezes" the encoder because we aren't updating its weights.
    optimizer = AdamW(classifier.parameters(), lr=LEARNING_RATE)
    loss_func = nn.CrossEntropyLoss()
    
    for epoch in range(EPOCH_COUNT):
        classifier.train()
        encoder.train() # Keep in train mode for dropout, but no grad update
        total_loss = 0
        loop = tqdm(train_loader, desc=f"MLP Epoch {epoch+1}")
        
        for batch in loop:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop('labels')
            
            # Forward pass through frozen encoder (no_grad for efficiency)
            with torch.no_grad():
                outputs = encoder(**batch)
                
                # Use Pooler output if available, else Mean Pool
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    # BERT/RoBERTa usually have this
                    features = outputs.pooler_output
                else:
                    # DistilBERT/Deberta might need mean pooling or CLS
                    # Using Mean Pool to be consistent with Encoder training
                    features = torch.mean(outputs.last_hidden_state, dim=1)

            # Forward pass through MLP
            logits = classifier(features)
            
            loss = loss_func(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        print(f"Epoch {epoch+1} Avg Loss: {total_loss/len(train_loader):.4f}")

    # Save ONLY the classifier state dict (or the whole thing if you prefer)
    # Here we save the state dict as requested in your snippet
    clean_name = get_clean_name(original_model_name)
    save_path = os.path.join(SAVE_DIR, f"{clean_name}_contrastive_classifier_mlp.pth")
    torch.save(classifier.state_dict(), save_path)
    print(f"[LAST LAYER] MLP Saved to {save_path}")

# -----------------------------
# 4. Main Execution
# -----------------------------
def main():
    # Load Data
    texts, labels = load_data_from_databricks()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_names = [
        "microsoft/deberta-v3-base", 
        "emilyalsentzer/Bio_ClinicalBERT", 
        "zzxslp/RadBERT-RoBERTa-4m"
    ]

    for model_name in model_names:
        print("="*60)
        print(f"PROCESSING MODEL: {model_name}")
        print("="*60)

        # Tokenize (Fresh for each model)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=120,
            return_tensors="pt"
        )
        
        # Save Tokenizer once per model family
        clean_name = get_clean_name(model_name)
        tokenizer_save_path = os.path.join(SAVE_DIR, f"{clean_name}_tokenizer")
        tokenizer.save_pretrained(tokenizer_save_path)

        dataset = EMRDataset(encodings, labels)
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 1. Run Baseline Training
        train_baseline(model_name, train_loader, device)
        
        # 2. Run Contrastive Encoder Training
        # We capture the path where it was saved to load it for step 3
        encoder_save_path = train_contrastive_encoder(model_name, train_loader, device)
        
        # 3. Run Contrastive Last Layer (MLP) Training
        train_contrastive_last_layer(model_name, encoder_save_path, train_loader, device)

    print("\nAll training runs completed.")

if __name__ == "__main__":
    main()