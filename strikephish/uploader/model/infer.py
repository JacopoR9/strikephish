import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from pprint import pprint
import os

# Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN_META = 128
MAX_LEN_BODY = 512

# Define the model (same as in training)
class BertClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids = input_ids,
            attention_mask = attention_mask
        )
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits


# Create Emial dataset (single mail)
class EmailDataset(Dataset):
    def __init__(self, text, tokenizer, max_len):
        self.texts = [text]
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding = "max_length",
            truncation = True,
            max_length = self.max_len,
            return_tensors = "pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze()
        }
    

# Combine metadata fields into merged string
def build_meta_text(email_dict):
    return f"Sender: {email_dict['sender']}\nReceiver: {email_dict['receiver']}\nDate: {email_dict['datetime']}\nSubject: {email_dict['subject']}"


# Use the models to perform prediction
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
META_MODEL_PATH = os.path.join(BASE_DIR, "meta_model.pth")
BODY_MODEL_PATH = os.path.join(BASE_DIR, "body_model.pth")
def predict_email(email_dict):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Process text
    meta_text = build_meta_text(email_dict)
    body_text = email_dict["body"]

    # Build datasets
    meta_dataset = EmailDataset(meta_text, tokenizer, MAX_LEN_META)
    body_dataset = EmailDataset(body_text, tokenizer, MAX_LEN_BODY)

    # Build dataloaders
    meta_loader = DataLoader(meta_dataset, batch_size=1)
    body_loader = DataLoader(body_dataset, batch_size=1)

    # Load models and prepare for inference
    meta_model = BertClassifier().to(DEVICE)
    body_model = BertClassifier().to(DEVICE)
    meta_model.load_state_dict(torch.load(META_MODEL_PATH, map_location=DEVICE))
    body_model.load_state_dict(torch.load(BODY_MODEL_PATH, map_location=DEVICE))
    meta_model.eval()
    body_model.eval()

    # Inference for metadata model
    with torch.no_grad():
        for batch in meta_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            meta_logits = meta_model(input_ids, attention_mask)
            meta_probs = F.softmax(meta_logits, dim=1)
            meta_pred = torch.argmax(meta_probs, dim=1).item()
            meta_confidence = meta_probs[0, meta_pred].item()

    # Inference for body model
    with torch.no_grad():
        for batch in body_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            body_logits = body_model(input_ids, attention_mask)
            body_probs = F.softmax(body_logits, dim=1)
            body_pred = torch.argmax(body_probs, dim=1).item()
            body_confidence = body_probs[0, body_pred].item()

    # Process predictions
    combined_logits = (meta_logits + body_logits) / 2
    combined_probs = F.softmax(combined_logits, dim=1)
    combined_pred = torch.argmax(combined_probs, dim=1).item()
    combined_confidence = combined_probs[0, combined_pred].item()
    label_map = {0: "legitimate", 1: "phishing"}

    # Format results and return
    return {
        "meta_model_prediction": label_map[meta_pred],
        "meta_model_confidence": meta_confidence,
        "body_model_prediction": label_map[body_pred],
        "body_model_confidence": body_confidence,
        "combined_prediction": label_map[combined_pred],
        "combined_confidence": combined_confidence
    }