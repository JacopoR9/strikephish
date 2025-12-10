import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm

# Number of epochs to run the training loop
NUMBER_OF_EPOCHS = 6


# Define metadata dataset
class MetaDataset(Dataset):

    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df["meta_text"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)
    
# Define body dataset
class BodyDataset(Dataset):

    def __init__(self, df, tokenizer, max_len=512):
        self.texts = df["body"].tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        text = smart_truncate(self.texts[idx], self.tokenizer, self.max_len)
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

    def __len__(self):
        return len(self.labels)
    
# Define the model
class BertClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits
    
# Freeze model lower layers (stability)
# Keep only last 2 encoder layers as trainable layers
def freeze_lower_layers(model):

    for name, param in model.bert.named_parameters():
        if not name.startswith("encoder.layer.11") and not name.startswith("encoder.layer.10"):
            param.requires_grad = False

# Merge metadata fields into one string
def build_meta_text(row):

    return (
        f"Sender: {row['sender']}\n"
        f"Receiver: {row['receiver']}\n"
        f"Date: {row['date']}\n"
        f"Subject: {row['subject']}"
    )

# Perform head-tail truncation
# (truncate email body by taking first and last 512 tokens)
def smart_truncate(text, tokenizer, max_len=512):

    tokens = tokenizer.tokenize(text)
    if len(tokens) <= max_len:
        return text
    kept_tokens = tokens[:256] + tokens[-256:]
    return tokenizer.convert_tokens_to_string(kept_tokens)

# Model training function
def train_model(model, train_loader, val_loader, epochs=NUMBER_OF_EPOCHS, lr=2e-5, accum_steps=1):

    freeze_lower_layers(model)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        
        # Training stage

        model.train()
        train_loss = 0
        preds, truths = [], []
        optimizer.zero_grad()

        for i, batch in enumerate(tqdm(train_loader)):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss = loss / accum_steps
            loss.backward()

            if (i + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            train_loss += loss.item() * input_ids.size(0) * accum_steps
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            truths.extend(labels.cpu().tolist( ))

        train_loss /= len(train_loader.dataset)
        train_acc = accuracy_score(truths, preds)
        train_f1 = f1_score(truths, preds)


        # Validation stage

        model.eval()
        val_loss = 0
        preds, truths = [], []

        with torch.no_grad():

            for batch in val_loader:

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                val_loss += loss.item() * input_ids.size(0)

                preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                truths.extend(labels.cpu().tolist())

        val_loss /= len(val_loader.dataset)
        val_acc = accuracy_score(truths, preds)
        val_f1 = f1_score(truths, preds)

        # Print progress data

        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")

    return model


# MAIN

# Define device for GPU accelleration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset file loading
df_global = pd.read_csv("CEAS_08.csv", quotechar='"', escapechar="\\")

# Replace null values
df_global = df_global.fillna("")

# Split dataset into metadata and body
# metadata => sender, receiver, date, subject, label
# body => body, label
df_meta = df_global[["sender", "receiver", "date", "subject", "label"]]
df_body = df_global[["body", "label"]]

# Reformat metadata dataframe with merged strings
df_meta["meta_text"] = df_meta.apply(build_meta_text, axis=1)
df_meta = df_meta[["meta_text", "label"]]

# Split datasets into training and validation
train_meta, val_meta = train_test_split(df_meta, test_size=0.2, stratify=df_meta["label"])
train_body, val_body = train_test_split(df_body, test_size=0.2, stratify=df_body["label"])

# Define BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Define metadata training and validation loaders
train_loader_meta = DataLoader(MetaDataset(train_meta, tokenizer), batch_size=32, shuffle=True)
val_loader_meta = DataLoader(MetaDataset(val_meta, tokenizer), batch_size=32)

# Define body training and validation loaders
train_loader_body = DataLoader(BodyDataset(train_body, tokenizer), batch_size=32, shuffle=True)
val_loader_body = DataLoader(BodyDataset(val_body, tokenizer), batch_size=32)

# Train and save metadata model
print("=============== TRAINING METADATA MODEL ===============")
meta_model = BertClassifier().to(device)
meta_model = train_model(meta_model, train_loader_meta, val_loader_meta)
torch.save(meta_model.state_dict(), "meta_model.pth")

# Train and save body model
print("=============== TRAINING BODY MODEL ===============")
body_model = BertClassifier().to(device)
body_model = train_model(body_model, train_loader_body, val_loader_body)
torch.save(body_model.state_dict(), "body_model.pth")

print("DONE!")
