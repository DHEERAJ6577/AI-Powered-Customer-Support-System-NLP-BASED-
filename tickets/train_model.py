import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments

# 1. Load dataset
df = pd.read_csv("customer_support_tickets.csv")

# Encode categories as numbers
le = LabelEncoder()
df["label"] = le.fit_transform(df["Category"])

# 2. Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Ticket Description"].tolist(), df["label"].tolist(), test_size=0.2
)

# 3. Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

class TicketDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = TicketDataset(train_texts, train_labels)
val_dataset = TicketDataset(val_texts, val_labels)

# 4. Model
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=len(le.classes_)
)

# 5. Training
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()

# 6. Save model + label encoder
model.save_pretrained("tickets/model")
tokenizer.save_pretrained("tickets/model")
import joblib
joblib.dump(le, "tickets/label_encoder.pkl")
