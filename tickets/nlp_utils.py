import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import joblib

# Load trained model + tokenizer + label encoder
model_path = "model"
tokenizer = DistilBertTokenizer.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)
le = joblib.load("label_encoder.pkl")

def predict_category(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    return le.inverse_transform([prediction])[0]
