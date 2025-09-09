MODEL_PATH = r"C:/Users/Admin/Support_System/tickets/model"

import torch
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)

# Make sure model is in evaluation mode
model.eval()

# Define label mapping (must match your training labels)
LABELS = ["Billing", "Technical", "Account", "Other"]

def predict_category(text):
    """
    Predict ticket category using fine-tuned BERT classifier.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = predictions.argmax().item()

    return LABELS[predicted_class]
