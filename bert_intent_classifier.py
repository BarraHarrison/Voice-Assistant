from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained("bert_intent_model")
model = BertForSequenceClassification.from_pretrained("bert_intent_model")

def predict_intent(text):
    """
    Predict the intent for a given text using the fine-tuned BERT model.
    """
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    predictions = torch.softmax(outputs.logits, dim=1)
    predicted_label = predictions.argmax().item()
    intent = model.config.id2label[predicted_label]
    print(f"Predicted Intent: {intent}")
    return intent



