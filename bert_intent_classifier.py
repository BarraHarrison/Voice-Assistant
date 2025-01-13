from transformers import BertTokenizer, BertForSequenceClassification
import torch
import speech_recognition


tokenizer = BertTokenizer.from_pretrained("bert_intent_model")
model = BertForSequenceClassification.from_pretrained("bert_intent_model")

with speech_recognition.Microphone() as mic:
    recognizer.adjust_for_ambient_noise(mic, duration=0.2)
    audio = recognizer.listen(mic)
    text = recognizer.recognize_google(audio)

inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
predictions = torch.softmax(outputs.logits, dim=1)
predicted_label = predictions.argmax().item()
# Labels ID to the Intent
intent = model.config.id2label[predicted_label]
print(f"Predicted Intent: {intent}")

intent_to_action = {
    "greeting": hello_function,
    "show_todos": show_todos,
    "create_todo": create_todo,
    "goodbye": quit_function
}

if intent in intent_to_action:
    intent_to_action[intent]()
else:
    speaker.say("I am not sure how to help with that.")
    speaker.runAndWait()