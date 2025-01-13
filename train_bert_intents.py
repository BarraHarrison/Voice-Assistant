from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, ClassLabel

dataset = load_dataset('csv', data_files="intent_dataset.csv")
dataset = dataset.map(lambda x: {"labels": x["intent"]}, batched=True)
intent_classes = dataset["train"].features["intent"] = ClassLabel(names=["greeting", "show_todos", "goodbye"])


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_data(examples):
    return tokenizer(examples['text'], padding=True, trunucation=True)

tokenized_dataset = dataset.map(preprocess_data, batched=True)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(dataset['train'].features['intent'].names))

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
)

trainer.train()
trainer.save_model("bert_intent_model")