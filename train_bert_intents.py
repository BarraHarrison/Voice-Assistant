from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, ClassLabel

dataset = load_dataset('csv', data_files="intent_dataset.csv")
intent_classes = ClassLabel(names=["greeting", "show_todos", "goodbye"])
dataset = dataset.map(lambda x: {"labels": intent_classes.str2int(x["intent"])}, batched=True)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(intent_classes.names))

def preprocess_data(examples):
    return tokenizer(examples['text'], padding=True, trunucation=True)

tokenized_dataset = dataset.map(preprocess_data, batched=True)

if "validation" not in tokenized_dataset:
    tokenized_dataset = tokenized_dataset["train"].train_test_split(test_size=0.2)


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