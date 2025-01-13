from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

dataset = load_dataset('csv', data_files="intent_dataset.csv")
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