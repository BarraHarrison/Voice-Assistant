# Python Voice Assistant
This Python project aims to combine natural language understanding and voice recognition to create a voice assistant

# Feature
- Speech Recognition: Converts voice input using speech_recognition library
- Intent Classification: Classifies user intent using BERT
- Dynamic Responses: Specific responses based on the classified intent
- Enhancable: New intents and responses can be created

# Project Structure
- main.py: Entry point of the voice assistant. Integrates speech recognition, intent classification and action handling.

- train_model.py: Initially used to train the assistant_intents.json data before I decided to use BERT. Worked well!

- bert_intent_classifier.py: Contains the logic for loading the fine-tuned BERT model and predicting intents.

- train_bert_intents.py: Script to fine-tune (train) the BERT model

- intent_dataset.csv: Responses and patterns which replaced the assistant_intents.json once I switched to BERT

- bert_intent_model: Directory where the trained BERT model is stored

# Challenges Faced
- Adapting the BERT model for intent classification which caused problems with tokenization and intent labelling

- Dependency conflicts with the libraries 'transformers' and 'accelerate'.

- Inconsistencies in the intent_dataset.csv

- The usual errors seemed to pop up during model training, things like invalid labelling.
