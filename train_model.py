import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pickle

with open("assistant_intents.json") as file:
    data = json.load(file)

patterns = []
tags = []
responses = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])
    responses[intent["tag"]] = intent["responses"]

# Encode tags
label_encoder = LabelEncoder()
tags_encoded = label_encoder.fit_transform(tags)

tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(patterns)
sequences = tokenizer.texts_to_sequences(patterns)
padded_sequences = tf.keras.prepocessing.sequence.pad_sequences(sequences, maxlen=20)

# Model definition
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=128, activation='relu', input_dim=padded_sequences.shape[1]))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=len(set(tags)), activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(padded_sequences, np.array(tags_encoded), epochs=200, verbose=1)

model.save("basic_model.keras")

with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("label_encoder.pickle", "wb") as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model, tokenizer, and label encoder saved successfully!")