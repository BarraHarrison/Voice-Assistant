# Voice Assistant in Python
import os
import speech_recognition 
import pyttsx3 as tts 
import sys
import pickle
import numpy as np 
import tensorflow as tf


recognizer = speech_recognition.Recognizer()
speaker = tts.init()
speaker.setProperty('rate', 150) # Speech Speed

todo_list = ["Go Shopping", "Clean Room", "Study Korean"]

def create_todo():
    global recognizer

    speaker.say("What to-do do you want to create?")
    speaker.runAndWait()

    done = False
    while not done:
        try:
        
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)
                todo = recognizer.recognize_google(audio)
                todo = todo.lower()

                todo_list.append(todo)
                done = True

                speaker.say(f"I successfully added {todo} to the to-do list.")
                speaker.runAndWait()

        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            speaker.say("I did not understand you. Please try again.")
            speaker.runAndWait()

def show_todos():
    speaker.say("Here are the following to-dos on your to-do list")
    for todo in todo_list:
        speaker.say(todo)
    speaker.runAndWait()

def hello_function():
    speaker.say("Hello Barra. What can I do for you today?")
    speaker.runAndWait()

def quit_function():
    speaker.say("Bye Bye!")
    speaker.runAndWait()
    sys.exit(1)

model = tf.keras.models.load_model(model_file)

with open(tokenizer_file, "rb") as handle:
    tokenizer = pickle.load(handle)

with open(label_encoder_file, "rb") as handle:
    label_encoder = pickle.load(handle)

mappings_dictionary = {
    "greeting": hello_function,
    "create_todo": create_todo,
    "show_todos": show_todos,
    "exit": quit_function
}

while True:
    try:

        with speech_recognition.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=0.2)
            audio = recognizer.listen(mic)
            message = recognizer.recognize_google(audio)
            message = message.lower()

        assistant.request(message)

    except speech_recognition.UnknownValueError:
        recognizer = speech_recognition.Recognizer()
        speaker.say("I did not understand you. Please try again.")
        speaker.runAndWait()
