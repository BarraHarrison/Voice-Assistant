# Voice Assistant in Python
import speech_recognition 
import pyttsx3 as tts 
import sys
import pickle
import numpy as np 
import tensorflow as tf
from datetime import datetime


recognizer = speech_recognition.Recognizer()
speaker = tts.init()
speaker.setProperty('rate', 150) # Speech Speed

todo_list = ["Go Shopping", "Clean Room", "Study Korean"]


def hello_function():
    speaker.say("Hello Barra. What can I do for you today?")
    speaker.runAndWait()

def show_todos():
    speaker.say("Here are the following to-dos on your to-do list")
    for todo in todo_list:
        speaker.say(todo)
    speaker.runAndWait()

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


def say_name():
    speaker.say("I'm your personal assistant. You can call me MacBook Voice Assistant.")
    speaker.runAndWait()

def say_thanks():
    speaker.say(("You are welcome. Happy to help!"))
    speaker.runAndWait()

def give_time():
    current_time = datetime.now().strftime("%H:%M")
    speaker.say(f"The current time is {current_time}")
    speaker.runAndWait()

def quit_function():
    speaker.say("Bye Bye! Let me know if you need anything else.")
    speaker.runAndWait()
    sys.exit(1)



mappings_dictionary = {
    "greeting": hello_function,
    "goodbye": quit_function,
    "thanks": say_thanks,
    "create_todo": create_todo,
    "show_todos": show_todos,
    "name": say_name,
    "time": give_time,
    "exit": quit_function
}

while True:
    try:
        with speech_recognition.Microphone() as mic:
            recognizer.adjust_for_ambient_noise(mic, duration=0.2)
            audio = recognizer.listen(mic)
            message = recognizer.recognize_google(audio)
            message = message.lower()
            print(f"Recognized message: {message}")

        sequence = tokenizer.texts_to_sequences([message])
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=20)
        prediction = model.predict(padded_sequences)
        predicted_tag = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        print(f"Predicted Tag: {predicted_tag}")


        if predicted_tag in mappings_dictionary:
            mappings_dictionary[predicted_tag]()
        else:
            print(f"Unrecognized Tag: {predicted_tag}")
            speaker.say("I'm not sure how to help with that.")
            speaker.runAndWait()


    except speech_recognition.UnknownValueError:
        recognizer = speech_recognition.Recognizer()
        speaker.say("I did not understand you. Please try again.")
        speaker.runAndWait()
