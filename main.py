# Voice Assistant in Python
from neuralintents import GenericAssistant
import speech_recognition 
import pyttsx3 as tts 
import sys

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
                todo = recognizer.recognize_bing(audio)
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
    sys.exit()

assistant = GenericAssistant('assistant_intents.json')
assistant.train_model()
