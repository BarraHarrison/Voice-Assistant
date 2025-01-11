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

    speaker.say("What todo do you want to create?")
    speaker.runAndWait()

    done = False
    while not done:
        try:
        
            with speech_recognition.Microphone() as mic:
                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)
                todo = recognizer.recognize_bing(audio)
                todo = todo.lower()

                speaker.say("Choose a file name!")
                speaker.runAndWait()

                recognizer.adjust_for_ambient_noise(mic, duration=0.2)
                audio = recognizer.listen(mic)

                filename = recognizer.recognize_bing(audio)
                filename = filename.lower()

            with open(filename, 'w') as f:
                f.write(todo)
                done = True
                speaker.say(f"I successfully created the todo {filename}")
                speaker.runAndWait()

        except speech_recognition.UnknownValueError:
            recognizer = speech_recognition.Recognizer()
            speaker.say("I did not understand you. Please try again.")
            speaker.runAndWait()


assistant = GenericAssistant('assistant_intents.json')
assistant.train_model()
