# Voice Assistant in Python
from neuralintents import GenericAssistant
import speech_recognition 
import pyttsx3 as tts 
import sys

recognizer = speech_recognition.Recognizer()
speaker = tts.init()
speaker.setProperty('rate', 150) # Speech Speed

todo_list = ["Go Shopping", "Clean Room", "Study Korean"]

assistant = GenericAssistant('assistant_intents.json')
assistant.train_model()
assistant.request("Hello")