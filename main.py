import pyttsx3
from datetime import datetime
engine = pyttsx3.init()
engine.setProperty('rate', 150)
def speak(text):
    print("Assistant:", text)
    engine.say(text)
    engine.runAndWait()
def get_command():
    command = input("You: ")
    return command.lower()
def respond_to_command(command):
    if "hello" in command:
        speak("Hi there! How can I help you today?")
    elif "your name" in command:
        speak("I am your Python Voice Assistant")
    elif "time" in command:
        now = datetime.now().strftime("%H:%M")
        speak(f"The time is {now}")
    elif "date" in command:
        today = datetime.now().strftime("%d %B %Y")
        speak(f"Todays date is {today}")
    elif "exit" in command or "stop" in command:
        speak("Goodbye!")
        return False
    else:
        speak("I don't understand that command")
        return True
def main():
    speak("Assistant Activated")
    while True:
        command = get_command()
        if not respond_to_command(command):
            break
if __name__ == "__main__":
    main()