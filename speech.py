import speech_recognition as sr
import pyttsx3

def recognize_speech():
    recognizer = sr.Recognizer()

    with sr.Microphone() as source:
        print("Say something:")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5)

    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        print("Could not understand audio.")
    except sr.RequestError as e:
        print(f"Google Speech Recognition request failed: {e}")

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if __name__ == "__main__":
    while True:
        # Recognize speech
        speech_text = recognize_speech()

        # If speech is recognized, print and speak the same text
        if speech_text:
            print(f"You said: {speech_text}")
            speak(speech_text)
