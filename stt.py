import speech_recognition as sr

def listen_microphone():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎤 Speak now...")
        audio = recognizer.listen(source)
        try:
            # Use Indian English to support more accents, or change to "hi-IN", etc.
            text = recognizer.recognize_google(audio, language="en-IN")
            print("📝 You said:", text)
            return text
        except sr.UnknownValueError:
            return "Sorry, I could not understand."
        except sr.RequestError:
            return "Error accessing the Speech API."



###'https://models.readyplayer.me/68497e671a5c93881e8e6acd.glb'
