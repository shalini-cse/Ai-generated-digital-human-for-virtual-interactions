from gtts import gTTS
from playsound import playsound
from langdetect import detect
import tempfile
import os

def speak(text):
    try:
        lang = detect(text)  # auto detect language (hi, en, te, etc.)
        print(f"🔍 Detected language for TTS: {lang}")
        tts = gTTS(text=text, lang=lang)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            temp_path = fp.name
            tts.save(temp_path)
        playsound(temp_path)
        os.remove(temp_path)
    except Exception as e:
        print("TTS Error:", e)
