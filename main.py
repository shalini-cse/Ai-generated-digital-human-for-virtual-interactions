from flask import Flask, request, jsonify, send_from_directory
import os
from app import ask_phi
from stt import listen_microphone
from tts import speak
from vision import detect_and_talk
import ollama

app = Flask(__name__, static_folder='static')

@app.route('/')
def serve_ui():
    return send_from_directory('static', 'froted.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        user_msg = request.json['message']
        print("🎤 User:", user_msg)
        reply = ask_phi(user_msg)
        speak(reply)
        return jsonify({
            "response": reply,
            "emotion": "neutral",
            "emotion_intensity": 0.5
        })
    except Exception as e:
        print("❌ Error in /api/chat:", e)
        return jsonify({"response": "Error occurred.", "emotion": "neutral", "emotion_intensity": 0.5})


@app.route('/api/vision', methods=['POST'])
def vision():
    try:
        result = detect_and_talk()
        return jsonify({"result": result})
    except Exception as e:
        print("❌ Error in /api/vision:", e)
        return jsonify({"result": "Vision error"})

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(debug=True)
