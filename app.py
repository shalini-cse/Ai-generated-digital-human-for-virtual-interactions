
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import os
import logging
import time
import threading
from queue import Queue

# Import deep-translator
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
    logging.info("✅ deep-translator imported successfully")
except ImportError:
    TRANSLATOR_AVAILABLE = False
    logging.warning("⚠️ deep-translator not available. Run: pip install deep-translator")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["JSON_AS_ASCII"] = False
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024
app.secret_key = os.environ.get("SECRET_KEY", "change-this-in-production")

CORS(app, resources={r"/*": {"origins": "*"}})

vision_sessions = {}

class VisionSession:
    def __init__(self, session_id, language):
        self.session_id = session_id
        self.language = language
        self.active = True
        self.queue = Queue(maxsize=5)
        self.thread = None
        self.last_activity = time.time()
    
    def stop(self):
        self.active = False

# ==================== TRANSLATION HELPERS ====================

LANGUAGE_MAP = {
    'en-US': 'en',
    'en': 'en',
    'hi-IN': 'hi',
    'hi': 'hi',
    'ta-IN': 'ta',
    'ta': 'ta',
    'te-IN': 'te',
    'te': 'te',
    'kn-IN': 'kn',
    'kn': 'kn',
    'ml-IN': 'ml',
    'ml': 'ml',
    'auto': 'auto'
}

def translate_text(text, source_lang='auto', target_lang='en'):
    """
    Translate text using deep-translator (GoogleTranslator)
    """
    if not TRANSLATOR_AVAILABLE:
        logger.warning("⚠️ Translator not available, returning original text")
        return text
    
    if not text or not text.strip():
        return text
    
    # Map language codes
    src = LANGUAGE_MAP.get(source_lang, source_lang)
    tgt = LANGUAGE_MAP.get(target_lang, target_lang)
    
    # No translation needed if same language
    if src == tgt and src != 'auto':
        return text
    
    try:
        logger.info(f"🌐 Translating [{src}] → [{tgt}]: '{text[:50]}...'")
        
        translator = GoogleTranslator(source=src, target=tgt)
        translated = translator.translate(text)
        
        logger.info(f"✅ Translation result: '{translated[:50]}...'")
        return translated
    
    except Exception as e:
        logger.error(f"❌ Translation error: {e}")
        return text  # Return original on error

# ==================== ROUTES ====================

@app.route("/")
def index():
    logger.info("📄 Serving froted.html")
    return render_template("froted.html")

@app.route("/health", methods=["GET"])
def health_check():
    cleanup_inactive_sessions()
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "active_vision_sessions": len(vision_sessions),
        "translator_available": TRANSLATOR_AVAILABLE
    })

# ==================== TRANSLATION API ====================

@app.route("/api/translate", methods=["POST"])
def api_translate():
    """
    Translate text between languages
    Request: { "text": "...", "source": "hi", "target": "en" }
    Response: { "translated_text": "..." }
    """
    try:
        data = request.get_json(force=True) or {}
        text = data.get("text", "")
        source_lang = data.get("source", "auto")
        target_lang = data.get("target", "en")
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        translated = translate_text(text, source_lang, target_lang)
        
        return jsonify({
            "translated_text": translated,
            "source_lang": source_lang,
            "target_lang": target_lang
        })
    
    except Exception as e:
        logger.exception("❌ Translation endpoint error")
        return jsonify({"error": str(e)}), 500

# ==================== PHI ENDPOINT WITH TRANSLATION ====================

@app.route("/phi", methods=["POST"])
def phi_endpoint():
    """✅ Main Phi AI chat endpoint with automatic translation"""
    start_time = time.time()
    
    logger.info("=" * 80)
    logger.info("📥 /phi ENDPOINT CALLED")
    
    try:
        data = request.get_json(force=True) or {}
        logger.info(f"✅ Received: {data}")
    except Exception as e:
        logger.error(f"❌ JSON parse error: {e}")
        return jsonify({
            "response": "Invalid request.",
            "emotion": "neutral",
            "emotion_intensity": 0.5
        }), 400
    
    try:
        from phi import ask_phi_with_emotion
        logger.info("✅ Phi module imported")
    except Exception as e:
        logger.exception("❌ Failed to import phi")
        return jsonify({
            "response": "AI module failed. Is Ollama running? (ollama serve)",
            "emotion": "neutral",
            "emotion_intensity": 0.5,
            "error": str(e)
        }), 500
    
    try:
        user_input = data.get("user_input") or data.get("message") or data.get("text") or ""
        lang = data.get("language", "en")
        
        # Normalize language code
        if isinstance(lang, str) and "-" in lang:
            lang = lang.split("-")[0]
        
        logger.info(f"🗣️ User [{lang}]: '{user_input}'")
        
        if not user_input:
            return jsonify({
                "response": "Please provide a message.",
                "emotion": "neutral",
                "emotion_intensity": 0.5
            }), 400
        
        # 🔄 TRANSLATE TO ENGLISH if not English
        original_input = user_input
        if lang != 'en':
            logger.info(f"🌐 Translating user input to English...")
            user_input = translate_text(user_input, source_lang=lang, target_lang='en')
            logger.info(f"✅ Translated: '{original_input}' → '{user_input}'")
        
        # 🤖 Get AI response in English
        logger.info(f"🤖 Calling ask_phi_with_emotion with English input...")
        reply, emotion, intensity = ask_phi_with_emotion(user_input, lang='en')
        logger.info(f"✅ AI Response (English): '{reply[:50]}...'")
        
        # 🔄 TRANSLATE BACK to user's language if not English
        if lang != 'en':
            logger.info(f"🌐 Translating AI response to {lang}...")
            reply = translate_text(reply, source_lang='en', target_lang=lang)
            logger.info(f"✅ Translated Response ({lang}): '{reply[:50]}...'")
        
        gesture = detect_gesture(original_input)
        
        elapsed = time.time() - start_time
        logger.info(f"⏱️ Completed in {elapsed:.2f}s")
        logger.info("=" * 80)
        
        return jsonify({
            "response": reply,
            "emotion": emotion,
            "emotion_intensity": intensity,
            "gesture": gesture
        })
        
    except Exception as e:
        logger.exception("❌ Error in /phi")
        return jsonify({
            "response": f"Error: {str(e)}",
            "emotion": "neutral",
            "emotion_intensity": 0.5,
            "error": str(e)
        }), 500

@app.route("/api/vision", methods=["POST"])
def api_vision():
    start_time = time.time()
    data = request.get_json(force=True) or {}
    
    logger.info(f"📥 /api/vision - Image: {bool(data.get('image_data'))}, Input: {data.get('user_input', '')[:30]}")
    
    try:
        from vision import vision_assistant_cycle
    except Exception as e:
        logger.exception("Failed to import vision")
        return jsonify({
            "response": "Vision module not available.",
            "emotion": "neutral",
            "emotion_intensity": 0.5,
            "detections": [],
            "error": str(e)
        }), 500
    
    try:
        result = vision_assistant_cycle(data)
        
        response_text = result.get("response") or "I couldn't process that."
        emotion = result.get("emotion", "neutral")
        intensity = result.get("emotion_intensity", 0.5)
        source = result.get("source", "unknown")
        objects_count = result.get("objects_count", 0)
        detections = result.get("detections", [])
        
        elapsed = time.time() - start_time
        logger.info(f"✅ /api/vision [{elapsed:.2f}s] - {source}, objects: {objects_count}")
        # Get language from request
        lang = data.get("language", "en")
        if isinstance(lang, str) and "-" in lang:
            lang = lang.split("-")[0]
        
        result = vision_assistant_cycle(data)
        
        response_text = result.get("response") or "I couldn't process that."
        
        # 🔄 Translate vision response if not English
        if lang != 'en':
            logger.info(f"🌐 Translating vision response to {lang}...")
            response_text = translate_text(response_text, source_lang='en', target_lang=lang)
        
        emotion = result.get("emotion", "neutral")
        intensity = result.get("emotion_intensity", 0.5)
        source = result.get("source", "unknown")
        objects_count = result.get("objects_count", 0)
        detections = result.get("detections", [])
        
        elapsed = time.time() - start_time
        logger.info(f"✅ /api/vision [{elapsed:.2f}s] - {source}, objects: {objects_count}")
        
        return jsonify({
            "response": response_text,
            "emotion": emotion,
            "emotion_intensity": intensity,
            "source": source,
            "objects_count": objects_count,
            "detections": detections
        })

        
    except Exception as e:
        logger.exception("Error in /api/vision")
        return jsonify({
            "response": "Vision error.",
            "emotion": "neutral",
            "emotion_intensity": 0.5,
            "detections": [],
            "error": str(e)
        }), 500

@app.route("/api/vision/start", methods=["POST"])
def api_vision_start():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id") or str(time.time())
    language = data.get("language", "en-US")
    
    logger.info(f"👁️ Starting vision - Session: {session_id}")
    
    if session_id in vision_sessions:
        vision_sessions[session_id].stop()
        del vision_sessions[session_id]
    
    session = VisionSession(session_id, language)
    vision_sessions[session_id] = session
    
    thread = threading.Thread(
        target=vision_monitoring_worker,
        args=(session,),
        daemon=True
    )
    thread.start()
    session.thread = thread
    
    return jsonify({
        "status": "started",
        "session_id": session_id,
        "message": "Vision monitoring active"
    })

@app.route("/api/vision/poll", methods=["POST"])
def api_vision_poll():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id")
    
    if not session_id or session_id not in vision_sessions:
        return jsonify({"status": "no_session", "detections": []}), 404
    
    session = vision_sessions[session_id]
    session.last_activity = time.time()
    
    if not session.queue.empty():
        detection = session.queue.get()
        
        # 🔄 Translate detection response if needed
        lang = session.language.split("-")[0] if isinstance(session.language, str) else "en"
        if lang != 'en' and 'response' in detection:
            detection['response'] = translate_text(
                detection['response'], 
                source_lang='en', 
                target_lang=lang
            )
        
        return jsonify(detection)
    
    return jsonify({"status": "no_data", "detections": []})

@app.route("/api/vision/stop", methods=["POST"])
def api_vision_stop():
    data = request.get_json(force=True) or {}
    session_id = data.get("session_id")
    
    if session_id and session_id in vision_sessions:
        vision_sessions[session_id].stop()
        del vision_sessions[session_id]
        logger.info(f"🛑 Vision stopped - Session: {session_id}")
        return jsonify({"status": "stopped"})
    
    return jsonify({"status": "not_found"}), 404

def vision_monitoring_worker(session: VisionSession):
    try:
        from vision import detect_objects_from_camera, create_natural_description
        from phi import ask_phi, detect_emotion_from_text
    except Exception as e:
        logger.error(f"Failed to import: {e}")
        return
    
    lang_code = session.language.split("-")[0] if isinstance(session.language, str) else "en"
    
    logger.info(f"🔄 Vision worker started [{lang_code}]")
    
    while session.active:
        try:
            items, error = detect_objects_from_camera()
            
            if error:
                result = {
                    'status': 'error',
                    'timestamp': time.time(),
                    'response': error,
                    'detections': [],
                    'objects_count': 0,
                    'emotion': 'neutral'
                }
            elif not items or len(items) == 0:
                result = {
                    'status': 'clear',
                    'timestamp': time.time(),
                    'response': 'Path clear.',
                    'detections': [],
                    'objects_count': 0,
                    'emotion': 'happy'
                }
            else:
                detections_list = [
                    {
                        "label": item["label"],
                        "confidence": round(item["confidence"], 2),
                        "position": f"{item['direction']} - {item['distance']}"
                    }
                    for item in items[:5]
                ]
                
                description = create_natural_description(items, lang='en')
                prompt = f"Describe this scene: {description}"
                response_text = ask_phi(prompt, lang='en')
                
                emotion, intensity = detect_emotion_from_text(response_text)
                
                result = {
                    'status': 'detection',
                    'timestamp': time.time(),
                    'response': response_text,
                    'detections': detections_list,
                    'objects_count': len(detections_list),
                    'emotion': emotion,
                    'emotion_intensity': intensity
                }
            
            if session.queue.full():
                try:
                    session.queue.get_nowait()
                except:
                    pass
            
            session.queue.put(result)
            logger.info(f"📤 Queued: {result.get('objects_count', 0)} objects")
            
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"❌ Vision worker error: {e}")
            time.sleep(5)
    
    logger.info(f"🛑 Vision worker ended")

def cleanup_inactive_sessions():
    current_time = time.time()
    inactive = []
    
    for session_id, session in vision_sessions.items():
        if current_time - session.last_activity > 300:
            inactive.append(session_id)
    
    for session_id in inactive:
        vision_sessions[session_id].stop()
        del vision_sessions[session_id]
        logger.info(f"🧹 Cleaned: {session_id}")

def detect_gesture(text: str) -> str:
    lower = text.lower()
    
    if any(word in lower for word in ["hi", "hello", "hey", "नमस्ते", "வணக்கம்", "నమస్కారం"]):
        return "wave"
    elif any(word in lower for word in ["yes", "yeah", "हां", "ஆம்", "అవును"]):
        return "nod"
    elif any(word in lower for word in ["no", "nope", "नहीं", "இல்லை", "కాదు"]):
        return "shake_head"
    elif any(word in lower for word in ["thank", "thanks", "धन्यवाद", "நன்றி", "ధన్యవాదాలు"]):
        return "gratitude"
    elif any(word in lower for word in ["?", "what", "how", "why", "क्या", "என்ன", "ఏమిటి"]):
        return "thinking"
    else:
        return "talk"

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.exception("500 error")
    return jsonify({"error": "Internal error"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    
    logger.info("=" * 60)
    logger.info("🚀 AI Digital Human Starting...")
    logger.info("=" * 60)
    logger.info(f"📍 Port: {port}")
    logger.info(f"🌐 Translator: {'✅ Available' if TRANSLATOR_AVAILABLE else '❌ Not Available'}")
    logger.info("📋 Endpoints:")
    logger.info("   GET  /               - Frontend")
    logger.info("   POST /phi            - AI chat (with auto-translation)")
    logger.info("   POST /api/translate  - Manual translation")
    logger.info("   POST /api/vision     - Vision")
    logger.info("=" * 60)
    
    # ✅ TEST PHI ON STARTUP
    try:
        logger.info("🧪 Testing Phi...")
        from phi import ask_phi_with_emotion
        test_reply, test_emotion, test_intensity = ask_phi_with_emotion("Hello", lang="en")
        logger.info(f"✅ Phi test OK: '{test_reply}'")
    except Exception as e:
        logger.error(f"❌ PHI TEST FAILED: {e}")
        logger.error("⚠️ Ensure Ollama is running: ollama serve")
        logger.error("⚠️ Ensure Phi downloaded: ollama pull phi")
    
    # ✅ TEST TRANSLATOR
    if TRANSLATOR_AVAILABLE:
        try:
            logger.info("🧪 Testing Translator...")
            test_translation = translate_text("Hello", source_lang='en', target_lang='hi')
            logger.info(f"✅ Translator test OK: 'Hello' → '{test_translation}'")
        except Exception as e:
            logger.error(f"❌ TRANSLATOR TEST FAILED: {e}")
    
    logger.info("=" * 60)
    
    app.run(host="0.0.0.0", port=port, debug=True, threaded=True)