import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_RETRIES = 1
TIMEOUT = 20

# ✅ TRY IMPORTING OLLAMA WITH ERROR HANDLING
try:
    import ollama
    OLLAMA_AVAILABLE = True
    logger.info("✅ Ollama library imported successfully")
except ImportError as e:
    OLLAMA_AVAILABLE = False
    logger.error(f"❌ Failed to import ollama: {e}")
    logger.error("Run: pip install ollama --upgrade")

def detect_emotion_from_text(text: str):
    """Detect emotion from text"""
    if not text:
        return "neutral", 0.5
    
    t = text.lower()
    
    # Happy
    if any(w in t for w in ["thank", "thanks", "great", "good", "happy", "glad", "nice", 
                             "धन्यवाद", "अच्छा", "நன்றி", "ధన్యవాదాలు", "ಧನ್ಯವಾದ", "നന്ദി"]):
        return "happy", 0.8
    
    # Sad
    if any(w in t for w in ["sorry", "sad", "unhappy", "खेद", "दुखी", "வருத்தம்", 
                             "క్షమించండి", "క్షమించి", "క్షమిక్కణం"]):
        return "sad", 0.7
    
    # Surprised
    if any(w in t for w in ["wow", "amazing", "surprise", "वाह", "ஆச்சரியம்", 
                             "ఆశ్చర్యం", "ಆಶ್ಚರ್ಯ", "ആശ്ചര്യം"]):
        return "surprised", 0.75
    
    # Curious
    if any(w in t for w in ["wonder", "curious", "think", "?", "जिज्ञासा", "ஆர்வம்", 
                             "ఉత్సుకత", "ಕುತೂಹಲ", "കൗതുകം"]):
        return "curious", 0.7
    
    return "neutral", 0.5

def ask_phi(message: str, lang: str = "en", timeout: int = TIMEOUT) -> str:
    """
    ✅ FIXED: Ask Phi AI with proper error handling
    """
    if not OLLAMA_AVAILABLE:
        logger.error("❌ Ollama library not available")
        return "AI service is not available. Please contact administrator."
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            logger.info(f"🧠 Phi request [{lang}] (attempt {attempt + 1}): {message[:50]}...")
            start_time = time.time()
            
            # Language instructions
            lang_instructions = {
                "hi": "You are a helpful AI assistant. Respond in Hindi (हिन्दी में). Be conversational and natural. Keep answer very brief (1-2 sentences max).",
                "ta": "You are a helpful AI assistant. Respond in Tamil (தமிழில்). Be conversational and natural. Keep answer very brief (1-2 sentences max).",
                "te": "You are a helpful AI assistant. Respond in Telugu (తెలుగులో). Be conversational and natural. Keep answer very brief (1-2 sentences max).",
                "kn": "You are a helpful AI assistant. Respond in Kannada (ಕನ್ನಡದಲ್ಲಿ). Be conversational and natural. Keep answer very brief (1-2 sentences max).",
                "ml": "You are a helpful AI assistant. Respond in Malayalam (മലയാളത്തിൽ). Be conversational and natural. Keep answer very brief (1-2 sentences max).",
                "en": "You are a helpful AI assistant. Respond in English. Be conversational, friendly and natural. Keep answer very brief (1-2 sentences max)."
            }
            
            system_prompt = lang_instructions.get(lang, lang_instructions["en"])
            
            logger.info(f"💬 Calling Ollama Phi...")
            
            # ✅ SIMPLE OLLAMA CALL - Uses default port 11434
            response = ollama.chat(
                model="phi",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                options={
                    "temperature": 0.8,
                    "num_predict": 80,
                    "top_k": 40,
                    "top_p": 0.9
                }
            )
            
            # Extract response
            reply = ""
            
            if isinstance(response, dict):
                if "message" in response and isinstance(response["message"], dict):
                    reply = response["message"].get("content", "")
                elif "response" in response:
                    reply = response["response"]
            elif hasattr(response, "message"):
                reply = response.message.content if hasattr(response.message, "content") else str(response.message)
            else:
                reply = str(response)
            
            reply = reply.strip()
            
            if not reply:
                raise ValueError("Empty response from Phi")
            
            # Limit to 2 sentences max
            sentences = [s.strip() for s in reply.replace('। ', '. ').split('. ') if s.strip()]
            if len(sentences) > 2:
                reply = '. '.join(sentences[:2]) + '.'
            
            elapsed = time.time() - start_time
            logger.info(f"✅ Phi response [{elapsed:.2f}s]: {reply[:60]}...")
            
            return reply
            
        except Exception as e:
            logger.error(f"❌ Phi error (attempt {attempt + 1}): {type(e).__name__}: {e}")
            if attempt < MAX_RETRIES:
                logger.info("🔄 Retrying...")
                time.sleep(1)
                continue
            else:
                # Final attempt failed
                logger.exception("❌ All retry attempts failed")
                break
    
    # Fallback error message
    error_messages = {
        "hi": "क्षमा करें, मुझे समस्या हो रही है। कृपया पुनः प्रयास करें।",
        "ta": "மன்னிக்கவும், எனக்கு சிக்கல் உள்ளது. மீண்டும் முயற்சிக்கவும்.",
        "te": "క్షమించండి, నాకు సమస్య ఉంది. మళ్లీ ప్రయత్నించండి.",
        "kn": "ಕ್ಷಮಿಸಿ, ನನಗೆ ಸಮಸ್ಯೆ ಇದೆ. ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
        "ml": "ക്ഷമിക്കണം, എനിക്ക് പ്രശ്നമുണ്ട്. വീണ്ടും ശ്രമിക്കുക.",
        "en": "Sorry, I'm having trouble. Please try again."
    }
    return error_messages.get(lang, error_messages["en"])

def translate_to_english(text, source_lang):
    """Translate to English"""
    if source_lang == "en":
        return text
    
    try:
        logger.info(f"🌐 Translating [{source_lang}] → [en]")
        
        response = ollama.chat(
            model="phi",
            messages=[
                {"role": "system", "content": "Translate to English. Only output the translation."},
                {"role": "user", "content": text}
            ],
            options={"temperature": 0.3, "num_predict": 100}
        )
        
        if isinstance(response, dict) and "message" in response:
            return response["message"].get("content", "").strip() or text
        return text
    except:
        return text

def translate_from_english(text, target_lang):
    """Translate from English"""
    if target_lang == "en":
        return text
    
    try:
        logger.info(f"🌐 Translating [en] → [{target_lang}]")
        
        lang_names = {"hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam"}
        
        response = ollama.chat(
            model="phi",
            messages=[
                {"role": "system", "content": f"Translate to {lang_names.get(target_lang, target_lang)}. Only output the translation."},
                {"role": "user", "content": text}
            ],
            options={"temperature": 0.3, "num_predict": 150}
        )
        
        if isinstance(response, dict) and "message" in response:
            return response["message"].get("content", "").strip() or text
        return text
    except:
        return text

def ask_phi_with_emotion(message: str, lang: str = "en"):
    """✅ Ask Phi with translation support"""
    try:
        # Translate to English if needed
        english_msg = translate_to_english(message, lang) if lang != "en" else message
        
        logger.info(f"🤖 Phi (English): '{english_msg[:50]}...'")
        
        # Get response in English
        reply = ask_phi(english_msg, lang="en")
        
        # Translate back to user's language
        if lang != "en":
            reply = translate_from_english(reply, lang)
            logger.info(f"✅ Translated [{lang}]: '{reply}'")
        
        emotion, intensity = detect_emotion_from_text(reply)
        return reply, emotion, intensity
        
    except Exception as e:
        logger.exception(f"❌ Error: {e}")
        
        error_messages = {
            "hi": "मुझे तकनीकी समस्या है।",
            "ta": "எனக்கு தொழில்நுட்ப சிக்கல் உள்ளது.",
            "te": "నాకు సాంకేతిక సమస్య ఉంది.",
            "kn": "ನನಗೆ ತಾಂತ್ರಿಕ ಸಮಸ್ಯೆ ಇದೆ.",
            "ml": "എനിക്ക് സാങ്കേതിക പ്രശ്നമുണ്ട്.",
            "en": "Technical difficulty."
        }
        return error_messages.get(lang, error_messages["en"]), "neutral", 0.5

# ✅ LANGUAGE INSTRUCTIONS (for vision.py)
LANGUAGE_INSTRUCTIONS = {
    "en": "Respond in English.",
    "hi": "Respond in Hindi (हिन्दी में).",
    "ta": "Respond in Tamil (தமிழில்).",
    "te": "Respond in Telugu (తెలుగులో).",
    "kn": "Respond in Kannada (ಕನ್ನಡದಲ್ಲಿ).",
    "ml": "Respond in Malayalam (മലയാളത്തിൽ)."
}

if __name__ == "__main__":
    print("🧪 Testing Phi AI...\n")
    
    # Check if ollama library is available
    if not OLLAMA_AVAILABLE:
        print("❌ Ollama library not installed or incompatible")
        print("\n🔧 SOLUTION:")
        print("   pip uninstall ollama")
        print("   pip install ollama")
        exit(1)
    
    # Test Ollama connection
    print("=" * 70)
    print("TEST 1: Checking Ollama connection...")
    try:
        models = ollama.list()
        print("✅ Ollama is running")
        print(f"📦 Available models:")
        if hasattr(models, 'models'):
            for model in models.models:
                print(f"   - {model.model if hasattr(model, 'model') else model}")
        else:
            print(f"   {models}")
    except Exception as e:
        print(f"❌ Ollama connection failed: {type(e).__name__}: {e}")
        print("\n🔧 SOLUTION:")
        print("   1. Check if Ollama is running: ollama list")
        print("   2. If you see 'port in use' error, Ollama IS running")
        print("   3. Test directly: ollama run phi 'Hello'")
        exit(1)
    
    # Test English
    print("\n" + "=" * 70)
    print("TEST 2: English query")
    try:
        start = time.time()
        reply, emotion, intensity = ask_phi_with_emotion("Hello, how are you?", lang="en")
        elapsed = time.time() - start
        print(f"✅ Response ({elapsed:.2f}s): {reply}")
        print(f"   Emotion: {emotion} ({intensity})")
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")
    
    # Test Hindi
    print("\n" + "=" * 70)
    print("TEST 3: Hindi query")
    try:
        start = time.time()
        reply, emotion, intensity = ask_phi_with_emotion("नमस्ते, आप कैसे हैं?", lang="hi")
        elapsed = time.time() - start
        print(f"✅ Response ({elapsed:.2f}s): {reply}")
        print(f"   Emotion: {emotion} ({intensity})")
    except Exception as e:
        print(f"❌ Failed: {type(e).__name__}: {e}")
    
    # Test unique answers
    print("\n" + "=" * 70)
    print("TEST 4: Testing unique answers (same question 3 times)")
    for i in range(3):
        try:
            start = time.time()
            reply, _, _ = ask_phi_with_emotion("What is AI?", lang="en")
            elapsed = time.time() - start
            print(f"   Answer {i+1} ({elapsed:.2f}s): {reply[:80]}...")
        except Exception as e:
            print(f"   Answer {i+1}: Failed - {e}")
    
    print("\n" + "=" * 70)
    print("✅ Testing complete!")
    print("If all tests passed, the Phi AI integration is working correctly.")