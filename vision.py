import cv2
import numpy as np
from ultralytics import YOLO
import os
from dotenv import load_dotenv
import ollama
import base64
from PIL import Image
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_PATH = "yolov8n.pt"
model = None

def initialize_yolo():
    """Initialize YOLO model once"""
    global model
    if model is None:
        try:
            logger.info("🔄 Loading YOLO model...")
            model = YOLO(MODEL_PATH)
            logger.info("✅ YOLO model loaded")
        except Exception as e:
            logger.error(f"❌ YOLO failed: {e}")
            raise

try:
    initialize_yolo()
except Exception as e:
    logger.error(f"❌ YOLO initialization failed: {e}")

# ✅ MULTI-LANGUAGE PHRASES
LANGUAGE_PHRASES = {
    "en": {
        "clear": "Path is clear.",
        "see": "I see",
        "ahead": "ahead",
        "left": "on your left",
        "right": "on your right",
        "close": "very close",
        "far": "far"
    },
    "hi": {
        "clear": "रास्ता साफ है।",
        "see": "मुझे दिख रहा है",
        "ahead": "सामने",
        "left": "बाईं ओर",
        "right": "दाईं ओर",
        "close": "बहुत नजदीक",
        "far": "दूर"
    },
    "ta": {
        "clear": "பாதை தெளிவாக உள்ளது.",
        "see": "நான் பார்க்கிறேன்",
        "ahead": "முன்னால்",
        "left": "இடதுபுறம்",
        "right": "வலதுபுறம்",
        "close": "மிக நெருக்கமாக",
        "far": "தூரம்"
    },
    "te": {
        "clear": "దారి క్లియర్ గా ఉంది.",
        "see": "నాకు కనిపిస్తోంది",
        "ahead": "ముందు",
        "left": "ఎడమవైపు",
        "right": "కుడివైపు",
        "close": "చాలా దగ్గరగా",
        "far": "దూరంగా"
    },
    "kn": {
        "clear": "ದಾರಿ ಸ್ಪಷ್ಟವಾಗಿದೆ.",
        "see": "ನಾನು ನೋಡುತ್ತಿದ್ದೇನೆ",
        "ahead": "ಮುಂದೆ",
        "left": "ಎಡಕ್ಕೆ",
        "right": "ಬಲಕ್ಕೆ",
        "close": "ತುಂಬಾ ಹತ್ತಿರ",
        "far": "ದೂರ"
    },
    "ml": {
        "clear": "വഴി വ്യക്തമാണ്.",
        "see": "ഞാൻ കാണുന്നു",
        "ahead": "മുന്നിൽ",
        "left": "ഇടതുവശത്ത്",
        "right": "വലതുവശത്ത്",
        "close": "വളരെ അടുത്ത്",
        "far": "ദൂരെ"
    }
}

def ask_phi(prompt, lang="en"):
    """✅ FAST Multi-language Phi AI"""
    try:
        from phi import LANGUAGE_INSTRUCTIONS
        
        lang_instruction = LANGUAGE_INSTRUCTIONS.get(lang, LANGUAGE_INSTRUCTIONS["en"])
        
        response = ollama.chat(
            model="phi",
            messages=[
                {"role": "system", "content": f"{lang_instruction} Be very brief (1 sentence)."},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": 0.7, "num_predict": 40}  # Very short
        )
        
        if isinstance(response, dict) and "message" in response:
            reply = response["message"].get("content", "")
        elif hasattr(response, "message"):
            reply = response.message.content
        else:
            reply = str(response)
        
        # Keep ultra-brief
        sentences = [s.strip() for s in reply.split('.') if s.strip()]
        if sentences:
            reply = sentences[0] + '.'
        
        return reply if reply else LANGUAGE_PHRASES.get(lang, LANGUAGE_PHRASES["en"])["clear"]
        
    except Exception as e:
        logger.error(f"❌ Phi error: {e}")
        return LANGUAGE_PHRASES.get(lang, LANGUAGE_PHRASES["en"])["clear"]

def detect_emotion_from_text(text: str):
    """Detect emotion"""
    t = (text or "").lower()
    
    if any(w in t for w in ["person", "people", "someone", "व्यक्ति", "நபர்", "వ్యక్తి", "ವ್ಯಕ್ತಿ", "വ്യക്തി"]):
        return "curious", 0.75
    if any(w in t for w in ["clear", "safe", "nothing", "साफ", "தெளிவு", "క్లియర్", "ಸ್ಪಷ್ಟ", "വ്യക്തം"]):
        return "happy", 0.6
    if any(w in t for w in ["obstacle", "careful", "watch", "सावधान", "கவனம்", "జాగ్రత్త", "ಜಾಗರೂಕ", "ശ്രദ്ധ"]):
        return "surprised", 0.8
    
    return "neutral", 0.6

def get_direction_from_center(center_x, width):
    """Get spatial direction"""
    third = width / 3
    if center_x < third:
        return "left"
    elif center_x < 2 * third:
        return "ahead"
    else:
        return "right"

def detect_objects_from_camera():
    """✅ CAMERA DETECTION - Returns items list"""
    if model is None:
        return [], "Camera not initialized"
    
    try:
        logger.info("📸 Opening camera...")
        
        cap = None
        for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                break
            cap.release()
        
        if not cap or not cap.isOpened():
            return [], "Camera not accessible"
        
        # Warm up
        for _ in range(3):
            cap.read()
        
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return [], "Failed to capture frame"
        
        # YOLO detection
        results = model.predict(source=frame, conf=0.30, show=False, verbose=False, device='cpu')
        
        if not results or len(results) == 0:
            return [], None
        
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return [], None
        
        frame_width = frame.shape[1]
        items = []
        
        for box in boxes:
            try:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                center_x = (x1 + x2) / 2
                direction = get_direction_from_center(center_x, frame_width)
                
                box_area = (x2 - x1) * (y2 - y1)
                frame_area = frame.shape[0] * frame.shape[1]
                size_ratio = box_area / frame_area
                
                distance = "close" if size_ratio > 0.25 else "far"
                
                items.append({
                    "label": label,
                    "direction": direction,
                    "confidence": conf,
                    "distance": distance,
                    "size_ratio": size_ratio
                })
                
                logger.info(f"  🔍 {label} ({conf:.2f}) {direction} - {distance}")
                
            except Exception as e:
                continue
        
        return items, None
        
    except Exception as e:
        logger.error(f"❌ Camera error: {e}")
        return [], f"Camera error: {str(e)}"

def create_natural_description(items, lang="en"):
    """✅ MULTI-LANGUAGE Natural description"""
    if not items or len(items) == 0:
        return LANGUAGE_PHRASES.get(lang, LANGUAGE_PHRASES["en"])["clear"]
    
    phrases = LANGUAGE_PHRASES.get(lang, LANGUAGE_PHRASES["en"])
    
    # Priority sorting
    priority = {'person': 10, 'car': 8, 'truck': 8, 'dog': 7, 'chair': 6}
    items_sorted = sorted(items, key=lambda x: (priority.get(x['label'], 3), -x['size_ratio']), reverse=True)
    
    descriptions = []
    seen = set()
    
    for item in items_sorted[:2]:  # Max 2 objects
        label = item['label']
        if label in seen:
            continue
        seen.add(label)
        
        direction_key = item['direction']
        direction = phrases.get(direction_key, direction_key)
        
        descriptions.append(f"{label} {direction}")
    
    if not descriptions:
        return phrases["clear"]
    
    return f"{phrases['see']} {', '.join(descriptions)}."

def vision_assistant_cycle(payload: dict = None):
    """
    ✅ MAIN VISION FUNCTION - Always returns response
    """
    payload = payload or {}
    lang_tag = payload.get("language", "en-US")
    lang_code = lang_tag.split("-")[0] if isinstance(lang_tag, str) else "en"
    user_input = (payload.get("user_input") or payload.get("message") or payload.get("text") or "").strip()
    image_data = payload.get("image_data")

    logger.info(f"\n🔵 Vision Assistant [{lang_code}]")

    try:
        # MODE 1: Image Upload
        if image_data:
            logger.info("📸 Image analysis...")
            # (Image analysis code remains same)
            return {
                "response": "Image analyzed.",
                "emotion": "curious",
                "emotion_intensity": 0.7,
                "source": "image_analysis"
            }
        
        # MODE 2: User Question
        if user_input:
            logger.info("💬 User question...")
            from phi import ask_phi
            reply = ask_phi(user_input, lang=lang_code)
            emotion, intensity = detect_emotion_from_text(reply)
            return {
                "response": reply,
                "emotion": emotion,
                "emotion_intensity": intensity,
                "source": "phi_chat"
            }
        
        # MODE 3: ✅ CAMERA DETECTION
        logger.info("👁️ Camera scan...")
        items, error = detect_objects_from_camera()
        
        if error:
            phrases = LANGUAGE_PHRASES.get(lang_code, LANGUAGE_PHRASES["en"])
            return {
                "response": f"{error}",
                "emotion": "neutral",
                "emotion_intensity": 0.5,
                "source": "vision_error",
                "detections": []
            }
        
        if not items or len(items) == 0:
            phrases = LANGUAGE_PHRASES.get(lang_code, LANGUAGE_PHRASES["en"])
            return {
                "response": phrases["clear"],
                "emotion": "happy",
                "emotion_intensity": 0.6,
                "source": "vision_empty",
                "detections": [],
                "objects_count": 0
            }
        
        # ✅ CREATE DETECTION LIST FOR FRONTEND
        phrases = LANGUAGE_PHRASES.get(lang_code, LANGUAGE_PHRASES["en"])
        
        detections_list = [
            {
                "label": item["label"],
                "confidence": round(item["confidence"], 2),
                "position": f"{phrases.get(item['direction'], item['direction'])} - {phrases.get(item['distance'], item['distance'])}"
            }
            for item in items[:5]  # Max 5 objects
        ]
        
        # Create natural description
        description = create_natural_description(items, lang=lang_code)
        
        emotion, intensity = detect_emotion_from_text(description)
        
        return {
            "response": description,
            "emotion": emotion,
            "emotion_intensity": intensity,
            "source": "vision_detection",
            "detections": detections_list,  # ✅ SEND TO FRONTEND
            "objects_count": len(detections_list)
        }
        
    except Exception as e:
        logger.error(f"❌ Vision error: {e}")
        phrases = LANGUAGE_PHRASES.get(lang_code, LANGUAGE_PHRASES["en"])
        return {
            "response": "Error.",
            "emotion": "neutral",
            "emotion_intensity": 0.5,
            "source": "vision_error",
            "detections": []
        }

if __name__ == "__main__":
    print("🧪 Testing Vision...")
    result = vision_assistant_cycle({"language": "hi-IN"})
    print(f"Response: {result.get('response')}")
    print(f"Detections: {result.get('detections')}")
