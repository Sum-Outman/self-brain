import logging
import threading
import json
from datetime import datetime
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('EmotionEngine')

class EmotionEngine:
    """Emotion engine for analyzing and generating emotional responses"""
    
    def __init__(self):
        self.current_emotion = 'neutral'
        self.emotion_history = []
        self.emotion_intensity = 0.5  # 0.0 to 1.0
        self.emotion_memory = {}
        self.lock = threading.RLock()
        
        # Predefined emotions and their transition probabilities
        self.emotions = {
            'neutral': 0.3,  # Base probability
            'happy': 0.2,
            'sad': 0.15,
            'angry': 0.1,
            'surprised': 0.1,
            'fearful': 0.05,
            'disgusted': 0.05,
            'excited': 0.05
        }
        
        # Emotional response patterns
        self.response_patterns = {
            'happy': [
                "That's wonderful!",
                "I'm glad to hear that!",
                "This makes me happy too!"
            ],
            'sad': [
                "I'm sorry to hear that.",
                "That sounds difficult.",
                "I wish things were better for you."
            ],
            'angry': [
                "I understand why you're upset.",
                "That must be frustrating.",
                "Let's try to find a solution."
            ],
            'surprised': [
                "Wow, that's unexpected!",
                "Oh really? That's interesting!",
                "I didn't see that coming!"
            ],
            'neutral': [
                "I see.",
                "Interesting.",
                "Tell me more."
            ]
        }
    
    def analyze_emotion(self, text):
        """Analyze emotion in text"""
        # In a real implementation, this would use NLP techniques to analyze emotion
        # For now, we'll just return a random emotion based on probabilities
        
        # Convert text to lowercase for analysis
        text = text.lower()
        
        # Simple keyword-based emotion detection
        if any(word in text for word in ['happy', 'great', 'wonderful', 'exciting']):
            return {'emotion': 'happy', 'confidence': 0.8}
        elif any(word in text for word in ['sad', 'sorry', 'terrible', 'bad']):
            return {'emotion': 'sad', 'confidence': 0.8}
        elif any(word in text for word in ['angry', 'frustrated', 'mad', 'upset']):
            return {'emotion': 'angry', 'confidence': 0.8}
        elif any(word in text for word in ['surprised', 'wow', 'unexpected']):
            return {'emotion': 'surprised', 'confidence': 0.8}
        else:
            # Random emotion based on probabilities
            emotions = list(self.emotions.keys())
            probabilities = list(self.emotions.values())
            emotion = random.choices(emotions, weights=probabilities, k=1)[0]
            return {'emotion': emotion, 'confidence': 0.5}
    
    def generate_emotional_response(self, text, context=None):
        """Generate an emotional response to text"""
        # Analyze the emotion in the input text
        analysis = self.analyze_emotion(text)
        input_emotion = analysis['emotion']
        
        # Update current emotion based on input
        with self.lock:
            self._update_emotion(input_emotion)
            
            # Choose a response pattern based on current emotion
            patterns = self.response_patterns.get(self.current_emotion, self.response_patterns['neutral'])
            response = random.choice(patterns)
            
            # Log the emotion and response
            self.emotion_history.append({
                'timestamp': datetime.now().isoformat(),
                'input_text': text,
                'input_emotion': input_emotion,
                'current_emotion': self.current_emotion,
                'response': response
            })
            
            # Keep history to a reasonable size
            if len(self.emotion_history) > 100:
                self.emotion_history.pop(0)
            
            logger.debug(f"Generated emotional response: {response} (emotion: {self.current_emotion})")
            
        return {
            'response': response,
            'emotion': self.current_emotion,
            'intensity': self.emotion_intensity
        }
    
    def _update_emotion(self, new_emotion):
        """Update current emotion based on new input"""
        # In a real implementation, this would use a more sophisticated emotion transition model
        # For now, we'll just transition to the new emotion with some probability
        if random.random() < 0.7:  # 70% chance to transition
            self.current_emotion = new_emotion
            self.emotion_intensity = min(1.0, self.emotion_intensity + 0.1)
        else:
            # Gradually return to neutral
            self.emotion_intensity = max(0.5, self.emotion_intensity - 0.05)
    
    def get_current_emotion(self):
        """Get the current emotion"""
        with self.lock:
            return {
                'emotion': self.current_emotion,
                'intensity': self.emotion_intensity
            }
    
    def get_emotion_history(self, limit=10):
        """Get emotion history"""
        with self.lock:
            return self.emotion_history[-limit:]

# Global instance cache
_emotion_engine_instance = None

from flask import Blueprint, jsonify, request

# Create blueprint
emotion_bp = Blueprint('emotion', __name__)

@emotion_bp.route('/api/emotion/analyze', methods=['POST'])
def analyze_text_emotion():
    """Analyze emotion in text"""
    data = request.json
    text = data.get('text')
    
    if not text:
        return jsonify({'status': 'error', 'message': 'Missing text parameter'}), 400
    
    emotion_engine = get_emotion_engine()
    result = emotion_engine.analyze_emotion(text)
    
    return jsonify({
        'status': 'success',
        'result': result
    })

@emotion_bp.route('/api/emotion/generate_response', methods=['POST'])
def generate_response():
    """Generate an emotional response"""
    data = request.json
    text = data.get('text')
    context = data.get('context', {})
    
    if not text:
        return jsonify({'status': 'error', 'message': 'Missing text parameter'}), 400
    
    emotion_engine = get_emotion_engine()
    result = emotion_engine.generate_emotional_response(text, context)
    
    return jsonify({
        'status': 'success',
        'result': result
    })

@emotion_bp.route('/api/emotion/current', methods=['GET'])
def get_current():  # Renamed from get_current_emotion to avoid conflict
    """Get the current emotion"""
    emotion_engine = get_emotion_engine()
    result = emotion_engine.get_current_emotion()
    
    return jsonify({
        'status': 'success',
        'result': result
    })

def get_emotion_engine():
    """Get a singleton instance of EmotionEngine"""
    global _emotion_engine_instance
    if _emotion_engine_instance is None:
        _emotion_engine_instance = EmotionEngine()
        logger.info("EmotionEngine initialized")
    return _emotion_engine_instance