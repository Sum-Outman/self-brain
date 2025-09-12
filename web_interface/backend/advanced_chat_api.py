from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import uuid
from datetime import datetime
import asyncio
import threading
import base64
import io
from PIL import Image
import cv2
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
import subprocess
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {
    'image': {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'},
    'video': {'mp4', 'avi', 'mov', 'wmv', 'flv', 'webm', 'mkv'},
    'document': {'pdf', 'txt', 'doc', 'docx', 'xls', 'xlsx', 'ppt', 'pptx'}
}

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'images'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'videos'), exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, 'documents'), exist_ok=True)

# Store conversation data
conversations = {}

# A Management Model Configuration
class AManagementModel:
    """A Management Model - Coordinates multiple internal AI models"""
    
    def __init__(self):
        self.models = {
            'language': 'LanguageModel',
            'vision': 'VisionModel', 
            'audio': 'AudioModel',
            'video': 'VideoModel',
            'document': 'DocumentModel'
        }
        
    def process_message(self, message, media=None, conversation_id=None):
        """Process user messages and coordinate relevant models"""
        
        response = {
            'success': True,
            'message': '',
            'models_used': [],
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        try:
            # Analyze message type and intent
            analysis = self.analyze_message(message, media)
            
            # Call appropriate models based on analysis
            if media:
                # Process multimedia content
                media_response = self.process_media(media, analysis)
                response['message'] = media_response['message']
                response['models_used'] = media_response['models_used']
            else:
                # Process plain text
                text_response = self.process_text(message, analysis)
                response['message'] = text_response['message']
                response['models_used'] = text_response['models_used']
                
            # Real-time voice/video processing
            if analysis.get('is_voice_request'):
                voice_response = self.process_voice(message)
                response['voice_data'] = voice_response
                
            response['processing_time'] = (datetime.now() - start_time).total_seconds()
            
        except Exception as e:
            logger.error(f"Failed to process message: {str(e)}")
            response = {
                'success': False,
                'message': 'Sorry, an error occurred while processing your request. Please try again later.',
                'error': str(e)
            }
            
        return response
    
    def analyze_message(self, message, media=None):
        """Analyze user message intent and type"""
        analysis = {
            'intent': 'general',
            'type': 'text',
            'confidence': 0.8,
            'is_voice_request': False,
            'is_visual_request': False
        }
        
        # Intent recognition
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['image', 'picture', 'photo', 'graph']):
            analysis['intent'] = 'image_analysis'
            analysis['is_visual_request'] = True
        elif any(word in message_lower for word in ['video', 'movie', 'film', 'clip']):
            analysis['intent'] = 'video_analysis'
            analysis['is_visual_request'] = True
        elif any(word in message_lower for word in ['voice', 'speech', 'audio', 'sound']):
            analysis['intent'] = 'voice_processing'
            analysis['is_voice_request'] = True
        elif any(word in message_lower for word in ['document', 'file', 'pdf', 'text']):
            analysis['intent'] = 'document_analysis'
        elif any(word in message_lower for word in ['translate', 'translation']):
            analysis['intent'] = 'translation'
        elif any(word in message_lower for word in ['summarize', 'summary', 'overview']):
            analysis['intent'] = 'summarization'
            
        # Media type analysis
        if media:
            analysis['type'] = media.get('type', 'unknown')
            
        return analysis
    
    def process_text(self, message, analysis):
        """Process plain text messages"""
        models_used = ['language_model']
        
        # Generate response based on intent
        intent_handlers = {
            'image_analysis': "I can help you analyze image content. Please upload an image, and I will provide a detailed description and analysis.",
            'video_analysis': "I can help you analyze video content. Please upload a video, and I will provide a summary and analysis.",
            'document_analysis': "I can help you process documents. Please upload a document, and I will extract key information and provide a summary.",
            'translation': "I can provide translation services. Please tell me the content to translate and the target language.",
            'summarization': "I can help you summarize content. Please provide the text or document to summarize.",
            'general': f"I understand your message: {message}. As an A Management Model, I will coordinate the most suitable internal AI models to provide you with accurate responses."
        }
        
        response_message = intent_handlers.get(
            analysis['intent'], 
            f"Received your message: {message}"
        )
        
        return {
            'message': response_message,
            'models_used': models_used
        }
    
    def process_media(self, media, analysis):
        """Process multimedia content"""
        models_used = []
        response_message = ""
        
        media_type = media.get('type')
        
        if media_type == 'image':
            models_used.extend(['vision_model', 'language_model'])
            response_message = f"Received image: {media.get('name', 'Unknown image')}. I will use vision AI models to analyze the image content, including object recognition, scene description, text extraction, etc."
            
        elif media_type == 'video':
            models_used.extend(['video_model', 'vision_model', 'language_model'])
            response_message = f"Received video: {media.get('name', 'Unknown video')}. I will use video AI models to analyze the video content, including keyframe extraction, action recognition, content summarization, etc."
            
        elif media_type == 'file':
            models_used.extend(['document_model', 'language_model'])
            response_message = f"Received document: {media.get('name', 'Unknown document')}. I will use document AI models to process the content, including text extraction, key information identification, content summarization, etc."
            
        return {
            'message': response_message,
            'models_used': models_used
        }
    
    def process_voice(self, message):
        """Process voice-related requests"""
        return {
            'type': 'voice_response',
            'text': 'Voice processing started',
            'audio_url': None  # This will generate the URL for voice response
        }

# Initialize A Management Model
a_model = AManagementModel()

def allowed_file(filename, file_type):
    """Check if file type is allowed"""
    if file_type in ALLOWED_EXTENSIONS:
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]
    return False

def process_image(file_path):
    """Process image file"""
    try:
        with Image.open(file_path) as img:
            # Generate thumbnail
            img.thumbnail((800, 600))
            
            # Save thumbnail
            thumb_path = file_path.replace('uploads/', 'uploads/thumbs/')
            os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
            img.save(thumb_path)
            
            return {
                'width': img.width,
                'height': img.height,
                'format': img.format,
                'mode': img.mode,
                'thumb_url': thumb_path
            }
    except Exception as e:
        logger.error(f"Failed to process image: {str(e)}")
        return None

def process_video(file_path):
    """Process video file"""
    try:
        cap = cv2.VideoCapture(file_path)
        
        # Get video information
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        # Extract first frame as thumbnail
        ret, frame = cap.read()
        if ret:
            thumb_path = file_path.replace('uploads/', 'uploads/thumbs/').replace('.mp4', '.jpg')
            os.makedirs(os.path.dirname(thumb_path), exist_ok=True)
            cv2.imwrite(thumb_path, frame)
            
        cap.release()
        
        return {
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'thumb_url': thumb_path
        }
    except Exception as e:
        logger.error(f"Failed to process video: {str(e)}")
        return None

def extract_text_from_document(file_path):
    """Extract text from document files"""
    try:
        file_extension = file_path.rsplit('.', 1)[1].lower()
        
        if file_extension == 'pdf':
            # Use pdfplumber or other PDF processing library
            import pdfplumber
            with pdfplumber.open(file_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                return text
                
        elif file_extension in ['txt', 'md']:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif file_extension in ['docx', 'doc']:
            # Use python-docx library
            from docx import Document
            doc = Document(file_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
            
    except Exception as e:
        logger.error(f"Failed to extract text from document: {str(e)}")
        return None

# API Routes

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get conversation list"""
    try:
        # Simulate conversation data
        conversations = [
            {
                'id': str(uuid.uuid4()),
                'title': 'Technical Consultation',
                'lastMessageTime': datetime.now().isoformat(),
                'messageCount': 15,
                'preview': 'Technical discussion about AI model deployment...'
            },
            {
                'id': str(uuid.uuid4()),
                'title': 'Image Analysis',
                'lastMessageTime': datetime.now().isoformat(),
                'messageCount': 8,
                'preview': 'Analyzed features of a product image...'
            },
            {
                'id': str(uuid.uuid4()),
                'title': 'Video Summary',
                'lastMessageTime': datetime.now().isoformat(),
                'messageCount': 12,
                'preview': 'Automatic summary of meeting video...'
            }
        ]
        
        return jsonify({
            'status': 'success',
            'conversations': conversations
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat messages"""
    try:
        data = request.json
        message = data.get('message', '')
        media = data.get('media')
        conversation_id = data.get('conversationId')
        
        if not message and not media:
            return jsonify({
                'status': 'error',
                'message': 'Message content cannot be empty'
            }), 400
        
        # Use A Management Model to process message
        response = a_model.process_message(message, media, conversation_id)
        
        if response['success']:
            return jsonify({
                'status': 'success',
                'message': {
                    'id': str(uuid.uuid4()),
                    'type': 'ai',
                    'content': response['message'],
                    'timestamp': datetime.now().isoformat(),
                    'models_used': response.get('models_used', []),
                    'processing_time': response.get('processing_time', 0)
                }
            })
        else:
            return jsonify({
                'status': 'error',
                'message': response['message']
            }), 500
            
    except Exception as e:
        logger.error(f"Chat API error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Internal server error'
        }), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'error',
                'message': 'No file was uploaded'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'error',
                'message': 'No file selected'
            }), 400
        
        # Determine file type
        file_type = None
        for type_name, extensions in ALLOWED_EXTENSIONS.items():
            if any(file.filename.lower().endswith('.' + ext) for ext in extensions):
                file_type = type_name
                break
        
        if not file_type:
            return jsonify({
                'status': 'error',
                'message': 'Unsupported file type'
            }), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4()}_{filename}"
        
        # Save file
        save_path = os.path.join(UPLOAD_FOLDER, file_type + 's', unique_filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)
        
        # Process file
        file_info = {
            'type': file_type,
            'name': filename,
            'size': os.path.getsize(save_path),
            'url': f"/uploads/{file_type}s/{unique_filename}",
            'original_path': save_path
        }
        
        # Additional processing based on file type
        if file_type == 'image':
            image_info = process_image(save_path)
            if image_info:
                file_info.update(image_info)
                
        elif file_type == 'video':
            video_info = process_video(save_path)
            if video_info:
                file_info.update(video_info)
                
        elif file_type == 'document':
            text_content = extract_text_from_document(save_path)
            if text_content:
                file_info['text_content'] = text_content[:1000]  # Limit to first 1000 characters
        
        return jsonify({
            'status': 'success',
            'message': 'File uploaded successfully',
            'file_info': file_info
        })
        
    except Exception as e:
        logger.error(f"File upload error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'File upload failed'
        }), 500

@app.route('/api/voice/process', methods=['POST'])
def process_voice():
    """Process voice data"""
    try:
        data = request.json
        audio_data = data.get('audio_data')
        
        if not audio_data:
            return jsonify({
                'status': 'error',
                'message': 'No audio data provided'
            }), 400
        
        # Decode base64 audio data
        audio_bytes = base64.b64decode(audio_data)
        
        # Use speech recognition
        recognizer = sr.Recognizer()
        audio_file = sr.AudioFile(io.BytesIO(audio_bytes))
        
        with audio_file as source:
            audio = recognizer.record(source)
        
        try:
            text = recognizer.recognize_google(audio, language='en-US')
            
            # Use A Management Model to process recognition result
            response = a_model.process_message(text)
            
            return jsonify({
                'status': 'success',
                'text': text,
                'response': response['message']
            })
            
        except sr.UnknownValueError:
            return jsonify({
                'status': 'error',
                'message': 'Unable to recognize speech'
            }), 400
        except sr.RequestError as e:
            return jsonify({
                'status': 'error',
                'message': 'Speech recognition service error'
            }), 500
            
    except Exception as e:
        logger.error(f"Voice processing error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Voice processing failed'
        }), 500

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_file(os.path.join(UPLOAD_FOLDER, filename))

@app.route('/api/health')
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': list(a_model.models.keys())
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
