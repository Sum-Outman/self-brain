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

# Import the external API configuration module
from external_api_config import get_external_api_config

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
    """A Management Model that handles various types of messages and coordinates with other models"""
    
    def __init__(self):
        # Initialize logger
        self.logger = logging.getLogger('AManagementModel')
        
        # Initialize conversation history storage
        self.conversation_history = {}
        
        # Initialize model components
        self.initialize_model_components()
        
        # Initialize external API config
        self.api_config = get_external_api_config()
        
        # Initialize task queue and processing thread
        self.task_queue = []
        self.processing_thread = threading.Thread(target=self._process_task_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Initialize knowledge base searcher
        self.knowledge_searcher = None
        
    def initialize_model_components(self):
        """Initialize the various model components"""
        # This would be where you initialize actual model components
        # For now, we'll just log that initialization is happening
        self.logger.info("Initializing model components...")
        
        # In a real implementation, you would load actual model weights here
        # For example:
        # self.language_model = load_language_model()
        # self.vision_model = load_vision_model()
        # etc.
        
        self.logger.info("Model components initialized successfully")
    
    def process_message(self, message, message_type="text", model_id="B", context=None):
        """Process a message based on its type"""
        self.logger.info(f"Processing {message_type} message: {message[:50]}...")
        
        # Handle different message types
        if message_type == "text":
            return self.process_text(message, model_id, context)
        elif message_type == "media":
            return self.process_media(message)
        elif message_type == "voice":
            return self.process_voice(message)
        else:
            return {"error": "Unsupported message type"}
        
    def process_text(self, text, model_id="B", context=None):
        """Process text messages with context awareness"""
        try:
            self.logger.info(f"Processing text with model {model_id}: {text[:50]}...")
            
            # Check if the model is configured to use external API
            if self.api_config.is_model_using_external_api(model_id):
                self.logger.info(f"Using external API for model {model_id}")
                
                # Prepare message with context for external API
                messages = []
                
                # Add system prompt
                system_prompt = "You are a helpful assistant."
                if model_id == "B":
                    system_prompt = "You are a helpful language assistant with emotional reasoning capabilities."
                elif model_id == "A":
                    system_prompt = "You are an AI management model that coordinates other models and handles emotional interactions."
                
                # Format messages based on provider requirements
                provider_config = self.api_config.get_model_external_config(model_id)
                if provider_config.get('provider') == 'anthropic':
                    # Anthropic format doesn't have explicit system role
                    messages.append({"role": "user", "content": system_prompt + "\n" + text})
                else:
                    # Standard format with system and user messages
                    messages.append({"role": "system", "content": system_prompt})
                    
                    # Add context messages if available
                    if context:
                        # Include relevant context messages (last 5 for brevity)
                        for msg in context[-5:]:
                            messages.append(msg)
                    
                    # Add the current user message
                    messages.append({"role": "user", "content": text})
                
                # Prepare request data
                request_data = {
                    "messages": messages,
                    "max_tokens": 1000,
                    "temperature": 0.7
                }
                
                # Call the external API
                response, error = self.api_config.call_external_api(model_id, request_data)
                
                if error:
                    self.logger.error(f"External API call failed: {error}")
                    # Fall back to default response generation if API call fails
                    return self._generate_default_response(text, context)
                
                if response and 'content' in response:
                    return {"response": response['content'], "model": model_id, "source": "external"}
                else:
                    self.logger.error("Invalid response from external API")
                    return self._generate_default_response(text, context)
            
            # If not using external API, generate default response
            return self._generate_default_response(text, context)
            
        except Exception as e:
            self.logger.error(f"Error processing text: {str(e)}")
            return {"error": str(e)}
            
    def _generate_default_response(self, text, context=None):
        """Generate responses using the main management model"""
        try:
            # Import and use the main management model for real interaction
            from manager_model.main_model import ManagementModel
            
            # Initialize management model if not already done
            if not hasattr(self, 'management_model'):
                self.management_model = ManagementModel()
            
            # Process input through the management model
            response = self.management_model.process_input(text, context)
            
            if response and 'response' in response:
                return response
            else:
                return {"response": "I'm processing your request. Please wait while I coordinate with my models.", "model": "A", "source": "local"}
                
        except Exception as e:
            self.logger.error(f"Error in management model processing: {str(e)}")
            return {"response": "I'm currently optimizing my systems. Please try again in a moment.", "model": "A", "source": "local"}
    
    def process_media(self, media_data):
        """Process media messages"""
        try:
            self.logger.info("Processing media message")
            
            # Check if media data is a dictionary with required fields
            if isinstance(media_data, dict) and "type" in media_data and "data" in media_data:
                media_type = media_data["type"]
                media_data = media_data["data"]
                
                # Process based on media type
                if media_type.startswith("image/"):
                    return self.process_image(media_data, media_type)
                elif media_type.startswith("video/"):
                    return self.process_video(media_data, media_type)
                elif media_type in ["application/pdf", "text/plain", "application/msword", 
                                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
                    return self.extract_text_from_document(media_data, media_type)
                else:
                    return {"error": f"Unsupported media type: {media_type}"}
            else:
                return {"error": "Invalid media data format"}
            
        except Exception as e:
            self.logger.error(f"Error processing media: {str(e)}")
            return {"error": str(e)}
            
    def process_image(self, image_data, image_type):
        """Process image data using visual processing model"""
        try:
            self.logger.info(f"Processing image of type {image_type}")
            
            # Use the actual image processing model (Model D)
            from sub_models.D_image.app import ImageProcessingModel
            
            if not hasattr(self, 'image_model'):
                self.image_model = ImageProcessingModel()
            
            # Process image through model D
            result = self.image_model.process_image(image_data)
            
            return {"response": result.get('analysis', 'Image processed successfully'), "model": "D", "source": "local"}
            
        except Exception as e:
            self.logger.error(f"Error processing image: {str(e)}")
            return {"error": str(e)}
            
    def process_video(self, video_data, video_type):
        """Process video data using video processing model"""
        try:
            self.logger.info(f"Processing video of type {video_type}")
            
            # Use the actual video processing model (Model E)
            from sub_models.E_video.app import VideoProcessingModel
            
            if not hasattr(self, 'video_model'):
                self.video_model = VideoProcessingModel()
            
            # Process video through model E
            result = self.video_model.process_video(video_data)
            
            return {"response": result.get('analysis', 'Video processed successfully'), "model": "E", "source": "local"}
            
        except Exception as e:
            self.logger.error(f"Error processing video: {str(e)}")
            return {"error": str(e)}
            
    def extract_text_from_document(self, document_data, document_type):
        """Extract text from document files using language model"""
        try:
            self.logger.info(f"Extracting text from document of type {document_type}")
            
            # Use the actual language model (Model B) for text extraction
            from sub_models.B_language.app import LanguageModel
            
            if not hasattr(self, 'language_model'):
                self.language_model = LanguageModel()
            
            # Extract text through model B
            result = self.language_model.process_document(document_data)
            
            return {"response": result.get('extracted_text', 'Text extracted successfully'), "model": "B", "source": "local"}
            
        except Exception as e:
            self.logger.error(f"Error extracting text from document: {str(e)}")
            return {"error": str(e)}
    
    def process_voice(self, voice_data):
        """Process voice messages using audio processing model"""
        try:
            self.logger.info("Processing voice message")
            
            # Use the actual audio processing model (Model C)
            from sub_models.C_audio.app import AudioProcessingModel
            
            if not hasattr(self, 'audio_model'):
                self.audio_model = AudioProcessingModel()
            
            # Process voice through model C
            result = self.audio_model.process_audio(voice_data)
            
            return {"response": result.get('transcription', 'Voice processed successfully'), "model": "C", "source": "local"}
            
        except Exception as e:
            self.logger.error(f"Error processing voice: {str(e)}")
            return {"error": str(e)}
            
    def _process_task_queue(self):
        """Process tasks in the queue"""
        while True:
            if self.task_queue:
                task = self.task_queue.pop(0)
                try:
                    # Process the task
                    self.logger.info(f"Processing task: {task[:50]}...")
                    # In a real implementation, you would do actual task processing here
                    time.sleep(1)  # Simulate processing time
                except Exception as e:
                    self.logger.error(f"Error processing task: {str(e)}")
            else:
                time.sleep(0.1)  # Sleep briefly to avoid busy waiting

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
@app.route('/api/chat/send', methods=['POST'])
def chat():
    try:
        data = request.json
        message = data.get('message', '')
        message_type = data.get('type', 'text')
        model_id = data.get('model', 'B')
        context = data.get('context', [])
        
        # Log the received message
        logger.info(f"Received chat message: {message[:50]}... from model {model_id}")
        
        # Process the message using the A model
        response = a_model.process_message(message, message_type, model_id, context)
        
        # Log the response
        if 'response' in response:
            logger.info(f"Sent response: {response['response'][:50]}...")
        elif 'message' in response:
            logger.info(f"Sent response: {response['message'][:50]}...")
        
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
            
            # Get the response text from the appropriate field
            response_text = response.get('response', response.get('message', ''))
            
            return jsonify({
                'status': 'success',
                'text': text,
                'response': response_text
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
        'model': 'AManagementModel'
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
