# -*- coding: utf-8 -*-
"""
Data Preprocessor Module
This module provides tools for preprocessing various types of training data.
"""

import logging
import json
import os
import numpy as np
import pandas as pd
import cv2
import librosa
import re
import nltk
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
import threading
import wave
import io
import base64
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DataPreprocessor')

# Download required NLTK resources
try:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    logger.warning("Failed to download NLTK resources, some text preprocessing features may not work")

class DataPreprocessor:
    """Base class for data preprocessing"""
    
    def __init__(self, model_id: str, data_type: str):
        """Initialize the data preprocessor"""
        self.model_id = model_id
        self.data_type = data_type
        self.preprocessing_config = {
            'normalize': True,
            'shuffle': True,
            'augment': False
        }
        self.lock = threading.Lock()
        
        # Temporary directory for processing files
        self.temp_dir = tempfile.gettempdir()
        
        # Initialize specific preprocessor based on data type
        self.specific_preprocessor = self._get_specific_preprocessor()
    
    def _get_specific_preprocessor(self):
        """Get the specific preprocessor based on data type"""
        preprocessors = {
            'text': TextPreprocessor,
            'audio': AudioPreprocessor,
            'image': ImagePreprocessor,
            'video': VideoPreprocessor,
            'sensor': SensorDataPreprocessor,
            'tabular': TabularDataPreprocessor,
            'mixed': MixedDataPreprocessor
        }
        
        if self.data_type in preprocessors:
            return preprocessors[self.data_type](self.model_id)
        else:
            logger.warning(f"No specific preprocessor found for data type {self.data_type}, using base preprocessor")
            return BaseDataPreprocessor(self.model_id)
    
    def set_config(self, config: Dict[str, Any]) -> bool:
        """Set preprocessing configuration"""
        try:
            with self.lock:
                self.preprocessing_config.update(config)
                # Also update specific preprocessor config
                self.specific_preprocessor.set_config(config)
            return True
        except Exception as e:
            logger.error(f"Error setting preprocessing config: {str(e)}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """Get current preprocessing configuration"""
        with self.lock:
            return self.preprocessing_config.copy()
    
    def preprocess(self, raw_data: Any) -> Dict[str, Any]:
        """Preprocess raw data"""
        try:
            # Delegate to specific preprocessor
            return self.specific_preprocessor.preprocess(raw_data)
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'preprocessed_data': None
            }
    
    def batch_preprocess(self, raw_data_list: List[Any], batch_size: int = 32) -> Dict[str, Any]:
        """Preprocess a batch of raw data"""
        try:
            # Split data into batches
            batches = [
                raw_data_list[i:i + batch_size] 
                for i in range(0, len(raw_data_list), batch_size)
            ]
            
            preprocessed_batches = []
            total_processed = 0
            
            # Process each batch
            for i, batch in enumerate(batches):
                logger.info(f"Processing batch {i+1}/{len(batches)}")
                
                # Preprocess batch
                batch_results = []
                for data in batch:
                    result = self.preprocess(data)
                    if result['status'] == 'success':
                        batch_results.append(result['preprocessed_data'])
                        total_processed += 1
                    else:
                        logger.warning(f"Failed to preprocess item: {result.get('message', 'Unknown error')}")
                
                # Only add non-empty batches
                if batch_results:
                    preprocessed_batches.append(batch_results)
            
            return {
                'status': 'success',
                'preprocessed_batches': preprocessed_batches,
                'total_processed': total_processed,
                'total_failed': len(raw_data_list) - total_processed
            }
        except Exception as e:
            logger.error(f"Error in batch preprocessing: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'preprocessed_batches': None
            }
    
    def validate_data(self, data: Any) -> Dict[str, Any]:
        """Validate data before preprocessing"""
        try:
            # Delegate to specific preprocessor
            return self.specific_preprocessor.validate_data(data)
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'is_valid': False
            }
    
    def split_data(self, data: Any, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, Any]:
        """Split data into training, validation, and test sets"""
        try:
            # Delegate to specific preprocessor
            return self.specific_preprocessor.split_data(data, train_ratio, val_ratio)
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'train_data': None,
                'val_data': None,
                'test_data': None
            }
    
    def normalize_data(self, data: Any) -> Any:
        """Normalize data"""
        try:
            # Delegate to specific preprocessor
            return self.specific_preprocessor.normalize_data(data)
        except Exception as e:
            logger.error(f"Error normalizing data: {str(e)}")
            return data

class BaseDataPreprocessor:
    """Base data preprocessor implementation"""
    
    def __init__(self, model_id: str):
        """Initialize the base data preprocessor"""
        self.model_id = model_id
        self.config = {
            'normalize': True,
            'shuffle': True,
            'augment': False
        }
    
    def set_config(self, config: Dict[str, Any]) -> None:
        """Set preprocessing configuration"""
        self.config.update(config)
    
    def preprocess(self, raw_data: Any) -> Dict[str, Any]:
        """Base preprocessing implementation"""
        # This is a placeholder implementation
        try:
            # Validate data
            validation = self.validate_data(raw_data)
            if not validation['is_valid']:
                return {
                    'status': 'error',
                    'message': validation.get('message', 'Data validation failed'),
                    'preprocessed_data': None
                }
            
            # Normalize data if configured
            if self.config.get('normalize', True):
                processed_data = self.normalize_data(raw_data)
            else:
                processed_data = raw_data
            
            # Return processed data
            return {
                'status': 'success',
                'preprocessed_data': processed_data,
                'data_type': 'base'
            }
        except Exception as e:
            logger.error(f"Error in base data preprocessing: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'preprocessed_data': None
            }
    
    def validate_data(self, data: Any) -> Dict[str, Any]:
        """Base data validation implementation"""
        # Basic validation - just check if data is not None
        if data is None:
            return {
                'status': 'error',
                'message': 'Data cannot be None',
                'is_valid': False
            }
        
        return {
            'status': 'success',
            'is_valid': True
        }
    
    def split_data(self, data: Any, train_ratio: float = 0.8, val_ratio: float = 0.1) -> Dict[str, Any]:
        """Base data splitting implementation"""
        try:
            if isinstance(data, list):
                # Shuffle if configured
                if self.config.get('shuffle', True):
                    np.random.shuffle(data)
                
                # Calculate split indices
                total_size = len(data)
                train_size = int(total_size * train_ratio)
                val_size = int(total_size * val_ratio)
                
                # Split data
                train_data = data[:train_size]
                val_data = data[train_size:train_size + val_size]
                test_data = data[train_size + val_size:]
                
                return {
                    'status': 'success',
                    'train_data': train_data,
                    'val_data': val_data,
                    'test_data': test_data,
                    'train_size': len(train_data),
                    'val_size': len(val_data),
                    'test_size': len(test_data)
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Data must be a list for splitting',
                    'train_data': None,
                    'val_data': None,
                    'test_data': None
                }
        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'train_data': None,
                'val_data': None,
                'test_data': None
            }
    
    def normalize_data(self, data: Any) -> Any:
        """Base data normalization implementation"""
        # This is a placeholder implementation
        # In a real system, this would be data type specific
        return data

class TextPreprocessor(BaseDataPreprocessor):
    """Text data preprocessor"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific configuration for text preprocessing
        self.config.update({
            'lowercase': True,
            'remove_punctuation': True,
            'remove_stopwords': False,
            'lemmatize': False,
            'max_length': None,
            'tokenize': True,
            'encoding': 'utf-8'
        })
    
    def preprocess(self, raw_data: Any) -> Dict[str, Any]:
        """Preprocess text data"""
        try:
            # Validate data
            validation = self.validate_data(raw_data)
            if not validation['is_valid']:
                return {
                    'status': 'error',
                    'message': validation.get('message', 'Text data validation failed'),
                    'preprocessed_data': None
                }
            
            # Ensure data is string
            text = str(raw_data)
            
            # Lowercase if configured
            if self.config.get('lowercase', True):
                text = text.lower()
            
            # Remove punctuation if configured
            if self.config.get('remove_punctuation', True):
                text = re.sub(r'[^\w\s]', '', text)
            
            # Tokenize if configured
            if self.config.get('tokenize', True):
                tokens = nltk.word_tokenize(text)
                
                # Remove stopwords if configured
                if self.config.get('remove_stopwords', False):
                    try:
                        from nltk.corpus import stopwords
                        nltk.download('stopwords', quiet=True)
                        stop_words = set(stopwords.words('english'))
                        tokens = [word for word in tokens if word not in stop_words]
                    except:
                        logger.warning("Failed to remove stopwords, proceeding without")
                
                # Lemmatize if configured
                if self.config.get('lemmatize', False):
                    try:
                        from nltk.stem import WordNetLemmatizer
                        lemmatizer = WordNetLemmatizer()
                        tokens = [lemmatizer.lemmatize(word) for word in tokens]
                    except:
                        logger.warning("Failed to lemmatize, proceeding without")
                
                # Apply max length if configured
                max_length = self.config.get('max_length')
                if max_length is not None and len(tokens) > max_length:
                    tokens = tokens[:max_length]
                
                processed_data = tokens
            else:
                # Apply max length if configured
                max_length = self.config.get('max_length')
                if max_length is not None and len(text) > max_length:
                    text = text[:max_length]
                
                processed_data = text
            
            return {
                'status': 'success',
                'preprocessed_data': processed_data,
                'data_type': 'text',
                'original_length': len(text),
                'processed_length': len(processed_data) if isinstance(processed_data, str) else len(' '.join(processed_data))
            }
        except Exception as e:
            logger.error(f"Error preprocessing text data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'preprocessed_data': None
            }
    
    def validate_data(self, data: Any) -> Dict[str, Any]:
        """Validate text data"""
        # Check if data is a string or can be converted to string
        try:
            str(data)
            return {
                'status': 'success',
                'is_valid': True
            }
        except:
            return {
                'status': 'error',
                'message': 'Data cannot be converted to text',
                'is_valid': False
            }

class AudioPreprocessor(BaseDataPreprocessor):
    """Audio data preprocessor"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific configuration for audio preprocessing
        self.config.update({
            'sample_rate': 16000,
            'mono': True,
            'duration': None,  # in seconds
            'normalize_volume': True,
            'noise_reduction': False,
            'feature_extraction': 'mfcc',  # 'mfcc', 'spectrogram', 'raw'
            'n_mfcc': 13
        })
    
    def preprocess(self, raw_data: Any) -> Dict[str, Any]:
        """Preprocess audio data"""
        try:
            # Validate data
            validation = self.validate_data(raw_data)
            if not validation['is_valid']:
                return {
                    'status': 'error',
                    'message': validation.get('message', 'Audio data validation failed'),
                    'preprocessed_data': None
                }
            
            # Get audio data and sample rate
            audio_data, sample_rate = self._load_audio(raw_data)
            
            # Resample if needed
            target_sample_rate = self.config.get('sample_rate', 16000)
            if sample_rate != target_sample_rate:
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=sample_rate, 
                    target_sr=target_sample_rate
                )
                sample_rate = target_sample_rate
            
            # Convert to mono if needed
            if self.config.get('mono', True) and len(audio_data.shape) > 1:
                audio_data = librosa.to_mono(audio_data)
            
            # Truncate or pad to duration if specified
            duration = self.config.get('duration')
            if duration is not None:
                target_length = int(duration * sample_rate)
                if len(audio_data) > target_length:
                    audio_data = audio_data[:target_length]
                elif len(audio_data) < target_length:
                    padding = np.zeros(target_length - len(audio_data))
                    audio_data = np.concatenate([audio_data, padding])
            
            # Normalize volume if configured
            if self.config.get('normalize_volume', True):
                audio_data = librosa.util.normalize(audio_data)
            
            # Apply noise reduction if configured
            if self.config.get('noise_reduction', False):
                try:
                    # Simple noise reduction using spectral subtraction
                    # This is a simplified implementation
                    S = np.abs(librosa.stft(audio_data))
                    S_db = librosa.amplitude_to_db(S)
                    # Estimate noise floor (simplified)
                    noise_floor = np.median(S_db)
                    # Apply spectral subtraction
                    S_db -= np.maximum(0, noise_floor - 10)  # Keep some headroom
                    # Convert back to audio
                    S_denoised = librosa.db_to_amplitude(S_db)
                    audio_data = librosa.istft(S_denoised)
                except:
                    logger.warning("Failed to apply noise reduction, proceeding without")
            
            # Extract features based on configuration
            feature_type = self.config.get('feature_extraction', 'mfcc')
            if feature_type == 'mfcc':
                n_mfcc = self.config.get('n_mfcc', 13)
                features = librosa.feature.mfcc(
                    y=audio_data, 
                    sr=sample_rate, 
                    n_mfcc=n_mfcc
                )
                # Normalize features
                features = self.normalize_data(features)
            elif feature_type == 'spectrogram':
                features = np.abs(librosa.stft(audio_data))
                features = librosa.amplitude_to_db(features)
                # Normalize features
                features = self.normalize_data(features)
            else:  # raw
                features = audio_data
            
            return {
                'status': 'success',
                'preprocessed_data': features,
                'data_type': 'audio',
                'sample_rate': sample_rate,
                'audio_length': len(audio_data) / sample_rate,  # in seconds
                'feature_type': feature_type
            }
        except Exception as e:
            logger.error(f"Error preprocessing audio data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'preprocessed_data': None
            }
    
    def _load_audio(self, data: Any) -> Tuple[np.ndarray, int]:
        """Load audio from various sources"""
        try:
            # Case 1: File path
            if isinstance(data, str) and os.path.isfile(data):
                return librosa.load(data, sr=None)
            
            # Case 2: Base64 encoded audio
            elif isinstance(data, str) and data.startswith('base64,'):
                # Extract base64 data
                base64_data = data.split(',')[1]
                # Decode base64 to bytes
                audio_bytes = base64.b64decode(base64_data)
                # Use temporary file to load audio
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(audio_bytes)
                    f.flush()
                    try:
                        audio_data, sample_rate = librosa.load(f.name, sr=None)
                    finally:
                        os.unlink(f.name)
                return audio_data, sample_rate
            
            # Case 3: Numpy array
            elif isinstance(data, np.ndarray):
                # Assume sample rate is 16000 if not specified
                return data, self.config.get('sample_rate', 16000)
            
            # Case 4: Bytes
            elif isinstance(data, bytes):
                # Use temporary file to load audio
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                    f.write(data)
                    f.flush()
                    try:
                        audio_data, sample_rate = librosa.load(f.name, sr=None)
                    finally:
                        os.unlink(f.name)
                return audio_data, sample_rate
            
            else:
                raise ValueError(f"Unsupported audio data type: {type(data)}")
        except Exception as e:
            raise ValueError(f"Failed to load audio data: {str(e)}")
    
    def validate_data(self, data: Any) -> Dict[str, Any]:
        """Validate audio data"""
        try:
            # Try to load the audio data
            self._load_audio(data)
            return {
                'status': 'success',
                'is_valid': True
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'is_valid': False
            }

class ImagePreprocessor(BaseDataPreprocessor):
    """Image data preprocessor"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific configuration for image preprocessing
        self.config.update({
            'target_size': (224, 224),
            'grayscale': False,
            'normalize': True,
            'mean': [0.485, 0.456, 0.406],  # ImageNet mean
            'std': [0.229, 0.224, 0.225],   # ImageNet std
            'augment': False,
            'rotation_range': 10,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'shear_range': 0.1,
            'zoom_range': 0.1,
            'horizontal_flip': False
        })
    
    def preprocess(self, raw_data: Any) -> Dict[str, Any]:
        """Preprocess image data"""
        try:
            # Validate data
            validation = self.validate_data(raw_data)
            if not validation['is_valid']:
                return {
                    'status': 'error',
                    'message': validation.get('message', 'Image data validation failed'),
                    'preprocessed_data': None
                }
            
            # Load image
            image = self._load_image(raw_data)
            
            # Convert to grayscale if configured
            if self.config.get('grayscale', False) and len(image.shape) > 2:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize to target size
            target_size = self.config.get('target_size', (224, 224))
            image = cv2.resize(image, target_size)
            
            # Add channel dimension if needed
            if len(image.shape) == 2:
                image = np.expand_dims(image, axis=-1)
            
            # Convert to RGB (cv2 loads as BGR)
            if len(image.shape) > 2 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply data augmentation if configured
            if self.config.get('augment', False):
                image = self._apply_augmentation(image)
            
            # Normalize if configured
            if self.config.get('normalize', True):
                image = self.normalize_data(image)
            
            return {
                'status': 'success',
                'preprocessed_data': image,
                'data_type': 'image',
                'original_shape': image.shape,
                'target_size': target_size
            }
        except Exception as e:
            logger.error(f"Error preprocessing image data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'preprocessed_data': None
            }
    
    def _load_image(self, data: Any) -> np.ndarray:
        """Load image from various sources"""
        try:
            # Case 1: File path
            if isinstance(data, str) and os.path.isfile(data):
                image = cv2.imread(data)
                if image is None:
                    raise ValueError(f"Failed to load image from file: {data}")
                return image
            
            # Case 2: Base64 encoded image
            elif isinstance(data, str) and data.startswith('base64,'):
                # Extract base64 data
                base64_data = data.split(',')[1]
                # Decode base64 to bytes
                image_bytes = base64.b64decode(base64_data)
                # Convert bytes to numpy array
                np_arr = np.frombuffer(image_bytes, np.uint8)
                # Decode image
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Failed to decode base64 image")
                return image
            
            # Case 3: Numpy array
            elif isinstance(data, np.ndarray):
                return data
            
            # Case 4: Bytes
            elif isinstance(data, bytes):
                # Convert bytes to numpy array
                np_arr = np.frombuffer(data, np.uint8)
                # Decode image
                image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Failed to decode bytes to image")
                return image
            
            else:
                raise ValueError(f"Unsupported image data type: {type(data)}")
        except Exception as e:
            raise ValueError(f"Failed to load image data: {str(e)}")
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation to the image"""
        try:
            # Get augmentation parameters
            rotation_range = self.config.get('rotation_range', 10)
            width_shift_range = self.config.get('width_shift_range', 0.1)
            height_shift_range = self.config.get('height_shift_range', 0.1)
            shear_range = self.config.get('shear_range', 0.1)
            zoom_range = self.config.get('zoom_range', 0.1)
            horizontal_flip = self.config.get('horizontal_flip', False)
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Random rotation
            if rotation_range > 0:
                angle = np.random.uniform(-rotation_range, rotation_range)
                M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
                image = cv2.warpAffine(image, M, (width, height))
            
            # Random translation
            if width_shift_range > 0 or height_shift_range > 0:
                tx = np.random.uniform(-width_shift_range, width_shift_range) * width
                ty = np.random.uniform(-height_shift_range, height_shift_range) * height
                M = np.float32([[1, 0, tx], [0, 1, ty]])
                image = cv2.warpAffine(image, M, (width, height))
            
            # Random shear
            if shear_range > 0:
                shear = np.random.uniform(-shear_range, shear_range)
                M = np.float32([[1, shear, 0], [0, 1, 0]])
                image = cv2.warpAffine(image, M, (width, height))
            
            # Random zoom
            if zoom_range > 0:
                zoom = np.random.uniform(1 - zoom_range, 1 + zoom_range)
                M = cv2.getRotationMatrix2D((width/2, height/2), 0, zoom)
                image = cv2.warpAffine(image, M, (width, height))
            
            # Random horizontal flip
            if horizontal_flip and np.random.random() < 0.5:
                image = cv2.flip(image, 1)
            
            return image
        except Exception as e:
            logger.warning(f"Failed to apply augmentation: {str(e)}, returning original image")
            return image
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize image data"""
        # Convert to float32
        image = data.astype(np.float32)
        
        # Scale pixel values to [0, 1]
        image = image / 255.0
        
        # Apply mean and std normalization if configured
        if self.config.get('normalize', True):
            mean = self.config.get('mean', [0.485, 0.456, 0.406])
            std = self.config.get('std', [0.229, 0.224, 0.225])
            
            # Check if image has 3 channels (RGB)
            if len(image.shape) > 2 and image.shape[2] == 3:
                # Subtract mean and divide by std
                for i in range(3):
                    image[..., i] = (image[..., i] - mean[i]) / std[i]
        
        return image
    
    def validate_data(self, data: Any) -> Dict[str, Any]:
        """Validate image data"""
        try:
            # Try to load the image
            self._load_image(data)
            return {
                'status': 'success',
                'is_valid': True
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'is_valid': False
            }

class VideoPreprocessor(BaseDataPreprocessor):
    """Video data preprocessor"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific configuration for video preprocessing
        self.config.update({
            'target_size': (224, 224),
            'fps': 16,
            'max_frames': 32,
            'grayscale': False,
            'normalize': True,
            'extract_frames': True,
            'optical_flow': False
        })
        # Initialize image preprocessor for frame processing
        self.image_preprocessor = ImagePreprocessor(model_id)
    
    def preprocess(self, raw_data: Any) -> Dict[str, Any]:
        """Preprocess video data"""
        try:
            # Validate data
            validation = self.validate_data(raw_data)
            if not validation['is_valid']:
                return {
                    'status': 'error',
                    'message': validation.get('message', 'Video data validation failed'),
                    'preprocessed_data': None
                }
            
            # Extract frames from video
            frames, original_fps = self._extract_frames(raw_data)
            
            # Process frames
            processed_frames = []
            target_fps = self.config.get('fps', 16)
            max_frames = self.config.get('max_frames', 32)
            
            # Calculate frame sampling rate
            if original_fps > 0:
                frame_step = max(1, int(original_fps / target_fps))
            else:
                frame_step = 1
            
            # Process frames at the target FPS
            for i in range(0, len(frames), frame_step):
                frame = frames[i]
                
                # Preprocess frame using image preprocessor
                frame_result = self.image_preprocessor.preprocess(frame)
                if frame_result['status'] == 'success':
                    processed_frames.append(frame_result['preprocessed_data'])
                
                # Stop if we've reached the maximum number of frames
                if len(processed_frames) >= max_frames:
                    break
            
            # If we have fewer frames than max_frames, pad with zeros
            while len(processed_frames) < max_frames:
                # Create a zero frame with the same shape as the first frame
                if processed_frames:
                    zero_frame = np.zeros_like(processed_frames[0])
                    processed_frames.append(zero_frame)
                else:
                    # If no frames were processed, create a dummy frame
                    target_size = self.config.get('target_size', (224, 224))
                    grayscale = self.config.get('grayscale', False)
                    channels = 1 if grayscale else 3
                    zero_frame = np.zeros((target_size[0], target_size[1], channels))
                    processed_frames.append(zero_frame)
            
            # Stack frames into a single array
            processed_video = np.array(processed_frames)
            
            # Add time dimension if needed
            if len(processed_video.shape) == 4:
                # (frames, height, width, channels) -> (frames, height, width, channels)
                pass  # Already in the correct format
            
            return {
                'status': 'success',
                'preprocessed_data': processed_video,
                'data_type': 'video',
                'original_fps': original_fps,
                'target_fps': target_fps,
                'processed_frames': len(processed_frames),
                'video_shape': processed_video.shape
            }
        except Exception as e:
            logger.error(f"Error preprocessing video data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'preprocessed_data': None
            }
    
    def _extract_frames(self, data: Any) -> Tuple[List[np.ndarray], float]:
        """Extract frames from video data"""
        frames = []
        fps = 0.0
        
        try:
            # Case 1: File path
            if isinstance(data, str) and os.path.isfile(data):
                cap = cv2.VideoCapture(data)
                if not cap.isOpened():
                    raise ValueError(f"Failed to open video file: {data}")
                
                # Get original FPS
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                # Read frames
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(frame)
                
                cap.release()
            
            # Case 2: List of frames
            elif isinstance(data, list) and data and isinstance(data[0], np.ndarray):
                frames = data
                fps = self.config.get('fps', 16)  # Default FPS if not provided
            
            # Case 3: Numpy array (already processed frames)
            elif isinstance(data, np.ndarray) and len(data.shape) >= 3:
                # Assuming shape is (frames, height, width, channels) or similar
                if len(data.shape) == 4:
                    # (frames, height, width, channels)
                    for i in range(data.shape[0]):
                        frames.append(data[i])
                elif len(data.shape) == 5:
                    # (batch, frames, height, width, channels) - take first batch
                    for i in range(data.shape[1]):
                        frames.append(data[0, i])
                fps = self.config.get('fps', 16)  # Default FPS if not provided
            
            else:
                raise ValueError(f"Unsupported video data type: {type(data)}")
            
            return frames, fps
        except Exception as e:
            raise ValueError(f"Failed to extract frames: {str(e)}")
    
    def validate_data(self, data: Any) -> Dict[str, Any]:
        """Validate video data"""
        try:
            # Try to extract frames
            self._extract_frames(data)
            return {
                'status': 'success',
                'is_valid': True
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'is_valid': False
            }

class SensorDataPreprocessor(BaseDataPreprocessor):
    """Sensor data preprocessor"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific configuration for sensor data preprocessing
        self.config.update({
            'sample_rate': 100,  # Hz
            'window_size': 100,  # Number of samples
            'hop_size': 50,      # Number of samples
            'feature_extraction': 'statistics',  # 'statistics', 'fft', 'raw'
            'normalize': True,
            'impute_missing': True,
            'outlier_removal': False
        })
    
    def preprocess(self, raw_data: Any) -> Dict[str, Any]:
        """Preprocess sensor data"""
        try:
            # Validate data
            validation = self.validate_data(raw_data)
            if not validation['is_valid']:
                return {
                    'status': 'error',
                    'message': validation.get('message', 'Sensor data validation failed'),
                    'preprocessed_data': None
                }
            
            # Convert to numpy array
            sensor_data = self._convert_to_array(raw_data)
            
            # Impute missing values if configured
            if self.config.get('impute_missing', True):
                sensor_data = self._impute_missing(sensor_data)
            
            # Remove outliers if configured
            if self.config.get('outlier_removal', False):
                sensor_data = self._remove_outliers(sensor_data)
            
            # Resample if needed
            # (This would be implemented based on actual time information)
            
            # Extract features based on configuration
            feature_type = self.config.get('feature_extraction', 'statistics')
            if feature_type == 'statistics':
                features = self._extract_statistical_features(sensor_data)
            elif feature_type == 'fft':
                features = self._extract_fft_features(sensor_data)
            else:  # raw
                features = sensor_data
            
            # Normalize if configured
            if self.config.get('normalize', True):
                features = self.normalize_data(features)
            
            return {
                'status': 'success',
                'preprocessed_data': features,
                'data_type': 'sensor',
                'original_shape': sensor_data.shape,
                'feature_type': feature_type
            }
        except Exception as e:
            logger.error(f"Error preprocessing sensor data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'preprocessed_data': None
            }
    
    def _convert_to_array(self, data: Any) -> np.ndarray:
        """Convert sensor data to numpy array"""
        try:
            # Case 1: List of lists or list of numbers
            if isinstance(data, list):
                return np.array(data)
            
            # Case 2: Numpy array
            elif isinstance(data, np.ndarray):
                return data
            
            # Case 3: Pandas DataFrame
            elif hasattr(data, 'values'):
                return data.values
            
            else:
                raise ValueError(f"Unsupported sensor data type: {type(data)}")
        except Exception as e:
            raise ValueError(f"Failed to convert data to array: {str(e)}")
    
    def _impute_missing(self, data: np.ndarray) -> np.ndarray:
        """Impute missing values in the data"""
        try:
            # Create a copy to avoid modifying the original data
            imputed_data = data.copy()
            
            # Iterate over each column
            for i in range(imputed_data.shape[1] if len(imputed_data.shape) > 1 else 1):
                # Get the column data
                if len(imputed_data.shape) > 1:
                    col_data = imputed_data[:, i]
                else:
                    col_data = imputed_data
                
                # Find NaN values
                nan_mask = np.isnan(col_data)
                
                # If there are NaN values, impute with mean
                if np.any(nan_mask):
                    # Calculate mean of non-NaN values
                    mean_val = np.nanmean(col_data)
                    # Replace NaN values with mean
                    col_data[nan_mask] = mean_val
                    
                    # Update the column in the imputed data
                    if len(imputed_data.shape) > 1:
                        imputed_data[:, i] = col_data
                    else:
                        imputed_data = col_data
            
            return imputed_data
        except Exception as e:
            logger.warning(f"Failed to impute missing values: {str(e)}, returning original data")
            return data
    
    def _remove_outliers(self, data: np.ndarray) -> np.ndarray:
        """Remove outliers from the data"""
        try:
            # Create a copy to avoid modifying the original data
            filtered_data = data.copy()
            
            # Iterate over each column
            for i in range(filtered_data.shape[1] if len(filtered_data.shape) > 1 else 1):
                # Get the column data
                if len(filtered_data.shape) > 1:
                    col_data = filtered_data[:, i]
                else:
                    col_data = filtered_data
                
                # Calculate Q1, Q3, and IQR
                q1 = np.percentile(col_data, 25)
                q3 = np.percentile(col_data, 75)
                iqr = q3 - q1
                
                # Define outlier boundaries
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                # Replace outliers with NaN
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                col_data[outlier_mask] = np.nan
                
                # Update the column in the filtered data
                if len(filtered_data.shape) > 1:
                    filtered_data[:, i] = col_data
                else:
                    filtered_data = col_data
            
            # Impute the NaN values created by outlier removal
            filtered_data = self._impute_missing(filtered_data)
            
            return filtered_data
        except Exception as e:
            logger.warning(f"Failed to remove outliers: {str(e)}, returning original data")
            return data
    
    def _extract_statistical_features(self, data: np.ndarray) -> np.ndarray:
        """Extract statistical features from the data"""
        try:
            features = []
            
            # Calculate features for each channel
            if len(data.shape) > 1:
                # Multiple channels
                for i in range(data.shape[1]):
                    col_data = data[:, i]
                    
                    # Calculate statistical features
                    mean_val = np.mean(col_data)
                    std_val = np.std(col_data)
                    min_val = np.min(col_data)
                    max_val = np.max(col_data)
                    median_val = np.median(col_data)
                    q1_val = np.percentile(col_data, 25)
                    q3_val = np.percentile(col_data, 75)
                    
                    # Add features
                    features.extend([mean_val, std_val, min_val, max_val, median_val, q1_val, q3_val])
            else:
                # Single channel
                mean_val = np.mean(data)
                std_val = np.std(data)
                min_val = np.min(data)
                max_val = np.max(data)
                median_val = np.median(data)
                q1_val = np.percentile(data, 25)
                q3_val = np.percentile(data, 75)
                
                # Add features
                features.extend([mean_val, std_val, min_val, max_val, median_val, q1_val, q3_val])
            
            return np.array(features)
        except Exception as e:
            logger.error(f"Failed to extract statistical features: {str(e)}")
            # Return original data as fallback
            return data
    
    def _extract_fft_features(self, data: np.ndarray) -> np.ndarray:
        """Extract FFT features from the data"""
        try:
            features = []
            
            # Calculate FFT for each channel
            if len(data.shape) > 1:
                # Multiple channels
                for i in range(data.shape[1]):
                    col_data = data[:, i]
                    
                    # Compute FFT
                    fft_vals = np.fft.fft(col_data)
                    # Take absolute values for magnitude
                    fft_magnitude = np.abs(fft_vals)
                    # Only take the first half (real signals are symmetric)
                    fft_magnitude = fft_magnitude[:len(fft_magnitude)//2]
                    
                    # Calculate features from FFT
                    mean_fft = np.mean(fft_magnitude)
                    std_fft = np.std(fft_magnitude)
                    max_fft = np.max(fft_magnitude)
                    peak_freq = np.argmax(fft_magnitude)  # Index of peak frequency
                    
                    # Add features
                    features.extend([mean_fft, std_fft, max_fft, peak_freq])
                    # Also add the first few FFT coefficients
                    features.extend(fft_magnitude[:10].tolist())  # Take first 10 coefficients
            else:
                # Single channel
                # Compute FFT
                fft_vals = np.fft.fft(data)
                # Take absolute values for magnitude
                fft_magnitude = np.abs(fft_vals)
                # Only take the first half (real signals are symmetric)
                fft_magnitude = fft_magnitude[:len(fft_magnitude)//2]
                
                # Calculate features from FFT
                mean_fft = np.mean(fft_magnitude)
                std_fft = np.std(fft_magnitude)
                max_fft = np.max(fft_magnitude)
                peak_freq = np.argmax(fft_magnitude)  # Index of peak frequency
                
                # Add features
                features.extend([mean_fft, std_fft, max_fft, peak_freq])
                # Also add the first few FFT coefficients
                features.extend(fft_magnitude[:10].tolist())  # Take first 10 coefficients
            
            return np.array(features)
        except Exception as e:
            logger.error(f"Failed to extract FFT features: {str(e)}")
            # Return original data as fallback
            return data
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """Normalize sensor data"""
        try:
            # Avoid division by zero
            if np.std(data) == 0:
                return data
            
            # Z-score normalization: (x - mean) / std
            normalized_data = (data - np.mean(data)) / np.std(data)
            
            return normalized_data
        except Exception as e:
            logger.error(f"Failed to normalize sensor data: {str(e)}")
            return data
    
    def validate_data(self, data: Any) -> Dict[str, Any]:
        """Validate sensor data"""
        try:
            # Try to convert to array
            self._convert_to_array(data)
            return {
                'status': 'success',
                'is_valid': True
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'is_valid': False
            }

class TabularDataPreprocessor(BaseDataPreprocessor):
    """Tabular data preprocessor"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific configuration for tabular data preprocessing
        self.config.update({
            'normalize': True,
            'one_hot_encode': True,
            'impute_missing': True,
            'outlier_removal': False,
            'feature_scaling': 'standard',  # 'standard', 'minmax', 'robust'
            'drop_duplicates': True,
            'drop_columns': [],
            'target_column': None
        })
        # Store fitted parameters for scaling and encoding
        self.fitted_params = {}
    
    def preprocess(self, raw_data: Any) -> Dict[str, Any]:
        """Preprocess tabular data"""
        try:
            # Validate data
            validation = self.validate_data(raw_data)
            if not validation['is_valid']:
                return {
                    'status': 'error',
                    'message': validation.get('message', 'Tabular data validation failed'),
                    'preprocessed_data': None
                }
            
            # Convert to DataFrame
            df = self._convert_to_dataframe(raw_data)
            
            # Drop duplicate rows if configured
            if self.config.get('drop_duplicates', True):
                initial_rows = len(df)
                df = df.drop_duplicates()
                logger.info(f"Dropped {initial_rows - len(df)} duplicate rows")
            
            # Drop specified columns if configured
            drop_columns = self.config.get('drop_columns', [])
            if drop_columns:
                df = df.drop(columns=drop_columns, errors='ignore')
            
            # Impute missing values if configured
            if self.config.get('impute_missing', True):
                df = self._impute_missing(df)
            
            # Remove outliers if configured
            if self.config.get('outlier_removal', False):
                df = self._remove_outliers(df)
            
            # One-hot encode categorical features if configured
            if self.config.get('one_hot_encode', True):
                df = self._one_hot_encode(df)
            
            # Split features and target if target column is specified
            target_column = self.config.get('target_column')
            if target_column and target_column in df.columns:
                X = df.drop(columns=[target_column])
                y = df[target_column]
            else:
                X = df
                y = None
            
            # Normalize/scale features if configured
            if self.config.get('normalize', True):
                X = self.normalize_data(X)
            
            # Convert back to numpy arrays
            X_array = X.values
            y_array = y.values if y is not None else None
            
            # Prepare results
            results = {
                'status': 'success',
                'preprocessed_data': {
                    'X': X_array,
                    'y': y_array
                },
                'data_type': 'tabular',
                'feature_columns': list(X.columns),
                'original_shape': df.shape,
                'processed_shape': X_array.shape
            }
            
            if y is not None:
                results['target_column'] = target_column
                results['target_shape'] = y_array.shape
            
            return results
        except Exception as e:
            logger.error(f"Error preprocessing tabular data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'preprocessed_data': None
            }
    
    def _convert_to_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert tabular data to pandas DataFrame"""
        try:
            # Case 1: Pandas DataFrame
            if isinstance(data, pd.DataFrame):
                return data.copy()
            
            # Case 2: List of dictionaries or list of lists
            elif isinstance(data, list):
                return pd.DataFrame(data)
            
            # Case 3: Dictionary of lists
            elif isinstance(data, dict):
                return pd.DataFrame(data)
            
            # Case 4: Numpy array
            elif isinstance(data, np.ndarray):
                # Assume first row is headers if it contains strings
                if data.ndim == 2 and isinstance(data[0, 0], str):
                    headers = data[0, :]
                    data_rows = data[1:, :]
                    return pd.DataFrame(data=data_rows, columns=headers)
                else:
                    # Generate default column names
                    columns = [f'feature_{i}' for i in range(data.shape[1])]
                    return pd.DataFrame(data=data, columns=columns)
            
            else:
                raise ValueError(f"Unsupported tabular data type: {type(data)}")
        except Exception as e:
            raise ValueError(f"Failed to convert data to DataFrame: {str(e)}")
    
    def _impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values in the DataFrame"""
        try:
            # Make a copy to avoid modifying the original DataFrame
            imputed_df = df.copy()
            
            # Iterate over each column
            for col in imputed_df.columns:
                # For numeric columns, impute with mean
                if pd.api.types.is_numeric_dtype(imputed_df[col]):
                    mean_val = imputed_df[col].mean()
                    imputed_df[col] = imputed_df[col].fillna(mean_val)
                # For categorical columns, impute with mode
                else:
                    mode_val = imputed_df[col].mode().iloc[0] if not imputed_df[col].mode().empty else 'Unknown'
                    imputed_df[col] = imputed_df[col].fillna(mode_val)
            
            return imputed_df
        except Exception as e:
            logger.warning(f"Failed to impute missing values: {str(e)}, returning original DataFrame")
            return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the DataFrame"""
        try:
            # Make a copy to avoid modifying the original DataFrame
            filtered_df = df.copy()
            
            # Iterate over each numeric column
            for col in filtered_df.columns:
                if pd.api.types.is_numeric_dtype(filtered_df[col]):
                    # Calculate Q1, Q3, and IQR
                    q1 = filtered_df[col].quantile(0.25)
                    q3 = filtered_df[col].quantile(0.75)
                    iqr = q3 - q1
                    
                    # Define outlier boundaries
                    lower_bound = q1 - 1.5 * iqr
                    upper_bound = q3 + 1.5 * iqr
                    
                    # Filter out outliers
                    filtered_df = filtered_df[(filtered_df[col] >= lower_bound) & (filtered_df[col] <= upper_bound)]
            
            return filtered_df
        except Exception as e:
            logger.warning(f"Failed to remove outliers: {str(e)}, returning original DataFrame")
            return df
    
    def _one_hot_encode(self, df: pd.DataFrame) -> pd.DataFrame:
        """One-hot encode categorical features"""
        try:
            # Make a copy to avoid modifying the original DataFrame
            encoded_df = df.copy()
            
            # Get categorical columns
            categorical_cols = encoded_df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # One-hot encode each categorical column
            if categorical_cols:
                encoded_df = pd.get_dummies(encoded_df, columns=categorical_cols, drop_first=True)
            
            return encoded_df
        except Exception as e:
            logger.warning(f"Failed to one-hot encode: {str(e)}, returning original DataFrame")
            return df
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize/scale tabular data"""
        try:
            # Make a copy to avoid modifying the original DataFrame
            normalized_df = data.copy()
            
            # Get feature scaling method
            feature_scaling = self.config.get('feature_scaling', 'standard')
            
            # Iterate over each numeric column
            for col in normalized_df.columns:
                if pd.api.types.is_numeric_dtype(normalized_df[col]):
                    # Standard scaling: (x - mean) / std
                    if feature_scaling == 'standard':
                        mean_val = normalized_df[col].mean()
                        std_val = normalized_df[col].std()
                        if std_val > 0:
                            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
                            # Store fitted parameters for inference
                            self.fitted_params[col] = {'mean': mean_val, 'std': std_val, 'type': 'standard'}
                    # Min-Max scaling: (x - min) / (max - min)
                    elif feature_scaling == 'minmax':
                        min_val = normalized_df[col].min()
                        max_val = normalized_df[col].max()
                        if max_val > min_val:
                            normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
                            # Store fitted parameters for inference
                            self.fitted_params[col] = {'min': min_val, 'max': max_val, 'type': 'minmax'}
                    # Robust scaling: (x - median) / IQR
                    elif feature_scaling == 'robust':
                        median_val = normalized_df[col].median()
                        q1 = normalized_df[col].quantile(0.25)
                        q3 = normalized_df[col].quantile(0.75)
                        iqr = q3 - q1
                        if iqr > 0:
                            normalized_df[col] = (normalized_df[col] - median_val) / iqr
                            # Store fitted parameters for inference
                            self.fitted_params[col] = {'median': median_val, 'q1': q1, 'q3': q3, 'type': 'robust'}
            
            return normalized_df
        except Exception as e:
            logger.error(f"Failed to normalize tabular data: {str(e)}")
            return data
    
    def validate_data(self, data: Any) -> Dict[str, Any]:
        """Validate tabular data"""
        try:
            # Try to convert to DataFrame
            self._convert_to_dataframe(data)
            return {
                'status': 'success',
                'is_valid': True
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'is_valid': False
            }

class MixedDataPreprocessor(BaseDataPreprocessor):
    """Mixed data preprocessor for handling multiple data types"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Initialize preprocessors for different data types
        self.text_preprocessor = TextPreprocessor(model_id)
        self.audio_preprocessor = AudioPreprocessor(model_id)
        self.image_preprocessor = ImagePreprocessor(model_id)
        self.video_preprocessor = VideoPreprocessor(model_id)
        self.sensor_preprocessor = SensorDataPreprocessor(model_id)
        self.tabular_preprocessor = TabularDataPreprocessor(model_id)
    
    def preprocess(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess mixed data (dictionary with different data types)"""
        try:
            # Validate data
            validation = self.validate_data(raw_data)
            if not validation['is_valid']:
                return {
                    'status': 'error',
                    'message': validation.get('message', 'Mixed data validation failed'),
                    'preprocessed_data': None
                }
            
            # Preprocess each data type
            preprocessed_data = {}
            
            for key, data_info in raw_data.items():
                # Extract data and type
                if isinstance(data_info, dict) and 'data' in data_info and 'type' in data_info:
                    data = data_info['data']
                    data_type = data_info['type'].lower()
                else:
                    # If data_info is not a dict with 'data' and 'type', skip or handle as raw
                    continue
                
                # Select appropriate preprocessor
                if data_type == 'text':
                    result = self.text_preprocessor.preprocess(data)
                elif data_type == 'audio':
                    result = self.audio_preprocessor.preprocess(data)
                elif data_type == 'image':
                    result = self.image_preprocessor.preprocess(data)
                elif data_type == 'video':
                    result = self.video_preprocessor.preprocess(data)
                elif data_type == 'sensor':
                    result = self.sensor_preprocessor.preprocess(data)
                elif data_type == 'tabular':
                    result = self.tabular_preprocessor.preprocess(data)
                else:
                    # Default to base preprocessor
                    result = super().preprocess(data)
                
                # Add to preprocessed data
                if result['status'] == 'success':
                    preprocessed_data[key] = result
                else:
                    logger.warning(f"Failed to preprocess {key}: {result.get('message', 'Unknown error')}")
            
            return {
                'status': 'success',
                'preprocessed_data': preprocessed_data,
                'data_type': 'mixed',
                'processed_keys': list(preprocessed_data.keys()),
                'total_processed': len(preprocessed_data)
            }
        except Exception as e:
            logger.error(f"Error preprocessing mixed data: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'preprocessed_data': None
            }
    
    def validate_data(self, data: Any) -> Dict[str, Any]:
        """Validate mixed data"""
        try:
            # Check if data is a dictionary
            if not isinstance(data, dict):
                return {
                    'status': 'error',
                    'message': 'Mixed data must be a dictionary',
                    'is_valid': False
                }
            
            # Check if dictionary is not empty
            if not data:
                return {
                    'status': 'error',
                    'message': 'Mixed data dictionary cannot be empty',
                    'is_valid': False
                }
            
            # Check each item in the dictionary
            for key, data_info in data.items():
                if not isinstance(data_info, dict) or 'data' not in data_info or 'type' not in data_info:
                    return {
                        'status': 'error',
                        'message': f'Item {key} must be a dict with "data" and "type" fields',
                        'is_valid': False
                    }
            
            return {
                'status': 'success',
                'is_valid': True
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'is_valid': False
            }

# Factory function to create data preprocessors
def create_data_preprocessor(model_id: str, data_type: str) -> DataPreprocessor:
    """Create a data preprocessor for the specified model and data type"""
    return DataPreprocessor(model_id, data_type)

# Initialize data preprocessor factory
def get_data_preprocessor(model_id: str, data_type: str) -> DataPreprocessor:
    """Get a data preprocessor instance"""
    return create_data_preprocessor(model_id, data_type)