# -*- coding: utf-8 -*-
"""
Data Loader Module
This module provides tools for loading and preprocessing training data for all models.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import cv2
import librosa
import random
import threading
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union, Generator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DataLoader')

class DataLoader:
    """Base class for data loading and preprocessing"""
    
    def __init__(self, data_dir: str, model_type: str):
        """Initialize the data loader"""
        self.data_dir = data_dir
        self.model_type = model_type
        self.cache = {}
        self.cache_lock = threading.Lock()
        self.data_stats = {
            'total_samples': 0,
            'loaded_samples': 0,
            'preprocessing_time': 0,
            'data_types': {},
            'last_updated': datetime.now().isoformat()
        }
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize specific data loader based on model type
        self.specific_loader = self._get_specific_loader()
    
    def _get_specific_loader(self):
        """Get the specific data loader based on model type"""
        loaders = {
            'management': ManagementDataLoader,
            'language': LanguageDataLoader,
            'audio': AudioDataLoader,
            'image': ImageDataLoader,
            'video': VideoDataLoader,
            'spatial': SpatialDataLoader,
            'sensor': SensorDataLoader,
            'computer': ComputerControlDataLoader,
            'motion': MotionControlDataLoader,
            'knowledge': KnowledgeBaseDataLoader,
            'programming': ProgrammingDataLoader
        }
        
        if self.model_type in loaders:
            return loaders[self.model_type](self.data_dir)
        else:
            logger.warning(f"No specific loader found for model type {self.model_type}, using base loader")
            return BaseDataLoader(self.data_dir)
    
    def load_data(self, split: str = 'train', batch_size: int = 32, shuffle: bool = True) -> Generator[Dict, None, None]:
        """Load data for training, validation, or testing"""
        try:
            # Delegate to specific loader
            for batch in self.specific_loader.load_data(split, batch_size, shuffle):
                # Update statistics
                self.data_stats['loaded_samples'] += len(batch.get('X', []))
                self.data_stats['last_updated'] = datetime.now().isoformat()
                
                yield batch
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            # Return empty batch in case of error
            yield {'X': [], 'y': []}
    
    def preprocess_data(self, raw_data: Any, **kwargs) -> Any:
        """Preprocess raw data"""
        start_time = datetime.now()
        
        try:
            # Delegate to specific loader
            preprocessed = self.specific_loader.preprocess_data(raw_data, **kwargs)
            
            # Update preprocessing time
            elapsed = (datetime.now() - start_time).total_seconds()
            self.data_stats['preprocessing_time'] += elapsed
            
            return preprocessed
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            return None
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded data"""
        # Update with specific loader stats
        specific_stats = self.specific_loader.get_data_stats()
        combined_stats = {**self.data_stats, **specific_stats}
        
        return combined_stats
    
    def save_data(self, data: Any, filename: str, split: str = 'train') -> bool:
        """Save processed data to disk"""
        try:
            # Create split directory
            split_dir = os.path.join(self.data_dir, split)
            os.makedirs(split_dir, exist_ok=True)
            
            # Save data using specific loader
            return self.specific_loader.save_data(data, filename, split_dir)
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of the available data"""
        try:
            # Delegate to specific loader
            return self.specific_loader.get_data_summary()
        except Exception as e:
            logger.error(f"Error getting data summary: {str(e)}")
            return {'error': str(e)}
    
    def clear_cache(self) -> None:
        """Clear the data cache"""
        with self.cache_lock:
            self.cache.clear()
            logger.info("Data cache cleared")
    
    def load_dataset_from_file(self, file_path: str) -> Any:
        """Load a dataset from a file"""
        # Check if file exists in cache
        if file_path in self.cache:
            return self.cache[file_path]
        
        try:
            # Delegate to specific loader
            data = self.specific_loader.load_dataset_from_file(file_path)
            
            # Cache the loaded data
            with self.cache_lock:
                self.cache[file_path] = data
            
            return data
        except Exception as e:
            logger.error(f"Error loading dataset from file: {str(e)}")
            return None

class BaseDataLoader:
    """Base data loader implementation"""
    
    def __init__(self, data_dir: str):
        """Initialize the base data loader"""
        self.data_dir = data_dir
        self.data_stats = {
            'total_samples': 0,
            'data_distribution': {},
            'file_formats': {}
        }
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize splits
        self.splits = ['train', 'validation', 'test']
        for split in self.splits:
            os.makedirs(os.path.join(self.data_dir, split), exist_ok=True)
    
    def load_data(self, split: str = 'train', batch_size: int = 32, shuffle: bool = True) -> Generator[Dict, None, None]:
        """Load data in batches"""
        # Validate split
        if split not in self.splits:
            logger.error(f"Invalid split: {split}")
            yield {'X': [], 'y': []}
            return
        
        # Get split directory
        split_dir = os.path.join(self.data_dir, split)
        
        # Get all files in the split directory
        try:
            files = [f for f in os.listdir(split_dir) if os.path.isfile(os.path.join(split_dir, f))]
            
            # Update statistics
            self.data_stats['total_samples'] = len(files)
            
            # Shuffle if requested
            if shuffle:
                random.shuffle(files)
            
            # Generate batches
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i + batch_size]
                batch = {'X': [], 'y': []}
                
                for file in batch_files:
                    file_path = os.path.join(split_dir, file)
                    
                    # Load file based on extension
                    if file.endswith('.json'):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            batch['X'].append(data.get('input', {}))
                            batch['y'].append(data.get('output', {}))
                    elif file.endswith('.npy'):
                        data = np.load(file_path, allow_pickle=True)
                        if len(data) >= 2:
                            batch['X'].append(data[0])
                            batch['y'].append(data[1])
                    
                # Only yield non-empty batches
                if batch['X']:
                    yield batch
        except Exception as e:
            logger.error(f"Error loading data from {split_dir}: {str(e)}")
            yield {'X': [], 'y': []}
    
    def preprocess_data(self, raw_data: Any, **kwargs) -> Any:
        """Preprocess raw data"""
        # Base implementation does minimal preprocessing
        return raw_data
    
    def get_data_stats(self) -> Dict[str, Any]:
        """Get data statistics"""
        return self.data_stats
    
    def save_data(self, data: Any, filename: str, split_dir: str) -> bool:
        """Save data to disk"""
        try:
            file_path = os.path.join(split_dir, filename)
            
            # Save based on data type
            if isinstance(data, dict) or isinstance(data, list):
                with open(file_path + '.json', 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            elif isinstance(data, np.ndarray):
                np.save(file_path + '.npy', data)
            else:
                # For other types, try to convert to JSON
                with open(file_path + '.json', 'w', encoding='utf-8') as f:
                    json.dump(str(data), f, ensure_ascii=False)
            
            # Update statistics
            self.data_stats['total_samples'] += 1
            
            return True
        except Exception as e:
            logger.error(f"Error saving data to {file_path}: {str(e)}")
            return False
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get a summary of the available data"""
        summary = {
            'data_dir': self.data_dir,
            'splits': {},
            'total_files': 0
        }
        
        # Count files in each split
        for split in self.splits:
            split_dir = os.path.join(self.data_dir, split)
            try:
                files = [f for f in os.listdir(split_dir) if os.path.isfile(os.path.join(split_dir, f))]
                summary['splits'][split] = len(files)
                summary['total_files'] += len(files)
            except Exception as e:
                logger.error(f"Error accessing {split_dir}: {str(e)}")
                summary['splits'][split] = 0
        
        return summary
    
    def load_dataset_from_file(self, file_path: str) -> Any:
        """Load dataset from file"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            elif file_path.endswith('.npy'):
                return np.load(file_path, allow_pickle=True)
            elif file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return None
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return None

class ManagementDataLoader(BaseDataLoader):
    """Data loader for management model"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.data_stats['model_type'] = 'management'

class LanguageDataLoader(BaseDataLoader):
    """Data loader for language model"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.data_stats['model_type'] = 'language'
        self.vocabulary = set()
    
    def preprocess_data(self, raw_data: str, **kwargs) -> Dict[str, Any]:
        """Preprocess text data"""
        # Basic text preprocessing
        text = raw_data.lower().strip()
        
        # Update vocabulary
        words = text.split()
        self.vocabulary.update(words)
        
        return {
            'text': text,
            'word_count': len(words),
            'char_count': len(text)
        }

class AudioDataLoader(BaseDataLoader):
    """Data loader for audio processing model"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.data_stats['model_type'] = 'audio'
    
    def load_data(self, split: str = 'train', batch_size: int = 32, shuffle: bool = True) -> Generator[Dict, None, None]:
        """Load audio data"""
        # Validate split
        if split not in self.splits:
            logger.error(f"Invalid split: {split}")
            yield {'X': [], 'y': []}
            return
        
        # Get split directory
        split_dir = os.path.join(self.data_dir, split)
        
        # Get all audio files in the split directory
        try:
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
            files = [f for f in os.listdir(split_dir) 
                     if os.path.isfile(os.path.join(split_dir, f)) 
                     and any(f.endswith(ext) for ext in audio_extensions)]
            
            # Update statistics
            self.data_stats['total_samples'] = len(files)
            
            # Shuffle if requested
            if shuffle:
                random.shuffle(files)
            
            # Generate batches
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i + batch_size]
                batch = {'X': [], 'y': []}
                
                for file in batch_files:
                    file_path = os.path.join(split_dir, file)
                    
                    try:
                        # Load audio file
                        y, sr = librosa.load(file_path, sr=16000)
                        
                        # Extract features (MFCC)
                        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                        mfcc = np.mean(mfcc.T, axis=0)
                        
                        batch['X'].append(mfcc)
                        # Assume filename contains label
                        label = file.split('_')[0] if '_' in file else 'unknown'
                        batch['y'].append(label)
                    except Exception as e:
                        logger.error(f"Error loading audio file {file_path}: {str(e)}")
                        continue
                    
                # Only yield non-empty batches
                if batch['X']:
                    yield batch
        except Exception as e:
            logger.error(f"Error loading audio data from {split_dir}: {str(e)}")
            yield {'X': [], 'y': []}

class ImageDataLoader(BaseDataLoader):
    """Data loader for image processing model"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.data_stats['model_type'] = 'image'
        self.image_size = (224, 224)  # Default image size for processing
    
    def load_data(self, split: str = 'train', batch_size: int = 32, shuffle: bool = True) -> Generator[Dict, None, None]:
        """Load image data"""
        # Validate split
        if split not in self.splits:
            logger.error(f"Invalid split: {split}")
            yield {'X': [], 'y': []}
            return
        
        # Get split directory
        split_dir = os.path.join(self.data_dir, split)
        
        # Get all image files in the split directory
        try:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            files = [f for f in os.listdir(split_dir) 
                     if os.path.isfile(os.path.join(split_dir, f)) 
                     and any(f.endswith(ext.lower()) for ext in image_extensions)]
            
            # Update statistics
            self.data_stats['total_samples'] = len(files)
            
            # Shuffle if requested
            if shuffle:
                random.shuffle(files)
            
            # Generate batches
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i + batch_size]
                batch = {'X': [], 'y': []}
                
                for file in batch_files:
                    file_path = os.path.join(split_dir, file)
                    
                    try:
                        # Load image
                        image = cv2.imread(file_path)
                        if image is None:
                            continue
                        
                        # Resize image
                        image = cv2.resize(image, self.image_size)
                        
                        # Convert to RGB (OpenCV loads as BGR)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        # Normalize pixel values
                        image = image / 255.0
                        
                        batch['X'].append(image)
                        # Assume filename contains label
                        label = file.split('_')[0] if '_' in file else 'unknown'
                        batch['y'].append(label)
                    except Exception as e:
                        logger.error(f"Error loading image file {file_path}: {str(e)}")
                        continue
                    
                # Only yield non-empty batches
                if batch['X']:
                    yield batch
        except Exception as e:
            logger.error(f"Error loading image data from {split_dir}: {str(e)}")
            yield {'X': [], 'y': []}

class VideoDataLoader(BaseDataLoader):
    """Data loader for video processing model"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.data_stats['model_type'] = 'video'
        self.frame_size = (224, 224)  # Default frame size for processing
        self.frames_per_video = 16  # Default number of frames per video sample
    
    def load_data(self, split: str = 'train', batch_size: int = 8, shuffle: bool = True) -> Generator[Dict, None, None]:
        """Load video data"""
        # Video loading is memory intensive, so use smaller batch size
        batch_size = min(batch_size, 8)
        
        # Validate split
        if split not in self.splits:
            logger.error(f"Invalid split: {split}")
            yield {'X': [], 'y': []}
            return
        
        # Get split directory
        split_dir = os.path.join(self.data_dir, split)
        
        # Get all video files in the split directory
        try:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            files = [f for f in os.listdir(split_dir) 
                     if os.path.isfile(os.path.join(split_dir, f)) 
                     and any(f.endswith(ext.lower()) for ext in video_extensions)]
            
            # Update statistics
            self.data_stats['total_samples'] = len(files)
            
            # Shuffle if requested
            if shuffle:
                random.shuffle(files)
            
            # Generate batches
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i + batch_size]
                batch = {'X': [], 'y': []}
                
                for file in batch_files:
                    file_path = os.path.join(split_dir, file)
                    
                    try:
                        # Load video
                        cap = cv2.VideoCapture(file_path)
                        frames = []
                        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        
                        # Sample frames uniformly
                        if frame_count > 0:
                            frame_indices = sorted(random.sample(range(frame_count), min(self.frames_per_video, frame_count)))
                            
                            for idx in frame_indices:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                                ret, frame = cap.read()
                                if ret:
                                    # Resize frame
                                    frame = cv2.resize(frame, self.frame_size)
                                    # Convert to RGB
                                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    # Normalize pixel values
                                    frame = frame / 255.0
                                    frames.append(frame)
                        
                        cap.release()
                        
                        # Only add if we have frames
                        if frames:
                            batch['X'].append(np.array(frames))
                            # Assume filename contains label
                            label = file.split('_')[0] if '_' in file else 'unknown'
                            batch['y'].append(label)
                    except Exception as e:
                        logger.error(f"Error loading video file {file_path}: {str(e)}")
                        continue
                    
                # Only yield non-empty batches
                if batch['X']:
                    yield batch
        except Exception as e:
            logger.error(f"Error loading video data from {split_dir}: {str(e)}")
            yield {'X': [], 'y': []}

class SpatialDataLoader(BaseDataLoader):
    """Data loader for spatial perception model"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.data_stats['model_type'] = 'spatial'

class SensorDataLoader(BaseDataLoader):
    """Data loader for sensor perception model"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.data_stats['model_type'] = 'sensor'
    
    def load_data(self, split: str = 'train', batch_size: int = 32, shuffle: bool = True) -> Generator[Dict, None, None]:
        """Load sensor data"""
        # Validate split
        if split not in self.splits:
            logger.error(f"Invalid split: {split}")
            yield {'X': [], 'y': []}
            return
        
        # Get split directory
        split_dir = os.path.join(self.data_dir, split)
        
        # Get all sensor data files in the split directory
        try:
            sensor_extensions = ['.csv', '.json', '.npy']
            files = [f for f in os.listdir(split_dir) 
                     if os.path.isfile(os.path.join(split_dir, f)) 
                     and any(f.endswith(ext.lower()) for ext in sensor_extensions)]
            
            # Update statistics
            self.data_stats['total_samples'] = len(files)
            
            # Shuffle if requested
            if shuffle:
                random.shuffle(files)
            
            # Generate batches
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i + batch_size]
                batch = {'X': [], 'y': []}
                
                for file in batch_files:
                    file_path = os.path.join(split_dir, file)
                    
                    try:
                        if file.endswith('.csv'):
                            # Load CSV data
                            df = pd.read_csv(file_path)
                            # Assume first columns are features, last column is target
                            if len(df.columns) > 1:
                                X = df.iloc[:, :-1].values
                                y = df.iloc[:, -1].values
                                batch['X'].append(X)
                                batch['y'].append(y)
                        elif file.endswith('.json'):
                            # Load JSON data
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                if 'features' in data and 'target' in data:
                                    batch['X'].append(data['features'])
                                    batch['y'].append(data['target'])
                        elif file.endswith('.npy'):
                            # Load numpy data
                            data = np.load(file_path, allow_pickle=True)
                            if len(data.shape) >= 2:
                                X = data[:, :-1]
                                y = data[:, -1]
                                batch['X'].append(X)
                                batch['y'].append(y)
                    except Exception as e:
                        logger.error(f"Error loading sensor data file {file_path}: {str(e)}")
                        continue
                    
                # Only yield non-empty batches
                if batch['X']:
                    yield batch
        except Exception as e:
            logger.error(f"Error loading sensor data from {split_dir}: {str(e)}")
            yield {'X': [], 'y': []}

class ComputerControlDataLoader(BaseDataLoader):
    """Data loader for computer control model"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.data_stats['model_type'] = 'computer'

class MotionControlDataLoader(BaseDataLoader):
    """Data loader for motion control model"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.data_stats['model_type'] = 'motion'

class KnowledgeBaseDataLoader(BaseDataLoader):
    """Data loader for knowledge base model"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.data_stats['model_type'] = 'knowledge'
        self.knowledge_categories = set()
    
    def load_data(self, split: str = 'train', batch_size: int = 32, shuffle: bool = True) -> Generator[Dict, None, None]:
        """Load knowledge base data"""
        # Validate split
        if split not in self.splits:
            logger.error(f"Invalid split: {split}")
            yield {'X': [], 'y': []}
            return
        
        # Get split directory
        split_dir = os.path.join(self.data_dir, split)
        
        # Get all knowledge files in the split directory
        try:
            files = []
            # Look for knowledge files in subdirectories (categories)
            for root, dirs, filenames in os.walk(split_dir):
                for filename in filenames:
                    if filename.endswith('.json') or filename.endswith('.txt'):
                        files.append(os.path.join(root, filename))
            
            # Update statistics
            self.data_stats['total_samples'] = len(files)
            
            # Shuffle if requested
            if shuffle:
                random.shuffle(files)
            
            # Generate batches
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i + batch_size]
                batch = {'X': [], 'y': []}
                
                for file_path in batch_files:
                    try:
                        # Get category from directory name
                        category = os.path.basename(os.path.dirname(file_path))
                        self.knowledge_categories.add(category)
                        
                        # Load knowledge content
                        if file_path.endswith('.json'):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                batch['X'].append(data.get('content', ''))
                                batch['y'].append({'category': category, 'metadata': data.get('metadata', {})})
                        elif file_path.endswith('.txt'):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                batch['X'].append(content)
                                batch['y'].append({'category': category, 'metadata': {}})
                    except Exception as e:
                        logger.error(f"Error loading knowledge file {file_path}: {str(e)}")
                        continue
                    
                # Only yield non-empty batches
                if batch['X']:
                    yield batch
        except Exception as e:
            logger.error(f"Error loading knowledge data from {split_dir}: {str(e)}")
            yield {'X': [], 'y': []}

class ProgrammingDataLoader(BaseDataLoader):
    """Data loader for programming model"""
    
    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.data_stats['model_type'] = 'programming'
        self.supported_languages = set()
    
    def load_data(self, split: str = 'train', batch_size: int = 32, shuffle: bool = True) -> Generator[Dict, None, None]:
        """Load programming data"""
        # Validate split
        if split not in self.splits:
            logger.error(f"Invalid split: {split}")
            yield {'X': [], 'y': []}
            return
        
        # Get split directory
        split_dir = os.path.join(self.data_dir, split)
        
        # Get all code files in the split directory
        try:
            code_extensions = ['.py', '.js', '.java', '.c', '.cpp', '.h', '.html', '.css', '.php', '.rb', '.go']
            files = [f for f in os.listdir(split_dir) 
                     if os.path.isfile(os.path.join(split_dir, f)) 
                     and any(f.endswith(ext) for ext in code_extensions)]
            
            # Update statistics
            self.data_stats['total_samples'] = len(files)
            
            # Shuffle if requested
            if shuffle:
                random.shuffle(files)
            
            # Generate batches
            for i in range(0, len(files), batch_size):
                batch_files = files[i:i + batch_size]
                batch = {'X': [], 'y': []}
                
                for file in batch_files:
                    file_path = os.path.join(split_dir, file)
                    
                    try:
                        # Get language from file extension
                        language = file.split('.')[-1]
                        self.supported_languages.add(language)
                        
                        # Load code content
                        with open(file_path, 'r', encoding='utf-8') as f:
                            code = f.read()
                            batch['X'].append(code)
                            batch['y'].append({'language': language, 'filename': file})
                    except Exception as e:
                        logger.error(f"Error loading code file {file_path}: {str(e)}")
                        continue
                    
                # Only yield non-empty batches
                if batch['X']:
                    yield batch
        except Exception as e:
            logger.error(f"Error loading programming data from {split_dir}: {str(e)}")
            yield {'X': [], 'y': []}

# Factory function to create data loaders
def create_data_loader(model_type: str, data_dir: str = None) -> DataLoader:
    """Create a data loader for the specified model type"""
    # Default data directory
    if data_dir is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, '../../training_data', model_type)
    
    return DataLoader(data_dir, model_type)

# Initialize data loader factory
def get_data_loader(model_type: str, data_dir: str = None) -> DataLoader:
    """Get a data loader instance"""
    return create_data_loader(model_type, data_dir)