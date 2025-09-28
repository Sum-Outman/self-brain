# -*- coding: utf-8 -*-
"""
Model Checkpoint Manager
This module provides functionality to manage model checkpoints during training.
"""

import os
import shutil
import logging
import json
import time
import datetime
import threading
import hashlib
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import tempfile
import zipfile
import pickle
import gc

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ModelCheckpointManager')

class ModelCheckpointManager:
    """Class for managing model checkpoints during training"""
    
    # Singleton instance
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelCheckpointManager, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the model checkpoint manager"""
        # Base directory for checkpoints
        self.checkpoint_dir = "d:\shiyan\web_interface\checkpoints"
        
        # Ensure checkpoint directory exists
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            logger.info(f"Created checkpoint directory: {self.checkpoint_dir}")
        
        # Dictionary to store checkpoint configurations for each model
        self.checkpoint_configs = {}
        
        # Dictionary to track the latest checkpoint for each model
        self.latest_checkpoints = {}
        
        # Dictionary to store checkpoint metadata
        self.checkpoint_metadata = {}
        
        # Load existing checkpoint configurations
        self._load_checkpoint_configs()
        
        # Load existing checkpoint metadata
        self._load_checkpoint_metadata()
    
    def _load_checkpoint_configs(self):
        """Load checkpoint configurations from disk"""
        try:
            config_file = os.path.join(self.checkpoint_dir, "checkpoint_configs.json")
            
            if os.path.isfile(config_file):
                with open(config_file, 'r', encoding='utf-8') as f:
                    self.checkpoint_configs = json.load(f)
                    logger.info(f"Loaded checkpoint configurations")
        except Exception as e:
            logger.error(f"Failed to load checkpoint configurations: {str(e)}")
    
    def _save_checkpoint_configs(self):
        """Save checkpoint configurations to disk"""
        try:
            config_file = os.path.join(self.checkpoint_dir, "checkpoint_configs.json")
            
            # Create a temporary file first to avoid partial writes
            temp_file = config_file + '.tmp'
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_configs, f, ensure_ascii=False, indent=2)
            
            # Replace the original file with the temporary file
            if os.path.exists(config_file):
                os.replace(temp_file, config_file)
            else:
                os.rename(temp_file, config_file)
                
            logger.info(f"Saved checkpoint configurations")
        except Exception as e:
            logger.error(f"Failed to save checkpoint configurations: {str(e)}")
    
    def _load_checkpoint_metadata(self):
        """Load checkpoint metadata from disk"""
        try:
            # Get all model IDs from the directory structure
            model_ids = []
            if os.path.exists(self.checkpoint_dir):
                for item in os.listdir(self.checkpoint_dir):
                    item_path = os.path.join(self.checkpoint_dir, item)
                    if os.path.isdir(item_path) and len(item) == 1 and item.isalpha():
                        model_ids.append(item)
            
            # Load metadata for each model
            for model_id in model_ids:
                metadata_file = os.path.join(self.checkpoint_dir, model_id, "checkpoint_metadata.json")
                
                if os.path.isfile(metadata_file):
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            self.checkpoint_metadata[model_id] = json.load(f)
                            logger.info(f"Loaded checkpoint metadata for model {model_id}")
                    except Exception as e:
                        logger.error(f"Failed to load checkpoint metadata for model {model_id}: {str(e)}")
                        self.checkpoint_metadata[model_id] = {}
                else:
                    self.checkpoint_metadata[model_id] = {}
                
                # Find the latest checkpoint
                self._update_latest_checkpoint(model_id)
        except Exception as e:
            logger.error(f"Failed to load checkpoint metadata: {str(e)}")
    
    def _save_checkpoint_metadata(self, model_id: str):
        """Save checkpoint metadata to disk"""
        try:
            # Ensure model ID is valid
            if model_id not in self.checkpoint_metadata:
                logger.error(f"Invalid model ID: {model_id}")
                return
            
            # Ensure model directory exists
            model_dir = os.path.join(self.checkpoint_dir, model_id)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # Save metadata to file
            metadata_file = os.path.join(model_dir, "checkpoint_metadata.json")
            
            # Create a temporary file first to avoid partial writes
            temp_file = metadata_file + '.tmp'
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.checkpoint_metadata[model_id], f, ensure_ascii=False, indent=2)
            
            # Replace the original file with the temporary file
            if os.path.exists(metadata_file):
                os.replace(temp_file, metadata_file)
            else:
                os.rename(temp_file, metadata_file)
                
            logger.info(f"Saved checkpoint metadata for model {model_id}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint metadata for model {model_id}: {str(e)}")
    
    def _update_latest_checkpoint(self, model_id: str):
        """Update the latest checkpoint for a model"""
        try:
            # Ensure model ID is valid
            if model_id not in self.checkpoint_metadata:
                logger.error(f"Invalid model ID: {model_id}")
                return
            
            # Get all checkpoints for the model
            checkpoints = self.checkpoint_metadata[model_id].values()
            
            if not checkpoints:
                self.latest_checkpoints[model_id] = None
                return
            
            # Find the checkpoint with the highest epoch or latest timestamp
            latest_checkpoint = None
            latest_epoch = -1
            latest_timestamp = 0
            
            for checkpoint in checkpoints:
                # Check if checkpoint has epoch information
                if 'epoch' in checkpoint:
                    if checkpoint['epoch'] > latest_epoch:
                        latest_epoch = checkpoint['epoch']
                        latest_checkpoint = checkpoint
                # Otherwise, use timestamp
                elif 'timestamp' in checkpoint:
                    timestamp = datetime.datetime.fromisoformat(checkpoint['timestamp']).timestamp()
                    if timestamp > latest_timestamp:
                        latest_timestamp = timestamp
                        latest_checkpoint = checkpoint
            
            # Update latest checkpoint
            if latest_checkpoint:
                self.latest_checkpoints[model_id] = latest_checkpoint['checkpoint_id']
                logger.info(f"Updated latest checkpoint for model {model_id}: {self.latest_checkpoints[model_id]}")
            else:
                self.latest_checkpoints[model_id] = None
        except Exception as e:
            logger.error(f"Failed to update latest checkpoint for model {model_id}: {str(e)}")
    
    def set_checkpoint_config(self, model_id: str, config: Dict[str, Any]):
        """Set checkpoint configuration for a model"""
        with self._lock:
            try:
                # Default configuration
                default_config = {
                    'save_interval': 1,  # Save every N epochs
                    'keep_checkpoints': 5,  # Keep the last N checkpoints
                    'save_best_only': True,  # Save only when performance improves
                    'metric_name': 'loss',  # Metric to use for 'save_best_only'
                    'metric_mode': 'min',  # 'min' or 'max' for the metric
                    'save_weights_only': False,  # Save only model weights
                    'save_optimizer': True,  # Save optimizer state
                    'save_metrics': True,  # Save metrics
                    'save_metadata': True,  # Save metadata
                    'checkpoint_dir': os.path.join(self.checkpoint_dir, model_id)  # Custom checkpoint directory
                }
                
                # Update default configuration with provided values
                default_config.update(config)
                
                # Ensure checkpoint directory exists
                if not os.path.exists(default_config['checkpoint_dir']):
                    os.makedirs(default_config['checkpoint_dir'])
                    logger.info(f"Created checkpoint directory: {default_config['checkpoint_dir']}")
                
                # Store configuration
                self.checkpoint_configs[model_id] = default_config
                
                # Ensure metadata is initialized
                if model_id not in self.checkpoint_metadata:
                    self.checkpoint_metadata[model_id] = {}
                
                # Save configuration
                self._save_checkpoint_configs()
                
                logger.info(f"Set checkpoint configuration for model {model_id}")
            except Exception as e:
                logger.error(f"Failed to set checkpoint configuration for model {model_id}: {str(e)}")
    
    def get_checkpoint_config(self, model_id: str) -> Dict[str, Any]:
        """Get checkpoint configuration for a model"""
        with self._lock:
            # Return default configuration if model ID is not found
            if model_id not in self.checkpoint_configs:
                default_config = {
                    'save_interval': 1,
                    'keep_checkpoints': 5,
                    'save_best_only': True,
                    'metric_name': 'loss',
                    'metric_mode': 'min',
                    'save_weights_only': False,
                    'save_optimizer': True,
                    'save_metrics': True,
                    'save_metadata': True,
                    'checkpoint_dir': os.path.join(self.checkpoint_dir, model_id)
                }
                return default_config
            
            return self.checkpoint_configs[model_id].copy()
    
    def save_checkpoint(self, model_id: str, model: Any, optimizer: Any = None, 
                       epoch: int = None, metrics: Dict[str, float] = None, 
                       metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Save a model checkpoint"""
        with self._lock:
            try:
                # Get checkpoint configuration
                config = self.get_checkpoint_config(model_id)
                
                # Create checkpoint ID
                timestamp = datetime.datetime.now().isoformat()
                checkpoint_id = f"checkpoint_{epoch if epoch is not None else 'step'}_{int(time.time())}"
                
                # Create checkpoint directory
                checkpoint_dir = os.path.join(config['checkpoint_dir'], checkpoint_id)
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                
                # Check if we should save this checkpoint (based on 'save_best_only')
                should_save = True
                
                if config['save_best_only'] and metrics and config['metric_name'] in metrics:
                    # Get the metric value
                    current_metric = metrics[config['metric_name']]
                    
                    # Find the best metric value among existing checkpoints
                    best_metric = None
                    
                    for checkpoint in self.checkpoint_metadata.get(model_id, {}).values():
                        if 'metrics' in checkpoint and config['metric_name'] in checkpoint['metrics']:
                            checkpoint_metric = checkpoint['metrics'][config['metric_name']]
                            
                            if best_metric is None:
                                best_metric = checkpoint_metric
                            elif config['metric_mode'] == 'min' and checkpoint_metric < best_metric:
                                best_metric = checkpoint_metric
                            elif config['metric_mode'] == 'max' and checkpoint_metric > best_metric:
                                best_metric = checkpoint_metric
                    
                    # Compare current metric with best metric
                    if best_metric is not None:
                        if config['metric_mode'] == 'min' and current_metric >= best_metric:
                            should_save = False
                        elif config['metric_mode'] == 'max' and current_metric <= best_metric:
                            should_save = False
                
                if not should_save:
                    logger.info(f"Not saving checkpoint {checkpoint_id} for model {model_id} (not better than previous)")
                    return {'error': 'Not better than previous checkpoint'}
                
                # Save model
                model_file = os.path.join(checkpoint_dir, "model.ckpt")
                
                try:
                    # Try to save using model's save method
                    if hasattr(model, 'save'):
                        model.save(model_file)
                    # Try to save using pickle
                    else:
                        with open(model_file, 'wb') as f:
                            pickle.dump(model, f)
                    
                    logger.info(f"Saved model to {model_file}")
                except Exception as e:
                    logger.error(f"Failed to save model: {str(e)}")
                    # Clean up checkpoint directory
                    shutil.rmtree(checkpoint_dir)
                    return {'error': f'Failed to save model: {str(e)}'}
                
                # Save optimizer if requested
                optimizer_file = None
                if optimizer and config['save_optimizer']:
                    optimizer_file = os.path.join(checkpoint_dir, "optimizer.ckpt")
                    
                    try:
                        # Try to save using optimizer's save method
                        if hasattr(optimizer, 'save'):
                            optimizer.save(optimizer_file)
                        # Try to save using pickle
                        else:
                            with open(optimizer_file, 'wb') as f:
                                pickle.dump(optimizer, f)
                        
                        logger.info(f"Saved optimizer to {optimizer_file}")
                    except Exception as e:
                        logger.error(f"Failed to save optimizer: {str(e)}")
                
                # Create checkpoint metadata
                checkpoint_metadata = {
                    'checkpoint_id': checkpoint_id,
                    'timestamp': timestamp,
                    'model_file': model_file,
                    'optimizer_file': optimizer_file,
                    'epoch': epoch,
                    'metrics': metrics if metrics else {},
                    'metadata': metadata if metadata else {},
                    'config': config
                }
                
                # Add metadata to storage
                self.checkpoint_metadata.setdefault(model_id, {})[checkpoint_id] = checkpoint_metadata
                
                # Save metadata
                self._save_checkpoint_metadata(model_id)
                
                # Update latest checkpoint
                self._update_latest_checkpoint(model_id)
                
                # Clean up old checkpoints
                self._cleanup_old_checkpoints(model_id)
                
                logger.info(f"Saved checkpoint {checkpoint_id} for model {model_id}")
                
                return checkpoint_metadata
            except Exception as e:
                logger.error(f"Failed to save checkpoint for model {model_id}: {str(e)}")
                return {'error': str(e)}
    
    def _cleanup_old_checkpoints(self, model_id: str):
        """Clean up old checkpoints"""
        try:
            # Get checkpoint configuration
            config = self.get_checkpoint_config(model_id)
            
            # Get all checkpoints for the model
            checkpoints = list(self.checkpoint_metadata.get(model_id, {}).items())
            
            # Sort checkpoints by epoch or timestamp (newest first)
            checkpoints.sort(key=lambda x: (
                x[1].get('epoch', -1) if x[1].get('epoch') is not None else -1,
                datetime.datetime.fromisoformat(x[1].get('timestamp', '1970-01-01T00:00:00')).timestamp()
            ), reverse=True)
            
            # Determine how many checkpoints to keep
            keep_count = config['keep_checkpoints']
            
            # Delete old checkpoints
            for checkpoint_id, checkpoint_metadata in checkpoints[keep_count:]:
                # Skip if this is the latest checkpoint
                if model_id in self.latest_checkpoints and self.latest_checkpoints[model_id] == checkpoint_id:
                    continue
                
                # Delete checkpoint directory
                checkpoint_dir = os.path.dirname(checkpoint_metadata['model_file'])
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir)
                    logger.info(f"Deleted old checkpoint {checkpoint_id} for model {model_id}")
                
                # Remove from metadata
                del self.checkpoint_metadata[model_id][checkpoint_id]
            
            # Save updated metadata
            self._save_checkpoint_metadata(model_id)
            
            # Update latest checkpoint
            self._update_latest_checkpoint(model_id)
        except Exception as e:
            logger.error(f"Failed to clean up old checkpoints for model {model_id}: {str(e)}")
    
    def load_checkpoint(self, model_id: str, checkpoint_id: str = None) -> Dict[str, Any]:
        """Load a model checkpoint"""
        with self._lock:
            try:
                # If no checkpoint ID is provided, load the latest one
                if checkpoint_id is None:
                    if model_id not in self.latest_checkpoints or self.latest_checkpoints[model_id] is None:
                        logger.error(f"No checkpoint found for model {model_id}")
                        return {'error': 'No checkpoint found'}
                    
                    checkpoint_id = self.latest_checkpoints[model_id]
                
                # Check if checkpoint exists
                if model_id not in self.checkpoint_metadata or checkpoint_id not in self.checkpoint_metadata[model_id]:
                    logger.error(f"Checkpoint {checkpoint_id} not found for model {model_id}")
                    return {'error': 'Checkpoint not found'}
                
                # Get checkpoint metadata
                checkpoint_metadata = self.checkpoint_metadata[model_id][checkpoint_id]
                
                # Check if model file exists
                model_file = checkpoint_metadata['model_file']
                if not os.path.isfile(model_file):
                    logger.error(f"Model file not found: {model_file}")
                    return {'error': 'Model file not found'}
                
                # Load model
                model = None
                try:
                    # Try to load using pickle first
                    try:
                        with open(model_file, 'rb') as f:
                            model = pickle.load(f)
                    except Exception:
                        # If pickle fails, try to use a more specific loading method
                        # This is a placeholder for framework-specific loading
                        # For example, for TensorFlow/Keras, you might use tf.keras.models.load_model
                        # For PyTorch, you might use torch.load
                        logger.error(f"Failed to load model with pickle, trying other methods")
                        
                        # For now, we'll just re-raise the exception
                        raise
                    
                    logger.info(f"Loaded model from {model_file}")
                except Exception as e:
                    logger.error(f"Failed to load model: {str(e)}")
                    return {'error': f'Failed to load model: {str(e)}'}
                
                # Load optimizer if available
                optimizer = None
                optimizer_file = checkpoint_metadata.get('optimizer_file')
                if optimizer_file and os.path.isfile(optimizer_file):
                    try:
                        # Try to load using pickle first
                        try:
                            with open(optimizer_file, 'rb') as f:
                                optimizer = pickle.load(f)
                        except Exception:
                            # Similar to model loading, this is a placeholder for framework-specific loading
                            logger.error(f"Failed to load optimizer with pickle, trying other methods")
                            
                            # For now, we'll just re-raise the exception
                            raise
                        
                        logger.info(f"Loaded optimizer from {optimizer_file}")
                    except Exception as e:
                        logger.warning(f"Failed to load optimizer: {str(e)}")
                
                # Return loaded objects and metadata
                result = {
                    'model': model,
                    'optimizer': optimizer,
                    'checkpoint_id': checkpoint_id,
                    'metadata': checkpoint_metadata
                }
                
                logger.info(f"Loaded checkpoint {checkpoint_id} for model {model_id}")
                
                return result
            except Exception as e:
                logger.error(f"Failed to load checkpoint for model {model_id}: {str(e)}")
                return {'error': str(e)}
    
    def list_checkpoints(self, model_id: str) -> List[Dict[str, Any]]:
        """List all checkpoints for a model"""
        with self._lock:
            # Check if model ID is valid
            if model_id not in self.checkpoint_metadata:
                return []
            
            # Get all checkpoints
            checkpoints = list(self.checkpoint_metadata[model_id].values())
            
            # Sort checkpoints by epoch or timestamp (newest first)
            checkpoints.sort(key=lambda x: (
                x.get('epoch', -1) if x.get('epoch') is not None else -1,
                datetime.datetime.fromisoformat(x.get('timestamp', '1970-01-01T00:00:00')).timestamp()
            ), reverse=True)
            
            return checkpoints
    
    def get_checkpoint(self, model_id: str, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific checkpoint for a model"""
        with self._lock:
            # Check if model ID and checkpoint ID are valid
            if model_id not in self.checkpoint_metadata or checkpoint_id not in self.checkpoint_metadata[model_id]:
                return None
            
            return self.checkpoint_metadata[model_id][checkpoint_id].copy()
    
    def delete_checkpoint(self, model_id: str, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint"""
        with self._lock:
            try:
                # Check if model ID and checkpoint ID are valid
                if model_id not in self.checkpoint_metadata or checkpoint_id not in self.checkpoint_metadata[model_id]:
                    logger.error(f"Checkpoint {checkpoint_id} not found for model {model_id}")
                    return False
                
                # Get checkpoint metadata
                checkpoint_metadata = self.checkpoint_metadata[model_id][checkpoint_id]
                
                # Check if this is the latest checkpoint
                if model_id in self.latest_checkpoints and self.latest_checkpoints[model_id] == checkpoint_id:
                    logger.error(f"Cannot delete the latest checkpoint {checkpoint_id} for model {model_id}")
                    return False
                
                # Delete checkpoint directory
                checkpoint_dir = os.path.dirname(checkpoint_metadata['model_file'])
                if os.path.exists(checkpoint_dir):
                    shutil.rmtree(checkpoint_dir)
                    logger.info(f"Deleted checkpoint directory {checkpoint_dir}")
                
                # Remove from metadata
                del self.checkpoint_metadata[model_id][checkpoint_id]
                
                # Save updated metadata
                self._save_checkpoint_metadata(model_id)
                
                # Update latest checkpoint
                self._update_latest_checkpoint(model_id)
                
                logger.info(f"Deleted checkpoint {checkpoint_id} for model {model_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete checkpoint {checkpoint_id} for model {model_id}: {str(e)}")
                return False
    
    def get_latest_checkpoint(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest checkpoint for a model"""
        with self._lock:
            # Check if model has a latest checkpoint
            if model_id not in self.latest_checkpoints or self.latest_checkpoints[model_id] is None:
                return None
            
            # Get the latest checkpoint ID
            latest_checkpoint_id = self.latest_checkpoints[model_id]
            
            # Return the checkpoint metadata
            return self.get_checkpoint(model_id, latest_checkpoint_id)
    
    def export_checkpoint(self, model_id: str, checkpoint_id: str, export_path: str) -> bool:
        """Export a checkpoint to a zip file"""
        try:
            # Get checkpoint metadata
            checkpoint_metadata = self.get_checkpoint(model_id, checkpoint_id)
            if checkpoint_metadata is None:
                return False
            
            # Get checkpoint directory
            checkpoint_dir = os.path.dirname(checkpoint_metadata['model_file'])
            if not os.path.exists(checkpoint_dir):
                logger.error(f"Checkpoint directory {checkpoint_dir} does not exist")
                return False
            
            # Ensure export directory exists
            export_dir = os.path.dirname(export_path)
            if export_dir and not os.path.exists(export_dir):
                os.makedirs(export_dir)
            
            # Create zip file
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk through the checkpoint directory
                for root, dirs, files in os.walk(checkpoint_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.join(self.checkpoint_dir, model_id))
                        zipf.write(file_path, arcname)
                
                # Add metadata
                metadata_path = os.path.join(tempfile.gettempdir(), f"{checkpoint_id}_metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(checkpoint_metadata, f, ensure_ascii=False, indent=2)
                zipf.write(metadata_path, f"{checkpoint_id}_metadata.json")
                os.remove(metadata_path)
            
            logger.info(f"Exported checkpoint {checkpoint_id} for model {model_id} to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export checkpoint {checkpoint_id} for model {model_id}: {str(e)}")
            return False
    
    def import_checkpoint(self, model_id: str, import_path: str) -> Dict[str, Any]:
        """Import a checkpoint from a zip file"""
        with self._lock:
            try:
                # Check if import file exists
                if not os.path.isfile(import_path):
                    logger.error(f"Import file does not exist: {import_path}")
                    return {'error': 'Import file does not exist'}
                
                # Create temporary directory for extraction
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Extract the zip file
                    with zipfile.ZipFile(import_path, 'r') as zipf:
                        zipf.extractall(temp_dir)
                    
                    # Find checkpoint directories and metadata
                    checkpoint_dirs = []
                    metadata_file = None
                    
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.endswith('_metadata.json'):
                                metadata_file = os.path.join(root, file)
                                break
                        for dir in dirs:
                            if dir.startswith('checkpoint_'):
                                checkpoint_dirs.append(os.path.join(root, dir))
                        if metadata_file:
                            break
                    
                    if not checkpoint_dirs or len(checkpoint_dirs) != 1:
                        logger.error(f"Invalid import file: {import_path}")
                        return {'error': 'Invalid import file format'}
                    
                    # Load metadata if available
                    if metadata_file:
                        try:
                            with open(metadata_file, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                        except Exception as e:
                            logger.warning(f"Failed to load metadata: {str(e)}")
                            metadata = None
                    else:
                        metadata = None
                    
                    # Get checkpoint directory
                    checkpoint_dir = checkpoint_dirs[0]
                    checkpoint_id = os.path.basename(checkpoint_dir)
                    
                    # Create target checkpoint directory
                    target_checkpoint_dir = os.path.join(self.checkpoint_dir, model_id, checkpoint_id)
                    
                    # Check if checkpoint already exists
                    if os.path.exists(target_checkpoint_dir):
                        # Generate a new checkpoint ID
                        timestamp = int(time.time())
                        base_name = "checkpoint_imported"
                        checkpoint_id = f"{base_name}_{timestamp}"
                        target_checkpoint_dir = os.path.join(self.checkpoint_dir, model_id, checkpoint_id)
                    
                    # Copy checkpoint data
                    shutil.copytree(checkpoint_dir, target_checkpoint_dir)
                    
                    # Create or update metadata
                    if metadata:
                        # Update metadata with new information
                        metadata['checkpoint_id'] = checkpoint_id
                        metadata['model_file'] = os.path.join(target_checkpoint_dir, "model.ckpt")
                        if 'optimizer_file' in metadata:
                            metadata['optimizer_file'] = os.path.join(target_checkpoint_dir, "optimizer.ckpt")
                        metadata['imported_at'] = datetime.datetime.now().isoformat()
                    else:
                        # Create new metadata
                        metadata = {
                            'checkpoint_id': checkpoint_id,
                            'timestamp': datetime.datetime.now().isoformat(),
                            'model_file': os.path.join(target_checkpoint_dir, "model.ckpt"),
                            'optimizer_file': os.path.join(target_checkpoint_dir, "optimizer.ckpt") if os.path.isfile(os.path.join(target_checkpoint_dir, "optimizer.ckpt")) else None,
                            'imported_at': datetime.datetime.now().isoformat(),
                            'metadata': {}
                        }
                    
                    # Add metadata to storage
                    self.checkpoint_metadata.setdefault(model_id, {})[checkpoint_id] = metadata
                    
                    # Save metadata
                    self._save_checkpoint_metadata(model_id)
                    
                    # Update latest checkpoint
                    self._update_latest_checkpoint(model_id)
                    
                    logger.info(f"Imported checkpoint {checkpoint_id} for model {model_id} from {import_path}")
                    
                    return metadata
            except Exception as e:
                logger.error(f"Failed to import checkpoint for model {model_id}: {str(e)}")
                return {'error': str(e)}
    
    def verify_checkpoint(self, model_id: str, checkpoint_id: str) -> Dict[str, Any]:
        """Verify the integrity of a checkpoint"""
        try:
            # Get checkpoint metadata
            checkpoint_metadata = self.get_checkpoint(model_id, checkpoint_id)
            if checkpoint_metadata is None:
                return {'is_valid': False, 'error': 'Checkpoint not found'}
            
            # Check if model file exists
            model_file = checkpoint_metadata['model_file']
            if not os.path.isfile(model_file):
                return {'is_valid': False, 'error': 'Model file not found'}
            
            # Check if optimizer file exists (if specified)
            optimizer_file = checkpoint_metadata.get('optimizer_file')
            if optimizer_file and not os.path.isfile(optimizer_file):
                return {'is_valid': False, 'error': 'Optimizer file not found'}
            
            # Try to load the model to verify integrity
            try:
                # This is a lightweight verification - we're just checking if the file can be opened
                with open(model_file, 'rb') as f:
                    # Read a small portion of the file
                    _ = f.read(4)
                
                # If optimizer file exists, verify it as well
                if optimizer_file:
                    with open(optimizer_file, 'rb') as f:
                        _ = f.read(4)
            except Exception as e:
                return {'is_valid': False, 'error': f'Failed to verify files: {str(e)}'}
            
            # Check file sizes
            model_size = os.path.getsize(model_file)
            optimizer_size = os.path.getsize(optimizer_file) if optimizer_file else 0
            
            # Return verification result
            return {
                'is_valid': True,
                'model_id': model_id,
                'checkpoint_id': checkpoint_id,
                'model_file_size_bytes': model_size,
                'optimizer_file_size_bytes': optimizer_size
            }
        except Exception as e:
            logger.error(f"Failed to verify checkpoint {checkpoint_id} for model {model_id}: {str(e)}")
            return {'is_valid': False, 'error': str(e)}
    
    def calculate_checkpoint_hash(self, model_id: str, checkpoint_id: str) -> str:
        """Calculate SHA256 hash of a checkpoint"""
        try:
            # Get checkpoint metadata
            checkpoint_metadata = self.get_checkpoint(model_id, checkpoint_id)
            if checkpoint_metadata is None:
                return ''
            
            # Get model file
            model_file = checkpoint_metadata['model_file']
            if not os.path.isfile(model_file):
                return ''
            
            # Calculate hash of model file
            sha256_hash = hashlib.sha256()
            with open(model_file, "rb") as f:
                # Read and update hash string value in blocks of 4K
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for checkpoint {checkpoint_id} of model {model_id}: {str(e)}")
            return ''
    
    def get_checkpoint_storage_usage(self) -> Dict[str, Any]:
        """Get summary of checkpoint storage usage"""
        summary = {
            'total_size_bytes': 0,
            'model_usage': {}
        }
        
        try:
            # Check each model
            model_ids = list(self.checkpoint_metadata.keys())
            
            for model_id in model_ids:
                model_usage = {
                    'checkpoint_count': len(self.checkpoint_metadata[model_id]),
                    'total_size_bytes': 0,
                    'latest_checkpoint_id': self.latest_checkpoints.get(model_id)
                }
                
                # Calculate total size for the model
                for checkpoint in self.checkpoint_metadata[model_id].values():
                    # Add model file size
                    if os.path.isfile(checkpoint['model_file']):
                        model_usage['total_size_bytes'] += os.path.getsize(checkpoint['model_file'])
                    
                    # Add optimizer file size if it exists
                    optimizer_file = checkpoint.get('optimizer_file')
                    if optimizer_file and os.path.isfile(optimizer_file):
                        model_usage['total_size_bytes'] += os.path.getsize(optimizer_file)
                
                # Update total size
                summary['total_size_bytes'] += model_usage['total_size_bytes']
                
                summary['model_usage'][model_id] = model_usage
        except Exception as e:
            logger.error(f"Failed to generate checkpoint storage summary: {str(e)}")
        
        return summary
    
    def clear_all_checkpoints(self, model_id: str) -> bool:
        """Clear all checkpoints for a model"""
        with self._lock:
            try:
                # Check if model ID is valid
                if model_id not in self.checkpoint_metadata:
                    logger.error(f"Invalid model ID: {model_id}")
                    return False
                
                # Get model's checkpoint directory
                config = self.get_checkpoint_config(model_id)
                model_checkpoint_dir = config['checkpoint_dir']
                
                # Delete all checkpoint directories
                if os.path.exists(model_checkpoint_dir):
                    shutil.rmtree(model_checkpoint_dir)
                    logger.info(f"Deleted all checkpoint directories for model {model_id}")
                
                # Clear metadata
                self.checkpoint_metadata[model_id] = {}
                
                # Clear latest checkpoint
                if model_id in self.latest_checkpoints:
                    del self.latest_checkpoints[model_id]
                
                # Save metadata
                self._save_checkpoint_metadata(model_id)
                
                logger.info(f"Cleared all checkpoints for model {model_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to clear all checkpoints for model {model_id}: {str(e)}")
                return False
    
    def garbage_collect(self):
        """Run garbage collection to free up memory"""
        try:
            # Force garbage collection
            gc.collect()
            logger.info("Ran garbage collection")
        except Exception as e:
            logger.error(f"Failed to run garbage collection: {str(e)}")

# Initialize the model checkpoint manager
def get_model_checkpoint_manager() -> ModelCheckpointManager:
    """Get the singleton instance of ModelCheckpointManager"""
    return ModelCheckpointManager()