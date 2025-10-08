# -*- coding: utf-8 -*-
"""
Data Version Control
This module provides functionality to manage versions of training data.
"""

import os
import shutil
import logging
import json
import hashlib
import time
import datetime
from typing import Dict, Any, List, Optional, Tuple, Union
import threading
import zipfile
import tempfile
import difflib
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DataVersionControl')

class DataVersionControl:
    """Class for managing versions of training data"""
    
    # Singleton instance
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DataVersionControl, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the data version control system"""
        # Base directory for data storage
        self.base_data_dir = "d:\shiyan\web_interface\data"
        
        # Directory for storing version information
        self.version_dir = os.path.join(self.base_data_dir, "versions")
        
        # Ensure the directories exist
        self._ensure_directories()
        
        # Dictionary to store version metadata
        self.versions = {}
        
        # Load existing versions
        self._load_versions()
        
        # Current active versions for each model
        self.active_versions = {}
        
        # Load active versions
        self._load_active_versions()
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        try:
            # Create base data directory if it doesn't exist
            if not os.path.exists(self.base_data_dir):
                os.makedirs(self.base_data_dir)
                logger.info(f"Created base data directory: {self.base_data_dir}")
            
            # Create version directory if it doesn't exist
            if not os.path.exists(self.version_dir):
                os.makedirs(self.version_dir)
                logger.info(f"Created version directory: {self.version_dir}")
            
            # Create model-specific data directories
            model_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
            for model_id in model_ids:
                model_data_dir = os.path.join(self.base_data_dir, model_id)
                if not os.path.exists(model_data_dir):
                    os.makedirs(model_data_dir)
                    logger.info(f"Created data directory for model {model_id}: {model_data_dir}")
                
                # Create train, val, test subdirectories
                for split in ['train', 'val', 'test']:
                    split_dir = os.path.join(model_data_dir, split)
                    if not os.path.exists(split_dir):
                        os.makedirs(split_dir)
                        logger.info(f"Created {split} directory for model {model_id}: {split_dir}")
        except Exception as e:
            logger.error(f"Failed to create directories: {str(e)}")
    
    def _load_versions(self):
        """Load version metadata from disk"""
        try:
            # Get all model IDs
            model_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
            
            # Load versions for each model
            for model_id in model_ids:
                model_version_file = os.path.join(self.version_dir, f"{model_id}_versions.json")
                
                # If version file exists, load it
                if os.path.isfile(model_version_file):
                    try:
                        with open(model_version_file, 'r', encoding='utf-8') as f:
                            model_versions = json.load(f)
                            self.versions[model_id] = model_versions
                            logger.info(f"Loaded {len(model_versions)} versions for model {model_id}")
                    except Exception as e:
                        logger.error(f"Failed to load versions for model {model_id}: {str(e)}")
                        self.versions[model_id] = []
                else:
                    self.versions[model_id] = []
        except Exception as e:
            logger.error(f"Error loading versions: {str(e)}")
    
    def _load_active_versions(self):
        """Load active versions from disk"""
        try:
            active_versions_file = os.path.join(self.version_dir, "active_versions.json")
            
            # If active versions file exists, load it
            if os.path.isfile(active_versions_file):
                try:
                    with open(active_versions_file, 'r', encoding='utf-8') as f:
                        self.active_versions = json.load(f)
                        logger.info(f"Loaded active versions: {self.active_versions}")
                except Exception as e:
                    logger.error(f"Failed to load active versions: {str(e)}")
                    self.active_versions = {}
        except Exception as e:
            logger.error(f"Error loading active versions: {str(e)}")
    
    def _save_versions(self, model_id: str):
        """Save version metadata to disk"""
        try:
            # Ensure model ID is valid
            if model_id not in self.versions:
                logger.error(f"Invalid model ID: {model_id}")
                return False
            
            # Save versions to file
            model_version_file = os.path.join(self.version_dir, f"{model_id}_versions.json")
            with open(model_version_file, 'w', encoding='utf-8') as f:
                json.dump(self.versions[model_id], f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved versions for model {model_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save versions for model {model_id}: {str(e)}")
            return False
    
    def _save_active_versions(self):
        """Save active versions to disk"""
        try:
            active_versions_file = os.path.join(self.version_dir, "active_versions.json")
            with open(active_versions_file, 'w', encoding='utf-8') as f:
                json.dump(self.active_versions, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Saved active versions")
            return True
        except Exception as e:
            logger.error(f"Failed to save active versions: {str(e)}")
            return False
    
    def create_version(self, model_id: str, version_name: str, description: str = "") -> Dict[str, Any]:
        """Create a new version of the data for a specific model"""
        with self._lock:
            try:
                # Check if model ID is valid
                if model_id not in self.versions:
                    self.versions[model_id] = []
                
                # Generate version ID
                version_id = f"v{len(self.versions[model_id]) + 1}_{int(time.time())}"
                
                # Create version directory
                version_dir = os.path.join(self.version_dir, model_id, version_id)
                if not os.path.exists(version_dir):
                    os.makedirs(version_dir)
                
                # Copy current data to version directory
                model_data_dir = os.path.join(self.base_data_dir, model_id)
                for split in ['train', 'val', 'test']:
                    source_dir = os.path.join(model_data_dir, split)
                    target_dir = os.path.join(version_dir, split)
                    
                    if os.path.exists(source_dir) and os.listdir(source_dir):
                        shutil.copytree(source_dir, target_dir)
                        logger.info(f"Copied {split} data for model {model_id} to version {version_id}")
                
                # Generate data statistics
                statistics = self._generate_statistics(version_dir)
                
                # Create version metadata
                version_metadata = {
                    'version_id': version_id,
                    'version_name': version_name,
                    'description': description,
                    'created_at': datetime.datetime.now().isoformat(),
                    'directory': version_dir,
                    'statistics': statistics
                }
                
                # Add version to metadata
                self.versions[model_id].append(version_metadata)
                
                # Save versions
                self._save_versions(model_id)
                
                logger.info(f"Created new version {version_id} for model {model_id}")
                
                return version_metadata
            except Exception as e:
                logger.error(f"Failed to create version for model {model_id}: {str(e)}")
                return {'error': str(e)}
    
    def _generate_statistics(self, data_dir: str) -> Dict[str, Any]:
        """Generate statistics for the data"""
        statistics = {
            'total_files': 0,
            'total_size_bytes': 0,
            'by_split': {}
        }
        
        try:
            # Check each split directory
            for split in ['train', 'val', 'test']:
                split_dir = os.path.join(data_dir, split)
                
                if os.path.exists(split_dir):
                    split_stats = {
                        'file_count': 0,
                        'size_bytes': 0,
                        'file_types': {}
                    }
                    
                    # Walk through the directory
                    for root, dirs, files in os.walk(split_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            
                            # Update file count and size
                            split_stats['file_count'] += 1
                            file_size = os.path.getsize(file_path)
                            split_stats['size_bytes'] += file_size
                            
                            # Update file types
                            file_ext = os.path.splitext(file)[1].lower()
                            if file_ext not in split_stats['file_types']:
                                split_stats['file_types'][file_ext] = {'count': 0, 'size_bytes': 0}
                            split_stats['file_types'][file_ext]['count'] += 1
                            split_stats['file_types'][file_ext]['size_bytes'] += file_size
                    
                    statistics['total_files'] += split_stats['file_count']
                    statistics['total_size_bytes'] += split_stats['size_bytes']
                    statistics['by_split'][split] = split_stats
        except Exception as e:
            logger.error(f"Failed to generate statistics for {data_dir}: {str(e)}")
        
        return statistics
    
    def list_versions(self, model_id: str) -> List[Dict[str, Any]]:
        """List all versions for a specific model"""
        with self._lock:
            # Check if model ID is valid
            if model_id not in self.versions:
                logger.error(f"Invalid model ID: {model_id}")
                return []
            
            return self.versions[model_id]
    
    def get_version(self, model_id: str, version_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific version for a model"""
        with self._lock:
            # Check if model ID is valid
            if model_id not in self.versions:
                logger.error(f"Invalid model ID: {model_id}")
                return None
            
            # Find the version
            for version in self.versions[model_id]:
                if version['version_id'] == version_id:
                    return version
            
            logger.error(f"Version {version_id} not found for model {model_id}")
            return None
    
    def delete_version(self, model_id: str, version_id: str) -> bool:
        """Delete a specific version"""
        with self._lock:
            try:
                # Check if model ID is valid
                if model_id not in self.versions:
                    logger.error(f"Invalid model ID: {model_id}")
                    return False
                
                # Find the version
                version_index = -1
                version_dir = None
                
                for i, version in enumerate(self.versions[model_id]):
                    if version['version_id'] == version_id:
                        version_index = i
                        version_dir = version['directory']
                        break
                
                if version_index == -1:
                    logger.error(f"Version {version_id} not found for model {model_id}")
                    return False
                
                # Check if this is the active version
                if model_id in self.active_versions and self.active_versions[model_id] == version_id:
                    logger.error(f"Cannot delete active version {version_id} for model {model_id}")
                    return False
                
                # Delete version directory
                if version_dir and os.path.exists(version_dir):
                    shutil.rmtree(version_dir)
                    logger.info(f"Deleted version directory {version_dir}")
                
                # Remove version from metadata
                self.versions[model_id].pop(version_index)
                
                # Save versions
                self._save_versions(model_id)
                
                logger.info(f"Deleted version {version_id} for model {model_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete version {version_id} for model {model_id}: {str(e)}")
                return False
    
    def restore_version(self, model_id: str, version_id: str) -> bool:
        """Restore a specific version to the active data directory"""
        with self._lock:
            try:
                # Get the version
                version = self.get_version(model_id, version_id)
                if version is None:
                    return False
                
                # Get version directory
                version_dir = version['directory']
                if not os.path.exists(version_dir):
                    logger.error(f"Version directory {version_dir} does not exist")
                    return False
                
                # Get model data directory
                model_data_dir = os.path.join(self.base_data_dir, model_id)
                
                # Clear existing data
                for split in ['train', 'val', 'test']:
                    split_dir = os.path.join(model_data_dir, split)
                    
                    if os.path.exists(split_dir):
                        # Remove all files and subdirectories
                        for root, dirs, files in os.walk(split_dir, topdown=False):
                            for file in files:
                                os.remove(os.path.join(root, file))
                            for dir in dirs:
                                os.rmdir(os.path.join(root, dir))
                
                # Copy version data to active directory
                for split in ['train', 'val', 'test']:
                    source_dir = os.path.join(version_dir, split)
                    target_dir = os.path.join(model_data_dir, split)
                    
                    if os.path.exists(source_dir) and os.listdir(source_dir):
                        shutil.copytree(source_dir, target_dir, dirs_exist_ok=True)
                        logger.info(f"Restored {split} data for model {model_id} from version {version_id}")
                
                # Update active version
                self.active_versions[model_id] = version_id
                self._save_active_versions()
                
                logger.info(f"Restored version {version_id} as active for model {model_id}")
                return True
            except Exception as e:
                logger.error(f"Failed to restore version {version_id} for model {model_id}: {str(e)}")
                return False
    
    def get_active_version(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get the currently active version for a model"""
        with self._lock:
            # Check if model has an active version
            if model_id not in self.active_versions:
                return None
            
            # Get the active version ID
            active_version_id = self.active_versions[model_id]
            
            # Return the version metadata
            return self.get_version(model_id, active_version_id)
    
    def compare_versions(self, model_id: str, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """Compare two versions of the data"""
        with self._lock:
            try:
                # Get the versions
                version1 = self.get_version(model_id, version_id1)
                version2 = self.get_version(model_id, version_id2)
                
                if version1 is None or version2 is None:
                    logger.error(f"One or both versions not found for model {model_id}")
                    return {'error': 'One or both versions not found'}
                
                # Compare statistics
                comparison = {
                    'version1': {
                        'version_id': version1['version_id'],
                        'version_name': version1['version_name'],
                        'created_at': version1['created_at']
                    },
                    'version2': {
                        'version_id': version2['version_id'],
                        'version_name': version2['version_name'],
                        'created_at': version2['created_at']
                    },
                    'statistics_diff': self._compare_statistics(version1['statistics'], version2['statistics'])
                }
                
                logger.info(f"Compared versions {version_id1} and {version_id2} for model {model_id}")
                return comparison
            except Exception as e:
                logger.error(f"Failed to compare versions {version_id1} and {version_id2} for model {model_id}: {str(e)}")
                return {'error': str(e)}
    
    def _compare_statistics(self, stats1: Dict[str, Any], stats2: Dict[str, Any]) -> Dict[str, Any]:
        """Compare two sets of statistics"""
        diff = {
            'total_files': stats2.get('total_files', 0) - stats1.get('total_files', 0),
            'total_size_bytes': stats2.get('total_size_bytes', 0) - stats1.get('total_size_bytes', 0),
            'by_split': {}
        }
        
        # Compare split statistics
        all_splits = set(stats1.get('by_split', {}).keys()) | set(stats2.get('by_split', {}).keys())
        
        for split in all_splits:
            split_stats1 = stats1.get('by_split', {}).get(split, {'file_count': 0, 'size_bytes': 0})
            split_stats2 = stats2.get('by_split', {}).get(split, {'file_count': 0, 'size_bytes': 0})
            
            split_diff = {
                'file_count': split_stats2['file_count'] - split_stats1['file_count'],
                'size_bytes': split_stats2['size_bytes'] - split_stats1['size_bytes']
            }
            
            diff['by_split'][split] = split_diff
        
        return diff
    
    def export_version(self, model_id: str, version_id: str, export_path: str) -> bool:
        """Export a version to a zip file"""
        try:
            # Get the version
            version = self.get_version(model_id, version_id)
            if version is None:
                return False
            
            # Get version directory
            version_dir = version['directory']
            if not os.path.exists(version_dir):
                logger.error(f"Version directory {version_dir} does not exist")
                return False
            
            # Ensure export directory exists
            export_dir = os.path.dirname(export_path)
            if export_dir and not os.path.exists(export_dir):
                os.makedirs(export_dir)
            
            # Create zip file
            with zipfile.ZipFile(export_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk through the directory
                for root, dirs, files in os.walk(version_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.join(self.version_dir, model_id))
                        zipf.write(file_path, arcname)
                
                # Add version metadata
                metadata_path = os.path.join(tempfile.gettempdir(), f"{version_id}_metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(version, f, ensure_ascii=False, indent=2)
                zipf.write(metadata_path, f"{version_id}_metadata.json")
                os.remove(metadata_path)
            
            logger.info(f"Exported version {version_id} for model {model_id} to {export_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to export version {version_id} for model {model_id}: {str(e)}")
            return False
    
    def import_version(self, model_id: str, import_path: str) -> Dict[str, Any]:
        """Import a version from a zip file"""
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
                    
                    # Find version directory and metadata
                    version_dirs = []
                    metadata_file = None
                    
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.endswith('_metadata.json'):
                                metadata_file = os.path.join(root, file)
                                break
                        for dir in dirs:
                            if dir.startswith('v'):
                                version_dirs.append(os.path.join(root, dir))
                        if metadata_file:
                            break
                    
                    if not version_dirs or len(version_dirs) != 1:
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
                    
                    # Get version directory
                    version_dir = version_dirs[0]
                    version_id = os.path.basename(version_dir)
                    
                    # Create target version directory
                    target_version_dir = os.path.join(self.version_dir, model_id, version_id)
                    if os.path.exists(target_version_dir):
                        # Generate a new version ID
                        version_id = f"v{len(self.versions[model_id]) + 1}_{int(time.time())}"
                        target_version_dir = os.path.join(self.version_dir, model_id, version_id)
                    
                    # Copy version data
                    shutil.copytree(version_dir, target_version_dir)
                    
                    # Generate statistics
                    statistics = self._generate_statistics(target_version_dir)
                    
                    # Create or update metadata
                    if metadata:
                        # Update metadata with new information
                        metadata['version_id'] = version_id
                        metadata['directory'] = target_version_dir
                        metadata['statistics'] = statistics
                        metadata['imported_at'] = datetime.datetime.now().isoformat()
                    else:
                        # Create new metadata
                        metadata = {
                            'version_id': version_id,
                            'version_name': f'Imported version {version_id}',
                            'description': 'Version imported from external source',
                            'created_at': datetime.datetime.now().isoformat(),
                            'directory': target_version_dir,
                            'statistics': statistics,
                            'imported_at': datetime.datetime.now().isoformat()
                        }
                    
                    # Add version to metadata
                    self.versions[model_id].append(metadata)
                    
                    # Save versions
                    self._save_versions(model_id)
                    
                    logger.info(f"Imported version {version_id} for model {model_id} from {import_path}")
                    
                    return metadata
            except Exception as e:
                logger.error(f"Failed to import version for model {model_id}: {str(e)}")
                return {'error': str(e)}
    
    def calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file"""
        try:
            if not os.path.isfile(file_path):
                logger.error(f"File does not exist: {file_path}")
                return ''
            
            sha256_hash = hashlib.sha256()
            with open(file_path, "rb") as f:
                # Read and update hash string value in blocks of 4K
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            
            return sha256_hash.hexdigest()
        except Exception as e:
            logger.error(f"Failed to calculate hash for {file_path}: {str(e)}")
            return ''
    
    def verify_data_integrity(self, model_id: str, version_id: str) -> Dict[str, Any]:
        """Verify the integrity of a version's data"""
        try:
            # Get the version
            version = self.get_version(model_id, version_id)
            if version is None:
                return {'is_valid': False, 'error': 'Version not found'}
            
            # Check if version directory exists
            version_dir = version['directory']
            if not os.path.exists(version_dir):
                return {'is_valid': False, 'error': 'Version directory does not exist'}
            
            # Generate current statistics
            current_statistics = self._generate_statistics(version_dir)
            
            # Compare with stored statistics
            stored_statistics = version['statistics']
            
            is_valid = True
            issues = []
            
            # Check total files
            if current_statistics['total_files'] != stored_statistics['total_files']:
                is_valid = False
                issues.append(f"Total files mismatch: expected {stored_statistics['total_files']}, got {current_statistics['total_files']}")
            
            # Check total size
            if current_statistics['total_size_bytes'] != stored_statistics['total_size_bytes']:
                is_valid = False
                issues.append(f"Total size mismatch: expected {stored_statistics['total_size_bytes']} bytes, got {current_statistics['total_size_bytes']} bytes")
            
            # Check by split
            for split in stored_statistics.get('by_split', {}).keys():
                if split not in current_statistics.get('by_split', {}):
                    is_valid = False
                    issues.append(f"Split {split} is missing")
                    continue
                
                stored_split = stored_statistics['by_split'][split]
                current_split = current_statistics['by_split'][split]
                
                if stored_split['file_count'] != current_split['file_count']:
                    is_valid = False
                    issues.append(f"File count mismatch for split {split}: expected {stored_split['file_count']}, got {current_split['file_count']}")
                
                if stored_split['size_bytes'] != current_split['size_bytes']:
                    is_valid = False
                    issues.append(f"Size mismatch for split {split}: expected {stored_split['size_bytes']} bytes, got {current_split['size_bytes']} bytes")
            
            result = {
                'is_valid': is_valid,
                'version_id': version_id,
                'model_id': model_id,
                'issues': issues
            }
            
            if is_valid:
                logger.info(f"Data integrity verified for version {version_id} of model {model_id}")
            else:
                logger.warning(f"Data integrity issues found for version {version_id} of model {model_id}")
            
            return result
        except Exception as e:
            logger.error(f"Failed to verify data integrity for version {version_id} of model {model_id}: {str(e)}")
            return {'is_valid': False, 'error': str(e)}
    
    def clean_old_versions(self, model_id: str, keep_latest: int = 5) -> List[str]:
        """Clean old versions, keeping only the latest specified number"""
        with self._lock:
            try:
                # Check if model ID is valid
                if model_id not in self.versions:
                    logger.error(f"Invalid model ID: {model_id}")
                    return []
                
                # Get versions sorted by creation time (newest first)
                versions = sorted(self.versions[model_id], key=lambda v: v['created_at'], reverse=True)
                
                # Determine which versions to delete
                versions_to_delete = versions[keep_latest:]
                deleted_versions = []
                
                for version in versions_to_delete:
                    version_id = version['version_id']
                    
                    # Skip if this is the active version
                    if model_id in self.active_versions and self.active_versions[model_id] == version_id:
                        continue
                    
                    # Delete the version
                    if self.delete_version(model_id, version_id):
                        deleted_versions.append(version_id)
                
                logger.info(f"Cleaned up {len(deleted_versions)} old versions for model {model_id}")
                return deleted_versions
            except Exception as e:
                logger.error(f"Failed to clean old versions for model {model_id}: {str(e)}")
                return []
    
    def get_data_usage_summary(self) -> Dict[str, Any]:
        """Get summary of data usage across all models"""
        summary = {
            'total_size_bytes': 0,
            'model_usage': {}
        }
        
        try:
            # Check each model
            model_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
            
            for model_id in model_ids:
                model_usage = {
                    'active_data_size_bytes': 0,
                    'versions_size_bytes': 0,
                    'version_count': len(self.versions.get(model_id, []))
                }
                
                # Calculate active data size
                model_data_dir = os.path.join(self.base_data_dir, model_id)
                if os.path.exists(model_data_dir):
                    active_size = 0
                    for root, dirs, files in os.walk(model_data_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            active_size += os.path.getsize(file_path)
                    model_usage['active_data_size_bytes'] = active_size
                
                # Calculate versions data size
                model_version_dir = os.path.join(self.version_dir, model_id)
                if os.path.exists(model_version_dir):
                    versions_size = 0
                    for root, dirs, files in os.walk(model_version_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            versions_size += os.path.getsize(file_path)
                    model_usage['versions_size_bytes'] = versions_size
                
                # Update total size
                total_model_size = model_usage['active_data_size_bytes'] + model_usage['versions_size_bytes']
                model_usage['total_size_bytes'] = total_model_size
                summary['total_size_bytes'] += total_model_size
                
                summary['model_usage'][model_id] = model_usage
        except Exception as e:
            logger.error(f"Failed to generate data usage summary: {str(e)}")
        
        return summary
    
    def find_files(self, model_id: str, pattern: str) -> List[Dict[str, Any]]:
        """Find files matching a pattern in a model's data"""
        results = []
        
        try:
            # Compile the regex pattern
            regex = re.compile(pattern)
            
            # Search in active data directory
            model_data_dir = os.path.join(self.base_data_dir, model_id)
            if os.path.exists(model_data_dir):
                for root, dirs, files in os.walk(model_data_dir):
                    for file in files:
                        if regex.search(file):
                            file_path = os.path.join(root, file)
                            file_info = {
                                'file_name': file,
                                'path': file_path,
                                'size_bytes': os.path.getsize(file_path),
                                'last_modified': datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                                'source': 'active'
                            }
                            results.append(file_info)
            
            # Search in version directories
            model_version_dir = os.path.join(self.version_dir, model_id)
            if os.path.exists(model_version_dir):
                for version_dir in os.listdir(model_version_dir):
                    version_path = os.path.join(model_version_dir, version_dir)
                    if os.path.isdir(version_path):
                        for root, dirs, files in os.walk(version_path):
                            for file in files:
                                if regex.search(file):
                                    file_path = os.path.join(root, file)
                                    file_info = {
                                        'file_name': file,
                                        'path': file_path,
                                        'size_bytes': os.path.getsize(file_path),
                                        'last_modified': datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat(),
                                        'source': f'version_{version_dir}'
                                    }
                                    results.append(file_info)
        except Exception as e:
            logger.error(f"Failed to find files for model {model_id} with pattern {pattern}: {str(e)}")
        
        return results

# Initialize the data version control system
def get_data_version_control() -> DataVersionControl:
    """Get the singleton instance of DataVersionControl"""
    return DataVersionControl()