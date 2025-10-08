#!/usr/bin/env python
# Self Brain AGI System - Training Data Manager
# Copyright 2025 AGI System Team

import os
import sys
import logging
import json
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
import shutil
import zipfile
import hashlib
import re
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SelfBrainDataManager")

class DataManager:
    """Manages training data for all models"""
    def __init__(self, base_data_dir="data"):
        self.base_data_dir = base_data_dir
        self.supported_formats = [
            ".json", ".jsonl", ".csv", ".txt", ".yaml", ".yml",
            ".npy", ".npz", ".h5", ".pkl"
        ]
        
        # Create base data directory if it doesn't exist
        os.makedirs(self.base_data_dir, exist_ok=True)
        
    def get_model_data_dir(self, model_id):
        """Get data directory for a specific model"""
        return os.path.join(self.base_data_dir, model_id)
    
    def get_training_data_dir(self, model_id):
        """Get training data directory for a specific model"""
        return os.path.join(self.get_model_data_dir(model_id), "training")
    
    def get_validation_data_dir(self, model_id):
        """Get validation data directory for a specific model"""
        return os.path.join(self.get_model_data_dir(model_id), "validation")
    
    def get_test_data_dir(self, model_id):
        """Get test data directory for a specific model"""
        return os.path.join(self.get_model_data_dir(model_id), "test")
    
    def create_model_data_structure(self, model_id):
        """Create data directory structure for a model"""
        directories = [
            self.get_model_data_dir(model_id),
            self.get_training_data_dir(model_id),
            self.get_validation_data_dir(model_id),
            self.get_test_data_dir(model_id),
            os.path.join(self.get_model_data_dir(model_id), "raw"),
            os.path.join(self.get_model_data_dir(model_id), "processed")
        ]
        
        for dir_path in directories:
            os.makedirs(dir_path, exist_ok=True)
        
        logger.info(f"Created data directories for model {model_id}")
        return directories
    
    def validate_file_format(self, file_path):
        """Validate if file format is supported"""
        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.supported_formats
    
    def load_data(self, file_path):
        """Load data from various file formats"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
        
        if not self.validate_file_format(file_path):
            logger.error(f"Unsupported file format: {file_path}")
            return None
        
        try:
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() == ".json":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            elif ext.lower() == ".jsonl":
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                return data
            
            elif ext.lower() == ".csv":
                return pd.read_csv(file_path).to_dict('records')
            
            elif ext.lower() in [".yaml", ".yml"]:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
            
            elif ext.lower() == ".txt":
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.readlines()
            
            elif ext.lower() == ".npy":
                return np.load(file_path)
            
            elif ext.lower() == ".npz":
                return np.load(file_path)
            
            elif ext.lower() == ".h5":
                import h5py
                data = {}
                with h5py.File(file_path, 'r') as f:
                    for key in f.keys():
                        data[key] = f[key][()]
                return data
            
            elif ext.lower() == ".pkl":
                import pickle
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
            
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {str(e)}")
            return None
    
    def save_data(self, data, file_path):
        """Save data to various file formats"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() == ".json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            elif ext.lower() == ".jsonl":
                with open(file_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            elif ext.lower() == ".csv" and isinstance(data, list) and len(data) > 0:
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False, encoding='utf-8')
            
            elif ext.lower() in [".yaml", ".yml"]:
                with open(file_path, 'w', encoding='utf-8') as f:
                    yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            
            elif ext.lower() == ".txt" and isinstance(data, list):
                with open(file_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(str(item) + "\n")
            
            elif ext.lower() == ".npy" and isinstance(data, np.ndarray):
                np.save(file_path, data)
            
            elif ext.lower() == ".npz" and isinstance(data, dict):
                np.savez(file_path, **data)
            
            elif ext.lower() == ".h5" and isinstance(data, dict):
                import h5py
                with h5py.File(file_path, 'w') as f:
                    for key, value in data.items():
                        f.create_dataset(key, data=value)
            
            elif ext.lower() == ".pkl":
                import pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
            
            logger.info(f"Data saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save data to {file_path}: {str(e)}")
            return False
    
    def import_data(self, model_id, file_path, split_ratio=(0.7, 0.2, 0.1)):
        """Import data for a model and split into train/validation/test sets"""
        try:
            # Create model data structure
            self.create_model_data_structure(model_id)
            
            # Load data
            data = self.load_data(file_path)
            if data is None:
                return False, "Failed to load data"
            
            # Handle different data types
            if isinstance(data, np.ndarray):
                # For numpy arrays, we need to convert to a list of dictionaries
                if len(data.shape) == 1:
                    data = [{"value": float(x)} for x in data]
                else:
                    data = [{"values": x.tolist()} for x in data]
            elif isinstance(data, dict):
                # For dictionaries, convert to list of items
                data = [{"key": k, "value": v} for k, v in data.items()]
            elif isinstance(data, str):
                # For strings, split into lines
                data = [{"text": line.strip()} for line in data.split('\n') if line.strip()]
            
            # Check if data is a list
            if not isinstance(data, list):
                logger.error(f"Data must be a list or convertible to a list, got {type(data)}")
                return False, "Invalid data format"
            
            if len(data) == 0:
                logger.error("No data found in the file")
                return False, "No data found"
            
            # Split data
            train_data, temp_data = train_test_split(data, test_size=1 - split_ratio[0], random_state=42)
            val_data, test_data = train_test_split(temp_data, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]), random_state=42)
            
            # Save split data
            train_file = os.path.join(self.get_training_data_dir(model_id), "data.json")
            val_file = os.path.join(self.get_validation_data_dir(model_id), "data.json")
            test_file = os.path.join(self.get_test_data_dir(model_id), "data.json")
            
            train_success = self.save_data(train_data, train_file)
            val_success = self.save_data(val_data, val_file)
            test_success = self.save_data(test_data, test_file)
            
            if not (train_success and val_success and test_success):
                return False, "Failed to save split data"
            
            # Copy original file to raw directory
            raw_file = os.path.join(self.get_model_data_dir(model_id), "raw", os.path.basename(file_path))
            shutil.copy2(file_path, raw_file)
            
            logger.info(f"Imported and split data for model {model_id}: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
            
            return True, {
                "message": "Data imported and split successfully",
                "train_count": len(train_data),
                "validation_count": len(val_data),
                "test_count": len(test_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to import data for model {model_id}: {str(e)}")
            return False, str(e)
    
    def import_batch_data(self, model_id, directory_path, split_ratio=(0.7, 0.2, 0.1)):
        """Import multiple data files for a model"""
        try:
            if not os.path.isdir(directory_path):
                logger.error(f"Directory not found: {directory_path}")
                return False, "Directory not found"
            
            # Create model data structure
            self.create_model_data_structure(model_id)
            
            all_data = []
            imported_files = []
            
            # Iterate over all files in the directory
            for root, _, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    
                    if not self.validate_file_format(file_path):
                        logger.warning(f"Skipping unsupported file format: {file_path}")
                        continue
                    
                    # Load data from file
                    data = self.load_data(file_path)
                    if data is None:
                        continue
                    
                    # Convert to list format if needed
                    if not isinstance(data, list):
                        data = [data]
                    
                    all_data.extend(data)
                    imported_files.append(file_path)
                    
                    # Copy file to raw directory
                    raw_dir = os.path.join(self.get_model_data_dir(model_id), "raw")
                    os.makedirs(raw_dir, exist_ok=True)
                    shutil.copy2(file_path, os.path.join(raw_dir, file))
            
            if not all_data:
                logger.error("No valid data found in the directory")
                return False, "No valid data found"
            
            # Split data
            train_data, temp_data = train_test_split(all_data, test_size=1 - split_ratio[0], random_state=42)
            val_data, test_data = train_test_split(temp_data, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]), random_state=42)
            
            # Save split data
            train_file = os.path.join(self.get_training_data_dir(model_id), "batch_data.json")
            val_file = os.path.join(self.get_validation_data_dir(model_id), "batch_data.json")
            test_file = os.path.join(self.get_test_data_dir(model_id), "batch_data.json")
            
            train_success = self.save_data(train_data, train_file)
            val_success = self.save_data(val_data, val_file)
            test_success = self.save_data(test_data, test_file)
            
            if not (train_success and val_success and test_success):
                return False, "Failed to save split data"
            
            logger.info(f"Imported batch data for model {model_id}: {len(imported_files)} files, {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
            
            return True, {
                "message": "Batch data imported successfully",
                "file_count": len(imported_files),
                "train_count": len(train_data),
                "validation_count": len(val_data),
                "test_count": len(test_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to import batch data for model {model_id}: {str(e)}")
            return False, str(e)
    
    def generate_sample_data(self, model_id, sample_size=100):
        """Generate sample training data for a model"""
        try:
            # Create model data structure
            self.create_model_data_structure(model_id)
            
            # Generate sample data based on model type
            sample_data = []
            
            if model_id == "A_management":
                # Management model sample data
                emotions = ["happy", "sad", "angry", "surprised", "neutral", "helpful"]
                actions = ["summarize", "analyze", "coordinate", "delegate", "report", "assist"]
                
                for i in range(sample_size):
                    sample_data.append({
                        "id": i,
                        "input": f"Please {np.random.choice(actions)} the current system status",
                        "output": f"I've {np.random.choice(actions)}d the system status. All models are operating normally.",
                        "emotion": np.random.choice(emotions),
                        "context": {"models": [f"{chr(66 + j % 10)}_model" for j in range(np.random.randint(1, 5))]},
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif model_id == "B_language":
                # Language model sample data
                languages = ["English", "Chinese", "Spanish", "French", "German"]
                topics = ["technology", "science", "history", "art", "literature", "mathematics"]
                
                for i in range(sample_size):
                    sample_data.append({
                        "id": i,
                        "text": f"This is a {np.random.choice(languages)} text about {np.random.choice(topics)}. It contains sample information for training the language model.",
                        "language": np.random.choice(languages),
                        "topic": np.random.choice(topics),
                        "sentiment": np.random.uniform(-1, 1),
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif model_id.startswith("C_"):
                # Audio model sample data
                audio_types = ["speech", "music", "sound_effect", "noise"]
                emotions = ["happy", "sad", "angry", "calm", "excited"]
                
                for i in range(sample_size):
                    sample_data.append({
                        "id": i,
                        "audio_type": np.random.choice(audio_types),
                        "duration": np.random.uniform(1, 30),
                        "sample_rate": np.random.choice([8000, 16000, 22050, 44100, 48000]),
                        "channels": np.random.choice([1, 2]),
                        "content": f"This is a {np.random.choice(audio_types)} sample with {np.random.choice(emotions)} emotion.",
                        "emotion": np.random.choice(emotions),
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif model_id.startswith("D_") or model_id.startswith("E_"):
                # Image or video model sample data
                categories = ["nature", "city", "people", "animals", "objects", "abstract"]
                resolutions = ["240p", "480p", "720p", "1080p", "4K"]
                
                for i in range(sample_size):
                    sample_data.append({
                        "id": i,
                        "category": np.random.choice(categories),
                        "resolution": np.random.choice(resolutions),
                        "description": f"This is a {np.random.choice(categories)} image with {np.random.choice(resolutions)} resolution.",
                        "tags": [f"tag{i}" for i in range(np.random.randint(1, 5))],
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif model_id.startswith("F_"):
                # Spatial model sample data
                environments = ["indoor", "outdoor", "urban", "natural"]
                
                for i in range(sample_size):
                    sample_data.append({
                        "id": i,
                        "environment": np.random.choice(environments),
                        "points": [{
                            "x": np.random.uniform(-10, 10),
                            "y": np.random.uniform(-10, 10),
                            "z": np.random.uniform(-10, 10),
                            "label": f"point_{j}"
                        } for j in range(np.random.randint(5, 20))],
                        "description": f"This is a {np.random.choice(environments)} spatial model sample.",
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif model_id.startswith("G_"):
                # Sensor model sample data
                sensor_types = ["temperature", "humidity", "pressure", "light", "motion", "acceleration"]
                
                for i in range(sample_size):
                    sample_data.append({
                        "id": i,
                        "sensor_type": np.random.choice(sensor_types),
                        "value": np.random.uniform(0, 100),
                        "unit": f"unit_{np.random.randint(1, 10)}",
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif model_id.startswith("H_"):
                # Computer control model sample data
                commands = ["open", "close", "read", "write", "execute", "copy", "move", "delete"]
                targets = ["file", "folder", "application", "process", "system"]
                
                for i in range(sample_size):
                    sample_data.append({
                        "id": i,
                        "command": np.random.choice(commands),
                        "target": np.random.choice(targets),
                        "target_path": f"/path/to/{np.random.choice(targets)}",
                        "parameters": {"param1": f"value_{np.random.randint(1, 10)}"},
                        "expected_result": f"Successfully {np.random.choice(commands)}d the {np.random.choice(targets)}.",
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif model_id.startswith("I_"):
                # Motion control model sample data
                actions = ["move", "rotate", "lift", "push", "pull", "grab"]
                axes = ["x", "y", "z", "rx", "ry", "rz"]
                
                for i in range(sample_size):
                    sample_data.append({
                        "id": i,
                        "action": np.random.choice(actions),
                        "target_position": {axis: np.random.uniform(-1, 1) for axis in axes},
                        "speed": np.random.uniform(0, 1),
                        "acceleration": np.random.uniform(0, 1),
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif model_id.startswith("J_"):
                # Knowledge model sample data
                domains = ["science", "history", "literature", "mathematics", "philosophy", "art"]
                
                for i in range(sample_size):
                    sample_data.append({
                        "id": i,
                        "domain": np.random.choice(domains),
                        "question": f"What is the importance of {np.random.choice(domains)} in modern society?",
                        "answer": f"{np.random.choice(domains)} plays a crucial role in modern society by advancing knowledge and improving human understanding.",
                        "sources": [f"source_{j}" for j in range(np.random.randint(1, 3))],
                        "timestamp": datetime.now().isoformat()
                    })
            
            elif model_id.startswith("K_"):
                # Programming model sample data
                languages = ["Python", "JavaScript", "Java", "C++", "Go", "Rust"]
                tasks = ["sorting", "searching", "data_processing", "web_scraping", "api_integration"]
                
                for i in range(sample_size):
                    sample_data.append({
                        "id": i,
                        "language": np.random.choice(languages),
                        "task": np.random.choice(tasks),
                        "description": f"Write a {np.random.choice(languages)} program to perform {np.random.choice(tasks)}.",
                        "code": f"# {np.random.choice(languages)} code for {np.random.choice(tasks)}\nprint('Hello, world!')",
                        "timestamp": datetime.now().isoformat()
                    })
            
            else:
                # Generic sample data for unknown model types
                for i in range(sample_size):
                    sample_data.append({
                        "id": i,
                        "input": f"Sample input {i}",
                        "output": f"Sample output {i}",
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Split data
            train_data, temp_data = train_test_split(sample_data, test_size=0.3, random_state=42)
            val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
            
            # Save data
            train_file = os.path.join(self.get_training_data_dir(model_id), "sample_data.json")
            val_file = os.path.join(self.get_validation_data_dir(model_id), "sample_data.json")
            test_file = os.path.join(self.get_test_data_dir(model_id), "sample_data.json")
            
            self.save_data(train_data, train_file)
            self.save_data(val_data, val_file)
            self.save_data(test_data, test_file)
            
            logger.info(f"Generated sample data for model {model_id}: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
            
            return True, {
                "message": "Sample data generated successfully",
                "train_count": len(train_data),
                "validation_count": len(val_data),
                "test_count": len(test_data)
            }
            
        except Exception as e:
            logger.error(f"Failed to generate sample data for model {model_id}: {str(e)}")
            return False, str(e)
    
    def get_data_statistics(self, model_id):
        """Get statistics about the data for a model"""
        try:
            stats = {
                "model_id": model_id,
                "training_data": None,
                "validation_data": None,
                "test_data": None,
                "raw_files": [],
                "last_updated": datetime.now().isoformat()
            }
            
            # Check training data
            train_dir = self.get_training_data_dir(model_id)
            if os.path.exists(train_dir):
                train_files = [f for f in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, f))]
                train_count = 0
                for file in train_files:
                    file_path = os.path.join(train_dir, file)
                    data = self.load_data(file_path)
                    if data is not None:
                        if isinstance(data, list):
                            train_count += len(data)
                        else:
                            train_count += 1
                
                stats["training_data"] = {
                    "file_count": len(train_files),
                    "record_count": train_count
                }
            
            # Check validation data
            val_dir = self.get_validation_data_dir(model_id)
            if os.path.exists(val_dir):
                val_files = [f for f in os.listdir(val_dir) if os.path.isfile(os.path.join(val_dir, f))]
                val_count = 0
                for file in val_files:
                    file_path = os.path.join(val_dir, file)
                    data = self.load_data(file_path)
                    if data is not None:
                        if isinstance(data, list):
                            val_count += len(data)
                        else:
                            val_count += 1
                
                stats["validation_data"] = {
                    "file_count": len(val_files),
                    "record_count": val_count
                }
            
            # Check test data
            test_dir = self.get_test_data_dir(model_id)
            if os.path.exists(test_dir):
                test_files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
                test_count = 0
                for file in test_files:
                    file_path = os.path.join(test_dir, file)
                    data = self.load_data(file_path)
                    if data is not None:
                        if isinstance(data, list):
                            test_count += len(data)
                        else:
                            test_count += 1
                
                stats["test_data"] = {
                    "file_count": len(test_files),
                    "record_count": test_count
                }
            
            # Check raw files
            raw_dir = os.path.join(self.get_model_data_dir(model_id), "raw")
            if os.path.exists(raw_dir):
                raw_files = [f for f in os.listdir(raw_dir) if os.path.isfile(os.path.join(raw_dir, f))]
                stats["raw_files"] = raw_files
            
            logger.info(f"Retrieved data statistics for model {model_id}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get data statistics for model {model_id}: {str(e)}")
            return {"error": str(e)}
    
    def clear_model_data(self, model_id, include_raw=False):
        """Clear all data for a model"""
        try:
            # Clear training data
            train_dir = self.get_training_data_dir(model_id)
            if os.path.exists(train_dir):
                shutil.rmtree(train_dir)
                os.makedirs(train_dir)
            
            # Clear validation data
            val_dir = self.get_validation_data_dir(model_id)
            if os.path.exists(val_dir):
                shutil.rmtree(val_dir)
                os.makedirs(val_dir)
            
            # Clear test data
            test_dir = self.get_test_data_dir(model_id)
            if os.path.exists(test_dir):
                shutil.rmtree(test_dir)
                os.makedirs(test_dir)
            
            # Clear processed data
            processed_dir = os.path.join(self.get_model_data_dir(model_id), "processed")
            if os.path.exists(processed_dir):
                shutil.rmtree(processed_dir)
                os.makedirs(processed_dir)
            
            # Clear raw data if requested
            if include_raw:
                raw_dir = os.path.join(self.get_model_data_dir(model_id), "raw")
                if os.path.exists(raw_dir):
                    shutil.rmtree(raw_dir)
                    os.makedirs(raw_dir)
            
            logger.info(f"Cleared model data for {model_id} (include_raw={include_raw})")
            return True, "Model data cleared successfully"
            
        except Exception as e:
            logger.error(f"Failed to clear model data for {model_id}: {str(e)}")
            return False, str(e)

# Create a global instance of DataManager
data_manager = DataManager()

if __name__ == "__main__":
    # Example usage
    if len(sys.argv) > 2:
        action = sys.argv[1]
        model_id = sys.argv[2]
        
        if action == "generate":
            success, result = data_manager.generate_sample_data(model_id)
            print(f"Generate result: {success}, {result}")
        
        elif action == "stats":
            stats = data_manager.get_data_statistics(model_id)
            print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        elif action == "clear":
            include_raw = len(sys.argv) > 3 and sys.argv[3] == "--include-raw"
            success, message = data_manager.clear_model_data(model_id, include_raw)
            print(f"Clear result: {success}, {message}")
        
        else:
            print("Unknown action. Use 'generate', 'stats', or 'clear'.")
    else:
        print("Usage: python data_manager.py <action> <model_id> [options]")
        print("Actions: generate, stats, clear")