#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self Brain AGI System Initialization Script
This script initializes the system environment for Self Brain AGI."""

import os
import sys
import json
import logging
import numpy as np
from pathlib import Path
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SelfBrainInitializer")

# Define base directory
BASE_DIR = Path(__file__).parent.absolute()

# Generate sample training data
logger.info("Generating sample training data...")

# Sample language training data
def generate_language_data():
    """Generate sample language training data"""
    data_dir = BASE_DIR / "data" / "training" / "language"
    sample_data = [
        {"text": "Hello, how can I help you today?", "intent": "greeting"},
        {"text": "What is the weather like?", "intent": "weather_query"},
        {"text": "Tell me a joke", "intent": "request_joke"},
        {"text": "Explain quantum computing", "intent": "request_explanation"},
        {"text": "Goodbye", "intent": "farewell"}
    ]
    
    # Generate 100 samples
    all_samples = []
    for i in range(100):
        base = random.choice(sample_data)
        # Make slight variations
        if random.random() > 0.7:
            text = base["text"] + "?" if not base["text"].endswith("?") else base["text"][:-1]
        else:
            text = base["text"]
        all_samples.append({"text": text, "intent": base["intent"]})
    
    with open(data_dir / "sample_data.json", 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=2)
    logger.info(f"Generated language training data: {data_dir / 'sample_data.json'}")

# Sample audio training data
def generate_audio_data():
    """Generate placeholder for audio training data"""
    data_dir = BASE_DIR / "data" / "training" / "audio"
    # Create placeholder file
    with open(data_dir / "audio_placeholder.txt", 'w', encoding='utf-8') as f:
        f.write("Audio training data will be stored here.\n")
        f.write("This includes WAV files and corresponding transcriptions.")
    logger.info(f"Created audio training data placeholder: {data_dir / 'audio_placeholder.txt'}")

# Sample image training data
def generate_image_data():
    """Generate placeholder for image training data"""
    data_dir = BASE_DIR / "data" / "training" / "image"
    # Create placeholder file
    with open(data_dir / "image_placeholder.txt", 'w', encoding='utf-8') as f:
        f.write("Image training data will be stored here.\n")
        f.write("This includes image files and corresponding labels.")
    logger.info(f"Created image training data placeholder: {data_dir / 'image_placeholder.txt'}")

# Sample video training data
def generate_video_data():
    """Generate placeholder for video training data"""
    data_dir = BASE_DIR / "data" / "training" / "video"
    # Create placeholder file
    with open(data_dir / "video_placeholder.txt", 'w', encoding='utf-8') as f:
        f.write("Video training data will be stored here.\n")
        f.write("This includes video files and corresponding annotations.")
    logger.info(f"Created video training data placeholder: {data_dir / 'video_placeholder.txt'}")

# Sample knowledge training data
def generate_knowledge_data():
    """Generate sample knowledge training data"""
    data_dir = BASE_DIR / "data" / "training" / "knowledge"
    sample_knowledge = [
        {"domain": "physics", "content": "Newton's laws of motion describe the relationship between a physical object and the forces acting upon it."},
        {"domain": "mathematics", "content": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides."},
        {"domain": "chemistry", "content": "Water is composed of two hydrogen atoms and one oxygen atom, with the chemical formula H2O."},
        {"domain": "biology", "content": "DNA is a molecule that carries genetic instructions for the development, functioning, growth and reproduction of all known organisms."},
        {"domain": "computer science", "content": "Machine learning is a method of data analysis that automates analytical model building."}
    ]
    
    with open(data_dir / "sample_knowledge.json", 'w', encoding='utf-8') as f:
        json.dump(sample_knowledge, f, indent=2)
    logger.info(f"Generated knowledge training data: {data_dir / 'sample_knowledge.json'}")

# Create models registry
def create_models_registry():
    """Create models registry file"""
    registry_path = BASE_DIR / "training_manager" / "models_registry.json"
    registry_dir = registry_path.parent
    registry_dir.mkdir(parents=True, exist_ok=True)
    
    models_registry = {
        "models": [
            {"id": "A_management", "name": "Management Model", "type": "manager", "port": 5000},
            {"id": "B_language", "name": "Language Model", "type": "language", "port": 5001},
            {"id": "C_audio", "name": "Audio Model", "type": "audio", "port": 5002},
            {"id": "D_image", "name": "Image Model", "type": "vision", "port": 5003},
            {"id": "E_video", "name": "Video Model", "type": "vision", "port": 5004},
            {"id": "F_spatial", "name": "Spatial Model", "type": "spatial", "port": 5005},
            {"id": "G_sensor", "name": "Sensor Model", "type": "sensor", "port": 5006},
            {"id": "H_computer_control", "name": "Computer Control Model", "type": "control", "port": 5007},
            {"id": "I_knowledge", "name": "Knowledge Model", "type": "knowledge", "port": 5008},
            {"id": "J_motion", "name": "Motion Control Model", "type": "control", "port": 5009},
            {"id": "K_programming", "name": "Programming Model", "type": "programming", "port": 5010}
        ]
    }
    
    with open(registry_path, 'w', encoding='utf-8') as f:
        json.dump(models_registry, f, indent=2)
    logger.info(f"Created models registry: {registry_path}")

# Main initialization
if __name__ == "__main__":
    try:
        generate_language_data()
        generate_audio_data()
        generate_image_data()
        generate_video_data()
        generate_knowledge_data()
        create_models_registry()
        
        logger.info("Self Brain AGI System initialization completed successfully!")
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        sys.exit(1)
