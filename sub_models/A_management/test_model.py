#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for A_management model inference
"""

import os
import sys
import json
import torch
import numpy as np
from datetime import datetime

# Import the ManagementModel from train.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from train import ManagementModel

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_A_management.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("A_Management_Test")

def load_model(model_path, device):
    """Load the trained management model"""
    # Create a dummy instance to get the input size and architecture
    dummy_input_size = 41  # Based on the input size reported during training
    hidden_sizes = [128, 64, 32]
    strategy_output_size = 4
    emotion_output_size = 7
    
    # Initialize model
    model = ManagementModel(
        input_size=dummy_input_size,
        hidden_sizes=hidden_sizes,
        strategy_output_size=strategy_output_size,
        emotion_output_size=emotion_output_size
    )
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model

def create_sample_task():
    """Create a sample task for testing the model"""
    # Create a sample task with random values
    sample_task = {
        "id": "test_task_1",
        "type": "coordinate",  # coordinate, allocate, monitor, optimize, decision
        "priority": "high",     # low, medium, high, urgent
        "requirements": {
            "cpu": 0.8,
            "memory": 0.7,
            "disk": 0.3,
            "network": 0.6
        },
        "deadline": 120,
        "submodel": "B_language",
        "complexity_score": 0.75,
        "urgency_score": 0.9,
        "user_emotion": "happy",
        "feedback_format": "text",
        "interaction_type": "language"
    }
    
    return sample_task

def preprocess_task(task):
    """Convert a task dictionary into a tensor input for the model"""
    # Task type encoding
    task_type_encoding = [1 if task["type"] == t else 0 for t in ["coordinate", "allocate", "monitor", "optimize", "decision"]]
    
    # Priority encoding
    priority_encoding = [1 if task["priority"] == p else 0 for p in ["low", "medium", "high", "urgent"]]
    
    # Submodel encoding
    submodel_names = ["B_language", "C_audio", "D_image", "E_video", "F_spatial", "G_sensor", "H_computer_control", "I_knowledge", "J_motion", "K_programming"]
    submodel_encoding = [1 if task["submodel"] == sm else 0 for sm in submodel_names]
    
    # Emotion encoding
    emotions = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
    emotion_encoding = [1 if task["user_emotion"] == e else 0 for e in emotions]
    
    # Feedback format encoding
    feedback_encoding = [1 if task["feedback_format"] == f else 0 for f in ["text", "voice", "image", "video"]]
    
    # Interaction type encoding
    interaction_encoding = [1 if task["interaction_type"] == i else 0 for i in ["language", "visual", "spatial", "sensor"]]
    
    # Combine all features into a numpy array
    features = np.array([
        *task_type_encoding,
        *priority_encoding,
        task["requirements"]["cpu"],
        task["requirements"]["memory"],
        task["requirements"]["disk"],
        task["requirements"]["network"],
        task["deadline"],
        task["complexity_score"],
        task["urgency_score"],
        *submodel_encoding,
        *emotion_encoding,
        *feedback_encoding,
        *interaction_encoding
    ], dtype=np.float32)
    
    # Convert to torch tensor and add batch dimension
    input_tensor = torch.tensor(features).unsqueeze(0)  # Add batch dimension
    
    return input_tensor

def test_model(model, device):
    """Test the model with a sample task"""
    # Create a sample task
    sample_task = create_sample_task()
    logger.info(f"Testing model with sample task: {sample_task}")
    
    # Preprocess the task into model input
    input_tensor = preprocess_task(sample_task)
    input_tensor = input_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        start_time = datetime.now()
        strategy_output, emotion_output = model(input_tensor)
        inference_time = (datetime.now() - start_time).total_seconds() * 1000  # Convert to milliseconds
    
    # Process the outputs
    strategy_idx = torch.argmax(strategy_output).item()
    emotion_idx = torch.argmax(emotion_output).item()
    
    # Get the strategy and emotion labels
    strategy_mapping = {0: "sequential", 1: "parallel", 2: "hierarchical", 3: "adaptive"}
    emotion_mapping = {0: "happy", 1: "sad", 2: "angry", 3: "fear", 4: "surprise", 5: "disgust", 6: "neutral"}
    
    predicted_strategy = strategy_mapping[strategy_idx]
    predicted_emotion = emotion_mapping[emotion_idx]
    
    # Get confidence scores
    strategy_confidence = strategy_output[0][strategy_idx].item()
    emotion_confidence = emotion_output[0][emotion_idx].item()
    
    # Log the results
    logger.info(f"\n===== Model Prediction Results =====")
    logger.info(f"Predicted Strategy: {predicted_strategy} (confidence: {strategy_confidence:.4f})")
    logger.info(f"Predicted Response Emotion: {predicted_emotion} (confidence: {emotion_confidence:.4f})")
    logger.info(f"Inference Time: {inference_time:.2f} ms")
    logger.info(f"Strategy Distribution: {strategy_output.squeeze().tolist()}")
    logger.info(f"Emotion Distribution: {emotion_output.squeeze().tolist()}")
    logger.info(f"====================================\n")
    
    # Test the process_task method
    result = model.process_task(input_tensor)
    logger.info(f"\n===== Model process_task Method Results =====")
    logger.info(f"Processed Strategy: {result['strategy']}")
    logger.info(f"Strategy Confidence: {result['strategy_confidence']:.4f}")
    logger.info(f"Response Emotion: {result['response_emotion']}")
    logger.info(f"Emotion Confidence: {result['emotion_confidence']:.4f}")
    logger.info(f"=============================================\n")
    
    # Test the adjust_response_based_on_emotion method
    original_response = "I will coordinate the task according to your requirements."
    adjusted_response = model.adjust_response_based_on_emotion(
        original_response, 
        sample_task["user_emotion"], 
        result['response_emotion']
    )
    
    logger.info(f"\n===== Emotion-Adjusted Response =====")
    logger.info(f"Original Response: {original_response}")
    logger.info(f"Adjusted Response: {adjusted_response}")
    logger.info(f"====================================\n")
    
    return {
        'sample_task': sample_task,
        'predicted_strategy': predicted_strategy,
        'predicted_emotion': predicted_emotion,
        'adjusted_response': adjusted_response
    }

def find_latest_model():
    """Find the latest saved model in the models directory"""
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    
    if not os.path.exists(models_dir):
        logger.error(f"Models directory not found: {models_dir}")
        return None
    
    # Get all model files and sort by modification time
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
    if not model_files:
        logger.error(f"No model files found in {models_dir}")
        return None
    
    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(models_dir, x)), reverse=True)
    
    # Return the path to the latest model
    latest_model = os.path.join(models_dir, model_files[0])
    logger.info(f"Found latest model: {latest_model}")
    
    return latest_model

def main():
    """Main function for testing the model"""
    logger.info("Starting model test...")
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Find the latest model
    model_path = find_latest_model()
    if model_path is None:
        logger.error("Failed to find a model to test")
        return
    
    # Load the model
    try:
        model = load_model(model_path, device)
        logger.info(f"Successfully loaded model from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Test the model
    try:
        test_results = test_model(model, device)
        logger.info("Model test completed successfully!")
        
        # Save test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"test_result_{timestamp}.json")
        with open(test_result_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Saved test results to {test_result_path}")
        
    except Exception as e:
        logger.error(f"Error during model test: {e}")

if __name__ == "__main__":
    main()