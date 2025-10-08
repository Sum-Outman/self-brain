#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for the A_MANAGEMENT model
This script handles training the model from scratch using the specified parameters.
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from pathlib import Path
import time
from torch.utils.data import DataLoader, TensorDataset

# Add parent directory to path to import modules
sys.path.append(str(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - A_management_training - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("A_management_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SelfBrain.A_management.training")

# Base directory
BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
CONFIG_PATH = BASE_DIR / "config.json"

# Import the actual ManagementModel
from manager_model.main_model import ManagementModel

# Load configuration
def load_config():
    """Load model configuration"""
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

# Initialize model architecture
def create_model_architecture(config):
    """Create the actual ManagementModel architecture"""
    logger.info(f"Creating A_MANAGEMENT model architecture from scratch")
    
    # Create an instance of the ManagementModel
    model = ManagementModel()
    
    # Ensure it's using randomly initialized weights (from scratch)
    if config["training"].get("from_scratch", True):
        logger.info("Initializing model with random weights for training from scratch")
    else:
        # Attempt to load existing weights if available
        model.load_weights()
    
    return model

# Load training data
def load_training_data(data_dir):
    """Load and preprocess training data"""
    logger.info(f"Loading training data from {data_dir}")
    
    # Ensure data directory exists
    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.warning(f"Data directory not found: {data_dir}")
        # Create sample training data
        return create_sample_training_data()
    
    # Initialize data dictionaries
    data = {
        "train": {"data": [], "labels": []},
        "validation": {"data": [], "labels": []},
        "test": {"data": [], "labels": []}
    }
    
    # Load data from each split
    for split in ["train", "validation", "test"]:
        split_dir = data_dir / split
        if split_dir.exists():
            for file in split_dir.glob("*.json"):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        items = json.load(f)
                        for item in items:
                            if "input" in item and "target" in item:
                                data[split]["data"].append(item["input"])
                                data[split]["labels"].append(item["target"])
                except Exception as e:
                    logger.error(f"Error loading file {file}: {str(e)}")
    
    # Check if data was loaded, if not create sample data
    if all(len(data[split]["data"]) == 0 for split in data):
        logger.warning("No training data found, creating sample data")
        return create_sample_training_data()
    
    logger.info(f"Loaded training data: train={len(data['train']['data'])}, validation={len(data['validation']['data'])}, test={len(data['test']['data'])}")
    return data

# Create sample training data when real data is not available
def create_sample_training_data():
    """Create sample training data for the ManagementModel"""
    logger.info("Creating sample training data")
    
    # Sample conversations for training
    sample_conversations = [
        {"input": "Hello", "target": "Hello! How can I help you today?"},
        {"input": "How are you", "target": "I'm doing well, thank you! How can I assist you?"},
        {"input": "What can you do", "target": "I can help with answering questions, processing data, and managing other AI models."},
        {"input": "Thanks", "target": "You're welcome! Let me know if you need anything else."},
        {"input": "What is your status", "target": "System is running smoothly. All core services are operational."},
        {"input": "Help me", "target": "I'm here to help. What do you need assistance with?"},
        {"input": "Goodbye", "target": "Goodbye! Have a nice day!"},
        {"input": "Tell me a joke", "target": "Why don't scientists trust atoms? Because they make up everything!"},
        {"input": "What's the weather today", "target": "I would need to access weather data to answer that. Let me check..."},
        {"input": "Calculate 2+2", "target": "The result is 4."}
    ]
    
    # Split into train, validation, test
    train_size = int(0.7 * len(sample_conversations))
    val_size = int(0.15 * len(sample_conversations))
    
    train_data = sample_conversations[:train_size]
    val_data = sample_conversations[train_size:train_size+val_size]
    test_data = sample_conversations[train_size+val_size:]
    
    return {
        "train": {"data": [item["input"] for item in train_data], "labels": [item["target"] for item in train_data]},
        "validation": {"data": [item["input"] for item in val_data], "labels": [item["target"] for item in val_data]},
        "test": {"data": [item["input"] for item in test_data], "labels": [item["target"] for item in test_data]}
    }

# Create data loaders
def create_data_loaders(data, batch_size, model):
    """Create actual data loaders for training"""
    logger.info(f"Creating data loaders with batch size: {batch_size}")
    
    # Prepare training data
    def prepare_data(data_items, label_items):
        inputs = []
        targets = []
        
        for input_msg, target_msg in zip(data_items, label_items):
            # Process input message
            input_tensor = torch.tensor([ord(c) for c in input_msg[:100]]).float()
            
            # Pad to input features length
            if len(input_tensor) < model.input_features:
                padding = torch.zeros(model.input_features - len(input_tensor))
                input_tensor = torch.cat([input_tensor, padding])
            else:
                input_tensor = input_tensor[:model.input_features]
            
            # Process target (expected output)
            target_tensor = torch.tensor([ord(c) for c in target_msg[:100]]).float()
            
            # Pad to input features length
            if len(target_tensor) < model.input_features:
                padding = torch.zeros(model.input_features - len(target_tensor))
                target_tensor = torch.cat([target_tensor, padding])
            else:
                target_tensor = target_tensor[:model.input_features]
            
            inputs.append(input_tensor)
            targets.append(target_tensor)
        
        # Convert to tensors
        inputs_tensor = torch.stack(inputs)
        targets_tensor = torch.stack(targets)
        
        return TensorDataset(inputs_tensor, targets_tensor)
    
    # Create datasets
    train_dataset = prepare_data(data["train"]["data"], data["train"]["labels"])
    val_dataset = prepare_data(data["validation"]["data"], data["validation"]["labels"])
    test_dataset = prepare_data(data["test"]["data"], data["test"]["labels"])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return {
        "train": train_loader,
        "validation": val_loader,
        "test": test_loader
    }

# Define loss function
def get_loss_function():
    """Define the MSE loss function"""
    return nn.MSELoss()

# Define optimizer
def get_optimizer(model, learning_rate):
    """Define the Adam optimizer"""
    return optim.Adam(model.parameters(), lr=learning_rate)

# Train model
def train_model(model, data_loaders, loss_function, optimizer, epochs, config, stop_event=None, progress_dict=None):
    """Train the model with actual training logic"""
    logger.info(f"Starting training for {epochs} epochs")
    
    # Training loop
    for epoch in range(epochs):
        # Check if training should be stopped
        if stop_event and stop_event.is_set():
            logger.info("Training stopped by user request")
            if progress_dict:
                progress_dict['status'] = 'stopped'
            break
        
        start_time = time.time()
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(data_loaders["train"]):
            # Check if training should be stopped after each batch
            if stop_event and stop_event.is_set():
                logger.info("Training stopped by user request")
                if progress_dict:
                    progress_dict['status'] = 'stopped'
                return
            
            # Forward pass
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            
            # Log progress
            if batch_idx % 10 == 0:
                logger.info(f"Batch {batch_idx+1}/{len(data_loaders['train'])}, Batch Loss: {loss.item():.4f}")
        
        # Calculate average training loss
        train_loss /= len(data_loaders["train"].dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in data_loaders["validation"]:
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        # Calculate average validation loss
        val_loss /= len(data_loaders["validation"].dataset)
        
        # Update progress
        model.training_progress = int(((epoch + 1) / epochs) * 100)
        
        # Update progress dictionary if provided
        if progress_dict:
            progress_dict['epoch'] = epoch + 1
            progress_dict['loss'] = train_loss
            progress_dict['accuracy'] = 1.0 - min(0.9, train_loss)  # Simple accuracy approximation
            progress_dict['steps_completed'] = epoch + 1
            progress_dict['val_loss'] = val_loss
        
        # Log epoch results
        logger.info(f"Epoch {epoch+1} completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(model, optimizer, epoch, config, train_loss, val_loss)
    
    # Update final status
    if progress_dict:
        if not stop_event or not stop_event.is_set():
            progress_dict['status'] = 'completed'
    
    # Final checkpoint
    save_checkpoint(model, optimizer, epochs, config, train_loss, val_loss, is_best=True)
    logger.info("Training completed successfully!")

# Train management model with progress tracking and stop capability
def train_management_model(model, progress_dict, stop_event, epochs=10, learning_rate=0.001, batch_size=32):
    """Train the management model with progress tracking and stop capability"""
    try:
        logger.info(f"===== A_MANAGEMENT Model Training (Interactive) =====")
        
        # Load configuration
        config = load_config()
        config["training"]["epochs"] = epochs
        config["training"]["learning_rate"] = learning_rate
        config["training"]["batch_size"] = batch_size
        
        # Update progress dict
        progress_dict['total_steps'] = epochs
        
        # Load training data
        data_dir = BASE_DIR.parent / "training_data" / "management"
        data = load_training_data(data_dir)
        
        # Create data loaders
        data_loaders = create_data_loaders(data, batch_size, model)
        
        # Get loss function and optimizer
        loss_function = get_loss_function()
        optimizer = get_optimizer(model, learning_rate)
        
        # Train model with stop capability
        train_model(
            model,
            data_loaders,
            loss_function,
            optimizer,
            epochs,
            config,
            stop_event=stop_event,
            progress_dict=progress_dict
        )
        
        # Evaluate model
        evaluation_results = evaluate_model(model, data_loaders["test"])
        logger.info(f"Evaluation results: {evaluation_results}")
        
        # Update progress dict with evaluation results
        if progress_dict:
            progress_dict.update(evaluation_results)
            
        return evaluation_results
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        if progress_dict:
            progress_dict['status'] = 'error'
            progress_dict['error'] = str(e)
        return {'status': 'error', 'message': str(e)}

# Stop training process
def stop_training_process(stop_event):
    """Signal to stop the training process"""
    if stop_event:
        stop_event.set()
        logger.info("Sent stop signal to training process")
        return True
    return False

# Save model checkpoint
def save_checkpoint(model, optimizer, epoch, config, train_loss, val_loss, is_best=False):
    """Save the actual model weights and training metadata"""
    checkpoint_dir = BASE_DIR / "training" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the actual model weights
    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_id': "A_management",
        'from_scratch': config["training"].get("from_scratch", True),
        'timestamp': time.time()
    }, checkpoint_path)
    
    logger.info(f"Model checkpoint saved: {checkpoint_path}")
    
    # Save metadata as JSON for easy inspection
    metadata_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.json"
    metadata = {
        "epoch": epoch,
        "model_id": "A_management",
        "from_scratch": config["training"].get("from_scratch", True),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "timestamp": time.time(),
        "batch_size": config["training"].get("batch_size", 32),
        "learning_rate": config["training"].get("learning_rate", 0.001)
    }
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Save best model weights separately
    if is_best:
        best_model_path = checkpoint_dir / "best_model.pth"
        torch.save(model.state_dict(), best_model_path)
        logger.info(f"Best model weights saved: {best_model_path}")
        
        # Save best model metadata
        best_metadata_path = checkpoint_dir / "best_model.json"
        with open(best_metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

# Evaluate model
def evaluate_model(model, data_loader):
    """Evaluate model performance with actual metrics"""
    logger.info("Evaluating model performance")
    
    model.eval()
    loss_function = nn.MSELoss()
    test_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
    
    # Calculate average loss
    test_loss /= len(data_loader.dataset)
    
    # Calculate additional metrics
    mse = test_loss
    rmse = np.sqrt(mse)
    
    # Return to training mode
    model.train()
    
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'samples': len(data_loader.dataset),
        'status': 'success'
    }
    
    logger.info(f"Model evaluation: {metrics}")
    return metrics

# Main training function
def main():
    """Main training function"""
    logger.info(f"===== A_MANAGEMENT Model Training =====")
    
    # Load configuration
    config = load_config()
    training_config = config["training"]
    
    # Create model architecture from scratch
    model = create_model_architecture(config)
    
    # Load training data
    data_dir = BASE_DIR.parent / "training_data" / "management"
    data = load_training_data(data_dir)
    
    # Create data loaders
    data_loaders = create_data_loaders(data, training_config["batch_size"], model)
    
    # Get loss function and optimizer
    loss_function = get_loss_function()
    optimizer = get_optimizer(model, training_config["learning_rate"])
    
    # Train model
    train_model(
        model,
        data_loaders,
        loss_function,
        optimizer,
        training_config["epochs"],
        config
    )
    
    # Evaluate model
    evaluation_results = evaluate_model(model, data_loaders["test"])
    logger.info(f"Evaluation results: {evaluation_results}")
    
    # Save evaluation results
    results_dir = BASE_DIR / "training" / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / "evaluation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation results saved to {results_path}")

if __name__ == "__main__":
    main()