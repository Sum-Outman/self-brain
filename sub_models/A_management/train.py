#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self Brain AGI - A Management Model Training Script
This script implements从零开始训练 of the management model without using any pre-trained models.
Copyright 2025 Self Brain Team
Contact: silencecrowtom@qq.com
"""

import os
import sys
import json
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F
from tqdm import tqdm
import logging
from datetime import datetime
from pathlib import Path
import psutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_A_management.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("A_Management_Training")

class ManagementTaskDataset(Dataset):
    """Dataset for management model training"""
    def __init__(self, data_dir, from_scratch=True):
        self.data_dir = data_dir
        self.from_scratch = from_scratch
        self.task_data = self._load_data()
        self.submodel_names = ["B_language", "C_audio", "D_image", "E_video", "F_spatial", "G_sensor", "H_computer_control", "I_knowledge", "J_motion", "K_programming"]
        
    def _load_data(self):
        """Load training data or generate synthetic data if from_scratch"""
        if self.from_scratch or not os.path.exists(os.path.join(self.data_dir, "management_tasks.json")):
            logger.info("Generating synthetic training data for management model")
            return self._generate_synthetic_data()
        else:
            logger.info("Loading existing training data")
            with open(os.path.join(self.data_dir, "management_tasks.json"), 'r') as f:
                return json.load(f)
    
    def _generate_synthetic_data(self):
        """Generate synthetic training data for management tasks"""
        synthetic_data = []
        task_types = ["coordinate", "allocate", "monitor", "optimize", "decision"]
        priority_levels = ["low", "medium", "high", "urgent"]
        
        for i in range(10000):  # Generate 10,000 synthetic samples
            task = {
                "id": f"task_{i}",
                "type": random.choice(task_types),
                "priority": random.choice(priority_levels),
                "requirements": {
                    "cpu": random.uniform(0.1, 1.0),
                    "memory": random.uniform(0.1, 1.0),
                    "disk": random.uniform(0.1, 0.5),
                    "network": random.uniform(0.1, 0.8)
                },
                "deadline": random.randint(1, 1000),
                "submodel": random.choice(self.submodel_names),
                "expected_strategy": random.choice(["sequential", "parallel", "hierarchical", "adaptive"]),
                "complexity_score": random.uniform(0.1, 1.0),
                "urgency_score": random.uniform(0.1, 1.0)
            }
            synthetic_data.append(task)
        
        # Save synthetic data for future use
        os.makedirs(self.data_dir, exist_ok=True)
        with open(os.path.join(self.data_dir, "management_tasks.json"), 'w') as f:
            json.dump(synthetic_data, f)
        
        return synthetic_data
    
    def __len__(self):
        return len(self.task_data)
    
    def __getitem__(self, idx):
        task = self.task_data[idx]
        
        # Convert task data to numerical features
        task_type_encoding = [1 if task["type"] == t else 0 for t in ["coordinate", "allocate", "monitor", "optimize", "decision"]]
        priority_encoding = [1 if task["priority"] == p else 0 for p in ["low", "medium", "high", "urgent"]]
        submodel_encoding = [1 if task["submodel"] == sm else 0 for sm in self.submodel_names]
        
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
            *submodel_encoding
        ], dtype=np.float32)
        
        # Convert expected strategy to one-hot encoding
        strategy_encoding = [1 if task["expected_strategy"] == s else 0 for s in ["sequential", "parallel", "hierarchical", "adaptive"]]
        label = np.array(strategy_encoding, dtype=np.float32)
        
        return torch.tensor(features), torch.tensor(label)

class ManagementModel(nn.Module):
    """Management model neural network architecture"""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], output_size=4):
        super(ManagementModel, self).__init__()
        
        # Create layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.3))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        layers.append(nn.Softmax(dim=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class ModelTrainer:
    """Model trainer class for A_management"""
    def __init__(self, model, train_loader, val_loader, device, learning_rate=0.0001):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for features, labels in tqdm(self.train_loader, desc="Training"):
            features, labels = features.to(self.device), labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            _, actual = torch.max(labels, 1)
            correct += (predicted == actual).sum().item()
            total += actual.size(0)
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = correct / total
        
        self.history['train_loss'].append(epoch_loss)
        self.history['train_acc'].append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc="Validation"):
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                
                # Update statistics
                running_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                _, actual = torch.max(labels, 1)
                correct += (predicted == actual).sum().item()
                total += actual.size(0)
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = correct / total
        
        self.history['val_loss'].append(epoch_loss)
        self.history['val_acc'].append(epoch_acc)
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs, early_stopping_patience=10):
        """Train the model for multiple epochs"""
        no_improvement_count = 0
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Train and validate
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            logger.info(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                no_improvement_count = 0
                # Save the best model
                torch.save(self.model.state_dict(), "best_management_model.pth")
                logger.info("Saved best model")
            else:
                no_improvement_count += 1
                
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load the best model
        self.model.load_state_dict(torch.load("best_management_model.pth"))
        
        return self.history

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="A Management Model Training Script")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--from_scratch", type=bool, default=True, help="Train from scratch or use pre-trained model")
    parser.add_argument("--dataset", type=str, default="management_tasks", help="Dataset name or path")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--hidden_sizes", type=str, default="128,64,32", help="Comma-separated list of hidden layer sizes")
    
    return parser.parse_args()

def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Create dataset directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create dataset
    dataset = ManagementTaskDataset(data_dir, from_scratch=args.from_scratch)
    
    # Split dataset into train and validation
    train_size = int((1 - args.val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)
    
    # Determine input size
    input_size = len(dataset[0][0])
    logger.info(f"Input size: {input_size}")
    
    # Parse hidden sizes
    hidden_sizes = [int(size) for size in args.hidden_sizes.split(',')]
    
    # Create model
    model = ManagementModel(input_size=input_size, hidden_sizes=hidden_sizes)
    logger.info(f"Created model with architecture: {model}")
    
    # Create trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr
    )
    
    # Start training
    logger.info("Starting training...")
    start_time = time.time()
    
    history = trainer.train(epochs=args.epochs)
    
    end_time = time.time()
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds")
    
    # Save the final model
    model_save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_save_dir, "management_model_final.pth"))
    
    # Save training history
    with open(os.path.join(model_save_dir, "training_history.json"), 'w') as f:
        json.dump(history, f)
    
    logger.info("Training process completed successfully!")

if __name__ == "__main__":
    main()