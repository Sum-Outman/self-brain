#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Self Brain AGI - A Management Model Training Script
This script implements training from scratch for the management model without using any pre-trained models.

Copyright 2025 AGI System Team
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
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
        self.submodel_names = ["B_language", "C_audio", "D_image", "E_video", "F_spatial", "G_sensor", "H_computer_control", "I_knowledge", "J_motion", "K_programming"]
        self.emotions = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
        self.task_data = self._load_data()
        
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
        emotions = ["happy", "sad", "angry", "fear", "surprise", "disgust", "neutral"]
        feedback_types = ["text", "voice", "image", "video"]
        
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
                "urgency_score": random.uniform(0.1, 1.0),
                "user_emotion": random.choice(emotions),
                "required_response_emotion": random.choice(emotions),
                "feedback_format": random.choice(feedback_types),
                "interaction_type": random.choice(["language", "visual", "spatial", "sensor"])
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
        emotion_encoding = [1 if task["user_emotion"] == e else 0 for e in self.emotions]
        feedback_encoding = [1 if task["feedback_format"] == f else 0 for f in ["text", "voice", "image", "video"]]
        interaction_encoding = [1 if task["interaction_type"] == i else 0 for i in ["language", "visual", "spatial", "sensor"]]
        
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
        
        # Convert expected outputs to one-hot encodings
        strategy_encoding = [1 if task["expected_strategy"] == s else 0 for s in ["sequential", "parallel", "hierarchical", "adaptive"]]
        response_emotion_encoding = [1 if task["required_response_emotion"] == e else 0 for e in self.emotions]
        
        # Combine strategy and emotion as labels
        label = np.concatenate([
            np.array(strategy_encoding, dtype=np.float32),
            np.array(response_emotion_encoding, dtype=np.float32)
        ])
        
        return torch.tensor(features), torch.tensor(label)

class ManagementModel(nn.Module):
    """Management model neural network architecture with emotion capabilities"""
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], strategy_output_size=4, emotion_output_size=7):
        super(ManagementModel, self).__init__()
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Hidden layers
        hidden_layers = []
        prev_size = hidden_sizes[0]
        for i in range(1, len(hidden_sizes)):
            hidden_layers.append(nn.Linear(prev_size, hidden_sizes[i]))
            hidden_layers.append(nn.ReLU())
            hidden_layers.append(nn.Dropout(0.3))
            prev_size = hidden_sizes[i]
        
        self.hidden_layers = nn.Sequential(*hidden_layers)
        
        # Separate output heads
        self.strategy_head = nn.Sequential(
            nn.Linear(prev_size, strategy_output_size),
            nn.Softmax(dim=1)
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(prev_size, emotion_output_size),
            nn.Softmax(dim=1)
        )
        
        # Emotion response mapping
        self.emotion_mapping = {
            0: "happy",
            1: "sad",
            2: "angry",
            3: "fear",
            4: "surprise",
            5: "disgust",
            6: "neutral"
        }
        
        # Strategy mapping
        self.strategy_mapping = {
            0: "sequential",
            1: "parallel",
            2: "hierarchical",
            3: "adaptive"
        }
    
    def forward(self, x):
        features = self.feature_extractor(x)
        hidden = self.hidden_layers(features)
        strategy_output = self.strategy_head(hidden)
        emotion_output = self.emotion_head(hidden)
        return strategy_output, emotion_output
    
    def process_task(self, task_data):
        """Process a single management task with emotion understanding"""
        # Convert task data to tensor
        if isinstance(task_data, np.ndarray):
            input_tensor = torch.tensor(task_data, dtype=torch.float32).unsqueeze(0)
        else:
            input_tensor = task_data
        
        # Get predictions
        with torch.no_grad():
            strategy_pred, emotion_pred = self.forward(input_tensor)
        
        # Get most likely strategy and emotion
        strategy_idx = torch.argmax(strategy_pred).item()
        emotion_idx = torch.argmax(emotion_pred).item()
        
        return {
            'strategy': self.strategy_mapping[strategy_idx],
            'strategy_confidence': strategy_pred[0][strategy_idx].item(),
            'response_emotion': self.emotion_mapping[emotion_idx],
            'emotion_confidence': emotion_pred[0][emotion_idx].item(),
            'strategy_distribution': strategy_pred.squeeze().tolist(),
            'emotion_distribution': emotion_pred.squeeze().tolist()
        }
    
    def adjust_response_based_on_emotion(self, original_response, user_emotion, response_emotion):
        """Adjust response style based on user emotion and target response emotion"""
        # Emotion adjustment templates
        emotion_adjustments = {
            'happy': {'prefix': 'Great! ', 'suffix': ' 😊'},
            'sad': {'prefix': 'I understand. ', 'suffix': ' 😔'},
            'angry': {'prefix': 'I apologize for the inconvenience. ', 'suffix': ' Let me help resolve this.'},
            'fear': {'prefix': 'Don\'t worry, ', 'suffix': ' I\'m here to assist.'},
            'surprise': {'prefix': 'Wow! ', 'suffix': ' That\'s interesting!'},
            'disgust': {'prefix': 'I see your concern. ', 'suffix': ' Let\'s find a better solution.'},
            'neutral': {'prefix': '', 'suffix': ''}
        }
        
        # Apply adjustments
        adj = emotion_adjustments.get(response_emotion, emotion_adjustments['neutral'])
        adjusted_response = adj['prefix'] + original_response + adj['suffix']
        
        return adjusted_response

class ModelTrainer:
    """Model trainer class for A_management with emotion capabilities"""
    def __init__(self, model, train_loader, val_loader, device, learning_rate=0.0001, emotion_weight=0.5):
        # Basic setup
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        self.emotion_weight = emotion_weight  # Weight for emotion loss
        
        # Optimizer and scheduler
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss functions for strategy and emotion tasks
        self.strategy_criterion = nn.CrossEntropyLoss()
        self.emotion_criterion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss()  # For backward compatibility
        
        # Training metrics
        self.best_val_loss = float('inf')
        self.best_model_path = 'best_management_model.pth'
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def train_epoch(self):
        """Train for one epoch with strategy and emotion tasks"""
        self.model.train()
        running_total_loss = 0.0
        running_strategy_loss = 0.0
        running_emotion_loss = 0.0
        correct_strategy = 0
        correct_emotion = 0
        total = 0
        
        for features, labels in tqdm(self.train_loader, desc="Training"):
            features = features.to(self.device)
            # Separate strategy and emotion labels
            strategy_labels = labels[:, :4].to(self.device)  # First 4 columns for strategy
            emotion_labels = labels[:, 4:].to(self.device)  # Next 7 columns for emotion
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            strategy_outputs, emotion_outputs = self.model(features)
            
            # Calculate losses
            strategy_loss = self.strategy_criterion(strategy_outputs, strategy_labels)
            emotion_loss = self.emotion_criterion(emotion_outputs, emotion_labels)
            
            # Combine losses with weights
            total_loss = strategy_loss + self.emotion_weight * emotion_loss
            
            # Backward pass and optimize
            total_loss.backward()
            self.optimizer.step()
            
            # Update statistics
            running_total_loss += total_loss.item() * features.size(0)
            running_strategy_loss += strategy_loss.item() * features.size(0)
            running_emotion_loss += emotion_loss.item() * features.size(0)
            
            # Calculate accuracy for strategy
            _, strategy_predicted = torch.max(strategy_outputs, 1)
            _, strategy_actual = torch.max(strategy_labels, 1)
            correct_strategy += (strategy_predicted == strategy_actual).sum().item()
            
            # Calculate accuracy for emotion
            _, emotion_predicted = torch.max(emotion_outputs, 1)
            _, emotion_actual = torch.max(emotion_labels, 1)
            correct_emotion += (emotion_predicted == emotion_actual).sum().item()
            
            total += features.size(0)
        
        epoch_total_loss = running_total_loss / len(self.train_loader.dataset)
        epoch_strategy_loss = running_strategy_loss / len(self.train_loader.dataset)
        epoch_emotion_loss = running_emotion_loss / len(self.train_loader.dataset)
        epoch_strategy_acc = correct_strategy / total
        epoch_emotion_acc = correct_emotion / total
        
        # Update history
        if 'train_total_loss' not in self.history:
            self.history['train_total_loss'] = []
            self.history['train_strategy_loss'] = []
            self.history['train_emotion_loss'] = []
            self.history['train_strategy_acc'] = []
            self.history['train_emotion_acc'] = []
        
        self.history['train_total_loss'].append(epoch_total_loss)
        self.history['train_strategy_loss'].append(epoch_strategy_loss)
        self.history['train_emotion_loss'].append(epoch_emotion_loss)
        self.history['train_strategy_acc'].append(epoch_strategy_acc)
        self.history['train_emotion_acc'].append(epoch_emotion_acc)
        
        return epoch_total_loss, epoch_strategy_acc, epoch_emotion_acc
    
    def validate(self):
        """Validate the model with strategy and emotion tasks"""
        self.model.eval()
        running_total_loss = 0.0
        running_strategy_loss = 0.0
        running_emotion_loss = 0.0
        correct_strategy = 0
        correct_emotion = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in tqdm(self.val_loader, desc="Validation"):
                features = features.to(self.device)
                # Separate strategy and emotion labels
                strategy_labels = labels[:, :4].to(self.device)  # First 4 columns for strategy
                emotion_labels = labels[:, 4:].to(self.device)  # Next 7 columns for emotion
                
                # Forward pass
                strategy_outputs, emotion_outputs = self.model(features)
                
                # Calculate losses
                strategy_loss = self.strategy_criterion(strategy_outputs, strategy_labels)
                emotion_loss = self.emotion_criterion(emotion_outputs, emotion_labels)
                
                # Combine losses with weights
                total_loss = strategy_loss + self.emotion_weight * emotion_loss
                
                # Update statistics
                running_total_loss += total_loss.item() * features.size(0)
                running_strategy_loss += strategy_loss.item() * features.size(0)
                running_emotion_loss += emotion_loss.item() * features.size(0)
                
                # Calculate accuracy for strategy
                _, strategy_predicted = torch.max(strategy_outputs, 1)
                _, strategy_actual = torch.max(strategy_labels, 1)
                correct_strategy += (strategy_predicted == strategy_actual).sum().item()
                
                # Calculate accuracy for emotion
                _, emotion_predicted = torch.max(emotion_outputs, 1)
                _, emotion_actual = torch.max(emotion_labels, 1)
                correct_emotion += (emotion_predicted == emotion_actual).sum().item()
                
                total += features.size(0)
        
        epoch_total_loss = running_total_loss / len(self.val_loader.dataset)
        epoch_strategy_loss = running_strategy_loss / len(self.val_loader.dataset)
        epoch_emotion_loss = running_emotion_loss / len(self.val_loader.dataset)
        epoch_strategy_acc = correct_strategy / total
        epoch_emotion_acc = correct_emotion / total
        
        # Update history
        if 'val_total_loss' not in self.history:
            self.history['val_total_loss'] = []
            self.history['val_strategy_loss'] = []
            self.history['val_emotion_loss'] = []
            self.history['val_strategy_acc'] = []
            self.history['val_emotion_acc'] = []
        
        self.history['val_total_loss'].append(epoch_total_loss)
        self.history['val_strategy_loss'].append(epoch_strategy_loss)
        self.history['val_emotion_loss'].append(epoch_emotion_loss)
        self.history['val_strategy_acc'].append(epoch_strategy_acc)
        self.history['val_emotion_acc'].append(epoch_emotion_acc)
        
        return epoch_total_loss, epoch_strategy_acc, epoch_emotion_acc
    
    def train(self, epochs, early_stopping_patience=10):
        """Train the model for multiple epochs with strategy and emotion tasks"""
        no_improvement_count = 0
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Train and validate
            train_total_loss, train_strategy_acc, train_emotion_acc = self.train_epoch()
            val_total_loss, val_strategy_acc, val_emotion_acc = self.validate()
            
            # Log detailed metrics
            logger.info(f"Epoch {epoch+1}: \
                        Total Loss (Train/Val): {train_total_loss:.4f}/{val_total_loss:.4f} \
                        Strategy Acc (Train/Val): {train_strategy_acc:.4f}/{val_strategy_acc:.4f} \
                        Emotion Acc (Train/Val): {train_emotion_acc:.4f}/{val_emotion_acc:.4f}")
            
            # Check for improvement based on total validation loss
            if val_total_loss < self.best_val_loss:
                self.best_val_loss = val_total_loss
                no_improvement_count = 0
                # Save the best model
                torch.save(self.model.state_dict(), self.best_model_path)
                logger.info(f"Saved best model to {self.best_model_path}")
            else:
                no_improvement_count += 1
                
            # Update learning rate
            self.scheduler.step(val_total_loss)
            
            # Early stopping
            if no_improvement_count >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load the best model
        try:
            self.model.load_state_dict(torch.load(self.best_model_path))
            logger.info(f"Loaded best model from {self.best_model_path}")
        except FileNotFoundError:
            logger.warning(f"Best model file {self.best_model_path} not found. Using current model state.")
        
        return self.history

def evaluate_model(model, data_loader, device):
    """
    Evaluate the performance of a trained management model
    
    Args:
        model: The trained management model
        data_loader: DataLoader for evaluation data
        device: Device to run evaluation on
        
    Returns:
        Dictionary containing evaluation metrics
    """
    model.eval()
    correct_strategy = 0
    correct_emotion = 0
    total = 0
    
    # Initialize metrics
    metrics = {
        'strategy_accuracy': 0.0,
        'emotion_accuracy': 0.0,
        'total_accuracy': 0.0
    }
    
    with torch.no_grad():
        for features, labels in tqdm(data_loader, desc="Evaluating"):
            features = features.to(device)
            # Separate strategy and emotion labels
            strategy_labels = labels[:, :4].to(device)
            emotion_labels = labels[:, 4:].to(device)
            
            # Forward pass
            strategy_outputs, emotion_outputs = model(features)
            
            # Calculate accuracy for strategy
            _, strategy_predicted = torch.max(strategy_outputs, 1)
            _, strategy_actual = torch.max(strategy_labels, 1)
            correct_strategy += (strategy_predicted == strategy_actual).sum().item()
            
            # Calculate accuracy for emotion
            _, emotion_predicted = torch.max(emotion_outputs, 1)
            _, emotion_actual = torch.max(emotion_labels, 1)
            correct_emotion += (emotion_predicted == emotion_actual).sum().item()
            
            total += features.size(0)
    
    # Compute final metrics
    if total > 0:
        metrics['strategy_accuracy'] = correct_strategy / total
        metrics['emotion_accuracy'] = correct_emotion / total
        # Total accuracy is average of strategy and emotion accuracy
        metrics['total_accuracy'] = (metrics['strategy_accuracy'] + metrics['emotion_accuracy']) / 2
    
    return metrics

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
    parser.add_argument("--evaluate_only", action="store_true", help="Only evaluate the model without training")
    parser.add_argument("--model_path", type=str, help="Path to the model file for evaluation")
    
    return parser.parse_args()

def generate_training_report(history, args, timestamp, report_dir):
    """
    Generate a comprehensive training report with metrics summary
    
    Args:
        history: Dictionary containing training history metrics
        args: Command line arguments
        timestamp: Timestamp for the report filename
        report_dir: Directory to save the report
    """
    # Create report content
    report = {
        "timestamp": timestamp,
        "training_parameters": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "learning_rate": args.lr,
            "from_scratch": args.from_scratch,
            "validation_split": args.val_split,
            "hidden_sizes": args.hidden_sizes
        },
        "metrics_summary": {
            "final_train_total_loss": history.get('train_total_loss', [])[-1] if history.get('train_total_loss') else None,
            "final_val_total_loss": history.get('val_total_loss', [])[-1] if history.get('val_total_loss') else None,
            "final_train_strategy_accuracy": history.get('train_strategy_acc', [])[-1] if history.get('train_strategy_acc') else None,
            "final_val_strategy_accuracy": history.get('val_strategy_acc', [])[-1] if history.get('val_strategy_acc') else None,
            "final_train_emotion_accuracy": history.get('train_emotion_acc', [])[-1] if history.get('train_emotion_acc') else None,
            "final_val_emotion_accuracy": history.get('val_emotion_acc', [])[-1] if history.get('val_emotion_acc') else None,
            "best_val_total_loss": min(history.get('val_total_loss', [float('inf')])) if history.get('val_total_loss') else None
        },
        "history": history
    }
    
    # Save report
    report_path = os.path.join(report_dir, f"training_report_{timestamp}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Saved training report to {report_path}")
    
    # Print summary to console
    summary = (
        "\n===== Training Summary =====\n"
        f"Final Training Loss: {report['metrics_summary']['final_train_total_loss']:.4f}\n"
        f"Final Validation Loss: {report['metrics_summary']['final_val_total_loss']:.4f}\n"
        f"Best Validation Loss: {report['metrics_summary']['best_val_total_loss']:.4f}\n"
        f"Strategy Accuracy - Train: {report['metrics_summary']['final_train_strategy_accuracy']:.4f}, Val: {report['metrics_summary']['final_val_strategy_accuracy']:.4f}\n"
        f"Emotion Accuracy - Train: {report['metrics_summary']['final_train_emotion_accuracy']:.4f}, Val: {report['metrics_summary']['final_val_emotion_accuracy']:.4f}\n"
        "===========================\n"
    )
    logger.info(summary)

def main():
    """Main training function with evaluation support"""
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
    
    # Define output sizes
    strategy_output_size = 4  # sequential, parallel, hierarchical, adaptive
    emotion_output_size = 7  # happy, sad, angry, fear, surprise, disgust, neutral
    
    # Create model
    model = ManagementModel(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        strategy_output_size=strategy_output_size,
        emotion_output_size=emotion_output_size
    )
    
    # Handle evaluation only mode
    if args.evaluate_only:
        if not args.model_path:
            logger.error("Please provide a model path with --model_path when using --evaluate_only")
            sys.exit(1)
        
        if not os.path.exists(args.model_path):
            logger.error(f"Model file not found: {args.model_path}")
            sys.exit(1)
        
        try:
            # Load the model
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            model.to(device)
            logger.info(f"Loaded model from {args.model_path}")
            
            # Evaluate the model on validation set
            logger.info("Starting evaluation...")
            metrics = evaluate_model(model, val_loader, device)
            
            # Print evaluation results
            eval_summary = (
                "\n===== Model Evaluation Results =====\n"
                f"Strategy Accuracy: {metrics['strategy_accuracy']:.4f}\n"
                f"Emotion Accuracy: {metrics['emotion_accuracy']:.4f}\n"
                f"Total Accuracy: {metrics['total_accuracy']:.4f}\n"
                "====================================\n"
            )
            logger.info(eval_summary)
            
            # Save evaluation results
            base_dir = os.path.dirname(os.path.abspath(__file__))
            report_dir = os.path.join(base_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            eval_report = {
                "timestamp": timestamp,
                "model_path": args.model_path,
                "metrics": metrics
            }
            
            eval_report_path = os.path.join(report_dir, f"evaluation_report_{timestamp}.json")
            with open(eval_report_path, 'w') as f:
                json.dump(eval_report, f, indent=2)
            
            logger.info(f"Saved evaluation report to {eval_report_path}")
            
        except Exception as e:
            logger.error(f"Error during model evaluation: {e}")
            sys.exit(1)
        
        return
    
    # Proceed with training if not in evaluate only mode
    logger.info(f"Created model with architecture: {model}")
    
    # Create trainer with emotion weight
    emotion_weight = 0.5  # Weight for emotion loss relative to strategy loss
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.lr,
        emotion_weight=emotion_weight
    )
    logger.info(f"Created trainer with learning rate {args.lr} and emotion weight {emotion_weight}")
    
    # Start training
    logger.info("Starting training...")
    start_time = time.time()
    
    history = trainer.train(epochs=args.epochs)
    
    end_time = time.time()
    logger.info(f"Training completed in {end_time - start_time:.2f} seconds")
    
    # Create model and report directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_save_dir = os.path.join(base_dir, "models")
    report_dir = os.path.join(base_dir, "reports")
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save the final model
    final_model_path = os.path.join(model_save_dir, f"management_model_final_{timestamp}.pth")
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")
    
    # Save best model with timestamp
    best_model_path = os.path.join(model_save_dir, f"management_model_best_{timestamp}.pth")
    try:
        # Copy best model to timestamped path
        if os.path.exists("best_management_model.pth"):
            import shutil
            shutil.copy2("best_management_model.pth", best_model_path)
            logger.info(f"Saved best model to {best_model_path}")
    except Exception as e:
        logger.warning(f"Failed to save best model with timestamp: {e}")
    
    # Save training history
    history_path = os.path.join(report_dir, f"training_history_{timestamp}.json")
    with open(history_path, 'w') as f:
        json.dump(history, f)
    logger.info(f"Saved training history to {history_path}")
    
    # Generate and save training report
    generate_training_report(history, args, timestamp, report_dir)
    
    logger.info("Training process completed successfully!")

if __name__ == "__main__":
    main()