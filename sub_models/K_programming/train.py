# Copyright 2025 The AI Management System Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Programming Model Training Program

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Tuple
from .model import ProgrammingModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("K_programming_Trainer")

class ProgrammingDataset(Dataset):
    """Programming Dataset Class"""
    def __init__(self, data_dir: str, sequence_length: int = 10):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.data_files = []
        self.dataset_info = {}
        self.task_type_map = {}
        
        self._load_dataset_info()
    
    def _load_dataset_info(self):
        """Load dataset information"""
        info_file = os.path.join(self.data_dir, "dataset_info.json")
        
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    self.dataset_info = json.load(f)
                    self.data_files = self.dataset_info.get("data_files", [])
                    self.task_type_map = self.dataset_info.get("task_type_map", {})
            except Exception as e:
                logger.error(f"Error loading dataset information: {str(e)}")
        
        # If no dataset information is available, scan for data files in the directory
        if not self.data_files:
            self.data_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
            # If no data files found, create some dummy data information
            if not self.data_files:
                self.data_files = [f"programming_data_{i}.csv" for i in range(12)]
                self.task_type_map = {
                    0: "data_processing",
                    1: "web_development",
                    2: "machine_learning",
                    3: "database_operations",
                    4: "file_operations",
                    5: "network_communication",
                    6: "gui_development",
                    7: "system_administration",
                    8: "security_implementation",
                    9: "algorithm_implementation",
                    10: "api_development",
                    11: "testing_framework"
                }
                self.dataset_info = {
                    "data_files": self.data_files,
                    "task_type_map": self.task_type_map,
                    "description": "Simulated programming task data",
                    "created": time.time()
                }
    
    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        data_file = self.data_files[idx]
        data_path = os.path.join(self.data_dir, data_file)
        
        try:
            # Try to load data file
            if os.path.exists(data_path):
                data = pd.read_csv(data_path)
            else:
                # Generate dummy data
                data = self._generate_dummy_data()
            
            # Extract programming features
            programming_features = self._extract_programming_features(data)
            
            # Extract task type labels
            task_type_labels = self._extract_task_type_labels(data)
            
            # Convert to tensors
            programming_features = torch.tensor(programming_features, dtype=torch.float32)
            task_type_labels = torch.tensor(task_type_labels, dtype=torch.long)
            
            return programming_features, task_type_labels
            
        except Exception as e:
            logger.error(f"Error processing programming data: {str(e)}")
            # Return empty data
            dummy_features = torch.zeros((self.sequence_length, 6), dtype=torch.float32)
            dummy_labels = torch.zeros(self.sequence_length, dtype=torch.long)
            return dummy_features, dummy_labels
    
    def _extract_programming_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract programming features"""
        # Define programming feature columns
        feature_columns = ['lines_of_code', 'function_count', 'class_count', 
                          'variable_count', 'comment_density', 'complexity']
        
        # If some columns are missing from the data, fill with default values
        for col in feature_columns:
            if col not in data.columns:
                if col == 'lines_of_code':
                    data[col] = np.random.randint(10, 1000, len(data))
                elif col == 'function_count':
                    data[col] = np.random.randint(1, 20, len(data))
                elif col == 'class_count':
                    data[col] = np.random.randint(0, 10, len(data))
                elif col == 'variable_count':
                    data[col] = np.random.randint(5, 50, len(data))
                elif col == 'comment_density':
                    data[col] = np.random.uniform(0.1, 0.3, len(data))
                elif col == 'complexity':
                    data[col] = np.random.uniform(1, 10, len(data))
        
        # Extract features and ensure sequence length
        features = data[feature_columns].values
        if len(features) < self.sequence_length:
            # Pad to required length
            padding = np.zeros((self.sequence_length - len(features), len(feature_columns)))
            features = np.vstack([features, padding])
        elif len(features) > self.sequence_length:
            # Truncate to required length
            features = features[:self.sequence_length]
        
        return features
    
    def _extract_task_type_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Extract task type labels"""
        if 'task_type' in data.columns:
            # Map task type strings to numeric labels
            labels = []
            for task_type in data['task_type']:
                if task_type in self.task_type_map.values():
                    # Find the numeric label corresponding to the task type
                    label = [k for k, v in self.task_type_map.items() if v == task_type][0]
                    labels.append(label)
                else:
                    # Use random label
                    labels.append(np.random.randint(0, len(self.task_type_map)))
            
            labels = np.array(labels)
        else:
            # Generate random labels
            labels = np.random.randint(0, len(self.task_type_map), self.sequence_length)
        
        # Ensure sequence length
        if len(labels) < self.sequence_length:
            labels = np.pad(labels, (0, self.sequence_length - len(labels)), 'constant')
        elif len(labels) > self.sequence_length:
            labels = labels[:self.sequence_length]
        
        return labels
    
    def _generate_dummy_data(self) -> pd.DataFrame:
        """Generate dummy data"""
        # Create simulated programming task data
        num_samples = self.sequence_length
        data = {
            'lines_of_code': np.random.randint(10, 1000, num_samples),
            'function_count': np.random.randint(1, 20, num_samples),
            'class_count': np.random.randint(0, 10, num_samples),
            'variable_count': np.random.randint(5, 50, num_samples),
            'comment_density': np.random.uniform(0.1, 0.3, num_samples),
            'complexity': np.random.uniform(1, 10, num_samples),
            'task_type': np.random.choice(list(self.task_type_map.values()), num_samples)
        }
        return pd.DataFrame(data)

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
               epochs: int = 10, lr: float = 0.001, device: str = 'cpu') -> Dict:
    """Train programming model
    Args:
        model: Programming model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        lr: Learning rate
        device: Training device
    Returns:
        Training history dictionary
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    train_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'learning_rate': []
    }
    
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += targets.numel()
                correct += (predicted == targets).sum().item()
        
        # Update learning rate
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Record history
        avg_train_loss = running_loss / len(train_loader)
        avg_val_accuracy = 100 * correct / total if total > 0 else 0
        
        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        train_history['val_accuracy'].append(avg_val_accuracy)
        train_history['learning_rate'].append(current_lr)
        
        logger.info(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {avg_train_loss:.4f}, '
              f'Val Loss: {avg_val_loss:.4f}, '
              f'Val Acc: {avg_val_accuracy:.2f}%, '
              f'LR: {current_lr:.8f}')
    
    return train_history

def evaluate_model(model: nn.Module, test_loader: DataLoader, device: str = 'cpu') -> Dict:
    """Evaluate programming model
    Args:
        model: Programming model
        test_loader: Test data loader
        device: Evaluation device
    Returns:
        Evaluation result dictionary
    """
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += targets.numel()
            correct += (predicted == targets).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100 * correct / total if total > 0 else 0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }

def save_training_results(model: nn.Module, history: Dict, results: Dict, 
                         save_path: str = 'models/k_programming_model.pth'):
    """Save training results
    Args:
        model: Trained model
        history: Training history
        results: Evaluation results
        save_path: Save path
    """
    # Create model directory
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model),
        'training_history': history,
        'evaluation_results': results,
        'timestamp': time.time()
    }, save_path)
    
    # Save training log
    log_path = save_path.replace('.pth', '_log.json')
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'training_history': history,
            'evaluation_results': results,
            'timestamp': time.time()
        }, f, indent=2)
    
    logger.info(f"Model and training results saved: {save_path}")

def train_jointly(models: List[nn.Module], train_loaders: List[DataLoader], 
                  val_loaders: List[DataLoader], epochs: int = 10, 
                  lrs: List[float] = None, device: str = 'cpu', 
                  loss_weights: List[float] = None) -> Dict:
    """Train multiple models jointly
    Args:
        models: List of models to train jointly
        train_loaders: List of training data loaders
        val_loaders: List of validation data loaders
        epochs: Number of epochs
        lrs: List of learning rates for each model
        device: Training device
        loss_weights: List of weights for each model's loss
    Returns:
        Joint training history dictionary
    """
    # Check input consistency
    if len(models) != len(train_loaders) or len(models) != len(val_loaders):
        raise ValueError("Number of models, train loaders, and val loaders must match")
    
    # Initialize default learning rates if not provided
    if lrs is None:
        lrs = [0.001] * len(models)
    elif len(lrs) != len(models):
        raise ValueError("Number of learning rates must match number of models")
    
    # Initialize default loss weights if not provided
    if loss_weights is None:
        loss_weights = [1.0] * len(models)
    elif len(loss_weights) != len(models):
        raise ValueError("Number of loss weights must match number of models")
    
    # Normalize loss weights
    total_weight = sum(loss_weights)
    loss_weights = [w / total_weight for w in loss_weights]
    
    # Set up optimizers and schedulers
    optimizers = []
    schedulers = []
    criteria = []
    
    for i, model in enumerate(models):
        model.to(device)
        optimizers.append(optim.Adam(model.parameters(), lr=lrs[i]))
        schedulers.append(optim.lr_scheduler.ReduceLROnPlateau(optimizers[i], 
                                                             mode='min', factor=0.5, patience=3))
        criteria.append(nn.CrossEntropyLoss())
    
    # Initialize joint training history
    joint_history = {
        'joint_loss': [],
        'individual_losses': [[] for _ in range(len(models))],
        'val_loss': [],
        'val_accuracies': [[] for _ in range(len(models))],
        'learning_rates': [[] for _ in range(len(models))]
    }
    
    for epoch in range(epochs):
        # Training phase
        for model in models:
            model.train()
        
        running_joint_loss = 0.0
        running_individual_losses = [0.0] * len(models)
        
        # Get minimum length of loaders to ensure all models train for the same number of batches
        min_loader_len = min(len(loader) for loader in train_loaders)
        
        for batch_idx in range(min_loader_len):
            # Zero gradients for all optimizers
            for optimizer in optimizers:
                optimizer.zero_grad()
            
            batch_losses = []
            
            # Forward pass through each model
            for i in range(len(models)):
                train_iter = iter(train_loaders[i])
                inputs, targets = next(train_iter)
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = models[i](inputs)
                loss = criteria[i](outputs, targets)
                batch_losses.append(loss)
                
                # Apply loss weight
                if i == 0:  # First model's loss is added directly
                    joint_loss = loss * loss_weights[i]
                else:  # Subsequent losses are added with their weights
                    joint_loss += loss * loss_weights[i]
                
                running_individual_losses[i] += loss.item()
            
            # Backward pass and optimize
            joint_loss.backward()
            for optimizer in optimizers:
                optimizer.step()
            
            running_joint_loss += joint_loss.item()
        
        # Validation phase
        for model in models:
            model.eval()
        
        val_joint_loss = 0.0
        val_individual_losses = [0.0] * len(models)
        val_correct = [0] * len(models)
        val_total = [0] * len(models)
        
        with torch.no_grad():
            min_val_loader_len = min(len(loader) for loader in val_loaders)
            
            for batch_idx in range(min_val_loader_len):
                # Forward pass through each model for validation
                for i in range(len(models)):
                    val_iter = iter(val_loaders[i])
                    inputs, targets = next(val_iter)
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    
                    outputs = models[i](inputs)
                    loss = criteria[i](outputs, targets)
                    val_individual_losses[i] += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    val_total[i] += targets.numel()
                    val_correct[i] += (predicted == targets).sum().item()
            
            # Calculate joint validation loss
            for i in range(len(models)):
                avg_individual_val_loss = val_individual_losses[i] / min_val_loader_len
                val_joint_loss += avg_individual_val_loss * loss_weights[i]
            
            # Update learning rates based on individual model validation loss
            for i in range(len(models)):
                avg_individual_val_loss = val_individual_losses[i] / min_val_loader_len
                schedulers[i].step(avg_individual_val_loss)
        
        # Record history
        avg_joint_loss = running_joint_loss / min_loader_len
        avg_val_joint_loss = val_joint_loss
        
        joint_history['joint_loss'].append(avg_joint_loss)
        joint_history['val_loss'].append(avg_val_joint_loss)
        
        for i in range(len(models)):
            avg_individual_loss = running_individual_losses[i] / min_loader_len
            joint_history['individual_losses'][i].append(avg_individual_loss)
            
            avg_val_accuracy = 100 * val_correct[i] / val_total[i] if val_total[i] > 0 else 0
            joint_history['val_accuracies'][i].append(avg_val_accuracy)
            
            current_lr = optimizers[i].param_groups[0]['lr']
            joint_history['learning_rates'][i].append(current_lr)
        
        # Log epoch results
        logger.info(f'Epoch {epoch+1}/{epochs}, Joint Loss: {avg_joint_loss:.4f}, Val Loss: {avg_val_joint_loss:.4f}')
        for i in range(len(models)):
            logger.info(f'  Model {i+1}: Train Loss: {joint_history["individual_losses"][i][-1]:.4f}, ' +
                       f'Val Acc: {joint_history["val_accuracies"][i][-1]:.2f}%, LR: {joint_history["learning_rates"][i][-1]:.8f}')
    
    return joint_history

def main():
    """Main training function"""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = ProgrammingDataset('data/train')
    val_dataset = ProgrammingDataset('data/val')
    test_dataset = ProgrammingDataset('data/test')
    
    logger.info(f"Training set size: {len(train_dataset)}")
    logger.info(f"Validation set size: {len(val_dataset)}")
    logger.info(f"Test set size: {len(test_dataset)}")
    logger.info(f"Task type mapping: {train_dataset.task_type_map}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    num_classes = len(train_dataset.task_type_map) if train_dataset.task_type_map else 12
    model = ProgrammingModel(input_size=6, hidden_size=128, num_classes=num_classes, num_layers=3)
    
    # Train model (single model training)
    logger.info("Starting programming model training (single model mode)")
    history = train_model(model, train_loader, val_loader, epochs=20, lr=0.001, device=device)
    
    # Evaluate model
    logger.info("Starting programming model evaluation")
    results = evaluate_model(model, test_loader, device=device)
    
    # Save results
    save_training_results(model, history, results)
    
    logger.info("Programming model training completed")
    logger.info(f"Final evaluation results - Loss: {results['loss']:.4f}, Accuracy: {results['accuracy']:.2f}%")
    
    # Example: Joint training with multiple models (uncomment to use)
    # This is just an example - in practice, you would want to load different types of models
    # from different modules
    
    # # Prepare models for joint training
    # model2 = ProgrammingModel(input_size=6, hidden_size=128, num_classes=num_classes, num_layers=3)
    # model3 = ProgrammingModel(input_size=6, hidden_size=128, num_classes=num_classes, num_layers=3)
    # models = [model, model2, model3]
    # 
    # # Prepare loaders for each model
    # # Note: In a real scenario, each model might have its own specific dataset
    # train_loaders = [train_loader, train_loader, train_loader]
    # val_loaders = [val_loader, val_loader, val_loader]
    # 
    # # Set loss weights for each model
    # loss_weights = [0.5, 0.3, 0.2]  # Adjust according to your requirements
    # 
    # # Joint training
    # logger.info("Starting joint training of multiple programming models")
    # joint_history = train_jointly(models, train_loaders, val_loaders, 
    #                              epochs=15, lrs=[0.001, 0.001, 0.001], 
    #                              device=device, loss_weights=loss_weights)
    # 
    # # Evaluate each model after joint training
    # for i, m in enumerate(models):
    #     logger.info(f"Evaluating model {i+1} after joint training")
    #     model_results = evaluate_model(m, test_loader, device=device)
    #     logger.info(f"Model {i+1} - Loss: {model_results['loss']:.4f}, Accuracy: {model_results['accuracy']:.2f}%")
    #     
    #     # Save individual model results
    #     save_training_results(m, joint_history, model_results, 
    #                          save_path=f'models/k_programming_model_joint_{i+1}.pth')

if __name__ == '__main__':
    main()
