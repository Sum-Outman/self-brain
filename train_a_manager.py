#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A Manager Model Training Script
"""

import torch
import torch.nn as nn
import numpy as np
from trainable_a_manager import TrainableAManager, TrainingConfig
import json
import os
from datetime import datetime
import logging

class TrainingDataGenerator:
    """Training Data Generator with Real Data Support"""
    
    def __init__(self, data_dir='./training_data'):
        self.emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation']
        self.task_types = ['text', 'audio', 'image', 'video', 'spatial', 'sensor', 'control', 'motion', 'knowledge', 'programming']
        self.data_dir = data_dir
        self.real_training_data = self.load_real_training_data()
        self.current_index = 0
        
    def load_real_training_data(self):
        """Load real training data from files"""
        real_data = []
        data_file = os.path.join(self.data_dir, 'a_manager_training_data.json')
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
        
        # If real data file exists, load it
        if os.path.exists(data_file):
            try:
                with open(data_file, 'r', encoding='utf-8') as f:
                    real_data = json.load(f)
                print(f"Loaded {len(real_data)} real training samples")
            except Exception as e:
                print(f"Failed to load real training data: {e}")
        
        return real_data
    
    def save_real_training_data(self, data):
        """Save real training data to file"""
        data_file = os.path.join(self.data_dir, 'a_manager_training_data.json')
        try:
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(data)} training samples to {data_file}")
        except Exception as e:
            print(f"Failed to save training data: {e}")
    
    def generate_sample(self):
        """Generate or load training sample"""
        # Use real data if available
        if self.real_training_data and len(self.real_training_data) > 0:
            sample = self.real_training_data[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.real_training_data)
            
            # Convert to tensors
            return {
                'input': torch.tensor(sample['input'], dtype=torch.float32),
                'emotion_target': torch.tensor(sample['emotion_target'], dtype=torch.float32),
                'task_target': torch.tensor(sample['task_target'], dtype=torch.float32),
                'output_target': torch.tensor(sample['output_target'], dtype=torch.float32)
            }
        else:
            # Fallback to synthetic data if no real data available
            # Simulate input features
            input_features = torch.randn(1, 50, 768)
            
            # Generate emotion labels
            emotion_labels = torch.rand(8)
            emotion_labels = emotion_labels / emotion_labels.sum()
            
            # Generate task assignment labels
            task_labels = torch.zeros(11)
            task_idx = np.random.randint(0, 11)
            task_labels[task_idx] = 1.0
            
            # Generate target output
            target_output = torch.randn(1, 768)
            
            return {
                'input': input_features,
                'emotion_target': emotion_labels,
                'task_target': task_labels,
                'output_target': target_output
            }
    
    def generate_batch(self, batch_size=32):
        """Generate batch data"""
        batch = []
        for _ in range(batch_size):
            batch.append(self.generate_sample())
        return batch
    
    def add_training_example(self, input_data, emotion_target, task_target, output_target):
        """Add a new training example"""
        example = {
            'input': input_data.tolist(),
            'emotion_target': emotion_target.tolist(),
            'task_target': task_target.tolist(),
            'output_target': output_target.tolist(),
            'timestamp': datetime.now().isoformat()
        }
        self.real_training_data.append(example)
        return example

class AManagerTrainer:
    """A Manager Model Trainer with Real Data Collection"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.data_generator = TrainingDataGenerator(data_dir='./training_data/a_manager')
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
        # Loss functions
        self.emotion_criterion = nn.BCELoss()
        self.task_criterion = nn.CrossEntropyLoss()
        self.output_criterion = nn.MSELoss()
        
        # Training log
        self.setup_logging()
        
        # Data collection for real training examples
        self.collected_examples = []
        self.collection_threshold = 0.1  # Collect examples with loss above this threshold
        
    def setup_logging(self):
        """Setup training logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('a_manager_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, epoch):
        """Train one epoch with real data collection"""
        self.model.train()
        total_loss = 0
        num_batches = 100  # Train 100 batches per epoch
        
        for batch_idx in range(num_batches):
            batch = self.data_generator.generate_batch(self.config.batch_size)
            
            batch_loss = 0
            for sample in batch:
                self.optimizer.zero_grad()
                
                # Forward propagation
                outputs = self.model.forward(sample['input'])
                
                # Calculate loss
                emotion_loss = self.emotion_criterion(outputs['emotions'], sample['emotion_target'])
                task_loss = self.task_criterion(outputs['model_weights'], sample['task_target'].unsqueeze(0))
                output_loss = self.output_criterion(outputs['output'], sample['output_target'])
                
                # Total loss
                total_batch_loss = emotion_loss + task_loss + output_loss
                
                # Backward propagation
                total_batch_loss.backward()
                self.optimizer.step()
                
                batch_loss += total_batch_loss.item()
                
                # Collect examples with high loss for future training
                if total_batch_loss.item() > self.collection_threshold:
                    with torch.no_grad():
                        self.collected_examples.append({
                            'input': sample['input'].clone(),
                            'emotion_target': sample['emotion_target'].clone(),
                            'task_target': sample['task_target'].clone(),
                            'output_target': sample['output_target'].clone(),
                            'loss': total_batch_loss.item()
                        })
            
            avg_batch_loss = batch_loss / len(batch)
            total_loss += avg_batch_loss
            
            if batch_idx % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {avg_batch_loss:.4f}")
            
        # Save collected examples after each epoch
        if self.collected_examples and epoch % 5 == 0:
            for example in self.collected_examples:
                self.data_generator.add_training_example(
                    example['input'],
                    example['emotion_target'],
                    example['task_target'],
                    example['output_target']
                )
            self.data_generator.save_real_training_data(self.data_generator.real_training_data)
            self.logger.info(f"Saved {len(self.collected_examples)} training examples")
            self.collected_examples = []  # Reset collection after saving
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        val_loss = 0
        num_val_batches = 20
        
        with torch.no_grad():
            for _ in range(num_val_batches):
                batch = self.data_generator.generate_batch(self.config.batch_size)
                
                batch_loss = 0
                for sample in batch:
                    outputs = self.model.forward(sample['input'])
                    
                    emotion_loss = self.emotion_criterion(outputs['emotions'], sample['emotion_target'])
                    task_loss = self.task_criterion(outputs['model_weights'], sample['task_target'].unsqueeze(0))
                    output_loss = self.output_criterion(outputs['output'], sample['output_target'])
                    
                    total_loss = emotion_loss + task_loss + output_loss
                    batch_loss += total_loss.item()
                
                val_loss += batch_loss / len(batch)
        
        return val_loss / num_val_batches
    
    def train(self, num_epochs=None):
        """Complete training process"""
        if num_epochs is None:
            num_epochs = self.config.epochs
            
        self.logger.info("🚀 Starting A Manager Model training...")
        self.logger.info(f"Model parameter count: {sum(p.numel() for p in self.model.parameters())}")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()
            
            self.scheduler.step(val_loss)
            
            self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch, val_loss)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"🛑 Early stopping at epoch {epoch}")
                break
        
        self.logger.info("✅ A Manager Model training completed!")
    
    def save_model(self, epoch, val_loss):
        """Save model"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"a_manager_epoch_{epoch}_loss_{val_loss:.4f}_{timestamp}.pth"
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': vars(self.config)
        }, filename)
        
        # Also save the latest version
        torch.save(self.model.state_dict(), 'a_manager_latest.pth')
        
        self.logger.info(f"💾 Model saved: {filename}")
    
    def load_model(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.logger.info(f"📂 Model loaded: {filepath}")

def main():
    """Main training function"""
    print("🎯 A Manager Model Training System")
    print("=" * 50)
    
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Using device: {device}")
    
    # Initialize model and configuration
    model = TrainableAManager()
    config = TrainingConfig()
    
    if torch.cuda.is_available():
        model = model.to(device)
    
    # Initialize trainer
    trainer = AManagerTrainer(model, config)
    
    # Start training
    trainer.train()
    
    print("\n🎉 Training completed!")
    print("🚀 You can now start the interactive A Manager Model")
    print("Run: python trainable_a_manager.py")

if __name__ == "__main__":
    main()
