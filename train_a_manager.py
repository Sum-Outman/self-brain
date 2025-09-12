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
    """Training Data Generator"""
    
    def __init__(self):
        self.emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'trust', 'anticipation']
        self.task_types = ['text', 'audio', 'image', 'video', 'spatial', 'sensor', 'control', 'motion', 'knowledge', 'programming']
        
    def generate_sample(self):
        """Generate training sample"""
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

class AManagerTrainer:
    """A Manager Model Trainer"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.data_generator = TrainingDataGenerator()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
        # Loss functions
        self.emotion_criterion = nn.BCELoss()
        self.task_criterion = nn.CrossEntropyLoss()
        self.output_criterion = nn.MSELoss()
        
        # Training log
        self.setup_logging()
        
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
        """Train one epoch"""
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
            
            avg_batch_loss = batch_loss / len(batch)
            total_loss += avg_batch_loss
            
            if batch_idx % 10 == 0:
                self.logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {avg_batch_loss:.4f}")
        
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
