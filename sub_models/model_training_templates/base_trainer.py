# Copyright 2025 AGI System Team
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

import os
import json
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

class BaseModelTrainer:
    def __init__(self, model, config_path="training_config.json"):
        self.model = model
        self.load_config(config_path)
        self.writer = SummaryWriter(log_dir=self.create_log_dir())
        
    def load_config(self, path):
        """Load training configuration file"""
        with open(path, 'r') as f:
            self.config = json.load(f)
        print(f"Loaded training config: {self.config['model_type']}")
        
    def create_log_dir(self):
        """Create timestamped log directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/{self.config['model_type']}_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        return log_dir
    
    def train_epoch(self, data_loader):
        """Single training epoch implementation"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            # Training logic placeholder
            # Actual implementation needs to be customized for specific models
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backpropagation and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Record batch loss
            if batch_idx % self.config['log_interval'] == 0:
                self.writer.add_scalar('train/loss_batch', loss.item(), 
                                      self.global_step)
                self.global_step += 1
        
        avg_loss = total_loss / len(data_loader)
        self.writer.add_scalar('train/loss_epoch', avg_loss, self.epoch)
        return avg_loss
    
    def validate(self, data_loader):
        """Validation epoch implementation"""
        self.model.eval()
        # Validation logic placeholder
        return 0.0  # Return validation metric
    
    def save_checkpoint(self):
        """Save model checkpoint (including model configuration)"""
        # Save model configuration
        config_path = os.path.join(self.writer.log_dir, "model_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
            
        # Save checkpoint
        checkpoint_path = os.path.join(self.writer.log_dir, f"checkpoint_epoch{self.epoch}.pt")
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.best_loss,
        }, checkpoint_path)
    
    def train(self, train_loader, val_loader=None, epochs=10):
        """Complete training process"""
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config['learning_rate']
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            train_loss = self.train_epoch(train_loader)
            
            if val_loader:
                val_metric = self.validate(val_loader)
                self.writer.add_scalar('val/metric', val_metric, epoch)
                
                # Early stopping and checkpoint saving logic
                if val_metric < self.best_loss:
                    self.best_loss = val_metric
                    self.save_checkpoint()
            
            print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}")
        
        self.writer.close()
        return self.best_loss

# Usage example
if __name__ == "__main__":
    # 1. Initialize model
    # model = YourModelClass()
    
    # 2. Create trainer
    # trainer = BaseModelTrainer(model)
    
    # 3. Prepare data loaders
    # train_loader = ...
    # val_loader = ...
    
    # 4. Start training
    # trainer.train(train_loader, val_loader, epochs=20)
