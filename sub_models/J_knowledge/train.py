"""
Knowledge Base Expert Model Training
Core model with comprehensive knowledge systems and self-learning capabilities
"""

import os
import json
import time
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KnowledgeDomain:
    """Knowledge domain configuration"""
    name: str
    subdomains: List[str]
    complexity_levels: List[str]

class KnowledgeBaseDataset(Dataset):
    """Dataset for knowledge base expert model training"""
    
    def __init__(self, data_dir: str, knowledge_domains: Optional[List[KnowledgeDomain]] = None):
        self.data_dir = data_dir
        self.knowledge_domains = knowledge_domains or self._get_default_domains()
        
        # Check for dataset info file
        dataset_info_path = os.path.join(data_dir, 'dataset_info.json')
        if os.path.exists(dataset_info_path):
            with open(dataset_info_path, 'r', encoding='utf-8') as f:
                self.dataset_info = json.load(f)
        else:
            self.dataset_info = self._create_mock_dataset_info()
        
        # Initialize data samples
        self.samples = self._load_samples()
    
    def _get_default_domains(self) -> List[KnowledgeDomain]:
        """Get default knowledge domains"""
        return [
            KnowledgeDomain("physics", ["classical", "quantum", "relativity", "thermodynamics"], ["basic", "intermediate", "advanced"]),
            KnowledgeDomain("mathematics", ["algebra", "calculus", "geometry", "statistics"], ["basic", "intermediate", "advanced"]),
            KnowledgeDomain("chemistry", ["organic", "inorganic", "physical", "analytical"], ["basic", "intermediate", "advanced"]),
            KnowledgeDomain("medicine", ["anatomy", "physiology", "pharmacology", "pathology"], ["basic", "intermediate", "advanced"]),
            KnowledgeDomain("law", ["civil", "criminal", "constitutional", "international"], ["basic", "intermediate", "advanced"]),
            KnowledgeDomain("history", ["ancient", "medieval", "modern", "contemporary"], ["basic", "intermediate", "advanced"]),
            KnowledgeDomain("sociology", ["social_structure", "social_change", "social_institutions"], ["basic", "intermediate", "advanced"]),
            KnowledgeDomain("humanities", ["philosophy", "literature", "arts", "languages"], ["basic", "intermediate", "advanced"]),
            KnowledgeDomain("psychology", ["cognitive", "developmental", "clinical", "social"], ["basic", "intermediate", "advanced"]),
            KnowledgeDomain("economics", ["microeconomics", "macroeconomics", "finance", "development"], ["basic", "intermediate", "advanced"]),
            KnowledgeDomain("management", ["strategic", "operations", "human_resources", "marketing"], ["basic", "intermediate", "advanced"]),
            KnowledgeDomain("mechanical_engineering", ["thermodynamics", "mechanics", "materials", "design"], ["basic", "intermediate", "advanced"]),
            KnowledgeDomain("electrical_engineering", ["circuits", "electronics", "power", "control"], ["basic", "intermediate", "advanced"]),
            KnowledgeDomain("food_engineering", ["food_processing", "preservation", "safety", "nutrition"], ["basic", "intermediate", "advanced"]),
            KnowledgeDomain("chemical_engineering", ["process", "reaction", "transport", "materials"], ["basic", "intermediate", "advanced"])
        ]
    
    def _create_mock_dataset_info(self) -> Dict:
        """Create mock dataset info for demonstration"""
        return {
            "total_samples": 1000,
            "knowledge_domains": [domain.name for domain in self.knowledge_domains],
            "data_format": "text_embeddings",
            "description": "Comprehensive knowledge base training data"
        }
    
    def _load_samples(self) -> List[Dict]:
        """Load or generate data samples"""
        samples = []
        
        # Generate mock samples if no real data
        for i in range(self.dataset_info["total_samples"]):
            domain_idx = i % len(self.knowledge_domains)
            domain = self.knowledge_domains[domain_idx]
            subdomain_idx = (i // len(self.knowledge_domains)) % len(domain.subdomains)
            complexity_idx = (i // (len(self.knowledge_domains) * len(domain.subdomains))) % len(domain.complexity_levels)
            
            sample = {
                "id": i,
                "domain": domain.name,
                "subdomain": domain.subdomains[subdomain_idx],
                "complexity": domain.complexity_levels[complexity_idx],
                "text_embedding": torch.randn(512),  # Mock text embedding
                "knowledge_label": torch.tensor([domain_idx, subdomain_idx, complexity_idx])
            }
            samples.append(sample)
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        return sample["text_embedding"], sample["knowledge_label"]

class KnowledgeExpertModel(nn.Module):
    """Knowledge Base Expert Model with comprehensive knowledge representation"""
    
    def __init__(self, 
                 input_dim: int = 512,
                 hidden_dims: List[int] = [1024, 2048, 1024],
                 num_domains: int = 15,
                 max_subdomains: int = 4,
                 max_complexity: int = 3):
        super(KnowledgeExpertModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.num_domains = num_domains
        self.max_subdomains = max_subdomains
        self.max_complexity = max_complexity
        
        # Feature extraction layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Domain classification head
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_domains)
        )
        
        # Subdomain classification head
        self.subdomain_classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, max_subdomains)
        )
        
        # Complexity classification head
        self.complexity_classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, max_complexity)
        )
        
        # Knowledge embedding projection
        self.knowledge_embedding = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1024),
            nn.ReLU(),
            nn.Linear(1024, 512)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.feature_extractor(x)
        
        outputs = {
            "domain_logits": self.domain_classifier(features),
            "subdomain_logits": self.subdomain_classifier(features),
            "complexity_logits": self.complexity_classifier(features),
            "knowledge_embedding": self.knowledge_embedding(features)
        }
        
        return outputs

def train_model(model: nn.Module, 
                train_loader: DataLoader, 
                val_loader: DataLoader, 
                epochs: int = 20,
                lr: float = 0.001,
                device: str = 'cpu',
                joint_training: bool = False,
                joint_info: Optional[Dict] = None) -> Dict:
    """Train the knowledge base expert model"""
    
    # Define loss functions for multi-task learning
    criterion_domain = nn.CrossEntropyLoss()
    criterion_subdomain = nn.CrossEntropyLoss()
    criterion_complexity = nn.CrossEntropyLoss()
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'domain_accuracy': [],
        'subdomain_accuracy': [],
        'complexity_accuracy': [],
        'learning_rate': []
    }
    
    model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        running_domain_correct = 0
        running_subdomain_correct = 0
        running_complexity_correct = 0
        total_samples = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            domain_targets = targets[:, 0].long().to(device)
            subdomain_targets = targets[:, 1].long().to(device)
            complexity_targets = targets[:, 2].long().to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate multi-task losses
            loss_domain = criterion_domain(outputs['domain_logits'], domain_targets)
            loss_subdomain = criterion_subdomain(outputs['subdomain_logits'], subdomain_targets)
            loss_complexity = criterion_complexity(outputs['complexity_logits'], complexity_targets)
            
            # Combined loss
            total_loss = loss_domain + loss_subdomain + loss_complexity
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            
            # Calculate accuracies
            _, domain_predicted = torch.max(outputs['domain_logits'].data, 1)
            _, subdomain_predicted = torch.max(outputs['subdomain_logits'].data, 1)
            _, complexity_predicted = torch.max(outputs['complexity_logits'].data, 1)
            
            running_domain_correct += (domain_predicted == domain_targets).sum().item()
            running_subdomain_correct += (subdomain_predicted == subdomain_targets).sum().item()
            running_complexity_correct += (complexity_predicted == complexity_targets).sum().item()
            total_samples += domain_targets.size(0)
        
        avg_train_loss = running_loss / len(train_loader)
        domain_accuracy = 100 * running_domain_correct / total_samples
        subdomain_accuracy = 100 * running_subdomain_correct / total_samples
        complexity_accuracy = 100 * running_complexity_correct / total_samples
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_domain_correct = 0
        val_subdomain_correct = 0
        val_complexity_correct = 0
        val_total_samples = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                domain_targets = targets[:, 0].long().to(device)
                subdomain_targets = targets[:, 1].long().to(device)
                complexity_targets = targets[:, 2].long().to(device)
                
                outputs = model(inputs)
                
                loss_domain = criterion_domain(outputs['domain_logits'], domain_targets)
                loss_subdomain = criterion_subdomain(outputs['subdomain_logits'], subdomain_targets)
                loss_complexity = criterion_complexity(outputs['complexity_logits'], complexity_targets)
                
                total_val_loss = loss_domain + loss_subdomain + loss_complexity
                val_loss += total_val_loss.item()
                
                _, domain_predicted = torch.max(outputs['domain_logits'].data, 1)
                _, subdomain_predicted = torch.max(outputs['subdomain_logits'].data, 1)
                _, complexity_predicted = torch.max(outputs['complexity_logits'].data, 1)
                
                val_domain_correct += (domain_predicted == domain_targets).sum().item()
                val_subdomain_correct += (subdomain_predicted == subdomain_targets).sum().item()
                val_complexity_correct += (complexity_predicted == complexity_targets).sum().item()
                val_total_samples += domain_targets.size(0)
        
        avg_val_loss = val_loss / len(val_loader)
        val_domain_accuracy = 100 * val_domain_correct / val_total_samples
        val_subdomain_accuracy = 100 * val_subdomain_correct / val_total_samples
        val_complexity_accuracy = 100 * val_complexity_correct / val_total_samples
        
        # Update learning rate
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        
        # Record history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['domain_accuracy'].append(val_domain_accuracy)
        history['subdomain_accuracy'].append(val_subdomain_accuracy)
        history['complexity_accuracy'].append(val_complexity_accuracy)
        history['learning_rate'].append(current_lr)
        
        logger.info(f'Epoch {epoch+1}/{epochs}, '
              f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, '
              f'Domain Acc: {val_domain_accuracy:.2f}%, Subdomain Acc: {val_subdomain_accuracy:.2f}%, '
              f'Complexity Acc: {val_complexity_accuracy:.2f}%, LR: {current_lr:.8f}')
    
    return history

def evaluate_model(model: nn.Module, 
                  data_loader: DataLoader, 
                  device: str = 'cpu') -> Dict:
    """Evaluate the knowledge base expert model"""
    model.eval()
    
    total_loss = 0.0
    domain_correct = 0
    subdomain_correct = 0
    complexity_correct = 0
    total_samples = 0
    
    criterion_domain = nn.CrossEntropyLoss()
    criterion_subdomain = nn.CrossEntropyLoss()
    criterion_complexity = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
            domain_targets = targets[:, 0].long().to(device)
            subdomain_targets = targets[:, 1].long().to(device)
            complexity_targets = targets[:, 2].long().to(device)
            
            outputs = model(inputs)
            
            loss_domain = criterion_domain(outputs['domain_logits'], domain_targets)
            loss_subdomain = criterion_subdomain(outputs['subdomain_logits'], subdomain_targets)
            loss_complexity = criterion_complexity(outputs['complexity_logits'], complexity_targets)
            
            total_loss += (loss_domain + loss_subdomain + loss_complexity).item()
            
            _, domain_predicted = torch.max(outputs['domain_logits'].data, 1)
            _, subdomain_predicted = torch.max(outputs['subdomain_logits'].data, 1)
            _, complexity_predicted = torch.max(outputs['complexity_logits'].data, 1)
            
            domain_correct += (domain_predicted == domain_targets).sum().item()
            subdomain_correct += (subdomain_predicted == subdomain_targets).sum().item()
            complexity_correct += (complexity_predicted == complexity_targets).sum().item()
            total_samples += domain_targets.size(0)
    
    avg_loss = total_loss / len(data_loader)
    domain_accuracy = 100 * domain_correct / total_samples
    subdomain_accuracy = 100 * subdomain_correct / total_samples
    complexity_accuracy = 100 * complexity_correct / total_samples
    
    results = {
        'loss': avg_loss,
        'domain_accuracy': domain_accuracy,
        'subdomain_accuracy': subdomain_accuracy,
        'complexity_accuracy': complexity_accuracy,
        'overall_accuracy': (domain_accuracy + subdomain_accuracy + complexity_accuracy) / 3
    }
    
    logger.info(f'Evaluation Results: Loss: {avg_loss:.4f}, Domain Acc: {domain_accuracy:.2f}%, '
                f'Subdomain Acc: {subdomain_accuracy:.2f}%, Complexity Acc: {complexity_accuracy:.2f}%')
    
    return results

def save_training_results(model: nn.Module, 
                         history: Dict, 
                         evaluation_results: Dict, 
                         save_dir: str,
                         joint_training: bool = False,
                         joint_results: Optional[Dict] = None):
    """Save training results and model"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model state
    model_path = os.path.join(save_dir, 'knowledge_expert_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': history,
        'evaluation_results': evaluation_results,
        'joint_training': joint_training,
        'joint_results': joint_results
    }, model_path)
    
    # Save training history
    history_path = os.path.join(save_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save evaluation results
    eval_path = os.path.join(save_dir, 'evaluation_results.json')
    with open(eval_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Save training log
    log_path = os.path.join(save_dir, 'training_log.json')
    training_log = {
        'timestamp': time.time(),
        'model_type': 'KnowledgeExpertModel',
        'joint_training': joint_training,
        'final_domain_accuracy': evaluation_results['domain_accuracy'],
        'final_subdomain_accuracy': evaluation_results['subdomain_accuracy'],
        'final_complexity_accuracy': evaluation_results['complexity_accuracy'],
        'final_overall_accuracy': evaluation_results['overall_accuracy']
    }
    
    with open(log_path, 'w') as f:
        json.dump(training_log, f, indent=2)
    
    logger.info(f'Training results saved to {save_dir}')

def train_jointly(models: List[nn.Module], 
                  data_loaders: List[DataLoader], 
                  val_loaders: List[DataLoader], 
                  epochs: int = 20,
                  learning_rates: Optional[List[float]] = None,
                  loss_weights: Optional[List[float]] = None,
                  device: str = 'cpu') -> Dict:
    """Joint training for knowledge expert model with other models"""
    
    if len(models) != len(data_loaders) or len(models) != len(val_loaders):
        raise ValueError("Number of models must match number of data loaders")
    
    num_models = len(models)
    
    # Set default learning rates and loss weights
    if learning_rates is None:
        learning_rates = [0.001] * num_models
    if loss_weights is None:
        loss_weights = [1.0] * num_models
    
    # Normalize loss weights
    total_weight = sum(loss_weights)
    loss_weights = [w / total_weight for w in loss_weights]
    
    # Create optimizers and schedulers for each model
    optimizers = []
    schedulers = []
    
    for i, model in enumerate(models):
        optimizer = optim.Adam(model.parameters(), lr=learning_rates[i], weight_decay=1e-4)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
    
    # Define loss functions
    criterion_domain = nn.CrossEntropyLoss()
    criterion_subdomain = nn.CrossEntropyLoss()
    criterion_complexity = nn.CrossEntropyLoss()
    
    joint_history = {
        'joint_train_loss': [],
        'joint_val_loss': [],
        'model_train_losses': [[] for _ in range(num_models)],
        'model_val_losses': [[] for _ in range(num_models)],
        'learning_rates': [[] for _ in range(num_models)]
    }
    
    # Move all models to device
    for model in models:
        model.to(device)
    
    for epoch in range(epochs):
        # Training phase
        joint_train_loss = 0.0
        model_train_losses = [0.0] * num_models
        
        for model_idx in range(num_models):
            models[model_idx].train()
        
        # Get batch iterators for all data loaders
        batch_iterators = [iter(loader) for loader in data_loaders]
        
        # Process batches
        batch_count = 0
        while True:
            try:
                batch_losses = []
                
                for model_idx in range(num_models):
                    try:
                        inputs, targets = next(batch_iterators[model_idx])
                    except StopIteration:
                        # Reset iterator if we reach the end
                        batch_iterators[model_idx] = iter(data_loaders[model_idx])
                        inputs, targets = next(batch_iterators[model_idx])
                    
                    inputs = inputs.to(device)
                    domain_targets = targets[:, 0].long().to(device)
                    subdomain_targets = targets[:, 1].long().to(device)
                    complexity_targets = targets[:, 2].long().to(device)
                    
                    optimizers[model_idx].zero_grad()
                    outputs = models[model_idx](inputs)
                    
                    # Calculate multi-task losses
                    loss_domain = criterion_domain(outputs['domain_logits'], domain_targets)
                    loss_subdomain = criterion_subdomain(outputs['subdomain_logits'], subdomain_targets)
                    loss_complexity = criterion_complexity(outputs['complexity_logits'], complexity_targets)
                    
                    total_loss = loss_domain + loss_subdomain + loss_complexity
                    weighted_loss = total_loss * loss_weights[model_idx]
                    
                    weighted_loss.backward()
                    optimizers[model_idx].step()
                    
                    batch_losses.append(total_loss.item())
                    model_train_losses[model_idx] += total_loss.item()
                
                joint_train_loss += sum(batch_losses)
                batch_count += 1
                
            except StopIteration:
                break
        
        # Calculate average losses
        joint_train_loss /= batch_count
        for i in range(num_models):
            model_train_losses[i] /= batch_count
        
        # Validation phase
        joint_val_loss = 0.0
        model_val_losses = [0.0] * num_models
        
        for model_idx in range(num_models):
            models[model_idx].eval()
        
        val_batch_count = 0
        with torch.no_grad():
            while True:
                try:
                    val_batch_losses = []
                    
                    for model_idx in range(num_models):
                        try:
                            inputs, targets = next(batch_iterators[model_idx])
                        except StopIteration:
                            break
                        
                        inputs = inputs.to(device)
                        domain_targets = targets[:, 0].long().to(device)
                        subdomain_targets = targets[:, 1].long().to(device)
                        complexity_targets = targets[:, 2].long().to(device)
                        
                        outputs = models[model_idx](inputs)
                        
                        loss_domain = criterion_domain(outputs['domain_logits'], domain_targets)
                        loss_subdomain = criterion_subdomain(outputs['subdomain_logits'], subdomain_targets)
                        loss_complexity = criterion_complexity(outputs['complexity_logits'], complexity_targets)
                        
                        total_loss = loss_domain + loss_subdomain + loss_complexity
                        val_batch_losses.append(total_loss.item())
                        model_val_losses[model_idx] += total_loss.item()
                    
                    if len(val_batch_losses) == num_models:
                        joint_val_loss += sum(val_batch_losses)
                        val_batch_count += 1
                    else:
                        break
                        
                except StopIteration:
                    break
        
        if val_batch_count > 0:
            joint_val_loss /= val_batch_count
            for i in range(num_models):
                model_val_losses[i] /= val_batch_count
        
        # Update learning rates
        current_lrs = []
        for i in range(num_models):
            schedulers[i].step(joint_val_loss)
            current_lrs.append(optimizers[i].param_groups[0]['lr'])
        
        # Record history
        joint_history['joint_train_loss'].append(joint_train_loss)
        joint_history['joint_val_loss'].append(joint_val_loss)
        for i in range(num_models):
            joint_history['model_train_losses'][i].append(model_train_losses[i])
            joint_history['model_val_losses'][i].append(model_val_losses[i])
            joint_history['learning_rates'][i].append(current_lrs[i])
        
        logger.info(f'Joint Epoch {epoch+1}/{epochs}, '
              f'Joint Train Loss: {joint_train_loss:.4f}, Joint Val Loss: {joint_val_loss:.4f}')
        for i in range(num_models):
            logger.info(f'  Model {i}: Train Loss: {model_train_losses[i]:.4f}, Val Loss: {model_val_losses[i]:.4f}, LR: {current_lrs[i]:.8f}')
    
    return joint_history

class MockKnowledgeDataset(Dataset):
    """Mock dataset for knowledge base expert model testing"""
    
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples
        self.samples = []
        
        # Create mock knowledge samples
        domains = ['physics', 'mathematics', 'chemistry', 'medicine', 'law', 
                  'history', 'sociology', 'humanities', 'psychology', 'economics',
                  'management', 'mechanical_engineering', 'electrical_engineering', 
                  'food_engineering', 'chemical_engineering']
        
        subdomains = ['basic', 'intermediate', 'advanced']
        complexity_levels = ['low', 'medium', 'high']
        
        for i in range(num_samples):
            domain_idx = i % len(domains)
            subdomain_idx = (i // len(domains)) % len(subdomains)
            complexity_idx = (i // (len(domains) * len(subdomains))) % len(complexity_levels)
            
            sample = {
                'text_embedding': torch.randn(512),
                'knowledge_label': torch.tensor([domain_idx, subdomain_idx, complexity_idx])
            }
            self.samples.append(sample)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]['text_embedding'], self.samples[idx]['knowledge_label']

def main():
    """Main function for knowledge base expert model training"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create datasets and data loaders
    data_dir = 'data/knowledge_data'
    os.makedirs(data_dir, exist_ok=True)
    
    dataset = KnowledgeBaseDataset(data_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    model = KnowledgeExpertModel()
    
    # Training configuration
    epochs = 20
    learning_rate = 0.001
    
    try:
        # Train model
        logger.info('Starting knowledge base expert model training...')
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=learning_rate,
            device=device
        )
        
        # Evaluate model
        logger.info('Evaluating knowledge base expert model...')
        eval_results = evaluate_model(model, val_loader, device)
        
        # Save results
        save_dir = 'training_results/knowledge_expert'
        save_training_results(model, history, eval_results, save_dir)
        
        logger.info('Knowledge base expert model training completed successfully!')
        
        # Joint training demonstration
        logger.info('Starting joint training demonstration...')
        
        # Create multiple models for joint training
        model1 = KnowledgeExpertModel()
        model2 = KnowledgeExpertModel()
        
        # Create mock datasets for demonstration
        mock_dataset1 = MockKnowledgeDataset(500)
        mock_dataset2 = MockKnowledgeDataset(500)
        
        mock_train1, mock_val1 = torch.utils.data.random_split(mock_dataset1, [400, 100])
        mock_train2, mock_val2 = torch.utils.data.random_split(mock_dataset2, [400, 100])
        
        mock_train_loader1 = DataLoader(mock_train1, batch_size=16, shuffle=True)
        mock_val_loader1 = DataLoader(mock_val1, batch_size=16, shuffle=False)
        mock_train_loader2 = DataLoader(mock_train2, batch_size=16, shuffle=True)
        mock_val_loader2 = DataLoader(mock_val2, batch_size=16, shuffle=False)
        
        # Perform joint training
        joint_models = [model1, model2]
        joint_train_loaders = [mock_train_loader1, mock_train_loader2]
        joint_val_loaders = [mock_val_loader1, mock_val_loader2]
        
        joint_history = train_jointly(
            models=joint_models,
            data_loaders=joint_train_loaders,
            val_loaders=joint_val_loaders,
            epochs=10,
            learning_rates=[0.001, 0.001],
            loss_weights=[0.5, 0.5],
            device=device
        )
        
        # Save joint training results
        joint_save_dir = 'training_results/knowledge_joint'
        save_training_results(
            model=model1,  # Save one of the models
            history=history,
            evaluation_results=eval_results,
            save_dir=joint_save_dir,
            joint_training=True,
            joint_results=joint_history
        )
        
        logger.info('Joint training demonstration completed!')
        
    except Exception as e:
        logger.error(f'Training failed with error: {str(e)}')
        raise

if __name__ == '__main__':
    main()