# Management Model for Sub-Model Coordination

This repository contains the implementation of a management model designed to coordinate multiple sub-models for emotional analysis and strategy prediction tasks. The management model integrates outputs from various specialized sub-models to provide enhanced decision-making capabilities.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Overview

The Management Model serves as a central coordination system for multiple sub-models, primarily focusing on:

1. **Sub-model Registration and Management**: Register, track, and manage multiple specialized sub-models
2. **Emotion Analysis Integration**: Combine emotional analysis results from different sub-models
3. **Strategy Prediction**: Make informed decisions based on integrated emotional insights
4. **Performance Monitoring**: Track and evaluate the performance of individual sub-models and the overall system
5. **Dynamic Adaptation**: Adjust strategies based on real-time performance and environmental changes

## Directory Structure

```
A_management/
├── enhanced_manager.py    # Core management model implementation
├── enhanced_trainer.py    # Training utilities for the management model
├── enhanced_evaluator.py  # Evaluation tools for performance assessment
├── config.py              # Configuration classes and utilities
├── app.py                 # Main application with command-line interface
├── README.md              # Project documentation
└── logs/                  # Directory for log files
```

## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.9 or higher
- scikit-learn
- pandas
- numpy

### Setup

1. Clone the repository

```bash
# Navigate to the parent directory
cd d:\shiyan\sub_models

# The A_management directory should already exist
```

2. Install the required dependencies

```bash
pip install torch scikit-learn pandas numpy
```

## Configuration

The management model uses a hierarchical configuration system defined in `config.py`. Configuration options include:

- **Model Configuration**: Neural network architecture, hidden dimensions, dropout rates, etc.
- **Training Configuration**: Batch size, learning rate, epochs, optimizer settings, etc.
- **Evaluation Configuration**: Metrics, batch size, etc.
- **Base Configuration**: Output directories, logging settings, etc.

### Configuration Environments

The system supports multiple configuration environments:

- **Default**: General-purpose configuration
- **Development**: Optimized for development and debugging
- **Production**: Optimized for deployment
- **Testing**: Optimized for automated testing

To use a specific environment configuration, use the `--env` flag when running commands.

## Usage

The management model can be used in multiple ways, including training, evaluation, and inference.

### Training the Model
To train the model from scratch, use the `train.py` script:

```bash
python train.py --epochs 20 --batch 16 --lr 0.0001
```

#### Training Parameters
- `--epochs`: Number of training epochs (default: 20)
- `--batch`: Batch size (default: 16)
- `--lr`: Learning rate (default: 0.0001)
- `--from_scratch`: Whether to train from scratch (default: True)
- `--val_split`: Validation split ratio (default: 0.2)
- `--hidden_sizes`: Comma-separated list of hidden layer sizes (default: "128,64,32")

### Testing and Inference
To test the trained model with a sample task, use the `test_model.py` script:

```bash
python test_model.py
```

This will automatically find the latest trained model and perform inference on a sample task, showing the predicted strategy, emotional response, and emotion-adjusted message.

### Command-Line Interface

The management model provides a command-line interface through `app.py` with the following commands:

#### Load a Model

```bash
python app.py load --model-path /path/to/model.pth
```

#### Train the Model

```bash
python app.py train --train-data /path/to/train_data.json --val-data /path/to/val_data.json --env dev
```

#### Evaluate the Model

```bash
python app.py evaluate --data /path/to/test_data.json --model-path /path/to/model.pth
```

#### Make Predictions

```bash
python app.py predict --input "Customer is angry about product quality" --model-path /path/to/model.pth
```

#### Analyze Integration Effectiveness

```bash
python app.py analyze --base-model /path/to/base_model.pth --enhanced-model /path/to/enhanced_model.pth --data /path/to/analysis_data.json
```

#### Run Demo

```bash
python app.py demo --env dev
```

## API Reference

### ManagementModel Class

The core class that implements the management model functionality.

#### Methods

- `__init__(self, config)`: Initialize the management model with the given configuration
- `register_sub_model(self, model_id, model, model_type='language')`: Register a sub-model
- `deregister_sub_model(self, model_id)`: Deregister a sub-model
- `update_sub_model_weight(self, model_id, weight)`: Update the weight of a registered sub-model
- `forward(self, inputs, sub_model_outputs=None)`: Forward pass through the model
- `predict(self, inputs, sub_model_outputs=None)`: Generate predictions
- `save_model(self, path)`: Save the model to the specified path
- `load_model(self, path)`: Load the model from the specified path

### ModelTrainer Class

Handles the training process for the management model.

#### Methods

- `__init__(self, model, config)`: Initialize the trainer with the model and configuration
- `prepare_data(self, train_data_path, val_data_path=None, test_data_path=None)`: Prepare the training data
- `train(self)`: Run the training process
- `validate(self, data_loader)`: Validate the model on the given data
- `save_checkpoint(self, epoch, metrics, is_best=False)`: Save a training checkpoint
- `load_checkpoint(self, path)`: Load a training checkpoint

### ModelEvaluator Class

Provides evaluation functionality for the management model.

#### Methods

- `__init__(self, model, config)`: Initialize the evaluator with the model and configuration
- `evaluate(self, data_loader, save_results=True)`: Evaluate the model on the given data
- `calculate_metrics(self, predictions, targets)`: Calculate evaluation metrics
- `generate_confusion_matrix(self, predictions, targets, labels=None)`: Generate a confusion matrix
- `evaluate_sub_model_performance(self, data_loader)`: Evaluate the performance of individual sub-models
- `compare_models(self, model_a, model_b, data_loader)`: Compare the performance of two models

### ManagementApp Class

Provides a higher-level interface for using the management model.

#### Methods

- `__init__(self, config=None)`: Initialize the application with the given configuration
- `load_model(self, model_path=None)`: Load the management model
- `train(self, train_data_path, val_data_path=None, test_data_path=None)`: Train the management model
- `evaluate(self, data_path, model_path=None)`: Evaluate the management model
- `predict(self, input_data, sub_model_outputs=None)`: Make predictions with the management model
- `analyze_emotion_integration(self, base_model_path, enhanced_model_path, data_path)`: Analyze the effectiveness of emotion integration
- `run_demo(self)`: Run a demonstration of the management model

## Examples

### Basic Usage Example

```python
from A_management.app import ManagementApp
from A_management.config import get_dev_config

# Create the application with development configuration
app = ManagementApp(get_dev_config())

# Load the model
app.load_model()

# Make a prediction
input_data = {
    'text': 'The customer is very angry about the product quality.',
    'context': 'Customer service scenario'
}

# Simulated sub-model outputs
sub_model_outputs = {
    'B_language': {
        'emotion_pred': 3,  # anger
        'confidence': 0.92,
        'processing_time': 0.05
    },
    'B_multimodal': {
        'emotion_pred': 3,  # anger
        'confidence': 0.88,
        'processing_time': 0.12
    }
}

result = app.predict(input_data, sub_model_outputs)
print(result)
```

### Training Example

```python
from A_management.app import ManagementApp
from A_management.config import get_train_config

# Create the application with training configuration
app = ManagementApp(get_train_config())

# Train the model
history = app.train(
    train_data_path='/path/to/train_data.json',
    val_data_path='/path/to/val_data.json',
    test_data_path='/path/to/test_data.json'
)

# Print training history
print("Training completed. Final metrics:")
print(f"- Training loss: {history['train_loss'][-1]:.4f}")
print(f"- Validation loss: {history['val_loss'][-1]:.4f}")
print(f"- Strategy accuracy: {history['strategy_accuracy'][-1]:.4f}")
print(f"- Emotion accuracy: {history['emotion_accuracy'][-1]:.4f}")
```

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure the model path is correct
   - Check that the model file is not corrupted
   - Verify that you're using the correct version of PyTorch

2. **Training Issues**
   - If training is unstable, try reducing the learning rate
   - If the model is overfitting, increase dropout or add regularization
   - Ensure your training data is properly formatted

3. **Prediction Errors**
   - Check that input data is in the correct format
   - Verify that sub-model outputs (if provided) are compatible
   - Ensure the model was trained on similar data

### Logs

Logs are stored in the `logs` directory and can be helpful for debugging issues.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](http://www.apache.org/licenses/LICENSE-2.0) file for details.