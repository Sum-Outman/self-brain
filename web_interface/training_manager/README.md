# Self Brain Training Manager

The Training Manager is a comprehensive module for the Self Brain AGI system that handles model training, evaluation, and management. It provides a unified interface for training all the AI models (A-K) defined in the Self Brain architecture.

## Features

- **Model Training**: Train individual or multiple AI models with customizable hyperparameters
- **Data Management**: Upload, list, download, and delete training data for each model
- **Model Evaluation**: Evaluate trained models on test data
- **Model Storage**: Save and load trained models from disk
- **API Interface**: RESTful API for remote control and integration
- **Monitoring**: Real-time monitoring of training progress and status
- **Hyperparameter Tuning**: Adjust model hyperparameters dynamically
- **Multi-model Support**: Manage all Self Brain models (A-K) from a single interface

## Models Supported

The Training Manager supports training and management of all Self Brain models:

1. **Model A**: Management model (main interactive AI)
2. **Model B**: Large language model
3. **Model C**: Audio processing model
4. **Model D**: Image processing model
5. **Model E**: Video processing model
6. **Model F**: Binocular spatial perception model
7. **Model G**: Sensor perception model
8. **Model H**: Computer control model
9. **Model I**: Motion and actuator control model
10. **Model J**: Knowledge base expert model
11. **Model K**: Programming model

## Installation

To set up the Training Manager, follow these steps:

1. Clone the Self Brain repository
2. Navigate to the project directory
3. Install the required dependencies

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Starting the Training API Server

The Training Manager provides a RESTful API server that can be started with the following command:

```bash
python -m web_interface.training_manager.training_api
```

The server will start on `http://localhost:5000` by default.

### Using the ModelTrainer Directly

You can also use the `ModelTrainer` classes directly in your Python code:

```python
from web_interface.training_manager.model_trainer import training_manager

# Get training status for all models
status = training_manager.get_training_status()
print(f"Model status: {status}")

# Start training a specific model
success, message = training_manager.start_training("model_A")
print(f"Training started: {success}, {message}")

# Stop training a specific model
success, message = training_manager.stop_training("model_A")
print(f"Training stopped: {success}, {message}")

# Save a trained model
success, message = training_manager.trainers["model_A"].save_model()
print(f"Model saved: {success}, {message}")
```

<<<<<<< HEAD
=======
### Running Tests

You can test the Training API using the provided test script:

```bash
# Run all tests
python -m web_interface.training_manager.test_training_api

# Run a specific test
python -m web_interface.training_manager.test_training_api --test health

# Test with a different model
python -m web_interface.training_manager.test_training_api --model-id model_B
```
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13

## API Endpoints

The Training Manager provides the following API endpoints:

### Health Check
- **GET** `/api/training/health` - Check if the API server is running

### Model Management
- **GET** `/api/training/models` - Get information about all available models
- **GET** `/api/training/models/{model_id}` - Get detailed information about a specific model
- **POST** `/api/training/models/save-all` - Save all trained models to disk
- **POST** `/api/training/models/load-all` - Load all trained models from disk

### Training Control
- **POST** `/api/training/models/{model_id}/start` - Start training a specific model
- **POST** `/api/training/models/{model_id}/stop` - Stop training a specific model
- **GET** `/api/training/models/{model_id}/status` - Get the current training status of a specific model

### Hyperparameter Management
- **GET** `/api/training/models/{model_id}/hyperparameters` - Get the current hyperparameters of a specific model
- **POST** `/api/training/models/{model_id}/hyperparameters` - Update hyperparameters for a specific model

### Model Storage
- **POST** `/api/training/models/{model_id}/save` - Save a trained model to disk
- **POST** `/api/training/models/{model_id}/load` - Load a trained model from disk

### Model Evaluation
- **POST** `/api/training/models/{model_id}/evaluate` - Evaluate a trained model on test data

### Training Data Management
- **POST** `/api/training/models/{model_id}/data/upload` - Upload training data for a specific model
- **GET** `/api/training/models/{model_id}/data/list` - List all uploaded training data for a specific model
- **GET** `/api/training/models/{model_id}/data/download/{timestamp}/{filename}` - Download a specific training data file
- **DELETE** `/api/training/models/{model_id}/data/delete/{timestamp}/{filename}` - Delete a specific training data file

## Configuration

The Training Manager can be configured through the following settings:

### API Server Configuration
- **Host**: The hostname or IP address to bind the API server to (default: `0.0.0.0`)
- **Port**: The port number to run the API server on (default: `5001`)
- **Debug Mode**: Whether to enable debug mode for the API server (default: `False`)

### Training Configuration
- **Learning Rate**: The learning rate for model training (default: `0.001`)
- **Batch Size**: The batch size for model training (default: `32`)
- **Epochs**: The number of epochs to train the model for (default: `100`)
- **Validation Split**: The fraction of data to use for validation (default: `0.2`)
- **Checkpoint Interval**: How often to save model checkpoints during training (default: `5` epochs)

## Project Structure

The Training Manager is organized as follows:

```
web_interface/training_manager/
├── __init__.py          # Package initialization and exports
├── model_trainer.py     # ModelTrainer classes and training logic
├── training_api.py      # RESTful API server implementation
├── test_training_api.py # API test script
├── data_loader.py       # Data loading and preprocessing utilities
├── advanced_train_control.py # Advanced training controls
└── README.md            # Documentation (this file)
```

## Contributing

Contributions to the Training Manager are welcome! If you find any issues or have suggestions for improvements, please submit an issue or pull request on the Self Brain repository.

## License

Self Brain AGI System is released under the MIT License.

## Contact

For questions or support, please contact the Self Brain development team at [silencecrowtom@qq.com](mailto:silencecrowtom@qq.com).

---
<<<<<<< HEAD
**Self Brain AGI System** - Copyright 2025 AGI System Team
=======
**Self Brain AGI System** - Copyright 2025 AGI System Team
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
