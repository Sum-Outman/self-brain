# Self Brain AGI System - Training API
# Copyright 2025 AGI System Team

import os
import sys
import json
import logging
from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SelfBrainTrainingAPI")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

# Function to load configuration from file
def load_config(config_file):
    """Load configuration from a JSON file"""
    config_path = os.path.join(PROJECT_ROOT, config_file)
    if not os.path.exists(config_path):
        logger.warning(f"Configuration file {config_path} not found, using default settings")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load configuration from {config_path}: {str(e)}")
        return {}

# Load configuration
api_config = load_config("config/api_config.json")
training_config = load_config("config/training_config.json")

# Import training manager
from web_interface.training_manager.model_trainer import training_manager

# Initialize Flask app
app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = api_config.get("server", {}).get("upload_folder", "uploads")
ALLOWED_EXTENSIONS = api_config.get("server", {}).get("allowed_extensions", 
    {'txt', 'pdf', 'doc', 'docx', 'csv', 'json', 'h5', 'pb', 'zip', 'tar', 'gz'})

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set upload folder configuration
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Maximum allowed payload size (100MB)
app.config['MAX_CONTENT_LENGTH'] = api_config.get("server", {}).get("max_content_length", 100 * 1024 * 1024)

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Helper function to generate standard response
def standard_response(success, message, data=None):
    """Generate a standard API response format"""
    response = {
        "success": success,
        "message": message
    }
    if data is not None:
        response["data"] = data
    return jsonify(response)

# API endpoints
@app.route('/api/training/models', methods=['GET'])
def get_all_models():
    """
    Get information about all available models.
    """
    try:
        # Get training status for all models
        status = training_manager.get_training_status()
        
        # Format response data
        models_info = []
        for model_id, model_status in status.items():
            model_info = {
                "model_id": model_id,
                "is_training": model_status["is_training"],
                "current_epoch": model_status["current_epoch"],
                "epochs_completed": model_status["epochs_completed"],
                "total_epochs": model_status["total_epochs"]
            }
            models_info.append(model_info)
        
        return standard_response(True, "Models retrieved successfully", models_info)
        
    except Exception as e:
        logger.error(f"Failed to get models: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """
    Get detailed information about a specific model.
    """
    try:
        # Get training status for the specific model
        status = training_manager.get_training_status(model_id)
        
        # Check for errors
        if "error" in status:
            return standard_response(False, status["error"]), 404
        
        return standard_response(True, "Model retrieved successfully", status)
        
    except Exception as e:
        logger.error(f"Failed to get model {model_id}: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/models/<model_id>/start', methods=['POST'])
def start_training(model_id):
    """
    Start training for a specific model.
    """
    try:
        # Start training
        success, message = training_manager.start_training(model_id)
        
        if success:
            return standard_response(True, message)
        else:
            return standard_response(False, message), 400
        
    except Exception as e:
        logger.error(f"Failed to start training for model {model_id}: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/models/<model_id>/stop', methods=['POST'])
def stop_training(model_id):
    """
    Stop training for a specific model.
    """
    try:
        # Stop training
        success, message = training_manager.stop_training(model_id)
        
        if success:
            return standard_response(True, message)
        else:
            return standard_response(False, message), 400
        
    except Exception as e:
        logger.error(f"Failed to stop training for model {model_id}: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/models/<model_id>/status', methods=['GET'])
def get_training_status(model_id):
    """
    Get the current training status of a specific model.
    """
    try:
        # Get training status
        status = training_manager.get_training_status(model_id)
        
        # Check for errors
        if "error" in status:
            return standard_response(False, status["error"]), 404
        
        return standard_response(True, "Training status retrieved successfully", status)
        
    except Exception as e:
        logger.error(f"Failed to get training status for model {model_id}: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/models/<model_id>/hyperparameters', methods=['GET'])
def get_hyperparameters(model_id):
    """
    Get the current hyperparameters of a specific model.
    """
    try:
        # Check if model exists
        if model_id not in training_manager.trainers:
            return standard_response(False, f"Model {model_id} not found"), 404
        
        # Get hyperparameters
        trainer = training_manager.trainers[model_id]
        hyperparameters = trainer.config.get("hyperparameters", {})
        
        return standard_response(True, "Hyperparameters retrieved successfully", hyperparameters)
        
    except Exception as e:
        logger.error(f"Failed to get hyperparameters for model {model_id}: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/models/<model_id>/hyperparameters', methods=['POST'])
def set_hyperparameters(model_id):
    """
    Update hyperparameters for a specific model.
    """
    try:
        # Check if model exists
        if model_id not in training_manager.trainers:
            return standard_response(False, f"Model {model_id} not found"), 404
        
        # Get hyperparameters from request
        hyperparameters = request.json
        
        if not hyperparameters or not isinstance(hyperparameters, dict):
            return standard_response(False, "Invalid hyperparameters format"), 400
        
        # Set hyperparameters
        success, message = training_manager.set_hyperparameters(model_id, hyperparameters)
        
        if success:
            return standard_response(True, message)
        else:
            return standard_response(False, message), 400
        
    except Exception as e:
        logger.error(f"Failed to set hyperparameters for model {model_id}: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/models/<model_id>/save', methods=['POST'])
def save_model(model_id):
    """
    Save a trained model to disk.
    """
    try:
        # Check if model exists
        if model_id not in training_manager.trainers:
            return standard_response(False, f"Model {model_id} not found"), 404
        
        # Save model
        trainer = training_manager.trainers[model_id]
        success, message = trainer.save_model()
        
        if success:
            return standard_response(True, message)
        else:
            return standard_response(False, message), 400
        
    except Exception as e:
        logger.error(f"Failed to save model {model_id}: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/models/<model_id>/load', methods=['POST'])
def load_model(model_id):
    """
    Load a trained model from disk.
    """
    try:
        # Check if model exists
        if model_id not in training_manager.trainers:
            return standard_response(False, f"Model {model_id} not found"), 404
        
        # Load model
        trainer = training_manager.trainers[model_id]
        success, message = trainer.load_model()
        
        if success:
            return standard_response(True, message)
        else:
            return standard_response(False, message), 400
        
    except Exception as e:
        logger.error(f"Failed to load model {model_id}: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/models/<model_id>/evaluate', methods=['POST'])
def evaluate_model(model_id):
    """
    Evaluate a trained model on test data.
    """
    try:
        # Check if model exists
        if model_id not in training_manager.trainers:
            return standard_response(False, f"Model {model_id} not found"), 404
        
        # Evaluate model
        trainer = training_manager.trainers[model_id]
        metrics = trainer.evaluate_model()
        
        # Check for errors
        if "error" in metrics:
            return standard_response(False, metrics["error"]), 400
        
        return standard_response(True, "Model evaluation completed successfully", metrics)
        
    except Exception as e:
        logger.error(f"Failed to evaluate model {model_id}: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/models/<model_id>/data/upload', methods=['POST'])
def upload_training_data(model_id):
    """
    Upload training data for a specific model.
    """
    try:
        # Check if model exists
        if model_id not in training_manager.trainers:
            return standard_response(False, f"Model {model_id} not found"), 404
        
        # Check if the post request has the file part
        if 'file' not in request.files:
            return standard_response(False, "No file part in the request"), 400
        
        file = request.files['file']
        
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return standard_response(False, "No file selected for uploading"), 400
        
        # Check if file extension is allowed
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            
            # Create a timestamped subfolder for the model's data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_data_folder = os.path.join(
                app.config['UPLOAD_FOLDER'], 
                model_id, 
                timestamp
            )
            os.makedirs(model_data_folder, exist_ok=True)
            
            # Save the file
            file_path = os.path.join(model_data_folder, filename)
            file.save(file_path)
            
            # Log the upload
            logger.info(f"Training data uploaded for model {model_id}: {filename}")
            
            return standard_response(
                True, 
                f"File {filename} uploaded successfully",
                {"file_path": file_path}
            )
        else:
            return standard_response(False, "Allowed file types are txt, pdf, doc, docx, csv, json, h5, pb, zip, tar, gz"), 400
            
    except Exception as e:
        logger.error(f"Failed to upload training data for model {model_id}: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/models/<model_id>/data/list', methods=['GET'])
def list_training_data(model_id):
    """
    List all uploaded training data for a specific model.
    """
    try:
        # Check if model exists
        if model_id not in training_manager.trainers:
            return standard_response(False, f"Model {model_id} not found"), 404
        
        # Get the model's data directory
        model_data_dir = os.path.join(app.config['UPLOAD_FOLDER'], model_id)
        
        # Check if the directory exists
        if not os.path.exists(model_data_dir):
            return standard_response(True, "No training data found", [])
        
        # List all data directories and files
        data_files = []
        for timestamp_dir in os.listdir(model_data_dir):
            timestamp_path = os.path.join(model_data_dir, timestamp_dir)
            if os.path.isdir(timestamp_path):
                for filename in os.listdir(timestamp_path):
                    file_path = os.path.join(timestamp_path, filename)
                    if os.path.isfile(file_path):
                        data_files.append({
                            "filename": filename,
                            "timestamp": timestamp_dir,
                            "path": file_path,
                            "size": os.path.getsize(file_path),
                            "upload_time": datetime.fromtimestamp(
                                os.path.getctime(file_path)
                            ).isoformat()
                        })
        
        # Sort by upload time (newest first)
        data_files.sort(key=lambda x: x["upload_time"], reverse=True)
        
        return standard_response(True, "Training data list retrieved successfully", data_files)
        
    except Exception as e:
        logger.error(f"Failed to list training data for model {model_id}: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/models/<model_id>/data/download/<timestamp>/<filename>', methods=['GET'])
def download_training_data(model_id, timestamp, filename):
    """
    Download a specific training data file for a model.
    """
    try:
        # Check if model exists
        if model_id not in training_manager.trainers:
            return standard_response(False, f"Model {model_id} not found"), 404
        
        # Construct the file path
        file_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 
            model_id, 
            timestamp, 
            filename
        )
        
        # Check if the file exists
        if not os.path.exists(file_path):
            return standard_response(False, "File not found"), 404
        
        # Send the file
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Failed to download training data for model {model_id}: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/models/<model_id>/data/delete/<timestamp>/<filename>', methods=['DELETE'])
def delete_training_data(model_id, timestamp, filename):
    """
    Delete a specific training data file for a model.
    """
    try:
        # Check if model exists
        if model_id not in training_manager.trainers:
            return standard_response(False, f"Model {model_id} not found"), 404
        
        # Construct the file path
        file_path = os.path.join(
            app.config['UPLOAD_FOLDER'], 
            model_id, 
            timestamp, 
            filename
        )
        
        # Check if the file exists
        if not os.path.exists(file_path):
            return standard_response(False, "File not found"), 404
        
        # Delete the file
        os.remove(file_path)
        
        # Log the deletion
        logger.info(f"Training data deleted for model {model_id}: {filename}")
        
        return standard_response(True, f"File {filename} deleted successfully")
        
    except Exception as e:
        logger.error(f"Failed to delete training data for model {model_id}: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/models/save-all', methods=['POST'])
def save_all_models():
    """
    Save all trained models to disk.
    """
    try:
        # Save all models
        results = training_manager.save_all_models()
        
        # Check if all saves were successful
        all_success = all(result["success"] for result in results.values())
        
        if all_success:
            return standard_response(True, "All models saved successfully", results)
        else:
            return standard_response(False, "Some models failed to save", results), 400
        
    except Exception as e:
        logger.error(f"Failed to save all models: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/models/load-all', methods=['POST'])
def load_all_models():
    """
    Load all trained models from disk.
    """
    try:
        # Load all models
        results = training_manager.load_all_models()
        
        # Check if all loads were successful
        all_success = all(result["success"] for result in results.values())
        
        if all_success:
            return standard_response(True, "All models loaded successfully", results)
        else:
            return standard_response(False, "Some models failed to load", results), 400
        
    except Exception as e:
        logger.error(f"Failed to load all models: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for the training API.
    """
    return standard_response(True, "Training API is running")

@app.route('/api/training/load-example-data', methods=['POST'])
def load_example_data():
    """Load example training data for all models."""
    try:
        # Path to example data file
        example_data_path = os.path.join(
            os.path.dirname(__file__),
            "example_training_data.json"
        )
        
        # Check if example data file exists
        if not os.path.exists(example_data_path):
            return standard_response(False, "Example training data file not found"), 404
        
        # Load example data
        with open(example_data_path, 'r', encoding='utf-8') as f:
            example_data = json.load(f)
        
        # Get model-specific data
        loaded_models = []
        for model_id, model_data in example_data.items():
            # Skip metadata
            if model_id == "metadata" or model_id == "multimodal_data":
                continue
            
            # Check if model exists
            if model_id not in training_manager.trainers:
                logger.warning(f"Model {model_id} not found, skipping example data load")
                continue
            
            # Create a temporary directory for the example data
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_data_folder = os.path.join(
                app.config['UPLOAD_FOLDER'], 
                model_id, 
                f"example_data_{timestamp}"
            )
            os.makedirs(model_data_folder, exist_ok=True)
            
            # Save the example data to a file
            example_data_file = os.path.join(model_data_folder, "example_data.json")
            with open(example_data_file, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False)
            
            # Load the data into the model trainer
            trainer = training_manager.trainers[model_id]
            success, message = trainer.load_training_data(example_data_file)
            
            if success:
                loaded_models.append({
                    "model_id": model_id,
                    "message": message,
                    "file_path": example_data_file
                })
                logger.info(f"Example data loaded for model {model_id}")
            else:
                logger.error(f"Failed to load example data for model {model_id}: {message}")
        
        if not loaded_models:
            return standard_response(False, "Failed to load example data for any model"), 400
        
        return standard_response(
            True, 
            f"Example data loaded for {len(loaded_models)} models",
            loaded_models
        )
        
    except Exception as e:
        logger.error(f"Failed to load example training data: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/hyperparameter-presets', methods=['GET'])
def get_hyperparameter_presets():
    """Get available hyperparameter presets for all models."""
    try:
        # Extract presets from training configuration
        presets = training_config.get("model_specific_params", {})
        
        if not presets:
            # If no configuration available, return default presets
            presets = {
                "default": {
                    "epochs": 100,
                    "batch_size": 32,
                    "learning_rate": 0.001
                }
            }
        
        return standard_response(True, "Hyperparameter presets retrieved successfully", presets)
        
    except Exception as e:
        logger.error(f"Failed to get hyperparameter presets: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/config', methods=['GET'])
def get_training_config():
    """Get the complete training configuration."""
    try:
        return standard_response(True, "Training configuration retrieved successfully", training_config)
        
    except Exception as e:
        logger.error(f"Failed to get training configuration: {str(e)}")
        return standard_response(False, str(e)), 500

@app.route('/api/training/statistics', methods=['GET'])
def get_training_statistics():
    """Get overall training statistics."""
    try:
        # Calculate statistics
        stats = {
            "total_models": len(training_manager.trainers),
            "training_models": sum(1 for status in training_manager.get_training_status().values() if status["is_training"]),
            "completed_models": sum(1 for status in training_manager.get_training_status().values() if status["epochs_completed"] > 0),
            "total_training_time": "N/A",  # Would need to implement time tracking
            "avg_epoch_duration": "N/A"    # Would need to implement time tracking
        }
        
        return standard_response(True, "Training statistics retrieved successfully", stats)
        
    except Exception as e:
        logger.error(f"Failed to get training statistics: {str(e)}")
        return standard_response(False, str(e)), 500

if __name__ == '__main__':
    # Get server configuration
    host = api_config.get("server", {}).get("host", '0.0.0.0')
    port = api_config.get("server", {}).get("port", 5001)
    debug = api_config.get("server", {}).get("debug", False)
    
    # Run the Flask app
    app.run(host=host, port=port, debug=debug)