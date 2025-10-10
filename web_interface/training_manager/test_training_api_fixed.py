# Copyright 2025 AGI System Team

import os
import sys
import json
import time
import requests
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SelfBrainTrainingAPITest")

# API Base URL
API_BASE_URL = "http://localhost:5000/api/training"

# Test timeout (seconds)
TEST_TIMEOUT = 30

class TrainingAPITester:
    # Class to test the Training API
    
    def __init__(self, base_url=API_BASE_URL, timeout=TEST_TIMEOUT):
        # Initialize the tester with API base URL and timeout
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        
    def make_request(self, method, endpoint, data=None, files=None, params=None):
        # Make a request to the API and return the response
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"} if data and files is None else {}
        
        try:
            if method == "GET":
                response = self.session.get(url, headers=headers, params=params, timeout=self.timeout)
            elif method == "POST":
                if files:
                    response = self.session.post(url, files=files, params=params, timeout=self.timeout)
                else:
                    response = self.session.post(url, json=data, headers=headers, params=params, timeout=self.timeout)
            elif method == "PUT":
                response = self.session.put(url, json=data, headers=headers, params=params, timeout=self.timeout)
            elif method == "DELETE":
                response = self.session.delete(url, headers=headers, params=params, timeout=self.timeout)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None, False, f"Unsupported HTTP method: {method}"
            
            if response.status_code in [200, 201]:
                try:
                    return response.json(), True, "Success"
                except json.JSONDecodeError:
                    return response.text, True, "Success (non-JSON response)"
            else:
                error_msg = f"HTTP error {response.status_code}: {response.text}"
                logger.error(error_msg)
                return None, False, error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(error_msg)
            return None, False, error_msg
    
    def test_health_check(self):
        # Test the health check endpoint
        logger.info("Testing health check endpoint...")
        data, success, message = self.make_request("GET", "/health")
        
        if success:
            logger.info("Health check passed successfully")
        else:
            logger.error(f"Health check failed: {message}")
        
        return success
    
    def test_get_all_models(self):
        # Test getting all models
        logger.info("Testing get all models endpoint...")
        data, success, message = self.make_request("GET", "/models")
        
        if success:
            logger.info(f"Successfully retrieved {len(data.get('models', []))} models")
        else:
            logger.error(f"Failed to get models: {message}")
        
        return success
    
    def test_get_model(self, model_id="model_A"):
        # Test getting a specific model
        logger.info(f"Testing get model endpoint for {model_id}...")
        data, success, message = self.make_request("GET", f"/models/{model_id}")
        
        if success:
            logger.info(f"Successfully retrieved model {model_id}")
        else:
            logger.error(f"Failed to get model {model_id}: {message}")
        
        return success
    
    def test_get_training_status(self, model_id="model_A"):
        # Test getting training status
        logger.info(f"Testing get training status endpoint for {model_id}...")
        data, success, message = self.make_request("GET", f"/models/{model_id}/status")
        
        if success:
            status = data.get('status', 'unknown')
            logger.info(f"Successfully retrieved training status for {model_id}: {status}")
        else:
            logger.error(f"Failed to get training status for {model_id}: {message}")
        
        return success
    
    def test_get_hyperparameters(self, model_id="model_A"):
        # Test getting hyperparameters
        logger.info(f"Testing get hyperparameters endpoint for {model_id}...")
        data, success, message = self.make_request("GET", f"/models/{model_id}/hyperparams")
        
        if success:
            logger.info(f"Successfully retrieved hyperparameters for {model_id}")
        else:
            logger.error(f"Failed to get hyperparameters for {model_id}: {message}")
        
        return success
    
    def test_set_hyperparameters(self, model_id="model_A"):
        # Test setting hyperparameters
        logger.info(f"Testing set hyperparameters endpoint for {model_id}...")
        hyperparams = {
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 10,
            "optimizer": "adam",
            "early_stopping": True
        }
        
        data, success, message = self.make_request("PUT", f"/models/{model_id}/hyperparams", data=hyperparams)
        
        if success:
            logger.info(f"Successfully updated hyperparameters for {model_id}")
        else:
            logger.error(f"Failed to update hyperparameters for {model_id}: {message}")
        
        return success
    
    def test_upload_training_data(self, model_id="model_A"):
        # Test uploading training data
        logger.info(f"Testing upload training data endpoint for {model_id}...")
        
        # Create a temporary test file
        temp_file_path = "temp_test_data.json"
        with open(temp_file_path, 'w') as f:
            json.dump([{"text": "test", "label": 1}], f)
        
        try:
            # Upload the test file
            with open(temp_file_path, 'rb') as f:
                files = {'file': (os.path.basename(temp_file_path), f, 'application/json')}
                data, success, message = self.make_request("POST", f"/models/{model_id}/data/upload", files=files)
            
            if success:
                logger.info(f"Successfully uploaded training data for {model_id}")
            else:
                logger.error(f"Failed to upload training data for {model_id}: {message}")
        except Exception as e:
            logger.error(f"Error during upload test: {str(e)}")
            success = False
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        
        return success
    
    def test_list_training_data(self, model_id="model_A"):
        # Test listing training data for a model
        logger.info(f"Testing list training data endpoint for {model_id}...")
        data, success, message = self.make_request("GET", f"/models/{model_id}/data/list")
        
        if success:
            logger.info(f"Successfully retrieved {len(data.get('data', []))} training data files for {model_id}")
        else:
            logger.error(f"Failed to list training data for {model_id}: {message}")
        
        return success
    
    def run_all_tests(self):
        # Run all tests sequentially
        logger.info("Starting all tests...")
        
        tests = [
            self.test_health_check,
            self.test_get_all_models,
            lambda: self.test_get_model("model_A"),
            lambda: self.test_get_training_status("model_A"),
            lambda: self.test_get_hyperparameters("model_A"),
            lambda: self.test_set_hyperparameters("model_A"),
            lambda: self.test_upload_training_data("model_A"),
            lambda: self.test_list_training_data("model_A")
        ]
        
        results = []
        for i, test in enumerate(tests):
            logger.info(f"Running test {i+1}/{len(tests)}...")
            success = test()
            results.append(success)
            # Add a small delay between tests
            time.sleep(1)
        
        # Calculate success rate
        total_tests = len(results)
        passed_tests = sum(results)
        success_rate = (passed_tests / total_tests) * 100
        
        logger.info(f"\nTest Summary:")
        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed tests: {passed_tests}")
        logger.info(f"Failed tests: {total_tests - passed_tests}")
        logger.info(f"Success rate: {success_rate:.2f}%")
        
        return all(results)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Self Brain Training API Test Script")
    parser.add_argument("--url", type=str, default=API_BASE_URL, help="Base URL of the Training API")
    parser.add_argument("--timeout", type=int, default=TEST_TIMEOUT, help="Request timeout in seconds")
    parser.add_argument("--test", type=str, default="all", 
                      choices=["all", "health", "models", "model", "status", "hyperparams", "upload", "list"],
                      help="Specific test to run")
    parser.add_argument("--model-id", type=str, default="model_A", help="Model ID to test with")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = TrainingAPITester(base_url=args.url, timeout=args.timeout)
    
    # Run specified tests
    if args.test == "all":
        success = tester.run_all_tests()
    elif args.test == "health":
        success = tester.test_health_check()
    elif args.test == "models":
        success = tester.test_get_all_models()
    elif args.test == "model":
        success = tester.test_get_model(args.model_id)
    elif args.test == "status":
        success = tester.test_get_training_status(args.model_id)
    elif args.test == "hyperparams":
        success = tester.test_get_hyperparameters(args.model_id)
        if success:
            success = tester.test_set_hyperparameters(args.model_id)
    elif args.test == "upload":
        success = tester.test_upload_training_data(args.model_id)
    elif args.test == "list":
        success = tester.test_list_training_data(args.model_id)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)