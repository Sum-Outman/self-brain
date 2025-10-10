# Self Brain AGI System - Training API Test Script
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
    """Class to test the Training API"""
    
    def __init__(self, base_url=API_BASE_URL, timeout=TEST_TIMEOUT):
        """Initialize the tester with API base URL and timeout"""
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        
    def make_request(self, method, endpoint, data=None, files=None, params=None):
        """Make a request to the API and return the response"""
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"} if data and files is None else {}
        
        try:
            if method.upper() == "GET":
                response = self.session.get(url, params=params, timeout=self.timeout)
            elif method.upper() == "POST":
                if files:
                    response = self.session.post(url, files=files, timeout=self.timeout)
                else:
                    response = self.session.post(url, json=data, headers=headers, timeout=self.timeout)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, timeout=self.timeout)
            else:
                logger.error(f"Unsupported HTTP method: {method}")
                return None, False, f"Unsupported HTTP method: {method}"
            
            # Check if response is JSON
            try:
                response_data = response.json()
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON response from {url}: {response.text}")
                return None, False, f"Invalid JSON response"
            
            # Check response status
            if response.status_code >= 200 and response.status_code < 300:
                return response_data, True, "Request successful"
            else:
                error_msg = response_data.get("message", f"Request failed with status code {response.status_code}")
                logger.error(f"Request failed: {error_msg}")
                return response_data, False, error_msg
                
        except requests.exceptions.Timeout:
            logger.error(f"Request timed out after {self.timeout} seconds")
            return None, False, f"Request timed out"
        except requests.exceptions.ConnectionError:
            logger.error("Connection error: Could not connect to the API")
            return None, False, "Connection error"
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return None, False, f"Unexpected error: {str(e)}"
    
    def test_health_check(self):
        """Test the health check endpoint"""
        logger.info("Testing health check endpoint...")
        data, success, message = self.make_request("GET", "/health")
        
        if success:
            logger.info(f"Health check passed: {message}")
        else:
            logger.error(f"Health check failed: {message}")
        
        return success
    
    def test_get_all_models(self):
        """Test getting all models"""
        logger.info("Testing get all models endpoint...")
        data, success, message = self.make_request("GET", "/models")
        
        if success:
            logger.info(f"Successfully retrieved {len(data.get('data', []))} models")
        else:
            logger.error(f"Failed to retrieve models: {message}")
        
        return success
    
    def test_get_model(self, model_id="model_A"):
        """Test getting a specific model"""
        logger.info(f"Testing get model endpoint for {model_id}...")
        data, success, message = self.make_request("GET", f"/models/{model_id}")
        
        if success:
            logger.info(f"Successfully retrieved model {model_id}")
        else:
            logger.error(f"Failed to retrieve model {model_id}: {message}")
        
        return success
    
    def test_get_training_status(self, model_id="model_A"):
        """Test getting training status of a model"""
        logger.info(f"Testing get training status endpoint for {model_id}...")
        data, success, message = self.make_request("GET", f"/models/{model_id}/status")
        
        if success:
            logger.info(f"Successfully retrieved training status for {model_id}")
        else:
            logger.error(f"Failed to retrieve training status for {model_id}: {message}")
        
        return success
    
    def test_get_hyperparameters(self, model_id="model_A"):
        """Test getting hyperparameters of a model"""
        logger.info(f"Testing get hyperparameters endpoint for {model_id}...")
        data, success, message = self.make_request("GET", f"/models/{model_id}/hyperparameters")
        
        if success:
            logger.info(f"Successfully retrieved hyperparameters for {model_id}")
        else:
            logger.error(f"Failed to retrieve hyperparameters for {model_id}: {message}")
        
        return success
    
    def test_set_hyperparameters(self, model_id="model_A", hyperparams=None):
        """Test setting hyperparameters of a model"""
        if hyperparams is None:
            hyperparams = {"learning_rate": 0.001, "batch_size": 32, "epochs": 10}
            
        logger.info(f"Testing set hyperparameters endpoint for {model_id}...")
        data, success, message = self.make_request("POST", f"/models/{model_id}/hyperparameters", data=hyperparams)
        
        if success:
            logger.info(f"Successfully set hyperparameters for {model_id}")
        else:
            logger.error(f"Failed to set hyperparameters for {model_id}: {message}")
        
        return success
    
    def test_upload_training_data(self, model_id="model_A"):
        """Test uploading training data for a model"""
        logger.info(f"Testing upload training data endpoint for {model_id}...")
        
        # Create a sample training data file
        sample_data = "This is a sample training data file for testing purposes.\n"
        sample_data += "It contains dummy content to verify the upload functionality.\n"
        sample_data += f"Timestamp: {datetime.now().isoformat()}\n"
        
        # Save sample data to a temporary file
        temp_file_path = f"sample_training_data_{model_id}_{int(time.time())}.txt"
        with open(temp_file_path, "w") as f:
            f.write(sample_data)
        
        try:
            # Upload the file
            with open(temp_file_path, "rb") as f:
                files = {"file": (os.path.basename(temp_file_path), f, "text/plain")}
                data, success, message = self.make_request("POST", f"/models/{model_id}/data/upload", files=files)
            
            if success:
                logger.info(f"Successfully uploaded training data for {model_id}")
            else:
                logger.error(f"Failed to upload training data for {model_id}: {message}")
                
            return success
            
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    def test_list_training_data(self, model_id="model_A"):
        """Test listing training data for a model"""
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
