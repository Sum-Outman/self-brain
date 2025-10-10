#!/usr/bin/env python
# Self Brain AGI System - Training API Start Script
# Copyright 2025 AGI System Team

import os
import sys
import subprocess
import time
import argparse
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SelfBrainTrainingAPIStart")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Default settings
DEFAULT_HOST = '0.0.0.0'
DEFAULT_PORT = 5000
DEFAULT_DEBUG = False

class TrainingAPIServer:
    """Class to manage the Training API server"""
    
    def __init__(self, host=DEFAULT_HOST, port=DEFAULT_PORT, debug=DEFAULT_DEBUG):
        """Initialize the server settings"""
        self.host = host
        self.port = port
        self.debug = debug
        self.process = None
        
    def start(self):
        """Start the Training API server"""
        logger.info(f"Starting Training API server on {self.host}:{self.port} (debug={self.debug})...")
        
        # Construct the command to start the server
        command = [
            sys.executable,
            '-m', 'web_interface.training_manager.training_api',
            '--host', self.host,
            '--port', str(self.port)
        ]
        
        if self.debug:
            command.append('--debug')
            
        try:
            # Start the server in a new process
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
            
            logger.info(f"Training API server started with process ID: {self.process.pid}")
            logger.info(f"Server URL: http://{self.host}:{self.port}/api/training")
            logger.info("To stop the server, press Ctrl+C or terminate the process")
            
            # Monitor the server process
            self._monitor_server()
            
        except Exception as e:
            logger.error(f"Failed to start Training API server: {str(e)}")
            if self.process:
                self.stop()
            return False
        
        return True
    
    def _monitor_server(self):
        """Monitor the server process and log output"""
        if not self.process:
            return
            
        try:
            # Monitor the server until it's stopped
            while self.process.poll() is None:
                # Check for output from stdout
                stdout_line = self.process.stdout.readline()
                if stdout_line:
                    logger.info(f"SERVER OUTPUT: {stdout_line.strip()}")
                    
                # Check for output from stderr
                stderr_line = self.process.stderr.readline()
                if stderr_line:
                    logger.error(f"SERVER ERROR: {stderr_line.strip()}")
                    
                # Small delay to prevent CPU usage
                time.sleep(0.1)
                
            # Log the exit code when the server stops
            exit_code = self.process.returncode
            logger.info(f"Training API server stopped with exit code: {exit_code}")
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt detected, stopping server...")
            self.stop()
        except Exception as e:
            logger.error(f"Error monitoring server: {str(e)}")
            
    def stop(self):
        """Stop the Training API server"""
        if self.process and self.process.poll() is None:
            logger.info("Stopping Training API server...")
            
            try:
                # Terminate the process
                if os.name == 'nt':  # Windows
                    self.process.terminate()
                    # Give the process some time to terminate gracefully
                    time.sleep(2)
                    # If it's still running, kill it
                    if self.process.poll() is None:
                        self.process.kill()
                else:  # Unix-like systems
                    # First try to terminate gracefully
                    self.process.terminate()
                    # Wait for up to 5 seconds for the process to terminate
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # If it's still running after 5 seconds, kill it
                        self.process.kill()
                        
                logger.info("Training API server stopped successfully")
                
            except Exception as e:
                logger.error(f"Failed to stop server: {str(e)}")
                
        else:
            logger.info("Server is not running")

def main():
    """Main function to parse arguments and start the server"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Self Brain Training API Server')
    parser.add_argument('--host', type=str, default=DEFAULT_HOST, help='Host to bind the server to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='Port to run the server on (default: 5001)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--test', action='store_true', help='Run tests before starting the server')
    
    args = parser.parse_args()
    
    # Run tests if requested
    if args.test:
        logger.info("Running tests before starting the server...")
        
        # Construct the test command
        test_command = [
            sys.executable,
            '-m', 'web_interface.training_manager.test_training_api'
        ]
        
        try:
            # Run the tests
            test_result = subprocess.run(
                test_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            logger.info("Tests passed successfully")
            print(test_result.stdout)
            
        except subprocess.CalledProcessError as e:
            logger.error("Tests failed")
            print(e.stdout)
            print(e.stderr)
            logger.error("Server will not start due to test failures")
            sys.exit(1)
        
        # Add a small delay after tests
        time.sleep(1)
    # Start the server
    server = TrainingAPIServer(host=args.host, port=args.port, debug=args.debug)
    server.start()

if __name__ == '__main__':
    main()
