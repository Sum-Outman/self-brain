# -*- coding: utf-8 -*-
"""
Training Logger
This module provides functionality to log training activities and metrics.
"""

import os
import logging
import json
import time
import datetime
import threading
import re
from typing import Dict, Any, List, Optional, Union
from logging.handlers import RotatingFileHandler
import traceback
import sys
import socket

# Configure root logger
logging.basicConfig(level=logging.INFO)

class TrainingLogger:
    """Class for logging training activities and metrics"""
    
    # Singleton instance
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TrainingLogger, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize the training logger"""
        # Base directory for logs
        self.log_dir = "d:\shiyan\web_interface\logs"
        
        # Ensure log directory exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"Created log directory: {self.log_dir}")
        
        # Dictionary to store loggers for each model
        self.loggers = {}
        
        # Dictionary to store metrics for each training run
        self.metrics = {}
        
        # Dictionary to store training status for each model
        self.training_status = {}
        
        # Maximum log file size (50MB)
        self.max_log_size = 50 * 1024 * 1024
        
        # Backup count for log rotation
        self.backup_count = 5
        
        # Current hostname
        self.hostname = socket.gethostname()
        
        # Initialize loggers for all models
        self._initialize_model_loggers()
    
    def _initialize_model_loggers(self):
        """Initialize loggers for all models"""
        # Model IDs
        model_ids = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        
        for model_id in model_ids:
            self._get_model_logger(model_id)
            
            # Initialize training status
            self.training_status[model_id] = {
                'status': 'idle',  # idle, preparing, running, paused, completed, failed
                'progress': 0.0,
                'current_epoch': 0,
                'total_epochs': 0,
                'start_time': None,
                'end_time': None,
                'error_message': None
            }
            
            # Initialize metrics
            self.metrics[model_id] = []
    
    def _get_model_logger(self, model_id: str) -> logging.Logger:
        """Get or create a logger for a specific model"""
        with self._lock:
            # Check if logger already exists
            if model_id in self.loggers:
                return self.loggers[model_id]
            
            # Create logger
            logger = logging.getLogger(f"training_{model_id}")
            logger.setLevel(logging.DEBUG)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - [Host: %(hostname)s] - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            formatter.converter = time.localtime  # Use local time
            
            # Create file handler with rotation
            log_file = os.path.join(self.log_dir, f"training_{model_id}.log")
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=self.max_log_size, 
                backupCount=self.backup_count, 
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(logging.INFO)
            
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.INFO)
            
            # Add handlers to logger
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
            
            # Store logger
            self.loggers[model_id] = logger
            
            # Log initialization
            extra = {'hostname': self.hostname}
            logger.info(f"Initialized logger for model {model_id}", extra=extra)
            
            return logger
    
    def log(self, model_id: str, level: str, message: str, extra: Dict[str, Any] = None):
        """Log a message for a specific model"""
        with self._lock:
            # Check if model ID is valid
            if model_id not in self.loggers:
                self._get_model_logger(model_id)
            
            # Get logger
            logger = self.loggers[model_id]
            
            # Prepare extra data
            if extra is None:
                extra = {}
            extra['hostname'] = self.hostname
            
            # Log message based on level
            level = level.lower()
            if level == 'debug':
                logger.debug(message, extra=extra)
            elif level == 'info':
                logger.info(message, extra=extra)
            elif level == 'warning' or level == 'warn':
                logger.warning(message, extra=extra)
            elif level == 'error':
                logger.error(message, extra=extra)
            elif level == 'critical' or level == 'fatal':
                logger.critical(message, extra=extra)
            else:
                logger.info(message, extra=extra)
    
    def debug(self, model_id: str, message: str, extra: Dict[str, Any] = None):
        """Log a debug message"""
        self.log(model_id, 'debug', message, extra)
    
    def info(self, model_id: str, message: str, extra: Dict[str, Any] = None):
        """Log an info message"""
        self.log(model_id, 'info', message, extra)
    
    def warning(self, model_id: str, message: str, extra: Dict[str, Any] = None):
        """Log a warning message"""
        self.log(model_id, 'warning', message, extra)
    
    def error(self, model_id: str, message: str, extra: Dict[str, Any] = None):
        """Log an error message"""
        self.log(model_id, 'error', message, extra)
    
    def critical(self, model_id: str, message: str, extra: Dict[str, Any] = None):
        """Log a critical message"""
        self.log(model_id, 'critical', message, extra)
    
    def log_exception(self, model_id: str, exception: Exception, message: str = ""):
        """Log an exception"""
        with self._lock:
            # Check if model ID is valid
            if model_id not in self.loggers:
                self._get_model_logger(model_id)
            
            # Get logger
            logger = self.loggers[model_id]
            
            # Prepare extra data
            extra = {'hostname': self.hostname}
            
            # Log exception
            full_message = message
            if message and not message.endswith('.') and not message.endswith(':'):
                full_message += ': '
            full_message += str(exception)
            
            logger.error(full_message, exc_info=True, extra=extra)
            
            # Update training status
            self.update_training_status(model_id, 'failed', error_message=str(exception))
    
    def log_metrics(self, model_id: str, metrics: Dict[str, float], epoch: int = None):
        """Log training metrics"""
        with self._lock:
            # Check if model ID is valid
            if model_id not in self.metrics:
                self.metrics[model_id] = []
            
            # Create metric record
            metric_record = {
                'timestamp': datetime.datetime.now().isoformat(),
                'metrics': metrics
            }
            
            # Add epoch if provided
            if epoch is not None:
                metric_record['epoch'] = epoch
            
            # Add to metrics list
            self.metrics[model_id].append(metric_record)
            
            # Log metrics
            metrics_str = ', '.join([f'{k}={v}' for k, v in metrics.items()])
            if epoch is not None:
                self.info(model_id, f'Epoch {epoch}: {metrics_str}')
            else:
                self.info(model_id, f'Metrics: {metrics_str}')
            
            # Save metrics to file
            self._save_metrics(model_id)
    
    def _save_metrics(self, model_id: str):
        """Save metrics to a JSON file"""
        try:
            metrics_file = os.path.join(self.log_dir, f"metrics_{model_id}.json")
            
            # Create a temporary file first to avoid partial writes
            temp_file = metrics_file + '.tmp'
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.metrics[model_id], f, ensure_ascii=False, indent=2)
            
            # Replace the original file with the temporary file
            if os.path.exists(metrics_file):
                os.replace(temp_file, metrics_file)
            else:
                os.rename(temp_file, metrics_file)
        except Exception as e:
            # Log the error but don't re-raise
            if model_id in self.loggers:
                self.loggers[model_id].error(f"Failed to save metrics: {str(e)}", extra={'hostname': self.hostname})
    
    def get_metrics(self, model_id: str, start_time: str = None, end_time: str = None) -> List[Dict[str, Any]]:
        """Get metrics for a specific model within a time range"""
        with self._lock:
            # Check if model ID is valid
            if model_id not in self.metrics:
                return []
            
            # Return all metrics if no time range is specified
            if start_time is None and end_time is None:
                return self.metrics[model_id].copy()
            
            # Filter metrics by time range
            filtered_metrics = []
            for metric in self.metrics[model_id]:
                metric_time = metric['timestamp']
                
                # Check if metric is within the time range
                if start_time is not None and metric_time < start_time:
                    continue
                if end_time is not None and metric_time > end_time:
                    continue
                
                filtered_metrics.append(metric)
            
            return filtered_metrics
    
    def update_training_status(self, model_id: str, status: str, progress: float = None, 
                              current_epoch: int = None, total_epochs: int = None, 
                              error_message: str = None):
        """Update the training status for a model"""
        with self._lock:
            # Check if model ID is valid
            if model_id not in self.training_status:
                self.training_status[model_id] = {
                    'status': 'idle',
                    'progress': 0.0,
                    'current_epoch': 0,
                    'total_epochs': 0,
                    'start_time': None,
                    'end_time': None,
                    'error_message': None
                }
            
            # Update status
            if status != self.training_status[model_id]['status']:
                # Log status change
                old_status = self.training_status[model_id]['status']
                self.info(model_id, f'Training status changed: {old_status} -> {status}')
                
                # Update start or end time based on status
                if status == 'running' and old_status != 'running':
                    self.training_status[model_id]['start_time'] = datetime.datetime.now().isoformat()
                elif status in ['completed', 'failed', 'idle'] and old_status == 'running':
                    self.training_status[model_id]['end_time'] = datetime.datetime.now().isoformat()
            
            # Update fields
            self.training_status[model_id]['status'] = status
            
            if progress is not None:
                self.training_status[model_id]['progress'] = min(max(progress, 0.0), 100.0)
            
            if current_epoch is not None:
                self.training_status[model_id]['current_epoch'] = current_epoch
            
            if total_epochs is not None:
                self.training_status[model_id]['total_epochs'] = total_epochs
            
            if error_message is not None:
                self.training_status[model_id]['error_message'] = error_message
            
            # Save status to file
            self._save_training_status(model_id)
    
    def _save_training_status(self, model_id: str):
        """Save training status to a JSON file"""
        try:
            status_file = os.path.join(self.log_dir, f"status_{model_id}.json")
            
            # Create a temporary file first to avoid partial writes
            temp_file = status_file + '.tmp'
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(self.training_status[model_id], f, ensure_ascii=False, indent=2)
            
            # Replace the original file with the temporary file
            if os.path.exists(status_file):
                os.replace(temp_file, status_file)
            else:
                os.rename(temp_file, status_file)
        except Exception as e:
            # Log the error but don't re-raise
            if model_id in self.loggers:
                self.loggers[model_id].error(f"Failed to save training status: {str(e)}", extra={'hostname': self.hostname})
    
    def get_training_status(self, model_id: str) -> Dict[str, Any]:
        """Get the current training status for a model"""
        with self._lock:
            # Check if model ID is valid
            if model_id not in self.training_status:
                return {
                    'status': 'idle',
                    'progress': 0.0,
                    'current_epoch': 0,
                    'total_epochs': 0,
                    'start_time': None,
                    'end_time': None,
                    'error_message': None
                }
            
            return self.training_status[model_id].copy()
    
    def get_all_training_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get training statuses for all models"""
        with self._lock:
            return {model_id: status.copy() for model_id, status in self.training_status.items()}
    
    def export_logs(self, model_id: str, export_path: str) -> bool:
        """Export logs for a specific model"""
        try:
            # Check if model ID is valid
            if model_id not in self.loggers:
                return False
            
            # Get log file path
            log_file = os.path.join(self.log_dir, f"training_{model_id}.log")
            
            # Check if log file exists
            if not os.path.isfile(log_file):
                return False
            
            # Ensure export directory exists
            export_dir = os.path.dirname(export_path)
            if export_dir and not os.path.exists(export_dir):
                os.makedirs(export_dir)
            
            # Copy log file to export path
            with open(log_file, 'r', encoding='utf-8') as f_in:
                with open(export_path, 'w', encoding='utf-8') as f_out:
                    f_out.write(f_in.read())
            
            # Also export metrics if available
            metrics_file = os.path.join(self.log_dir, f"metrics_{model_id}.json")
            if os.path.isfile(metrics_file):
                metrics_export_path = os.path.splitext(export_path)[0] + '_metrics.json'
                with open(metrics_file, 'r', encoding='utf-8') as f_in:
                    with open(metrics_export_path, 'w', encoding='utf-8') as f_out:
                        f_out.write(f_in.read())
            
            # Also export status if available
            status_file = os.path.join(self.log_dir, f"status_{model_id}.json")
            if os.path.isfile(status_file):
                status_export_path = os.path.splitext(export_path)[0] + '_status.json'
                with open(status_file, 'r', encoding='utf-8') as f_in:
                    with open(status_export_path, 'w', encoding='utf-8') as f_out:
                        f_out.write(f_in.read())
            
            self.info(model_id, f"Exported logs to {export_path}")
            return True
        except Exception as e:
            # Log the error but don't re-raise
            if model_id in self.loggers:
                self.loggers[model_id].error(f"Failed to export logs: {str(e)}", extra={'hostname': self.hostname})
            return False
    
    def search_logs(self, model_id: str, search_term: str, start_time: str = None, end_time: str = None) -> List[Dict[str, Any]]:
        """Search logs for a specific model"""
        results = []
        
        try:
            # Check if model ID is valid
            if model_id not in self.loggers:
                return results
            
            # Get log file path
            log_file = os.path.join(self.log_dir, f"training_{model_id}.log")
            
            # Check if log file exists
            if not os.path.isfile(log_file):
                return results
            
            # Compile search pattern
            search_pattern = re.compile(search_term, re.IGNORECASE)
            
            # Parse start and end times if provided
            start_datetime = None
            end_datetime = None
            
            if start_time:
                start_datetime = datetime.datetime.fromisoformat(start_time)
            
            if end_time:
                end_datetime = datetime.datetime.fromisoformat(end_time)
            
            # Read and search log file
            with open(log_file, 'r', encoding='utf-8') as f:
                for line_number, line in enumerate(f, 1):
                    # Extract timestamp from log line
                    timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                    
                    if timestamp_match:
                        log_timestamp_str = timestamp_match.group(1)
                        log_timestamp = datetime.datetime.strptime(log_timestamp_str, '%Y-%m-%d %H:%M:%S')
                        
                        # Check if log entry is within the time range
                        if start_datetime and log_timestamp < start_datetime:
                            continue
                        if end_datetime and log_timestamp > end_datetime:
                            continue
                    
                    # Check if search term is in the line
                    if search_pattern.search(line):
                        result_entry = {
                            'line_number': line_number,
                            'timestamp': log_timestamp_str if timestamp_match else None,
                            'message': line.strip()
                        }
                        results.append(result_entry)
        except Exception as e:
            # Log the error but don't re-raise
            if model_id in self.loggers:
                self.loggers[model_id].error(f"Failed to search logs: {str(e)}", extra={'hostname': self.hostname})
        
        return results
    
    def clear_logs(self, model_id: str) -> bool:
        """Clear logs for a specific model"""
        try:
            # Check if model ID is valid
            if model_id not in self.loggers:
                return False
            
            # Close and remove handlers
            logger = self.loggers[model_id]
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
            
            # Delete log files
            log_files = [
                os.path.join(self.log_dir, f"training_{model_id}.log"),
                os.path.join(self.log_dir, f"metrics_{model_id}.json"),
                os.path.join(self.log_dir, f"status_{model_id}.json")
            ]
            
            for log_file in log_files:
                if os.path.isfile(log_file):
                    os.remove(log_file)
            
            # Re-initialize logger
            self._get_model_logger(model_id)
            
            # Reset metrics and status
            self.metrics[model_id] = []
            self.training_status[model_id] = {
                'status': 'idle',
                'progress': 0.0,
                'current_epoch': 0,
                'total_epochs': 0,
                'start_time': None,
                'end_time': None,
                'error_message': None
            }
            
            self.info(model_id, "Cleared all logs and reset metrics and status")
            return True
        except Exception as e:
            # Log the error but don't re-raise
            # Re-initialize logger in case of error
            self._get_model_logger(model_id)
            self.loggers[model_id].error(f"Failed to clear logs: {str(e)}", extra={'hostname': self.hostname})
            return False
    
    def get_log_file_size(self, model_id: str) -> int:
        """Get the size of the log file for a specific model"""
        try:
            # Check if model ID is valid
            if model_id not in self.loggers:
                return 0
            
            # Get log file path
            log_file = os.path.join(self.log_dir, f"training_{model_id}.log")
            
            # Check if log file exists
            if not os.path.isfile(log_file):
                return 0
            
            # Return file size in bytes
            return os.path.getsize(log_file)
        except Exception as e:
            # Log the error but don't re-raise
            if model_id in self.loggers:
                self.loggers[model_id].error(f"Failed to get log file size: {str(e)}", extra={'hostname': self.hostname})
            return 0
    
    def get_log_summary(self, model_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get a summary of the most recent log entries"""
        summary = []
        
        try:
            # Check if model ID is valid
            if model_id not in self.loggers:
                return summary
            
            # Get log file path
            log_file = os.path.join(self.log_dir, f"training_{model_id}.log")
            
            # Check if log file exists
            if not os.path.isfile(log_file):
                return summary
            
            # Read the last 'limit' lines from the log file
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()[-limit:]
            
            # Parse log lines
            for line in lines:
                # Extract timestamp, level, and message
                log_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - (\w+) - (DEBUG|INFO|WARNING|ERROR|CRITICAL) - .*? - (.*)', line)
                
                if log_match:
                    timestamp, logger_name, level, message = log_match.groups()
                    summary_entry = {
                        'timestamp': timestamp,
                        'level': level,
                        'message': message.strip()
                    }
                    summary.append(summary_entry)
        except Exception as e:
            # Log the error but don't re-raise
            if model_id in self.loggers:
                self.loggers[model_id].error(f"Failed to get log summary: {str(e)}", extra={'hostname': self.hostname})
        
        return summary
    
    def get_model_logger_names(self) -> List[str]:
        """Get the names of all model loggers"""
        with self._lock:
            return list(self.loggers.keys())
    
    def set_log_level(self, model_id: str, level: str):
        """Set the log level for a specific model"""
        with self._lock:
            # Check if model ID is valid
            if model_id not in self.loggers:
                return
            
            # Get logger
            logger = self.loggers[model_id]
            
            # Convert level to logging constant
            level = level.upper()
            if level == 'DEBUG':
                log_level = logging.DEBUG
            elif level == 'INFO':
                log_level = logging.INFO
            elif level == 'WARNING' or level == 'WARN':
                log_level = logging.WARNING
            elif level == 'ERROR':
                log_level = logging.ERROR
            elif level == 'CRITICAL' or level == 'FATAL':
                log_level = logging.CRITICAL
            else:
                log_level = logging.INFO
            
            # Set logger level
            logger.setLevel(log_level)
            
            # Set handler levels
            for handler in logger.handlers:
                handler.setLevel(log_level)
            
            self.info(model_id, f"Log level set to {level}")
    
    def get_log_level(self, model_id: str) -> str:
        """Get the current log level for a specific model"""
        with self._lock:
            # Check if model ID is valid
            if model_id not in self.loggers:
                return 'INFO'
            
            # Get logger level
            log_level = self.loggers[model_id].getEffectiveLevel()
            
            # Convert to string
            if log_level == logging.DEBUG:
                return 'DEBUG'
            elif log_level == logging.INFO:
                return 'INFO'
            elif log_level == logging.WARNING:
                return 'WARNING'
            elif log_level == logging.ERROR:
                return 'ERROR'
            elif log_level == logging.CRITICAL:
                return 'CRITICAL'
            else:
                return 'INFO'

# Initialize the training logger
def get_training_logger() -> TrainingLogger:
    """Get the singleton instance of TrainingLogger"""
    return TrainingLogger()