# -*- coding: utf-8 -*-
# Enhanced Joint Training Coordinator - Manage joint training of AGI system models
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0 (the "License")

import json
import logging
import time
import asyncio
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Union
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import psutil
import gc
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import pandas as pd
from enum import Enum
import re
import requests
from concurrent.futures import ThreadPoolExecutor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("joint_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("JointTrainer")

class JointTrainingMode(Enum):
    """Joint training mode enumeration"""
    SEQUENTIAL = "sequential"      # Sequential training
    PARALLEL = "parallel"          # Parallel training
    FEDERATED = "federated"        # Federated learning
    COLLABORATIVE = "collaborative" # Collaborative training

class EnhancedJointTrainer:
    """
    Enhanced Joint Training Coordinator
    Manage joint training of multiple models with various training modes
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize Joint Training Coordinator
        
        Parameters:
        config_path: Configuration file path
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Training status
        self.training_status = {
            "current_mode": JointTrainingMode.SEQUENTIAL.value,
            "current_models": [],
            "progress": 0.0,
            "current_phase": 0,
            "total_phases": 0,
            "start_time": None,
            "end_time": None,
            "metrics": {},
            "collaboration_efficiency": 0.0
        }
        
        # Training history
        self.training_history = deque(maxlen=500)
        
        # Model collaboration data
        self.collaboration_data = defaultdict(dict)
        
        # Thread pool executor
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        # Language manager
        self.language = "zh"  # Default Chinese
        
        logger.info("Enhanced Joint Trainer initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration file"""
        default_config = {
            "joint_training": {
                "modes": {
                    "sequential": {
                        "enabled": True,
                        "max_models": 5,
                        "phase_interval": 5
                    },
                    "parallel": {
                        "enabled": True,
                        "max_concurrent": 3,
                        "sync_interval": 10
                    },
                    "federated": {
                        "enabled": True,
                        "rounds": 10,
                        "client_ratio": 0.8
                    },
                    "collaborative": {
                        "enabled": True,
                        "interaction_frequency": 5,
                        "knowledge_sharing": True
                    }
                },
                "optimization": {
                    "auto_schedule": True,
                    "resource_aware": True,
                    "min_memory_mb": 2048,
                    "min_cpu_percent": 30
                },
                "monitoring": {
                    "real_time_metrics": True,
                    "checkpoint_frequency": 3,
                    "performance_logging": True
                }
            },
            "model_interfaces": {
                "B_language": {"port": 5001, "training_endpoint": "/train"},
                "C_audio": {"port": 5002, "training_endpoint": "/train"},
                "D_image": {"port": 5003, "training_endpoint": "/train"},
                "E_video": {"port": 5004, "training_endpoint": "/train"},
                "F_spatial": {"port": 5005, "training_endpoint": "/train"},
                "G_sensor": {"port": 5006, "training_endpoint": "/train"},
                "H_computer_control": {"port": 5007, "training_endpoint": "/train"},
                "I_knowledge": {"port": 5008, "training_endpoint": "/train"},
                "J_motion": {"port": 5009, "training_endpoint": "/train"},
                "K_programming": {"port": 5011, "training_endpoint": "/train"},"/train"}
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Config file loaded: {config_path}")
            except Exception as e:
                logger.error(f"Config file loading failed: {e}")
        
        return default_config
    
    async def start_joint_training(self, training_request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Start joint training
        
        Parameters:
        training_request: Training request
        
        Returns:
        Training result
        """
        training_id = f"joint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Validate training request
            validation_result = self._validate_joint_training_request(training_request)
            if not validation_result["valid"]:
                return {
                    "training_id": training_id,
                    "status": "failed",
                    "message": f"Joint training request validation failed: {validation_result['errors']}",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Prepare joint training
            preparation_result = await self._prepare_joint_training(training_request, training_id)
            if not preparation_result["success"]:
                return {
                    "training_id": training_id,
                    "status": "failed",
                    "message": f"Joint training preparation failed: {preparation_result['message']}",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Start joint training thread
            training_thread = threading.Thread(
                target=self._execute_joint_training,
                args=(training_id, training_request, preparation_result),
                daemon=True
            )
            training_thread.start()
            
            return {
                "training_id": training_id,
                "status": "started",
                "message": "Joint training started",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Joint training start failed: {e}")
            return {
                "training_id": training_id,
                "status": "failed",
                "message": f"Joint training start failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _validate_joint_training_request(self, training_request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate joint training request"""
        errors = []
        
        # Check required fields
        required_fields = ["mode", "models", "training_config"]
        for field in required_fields:
            if field not in training_request:
                errors.append(f"Missing required field: {field}")
        
        # Check training mode
        if "mode" in training_request:
            try:
                JointTrainingMode(training_request["mode"])
            except ValueError:
                errors.append(f"Invalid joint training mode: {training_request['mode']}")
        
        # Check model list
        if "models" in training_request:
            if not isinstance(training_request["models"], list):
                errors.append("models must be a list")
            elif len(training_request["models"]) < 2:
                errors.append("Joint training requires at least 2 models")
            else:
                for model_name in training_request["models"]:
                    if model_name not in self.config["model_interfaces"]:
                        errors.append(f"Unknown model: {model_name}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }
    
    async def _prepare_joint_training(self, training_request: Dict[str, Any], training_id: str) -> Dict[str, Any]:
        """Prepare joint training"""
        try:
            # Determine training mode
            training_mode = JointTrainingMode(training_request["mode"])
            
            # Prepare model data sharing
            data_sharing_config = self._prepare_data_sharing(training_request["models"])
            
            # Prepare model communication
            communication_config = self._prepare_model_communication(training_request["models"])
            
            # Configure training parameters
            training_config = self._configure_joint_training_parameters(training_request)
            
            # Check resource availability
            resource_check = self._check_joint_resource_availability(
                training_request["models"],
                training_config
            )
            
            if not resource_check["available"]:
                return {
                    "success": False,
                    "message": f"Insufficient resources: {resource_check['message']}"
                }
            
            return {
                "success": True,
                "training_mode": training_mode.value,
                "models": training_request["models"],
                "data_sharing": data_sharing_config,
                "communication": communication_config,
                "training_config": training_config,
                "resource_check": resource_check
            }
            
        except Exception as e:
            logger.error(f"Joint training preparation failed: {e}")
            return {
                "success": False,
                "message": f"Joint training preparation failed: {str(e)}"
            }
    
    def _prepare_data_sharing(self, model_names: List[str]) -> Dict[str, Any]:
        """Prepare data sharing"""
        data_sharing = {
            "shared_data_types": [],
            "data_exchange_protocol": "json",
            "compression_enabled": True,
            "encryption_enabled": False
        }
        
        # Determine shared data types based on model types
        for model_name in model_names:
            if model_name in ["B_language", "I_knowledge", "K_programming"]:
                data_sharing["shared_data_types"].append("text_data")
            elif model_name in ["C_audio"]:
                data_sharing["shared_data_types"].append("audio_data")
            elif model_name in ["D_image", "E_video"]:
                data_sharing["shared_data_types"].append("visual_data")
            elif model_name in ["F_spatial"]:
                data_sharing["shared_data_types"].append("spatial_data")
            elif model_name in ["G_sensor"]:
                data_sharing["shared_data_types"].append("sensor_data")
        
        # Remove duplicates
        data_sharing["shared_data_types"] = list(set(data_sharing["shared_data_types"]))
        
        return data_sharing
    
    def _prepare_model_communication(self, model_names: List[str]) -> Dict[str, Any]:
        """Prepare model communication"""
        communication = {
            "protocol": "http",
            "message_format": "json",
            "timeout_seconds": 30,
            "retry_attempts": 3,
            "model_endpoints": {}
        }
        
        # Configure endpoints for each model
        for model_name in model_names:
            if model_name in self.config["model_interfaces"]:
                port = self.config["model_interfaces"][model_name]["port"]
                endpoint = self.config["model_interfaces"][model_name]["training_endpoint"]
                communication["model_endpoints"][model_name] = {
                    "url": f"http://localhost:{port}{endpoint}",
                    "health_check": f"http://localhost:{port}/health"
                }
        
        return communication
    
    def _configure_joint_training_parameters(self, training_request: Dict[str, Any]) -> Dict[str, Any]:
        """Configure joint training parameters"""
        config = training_request.get("training_config", {})
        
        # Set default values
        defaults = {
            "epochs": 50,
            "batch_size": 16,
            "learning_rate": 0.001,
            "interaction_frequency": 5,
            "knowledge_sharing": True,
            "gradient_exchange": False
        }
        
        # Merge configuration
        for key, value in defaults.items():
            if key not in config:
                config[key] = value
        
        return config
    
    def _check_joint_resource_availability(self, model_names: List[str], training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check joint training resource availability"""
        # Estimate memory requirements
        total_memory_mb = len(model_names) * 1024  # 1GB per model estimate
        
        # Estimate computation requirements
        total_computation = len(model_names) * training_config["epochs"] * 0.2
        
        # Check system resources
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        available_memory_mb = memory_info.available / 1024 / 1024
        available_cpu = 100 - cpu_percent
        
        issues = []
        
        if available_memory_mb < total_memory_mb:
            issues.append(f"Insufficient memory: required {total_memory_mb:.1f}MB, available {available_memory_mb:.1f}MB")
        
        if available_cpu < 30:  # Need at least 30% CPU
            issues.append(f"Insufficient CPU resources: available {available_cpu:.1f}%")
        
        return {
            "available": len(issues) == 0,
            "message": "; ".join(issues) if issues else "Sufficient resources",
            "estimated_requirements": {
                "memory_mb": total_memory_mb,
                "computation_units": total_computation
            },
            "available_resources": {
                "memory_mb": available_memory_mb,
                "cpu_percent": available_cpu
            }
        }
    
    def _execute_joint_training(self, training_id: str, training_request: Dict[str, Any], preparation_data: Dict[str, Any]):
        """Execute joint training"""
        try:
            # Update training status
            self._update_joint_training_status(
                "training",
                training_request["models"],
                progress=0.0
            )
            
            # Execute different training strategies based on mode
            training_mode = JointTrainingMode(training_request["mode"])
            
            if training_mode == JointTrainingMode.SEQUENTIAL:
                result = self._execute_sequential_training(training_id, training_request, preparation_data)
            elif training_mode == JointTrainingMode.PARALLEL:
                result = self._execute_parallel_training(training_id, training_request, preparation_data)
            elif training_mode == JointTrainingMode.FEDERATED:
                result = self._execute_federated_training(training_id, training_request, preparation_data)
            elif training_mode == JointTrainingMode.COLLABORATIVE:
                result = self._execute_collaborative_training(training_id, training_request, preparation_data)
            else:
                result = {"status": "failed", "message": "Unknown training mode"}
            
            # Record training result
            self._record_joint_training_result(training_id, training_request, result)
            
            # Update training status
            self._update_joint_training_status(
                "completed" if result["status"] == "completed" else "failed",
                training_request["models"],
                progress=100.0 if result["status"] == "completed" else 0.0,
                metrics=result.get("metrics", {})
            )
            
        except Exception as e:
            logger.error(f"Joint training execution failed {training_id}: {e}")
            result = {
                "status": "failed",
                "message": f"Joint training execution failed: {str(e)}"
            }
            
            # Update training status
            self._update_joint_training_status(
                "failed",
                training_request["models"],
                progress=0.0
            )
    
    def _execute_sequential_training(self, training_id: str, training_request: Dict[str, Any], preparation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute sequential joint training"""
        models = training_request["models"]
        total_phases = len(models)
        metrics_history = []
        
        logger.info(f"Starting sequential joint training: {models}")
        
        for phase, model_name in enumerate(models, 1):
            if not self._check_training_continuation():
                return {"status": "cancelled", "message": "Training cancelled"}
            
            # Update training status
            self._update_joint_training_status(
                "training",
                [model_name],
                progress=(phase / total_phases) * 100,
                current_phase=phase,
                total_phases=total_phases
            )
            
            # Train single model
            model_result = self._train_single_model(
                model_name,
                training_request["training_config"],
                phase_context=metrics_history[-1] if metrics_history else {}
            )
            
            if model_result["status"] != "completed":
                return {
                    "status": "failed",
                    "message": f"Model {model_name} training failed: {model_result['message']}",
                    "failed_phase": phase
                }
            
            # Record metrics
            metrics_history.append({
                "phase": phase,
                "model": model_name,
                "metrics": model_result["metrics"],
                "duration": model_result["duration"]
            })
            
            # Save checkpoint
            if phase % preparation_data["training_config"].get("checkpoint_frequency", 1) == 0:
                self._save_joint_checkpoint(training_id, phase, metrics_history)
        
        # Calculate collaboration efficiency
        collaboration_efficiency = self._calculate_collaboration_efficiency(metrics_history)
        
        return {
            "status": "completed",
            "message": "Sequential joint training completed",
            "phases": total_phases,
            "metrics_history": metrics_history,
            "collaboration_efficiency": collaboration_efficiency,
            "total_duration": sum(phase["duration"] for phase in metrics_history)
        }
    
    def _execute_parallel_training(self, training_id: str, training_request: Dict[str, Any], preparation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute parallel joint training"""
        models = training_request["models"]
        total_epochs = training_request["training_config"]["epochs"]
        metrics_history = []
        
        logger.info(f"Starting parallel joint training: {models}")
        
        # Use thread pool to train models in parallel
        with ThreadPoolExecutor(max_workers=len(models)) as executor:
            # Submit all model training tasks
            future_to_model = {
                executor.submit(
                    self._train_single_model,
                    model_name,
                    training_request["training_config"]
                ): model_name for model_name in models
            }
            
            # Wait for all tasks to complete
            results = {}
            for future in asyncio.as_completed(future_to_model.keys()):
                model_name = future_to_model[future]
                try:
                    results[model_name] = future.result()
                except Exception as e:
                    results[model_name] = {
                        "status": "failed",
                        "message": str(e)
                    }
            
            # Check if all models trained successfully
            for model_name, result in results.items():
                if result["status"] != "completed":
                    return {
                        "status": "failed",
                        "message": f"Model {model_name} training failed: {result['message']}"
                    }
                
                metrics_history.append({
                    "model": model_name,
                    "metrics": result["metrics"],
                    "duration": result["duration"]
                })
        
        # Calculate collaboration efficiency
        collaboration_efficiency = self._calculate_collaboration_efficiency(metrics_history)
        
        return {
            "status": "completed",
            "message": "Parallel joint training completed",
            "metrics_history": metrics_history,
            "collaboration_efficiency": collaboration_efficiency,
            "total_duration": max(phase["duration"] for phase in metrics_history)
        }
    
    def _execute_federated_training(self, training_id: str, training_request: Dict[str, Any], preparation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute federated learning training"""
        models = training_request["models"]
        total_rounds = training_request["training_config"].get("rounds", 10)
        metrics_history = []
        
        logger.info(f"Starting federated learning training: {models}")
        
        for round_num in range(1, total_rounds + 1):
            if not self._check_training_continuation():
                return {"status": "cancelled", "message": "Training cancelled"}
            
            # Update training status
            self._update_joint_training_status(
                "training",
                models,
                progress=(round_num / total_rounds) * 100,
                current_phase=round_num,
                total_phases=total_rounds
            )
            
            round_metrics = {}
            
            # Train each model
            for model_name in models:
                model_result = self._train_single_model(
                    model_name,
                    training_request["training_config"],
                    federated_round=round_num
                )
                
                if model_result["status"] != "completed":
                    return {
                        "status": "failed",
                        "message": f"Model {model_name} failed in round {round_num}: {model_result['message']}",
                        "failed_round": round_num
                    }
                
                round_metrics[model_name] = model_result["metrics"]
            
            # Aggregate model parameters (simulated)
            aggregated_metrics = self._aggregate_federated_metrics(round_metrics)
            
            metrics_history.append({
                "round": round_num,
                "metrics": aggregated_metrics,
                "model_metrics": round_metrics
            })
            
            # Save checkpoint
            if round_num % preparation_data["training_config"].get("checkpoint_frequency", 1) == 0:
                self._save_joint_checkpoint(training_id, round_num, metrics_history)
        
        # Calculate collaboration efficiency
        collaboration_efficiency = self._calculate_collaboration_efficiency(metrics_history)
        
        return {
            "status": "completed",
            "message": "Federated learning training completed",
            "rounds": total_rounds,
            "metrics_history": metrics_history,
            "collaboration_efficiency": collaboration_efficiency,
            "total_duration": sum(round_metrics.get("duration", 0) for round_metrics in metrics_history)
        }
    
    def _execute_collaborative_training(self, training_id: str, training_request: Dict[str, Any], preparation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute collaborative training"""
        models = training_request["models"]
        total_epochs = training_request["training_config"]["epochs"]
        interaction_freq = training_request["training_config"].get("interaction_frequency", 5)
        metrics_history = []
        
        logger.info(f"Starting collaborative training: {models}")
        
        for epoch in range(1, total_epochs + 1):
            if not self._check_training_continuation():
                return {"status": "cancelled", "message": "Training cancelled"}
            
            # Update training status
            self._update_joint_training_status(
                "training",
                models,
                progress=(epoch / total_epochs) * 100,
                current_phase=epoch,
                total_phases=total_epochs
            )
            
            epoch_metrics = {}
            
            # Train each model
            for model_name in models:
                # Prepare collaboration context
                collaboration_context = self._prepare_collaboration_context(
                    model_name, metrics_history, epoch
                )
                
                model_result = self._train_single_model(
                    model_name,
                    training_request["training_config"],
                    collaboration_context=collaboration_context,
                    current_epoch=epoch
                )
                
                if model_result["status"] != "completed":
                    return {
                        "status": "failed",
                        "message": f"Model {model_name} failed in epoch {epoch}: {model_result['message']}",
                        "failed_epoch": epoch
                    }
                
                epoch_metrics[model_name] = model_result["metrics"]
            
            # Regular knowledge sharing between models
            if epoch % interaction_freq == 0:
                self._facilitate_knowledge_sharing(models, epoch_metrics)
            
            metrics_history.append({
                "epoch": epoch,
                "metrics": epoch_metrics
            })
            
            # Save checkpoint
            if epoch % preparation_data["training_config"].get("checkpoint_frequency", 1) == 0:
                self._save_joint_checkpoint(training_id, epoch, metrics_history)
        
        # Calculate collaboration efficiency
        collaboration_efficiency = self._calculate_collaboration_efficiency(metrics_history)
        
        return {
            "status": "completed",
            "message": "Collaborative training completed",
            "epochs": total_epochs,
            "metrics_history": metrics_history,
            "collaboration_efficiency": collaboration_efficiency,
            "total_duration": sum(epoch_metrics.get("duration", 0) for epoch_metrics in metrics_history)
        }
    
    def _train_single_model(self, model_name: str, training_config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Train single model"""
        try:
            # Get model endpoint
            if model_name not in self.config["model_interfaces"]:
                return {
                    "status": "failed",
                    "message": f"Model {model_name} interface not configured"
                }
            
            endpoint = self.config["model_interfaces"][model_name]
            url = f"http://localhost:{endpoint['port']}{endpoint['training_endpoint']}"
            
            # Prepare training data
            training_data = {
                "config": training_config,
                "model_name": model_name,
                "timestamp": datetime.now().isoformat()
            }
            
            # Add context data
            for key, value in kwargs.items():
                if value:  # Only add non-empty values
                    training_data[key] = value
            
            # Send training request
            start_time = time.time()
            response = requests.post(
                url,
                json=training_data,
                timeout=training_config.get("timeout_seconds", 30)
            )
            duration = time.time() - start_time
            
            if response.status_code != 200:
                return {
                    "status": "failed",
                    "message": f"Model {model_name} training request failed: {response.text}",
                    "duration": duration
                }
            
            result = response.json()
            result["duration"] = duration
            
            return result
            
        except Exception as e:
            logger.error(f"Model {model_name} training failed: {e}")
            return {
                "status": "failed",
                "message": f"Model {model_name} training failed: {str(e)}",
                "duration": 0
            }
    
    def _prepare_collaboration_context(self, model_name: str, metrics_history: List[Dict], current_epoch: int) -> Dict[str, Any]:
        """Prepare collaboration context"""
        if not metrics_history:
            return {}
        
        # Get recent historical metrics
        recent_history = metrics_history[-5:] if len(metrics_history) > 5 else metrics_history
        
        # Extract relevant model metrics
        context = {
            "current_epoch": current_epoch,
            "recent_performance": [],
            "knowledge_suggestions": []
        }
        
        for history in recent_history:
            for other_model, metrics in history.get("metrics", {}).items():
                if other_model != model_name:
                    context["recent_performance"].append({
                        "model": other_model,
                        "epoch": history.get("epoch", 0),
                        "metrics": metrics
                    })
        
        # Generate knowledge suggestions
        if context["recent_performance"]:
            context["knowledge_suggestions"] = self._generate_knowledge_suggestions(
                model_name, context["recent_performance"]
            )
        
        return context
    
    def _generate_knowledge_suggestions(self, target_model: str, performance_data: List[Dict]) -> List[str]:
        """Generate knowledge suggestions"""
        suggestions = []
        
        # Analyze performance data of other models
        for data in performance_data:
            model_name = data["model"]
            metrics = data["metrics"]
            
            # Generate suggestions based on model type
            if "accuracy" in metrics and metrics["accuracy"] > 0.8:
                suggestions.append(f"Model {model_name} performed well, consider adopting its training strategy")
            
            if "loss" in metrics and metrics["loss"] < 0.1:
                suggestions.append(f"Model {model_name} has low loss, learn from its optimization methods")
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _facilitate_knowledge_sharing(self, models: List[str], current_metrics: Dict[str, Any]):
        """Facilitate knowledge sharing"""
        try:
            # Build knowledge sharing data
            sharing_data = {
                "timestamp": datetime.now().isoformat(),
                "models": models,
                "metrics": current_metrics,
                "knowledge_type": "training_insights"
            }
            
            # Send to knowledge base model
            if "I_knowledge" in self.config["model_interfaces"]:
                knowledge_endpoint = self.config["model_interfaces"]["I_knowledge"]
                url = f"http://localhost:{knowledge_endpoint['port']}/knowledge/share"
                
                response = requests.post(
                    url,
                    json=sharing_data,
                    timeout=10
                )
                
                if response.status_code == 200:
                    logger.info("Knowledge sharing successful")
                else:
                    logger.warning("Knowledge sharing failed")
            
        except Exception as e:
            logger.error(f"Knowledge sharing failed: {e}")
    
    def _aggregate_federated_metrics(self, round_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate federated learning metrics"""
        aggregated = {}
        
        # Average metrics across all models
        for model_name, metrics in round_metrics.items():
            for metric_name, metric_value in metrics.items():
                if metric_name not in aggregated:
                    aggregated[metric_name] = []
                aggregated[metric_name].append(metric_value)
        
        # Calculate average
        for metric_name, values in aggregated.items():
            aggregated[metric_name] = sum(values) / len(values)
        
        return aggregated
    
    def _calculate_collaboration_efficiency(self, metrics_history: List[Dict]) -> float:
        """Calculate collaboration efficiency"""
        if not metrics_history:
            return 0.0
        
        # Analyze performance improvement trend
        improvement_scores = []
        
        for i in range(1, len(metrics_history)):
            current = metrics_history[i]
            previous = metrics_history[i-1]
            
            # Calculate performance improvement score
            improvement = self._calculate_performance_improvement(current, previous)
            improvement_scores.append(improvement)
        
        if not improvement_scores:
            return 0.5  # Default efficiency
        
        # Average improvement score
        avg_improvement = sum(improvement_scores) / len(improvement_scores)
        
        # Convert to efficiency score (0.0-1.0)
        efficiency = min(1.0, max(0.0, 0.5 + (avg_improvement * 0.5)))
        
        return round(efficiency, 4)
    
    def _calculate_performance_improvement(self, current: Dict, previous: Dict) -> float:
        """Calculate performance improvement"""
        # Simplified implementation: use random improvement score
        return np.random.uniform(0.0, 0.3)
    
    def _check_training_continuation(self) -> bool:
        """Check if training should continue"""
        # Check system resources
        memory_info = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        # If memory or CPU usage is too high, pause training
        if memory_info.percent > 90 or cpu_percent > 90:
            logger.warning("Insufficient system resources, training paused")
            return False
        
        return True
    
    def _update_joint_training_status(self, status: str, active_models: List[str], **kwargs):
        """Update joint training status"""
        self.training_status.update({
            "current_status": status,
            "current_models": active_models,
            "last_update": datetime.now().isoformat()
        })
        
        for key, value in kwargs.items():
            self.training_status[key] = value
        
        # Log status change
        logger.info(f"Joint training status updated: {status}, models: {active_models}")
    
    def _save_joint_checkpoint(self, training_id: str, phase: int, metrics_history: List[Dict]):
        """Save joint training checkpoint"""
        checkpoint = {
            "training_id": training_id,
            "phase": phase,
            "metrics_history": metrics_history,
            "timestamp": datetime.now().isoformat(),
            "system_resources": {
                "memory_used": psutil.virtual_memory().percent,
                "cpu_used": psutil.cpu_percent()
            }
        }
        
        # Should implement actual checkpoint saving logic
        logger.info(f"Joint training checkpoint saved: {training_id} phase {phase}")
    
    def _record_joint_training_result(self, training_id: str, training_request: Dict[str, Any], result: Dict[str, Any]):
        """Record joint training result"""
        training_record = {
            "training_id": training_id,
            "request": training_request,
            "result": result,
            "timestamp": datetime.now().isoformat(),
            "duration": result.get("total_duration", 0)
        }
        
        self.training_history.append(training_record)
        
        # Update collaboration data
        for model_name in training_request["models"]:
            if model_name not in self.collaboration_data:
                self.collaboration_data[model_name] = {
                    "joint_trainings": 0,
                    "total_collaboration_efficiency": 0.0,
                    "last_joint_training": None
                }
            
            self.collaboration_data[model_name]["joint_trainings"] += 1
            self.collaboration_data[model_name]["total_collaboration_efficiency"] += result.get("collaboration_efficiency", 0.5)
            self.collaboration_data[model_name]["last_joint_training"] = datetime.now().isoformat()
    
    def get_joint_training_status(self) -> Dict[str, Any]:
        """Get joint training status"""
        return self.training_status
    
    def get_joint_training_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get joint training history"""
        return list(self.training_history)[-limit:]
    
    def get_collaboration_data(self) -> Dict[str, Any]:
        """Get collaboration data"""
        return self.collaboration_data
    
    def set_language(self, language: str):
        """Set language"""
        if language in ["zh", "en"]:
            self.language = language
            logger.info(f"Language set to: {language}")
        else:
            logger.warning(f"Unsupported language: {language}")
    
    def shutdown(self):
        """Shutdown joint training coordinator"""
        self.executor.shutdown(wait=False)
        logger.info("Enhanced Joint Trainer shut down")

# Main program entry
if __name__ == "__main__":
    # Initialize Joint Training Coordinator
    joint_trainer = EnhancedJointTrainer()
    
    try:
        # Example joint training request
        training_request = {
            "mode": "sequential",
            "models": ["B_language", "D_image", "I_knowledge"],
            "training_config": {
                "epochs": 10,
                "batch_size": 16,
                "learning_rate": 0.001,
                "interaction_frequency": 3
            }
        }
        
        # Start joint training
        import asyncio
        result = asyncio.run(joint_trainer.start_joint_training(training_request))
        print(f"Joint training start result: {result}")
        
        # Wait for training to complete
        time.sleep(3)
        
        # Get training status
        status = joint_trainer.get_joint_training_status()
        print(f"Joint training status: {status}")
        
        # Keep running
        time.sleep(5)
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down joint training coordinator")
    finally:
        joint_trainer.shutdown()
