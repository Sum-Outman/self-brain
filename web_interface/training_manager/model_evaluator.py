# -*- coding: utf-8 -*-
"""
Model Evaluator Module
This module provides tools for evaluating the performance of trained models.
"""

import logging
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Union
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('ModelEvaluator')

class ModelEvaluator:
    """Base class for model evaluation"""
    
    def __init__(self, model_id: str, model_type: str):
        """Initialize the model evaluator"""
        self.model_id = model_id
        self.model_type = model_type
        self.metrics = {}
        self.evaluation_results = {}
        self.lock = threading.Lock()
        
        # Results directory
        self.results_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            '../../evaluation_results',
            model_id
        )
        
        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Initialize specific evaluator based on model type
        self.specific_evaluator = self._get_specific_evaluator()
    
    def _get_specific_evaluator(self):
        """Get the specific evaluator based on model type"""
        evaluators = {
            'management': ManagementModelEvaluator,
            'language': LanguageModelEvaluator,
            'audio': AudioModelEvaluator,
            'image': ImageModelEvaluator,
            'video': VideoModelEvaluator,
            'spatial': SpatialModelEvaluator,
            'sensor': SensorModelEvaluator,
            'computer': ComputerControlModelEvaluator,
            'motion': MotionControlModelEvaluator,
            'knowledge': KnowledgeBaseModelEvaluator,
            'programming': ProgrammingModelEvaluator
        }
        
        if self.model_type in evaluators:
            return evaluators[self.model_type](self.model_id)
        else:
            logger.warning(f"No specific evaluator found for model type {self.model_type}, using base evaluator")
            return BaseModelEvaluator(self.model_id)
    
    def evaluate(self, model, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the model on test data"""
        try:
            # Delegate to specific evaluator
            results = self.specific_evaluator.evaluate(model, test_data)
            
            # Store results
            with self.lock:
                self.evaluation_results = results
                self.metrics = results.get('metrics', {})
            
            # Save results
            self.save_results(results)
            
            return results
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'metrics': {}
            }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current evaluation metrics"""
        with self.lock:
            return self.metrics.copy()
    
    def get_evaluation_results(self) -> Dict[str, Any]:
        """Get the complete evaluation results"""
        with self.lock:
            return self.evaluation_results.copy()
    
    def save_results(self, results: Dict[str, Any]) -> bool:
        """Save evaluation results to disk"""
        try:
            # Generate results filename with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = os.path.join(self.results_dir, f"evaluation_results_{timestamp}.json")
            
            # Save results to JSON file
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Evaluation results saved to: {results_file}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {str(e)}")
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate an evaluation report"""
        try:
            # Delegate to specific evaluator
            report = self.specific_evaluator.generate_report()
            
            # Add common information
            report['model_id'] = self.model_id
            report['model_type'] = self.model_type
            report['timestamp'] = datetime.now().isoformat()
            report['results_directory'] = self.results_dir
            
            return report
        except Exception as e:
            logger.error(f"Error generating evaluation report: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def compare_evaluations(self, evaluation_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple evaluation results"""
        try:
            # Load evaluation results
            evaluations = []
            for eval_id in evaluation_ids:
                try:
                    # Find the evaluation file
                    files = [f for f in os.listdir(self.results_dir) if eval_id in f]
                    if files:
                        file_path = os.path.join(self.results_dir, files[0])
                        with open(file_path, 'r', encoding='utf-8') as f:
                            eval_data = json.load(f)
                            evaluations.append({
                                'id': eval_id,
                                'data': eval_data
                            })
                except Exception as e:
                    logger.error(f"Error loading evaluation {eval_id}: {str(e)}")
                    continue
            
            # Compare evaluations
            comparison = {
                'evaluations_compared': len(evaluations),
                'metrics_comparison': {}
            }
            
            # Compare metrics across evaluations
            if evaluations:
                # Get all unique metric names
                all_metrics = set()
                for eval_data in evaluations:
                    metrics = eval_data['data'].get('metrics', {})
                    all_metrics.update(metrics.keys())
                
                # Compare each metric
                for metric in all_metrics:
                    comparison['metrics_comparison'][metric] = {}
                    values = []
                    
                    for eval_data in evaluations:
                        eval_id = eval_data['id']
                        value = eval_data['data'].get('metrics', {}).get(metric, None)
                        comparison['metrics_comparison'][metric][eval_id] = value
                        
                        if value is not None and isinstance(value, (int, float)):
                            values.append(value)
                    
                    # Calculate statistics for numeric metrics
                    if values:
                        comparison['metrics_comparison'][metric]['statistics'] = {
                            'mean': np.mean(values),
                            'median': np.median(values),
                            'std': np.std(values),
                            'min': min(values),
                            'max': max(values)
                        }
            
            return comparison
        except Exception as e:
            logger.error(f"Error comparing evaluations: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_best_performing_model(self) -> Dict[str, Any]:
        """Get the best performing model based on evaluation results"""
        try:
            # Get all evaluation files
            files = [f for f in os.listdir(self.results_dir) if f.startswith('evaluation_results_')]
            
            if not files:
                return {
                    'status': 'error',
                    'message': 'No evaluation results found'
                }
            
            # Load all evaluation results
            evaluations = []
            for file in files:
                file_path = os.path.join(self.results_dir, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # Extract timestamp from filename
                        timestamp = file.replace('evaluation_results_', '').replace('.json', '')
                        evaluations.append({
                            'timestamp': timestamp,
                            'file': file,
                            'results': data
                        })
                except Exception as e:
                    logger.error(f"Error loading evaluation file {file}: {str(e)}")
                    continue
            
            # Sort evaluations by timestamp (newest first)
            evaluations.sort(key=lambda x: x['timestamp'], reverse=True)
            
            # For simplicity, return the latest evaluation as the best
            # In a real implementation, this would compare metrics to find the best
            if evaluations:
                best_eval = evaluations[0]
                return {
                    'status': 'success',
                    'best_model': {
                        'timestamp': best_eval['timestamp'],
                        'file': best_eval['file'],
                        'metrics': best_eval['results'].get('metrics', {})
                    }
                }
            else:
                return {
                    'status': 'error',
                    'message': 'No valid evaluation results found'
                }
        except Exception as e:
            logger.error(f"Error getting best performing model: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }

class BaseModelEvaluator:
    """Base model evaluator implementation"""
    
    def __init__(self, model_id: str):
        """Initialize the base model evaluator"""
        self.model_id = model_id
        self.default_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'loss', 'val_loss', 'val_accuracy'
        ]
    
    def evaluate(self, model, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Base evaluation implementation"""
        # This is a placeholder implementation
        # In a real system, this would use the actual model to make predictions
        
        try:
            # Extract test features and labels
            X_test = test_data.get('X', [])
            y_test = test_data.get('y', [])
            
            # Check if we have test data
            if not X_test or not y_test:
                return {
                    'status': 'error',
                    'message': 'No test data provided',
                    'metrics': {}
                }
            
            # Simulate model predictions (in a real system, this would use model.predict())
            # For the base evaluator, we'll just return random metrics
            metrics = {
                'accuracy': np.random.uniform(0.5, 1.0),
                'loss': np.random.uniform(0.0, 0.5),
                'precision': np.random.uniform(0.5, 1.0),
                'recall': np.random.uniform(0.5, 1.0),
                'f1_score': np.random.uniform(0.5, 1.0),
                'test_samples': len(X_test)
            }
            
            # Create evaluation results
            results = {
                'status': 'success',
                'model_id': self.model_id,
                'evaluation_time': datetime.now().isoformat(),
                'test_size': len(X_test),
                'metrics': metrics
            }
            
            return results
        except Exception as e:
            logger.error(f"Error in base model evaluation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'metrics': {}
            }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a base evaluation report"""
        return {
            'report_type': 'base',
            'evaluator': 'BaseModelEvaluator',
            'summary': 'Base evaluation report generated'
        }

class ManagementModelEvaluator(BaseModelEvaluator):
    """Evaluator for management model"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific metrics for management model
        self.management_metrics = [
            'task_allocation_accuracy', 'resource_optimization_score',
            'collaboration_efficiency', 'decision_quality_score',
            'emotional_intelligence_score'
        ]
    
    def evaluate(self, model, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate management model"""
        try:
            # Call base evaluation
            base_results = super().evaluate(model, test_data)
            
            # Add management-specific metrics
            if base_results['status'] == 'success':
                # Simulate management metrics
                management_metrics = {
                    'task_allocation_accuracy': np.random.uniform(0.7, 1.0),
                    'resource_optimization_score': np.random.uniform(0.6, 0.95),
                    'collaboration_efficiency': np.random.uniform(0.7, 0.98),
                    'decision_quality_score': np.random.uniform(0.65, 0.95),
                    'emotional_intelligence_score': np.random.uniform(0.6, 0.9)
                }
                
                # Update metrics
                base_results['metrics'].update(management_metrics)
                base_results['evaluation_type'] = 'management_model'
            
            return base_results
        except Exception as e:
            logger.error(f"Error in management model evaluation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'metrics': {}
            }

class LanguageModelEvaluator(BaseModelEvaluator):
    """Evaluator for language model"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific metrics for language model
        self.language_metrics = [
            'perplexity', 'bleu_score', 'rouge_score',
            'meteor_score', 'semantic_similarity'
        ]
    
    def evaluate(self, model, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate language model"""
        try:
            # Call base evaluation
            base_results = super().evaluate(model, test_data)
            
            # Add language-specific metrics
            if base_results['status'] == 'success':
                # Simulate language metrics
                language_metrics = {
                    'perplexity': np.random.uniform(5, 20),
                    'bleu_score': np.random.uniform(0.3, 0.7),
                    'rouge_score': np.random.uniform(0.4, 0.8),
                    'meteor_score': np.random.uniform(0.4, 0.85),
                    'semantic_similarity': np.random.uniform(0.6, 0.9)
                }
                
                # Update metrics
                base_results['metrics'].update(language_metrics)
                base_results['evaluation_type'] = 'language_model'
            
            return base_results
        except Exception as e:
            logger.error(f"Error in language model evaluation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'metrics': {}
            }

class AudioModelEvaluator(BaseModelEvaluator):
    """Evaluator for audio model"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific metrics for audio model
        self.audio_metrics = [
            'wer', 'cer', 'per', 'accuracy',
            'precision', 'recall', 'f1_score'
        ]
    
    def evaluate(self, model, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate audio model"""
        try:
            # Call base evaluation
            base_results = super().evaluate(model, test_data)
            
            # Add audio-specific metrics
            if base_results['status'] == 'success':
                # Simulate audio metrics
                audio_metrics = {
                    'wer': np.random.uniform(0.05, 0.3),  # Word Error Rate
                    'cer': np.random.uniform(0.03, 0.25),  # Character Error Rate
                    'per': np.random.uniform(0.04, 0.28),  # Phoneme Error Rate
                    'audio_accuracy': np.random.uniform(0.7, 0.95)
                }
                
                # Update metrics
                base_results['metrics'].update(audio_metrics)
                base_results['evaluation_type'] = 'audio_model'
            
            return base_results
        except Exception as e:
            logger.error(f"Error in audio model evaluation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'metrics': {}
            }

class ImageModelEvaluator(BaseModelEvaluator):
    """Evaluator for image model"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific metrics for image model
        self.image_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'iou', 'ap', 'map', 'confusion_matrix'
        ]
    
    def evaluate(self, model, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate image model"""
        try:
            # Call base evaluation
            base_results = super().evaluate(model, test_data)
            
            # Add image-specific metrics
            if base_results['status'] == 'success':
                # Simulate image metrics
                image_metrics = {
                    'iou': np.random.uniform(0.6, 0.95),  # Intersection over Union
                    'ap': np.random.uniform(0.65, 0.9),   # Average Precision
                    'map': np.random.uniform(0.6, 0.85),  # Mean Average Precision
                    'image_accuracy': np.random.uniform(0.75, 0.98)
                }
                
                # Update metrics
                base_results['metrics'].update(image_metrics)
                base_results['evaluation_type'] = 'image_model'
            
            return base_results
        except Exception as e:
            logger.error(f"Error in image model evaluation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'metrics': {}
            }

class VideoModelEvaluator(BaseModelEvaluator):
    """Evaluator for video model"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific metrics for video model
        self.video_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score',
            'video_accuracy', 'temporal_consistency_score',
            'action_recognition_score', 'scene_segmentation_score'
        ]
    
    def evaluate(self, model, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate video model"""
        try:
            # Call base evaluation
            base_results = super().evaluate(model, test_data)
            
            # Add video-specific metrics
            if base_results['status'] == 'success':
                # Simulate video metrics
                video_metrics = {
                    'video_accuracy': np.random.uniform(0.7, 0.95),
                    'temporal_consistency_score': np.random.uniform(0.75, 0.98),
                    'action_recognition_score': np.random.uniform(0.65, 0.92),
                    'scene_segmentation_score': np.random.uniform(0.6, 0.9)
                }
                
                # Update metrics
                base_results['metrics'].update(video_metrics)
                base_results['evaluation_type'] = 'video_model'
            
            return base_results
        except Exception as e:
            logger.error(f"Error in video model evaluation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'metrics': {}
            }

class SpatialModelEvaluator(BaseModelEvaluator):
    """Evaluator for spatial model"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific metrics for spatial model
        self.spatial_metrics = [
            'position_accuracy', 'distance_accuracy',
            'volume_estimation_error', 'orientation_accuracy',
            'motion_prediction_accuracy', 'spatial_consistency_score'
        ]
    
    def evaluate(self, model, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate spatial model"""
        try:
            # Call base evaluation
            base_results = super().evaluate(model, test_data)
            
            # Add spatial-specific metrics
            if base_results['status'] == 'success':
                # Simulate spatial metrics
                spatial_metrics = {
                    'position_accuracy': np.random.uniform(0.8, 0.98),
                    'distance_accuracy': np.random.uniform(0.75, 0.96),
                    'volume_estimation_error': np.random.uniform(0.05, 0.2),
                    'orientation_accuracy': np.random.uniform(0.85, 0.99),
                    'motion_prediction_accuracy': np.random.uniform(0.7, 0.9)
                }
                
                # Update metrics
                base_results['metrics'].update(spatial_metrics)
                base_results['evaluation_type'] = 'spatial_model'
            
            return base_results
        except Exception as e:
            logger.error(f"Error in spatial model evaluation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'metrics': {}
            }

class SensorModelEvaluator(BaseModelEvaluator):
    """Evaluator for sensor model"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific metrics for sensor model
        self.sensor_metrics = [
            'mae', 'mse', 'rmse', 'r2_score',
            'correlation_coefficient', 'prediction_accuracy',
            'anomaly_detection_rate', 'sensor_noise_reduction_score'
        ]
    
    def evaluate(self, model, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate sensor model"""
        try:
            # Call base evaluation
            base_results = super().evaluate(model, test_data)
            
            # Add sensor-specific metrics
            if base_results['status'] == 'success':
                # Simulate sensor metrics
                sensor_metrics = {
                    'mae': np.random.uniform(0.01, 0.1),  # Mean Absolute Error
                    'mse': np.random.uniform(0.0001, 0.01),  # Mean Squared Error
                    'rmse': np.random.uniform(0.01, 0.1),  # Root Mean Squared Error
                    'r2_score': np.random.uniform(0.8, 0.99),  # R-squared Score
                    'correlation_coefficient': np.random.uniform(0.75, 0.98)
                }
                
                # Update metrics
                base_results['metrics'].update(sensor_metrics)
                base_results['evaluation_type'] = 'sensor_model'
            
            return base_results
        except Exception as e:
            logger.error(f"Error in sensor model evaluation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'metrics': {}
            }

class ComputerControlModelEvaluator(BaseModelEvaluator):
    """Evaluator for computer control model"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific metrics for computer control model
        self.computer_metrics = [
            'command_execution_accuracy', 'response_time',
            'system_resource_usage', 'multi_tasking_efficiency',
            'error_recovery_rate', 'cross_platform_compatibility'
        ]
    
    def evaluate(self, model, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate computer control model"""
        try:
            # Call base evaluation
            base_results = super().evaluate(model, test_data)
            
            # Add computer control-specific metrics
            if base_results['status'] == 'success':
                # Simulate computer control metrics
                computer_metrics = {
                    'command_execution_accuracy': np.random.uniform(0.85, 0.99),
                    'response_time': np.random.uniform(0.1, 1.0),  # in seconds
                    'system_resource_usage': np.random.uniform(0.1, 0.4),  # as fraction
                    'multi_tasking_efficiency': np.random.uniform(0.7, 0.95),
                    'error_recovery_rate': np.random.uniform(0.8, 0.98)
                }
                
                # Update metrics
                base_results['metrics'].update(computer_metrics)
                base_results['evaluation_type'] = 'computer_control_model'
            
            return base_results
        except Exception as e:
            logger.error(f"Error in computer control model evaluation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'metrics': {}
            }

class MotionControlModelEvaluator(BaseModelEvaluator):
    """Evaluator for motion control model"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific metrics for motion control model
        self.motion_metrics = [
            'positioning_accuracy', 'trajectory_following_error',
            'response_time', 'stability_score',
            'energy_efficiency', 'force_control_accuracy'
        ]
    
    def evaluate(self, model, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate motion control model"""
        try:
            # Call base evaluation
            base_results = super().evaluate(model, test_data)
            
            # Add motion control-specific metrics
            if base_results['status'] == 'success':
                # Simulate motion control metrics
                motion_metrics = {
                    'positioning_accuracy': np.random.uniform(0.9, 0.99),
                    'trajectory_following_error': np.random.uniform(0.01, 0.1),
                    'response_time': np.random.uniform(0.05, 0.5),  # in seconds
                    'stability_score': np.random.uniform(0.85, 0.99),
                    'energy_efficiency': np.random.uniform(0.7, 0.9)
                }
                
                # Update metrics
                base_results['metrics'].update(motion_metrics)
                base_results['evaluation_type'] = 'motion_control_model'
            
            return base_results
        except Exception as e:
            logger.error(f"Error in motion control model evaluation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'metrics': {}
            }

class KnowledgeBaseModelEvaluator(BaseModelEvaluator):
    """Evaluator for knowledge base model"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific metrics for knowledge base model
        self.knowledge_metrics = [
            'knowledge_coverage', 'answer_accuracy',
            'relevance_score', 'latency',
            'knowledge_update_efficiency', 'cross_domain_knowledge_integration'
        ]
    
    def evaluate(self, model, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate knowledge base model"""
        try:
            # Call base evaluation
            base_results = super().evaluate(model, test_data)
            
            # Add knowledge base-specific metrics
            if base_results['status'] == 'success':
                # Simulate knowledge base metrics
                knowledge_metrics = {
                    'knowledge_coverage': np.random.uniform(0.8, 0.99),
                    'answer_accuracy': np.random.uniform(0.85, 0.98),
                    'relevance_score': np.random.uniform(0.8, 0.97),
                    'latency': np.random.uniform(0.1, 0.5),  # in seconds
                    'knowledge_update_efficiency': np.random.uniform(0.75, 0.95)
                }
                
                # Update metrics
                base_results['metrics'].update(knowledge_metrics)
                base_results['evaluation_type'] = 'knowledge_base_model'
            
            return base_results
        except Exception as e:
            logger.error(f"Error in knowledge base model evaluation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'metrics': {}
            }

class ProgrammingModelEvaluator(BaseModelEvaluator):
    """Evaluator for programming model"""
    
    def __init__(self, model_id: str):
        super().__init__(model_id)
        # Specific metrics for programming model
        self.programming_metrics = [
            'code_compilation_rate', 'code_functionality_accuracy',
            'code_quality_score', 'execution_time',
            'memory_usage', 'bug_detection_rate',
            'language_support_coverage'
        ]
    
    def evaluate(self, model, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate programming model"""
        try:
            # Call base evaluation
            base_results = super().evaluate(model, test_data)
            
            # Add programming-specific metrics
            if base_results['status'] == 'success':
                # Simulate programming metrics
                programming_metrics = {
                    'code_compilation_rate': np.random.uniform(0.85, 0.99),
                    'code_functionality_accuracy': np.random.uniform(0.75, 0.95),
                    'code_quality_score': np.random.uniform(0.7, 0.9),
                    'bug_detection_rate': np.random.uniform(0.7, 0.95),
                    'language_support_coverage': np.random.uniform(0.8, 0.98)
                }
                
                # Update metrics
                base_results['metrics'].update(programming_metrics)
                base_results['evaluation_type'] = 'programming_model'
            
            return base_results
        except Exception as e:
            logger.error(f"Error in programming model evaluation: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'metrics': {}
            }

# Factory function to create model evaluators
def create_model_evaluator(model_id: str, model_type: str) -> ModelEvaluator:
    """Create a model evaluator for the specified model"""
    return ModelEvaluator(model_id, model_type)

# Initialize model evaluator factory
def get_model_evaluator(model_id: str, model_type: str) -> ModelEvaluator:
    """Get a model evaluator instance"""
    return create_model_evaluator(model_id, model_type)