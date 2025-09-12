# -*- coding: utf-8 -*-
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0

"""
Unified Model Manager
Integrates all sub-models, providing unified interface and management functions
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

# Add sub-models path to system path
sub_models_path = Path(__file__).parent
sys.path.insert(0, str(sub_models_path))

# Import all unified models
from B_language.unified_language_model import UnifiedLanguageModel
from C_audio.unified_audio_model import UnifiedAudioProcessingModel
from D_image.unified_image_model import UnifiedImageModel
from E_video.unified_video_model import UnifiedVideoModel
from F_spatial.unified_spatial_model import UnifiedSpatialModel
from G_sensor.unified_sensor_model import UnifiedSensorModel
from H_computer_control.unified_computer_control_model import UnifiedComputerControlModel
from I_knowledge.unified_knowledge_model import UnifiedKnowledgeModel
from J_motion.unified_motion_model import UnifiedMotionModel
from K_programming.unified_programming_model import UnifiedProgrammingModel

class UnifiedModelManager:
    """
    Unified Model Manager
    Manages creation, configuration, and invocation of all sub-models
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.config = {}
        
        # Load configuration
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        # Initialize all models
        self._initialize_models()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "language": {"mode": "enhanced", "config": {}},
            "audio": {"mode": "enhanced", "config": {}},
            "image": {"mode": "enhanced", "config": {}},
            "video": {"mode": "enhanced", "config": {}},
            "spatial": {"mode": "enhanced", "config": {}},
            "sensor": {"mode": "enhanced", "config": {}},
            "computer_control": {"mode": "enhanced", "config": {}},
            "knowledge": {"mode": "enhanced", "config": {}},
            "motion": {"mode": "enhanced", "config": {}},
            "programming": {"mode": "enhanced", "config": {}}
        }
    
    def _initialize_models(self):
        """Initialize all sub-models"""
        model_mapping = {
            "language": UnifiedLanguageModel,
            "audio": UnifiedAudioProcessingModel,
            "image": UnifiedImageModel,
            "video": UnifiedVideoModel,
            "spatial": UnifiedSpatialModel,
            "sensor": UnifiedSensorModel,
            "computer_control": UnifiedComputerControlModel,
            "knowledge": UnifiedKnowledgeModel,
            "motion": UnifiedMotionModel,
            "programming": UnifiedProgrammingModel
        }
        
        for model_name, model_class in model_mapping.items():
            try:
                model_config = self.config.get(model_name, {})
                mode = model_config.get("mode", "enhanced")
                config = model_config.get("config", {})
                
                # Pass parameters based on model class signature
                try:
                    # Try to pass mode and config
                    self.models[model_name] = model_class(mode=mode, **config)
                except TypeError:
                    # If failed, try to pass only mode
                    try:
                        self.models[model_name] = model_class(mode=mode)
                    except TypeError:
                        # Finally try no parameters
                        self.models[model_name] = model_class()
                
                self.logger.info(f"Initialized model {model_name} successfully (mode: {mode})")
                
            except Exception as e:
                self.logger.error(f"Failed to initialize model {model_name}: {str(e)}")
                try:
                    # Create model in standard mode as fallback
                    self.models[model_name] = model_class(mode="standard")
                except:
                    self.models[model_name] = model_class()
    
    def get_model(self, model_name: str) -> Any:
        """Get specified model instance"""
        return self.models.get(model_name)
    
    def get_available_models(self) -> List[str]:
        """Get list of all available models"""
        return list(self.models.keys())
    
    def set_model_mode(self, model_name: str, mode: str) -> bool:
        """Set model operation mode"""
        if model_name not in self.models:
            return False
        
        try:
            model_class = type(self.models[model_name])
            config = self.config.get(model_name, {}).get("config", {})
            
            try:
                new_model = model_class(mode=mode, **config)
            except TypeError:
                try:
                    new_model = model_class(mode=mode)
                except TypeError:
                    new_model = model_class()
            
            self.models[model_name] = new_model
            return True
        except Exception as e:
            self.logger.error(f"Failed to set mode for model {model_name}: {str(e)}")
            return False
    
    def get_model_status(self) -> Dict[str, str]:
        """Get status of all models"""
        status = {}
        for name, model in self.models.items():
            status[name] = getattr(model, 'mode', 'unknown')
        return status
    
    def execute_cross_model_task(self, task_type: str, **kwargs) -> Dict[str, Any]:
        """Execute cross-model task"""
        results = {}
        
        if task_type == "comprehensive_analysis":
            # Comprehensive analysis task
            text = kwargs.get('text', '')
            audio_data = kwargs.get('audio_data', None)
            image_data = kwargs.get('image_data', None)
            
            if text:
                language_model = self.get_model('language')
                if language_model:
                    results['language_analysis'] = language_model.analyze_sentiment(text)
            
            if audio_data:
                audio_model = self.get_model('audio')
                if audio_model:
                    results['audio_analysis'] = audio_model.process_audio(audio_data)
            
            if image_data:
                image_model = self.get_model('image')
                if image_model:
                    results['image_analysis'] = image_model.analyze_image(image_data)
        
        elif task_type == "intelligent_response":
            # Intelligent response task
            query = kwargs.get('query', '')
            context = kwargs.get('context', {})
            
            language_model = self.get_model('language')
            knowledge_model = self.get_model('knowledge')
            
            if language_model and knowledge_model:
                # Generate response combined with knowledge base
                knowledge = knowledge_model.search_knowledge(query)
                response = language_model.generate_response(query, context)
                results['response'] = response
                results['knowledge_used'] = knowledge
        
        return results
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "total_models": len(self.models),
            "available_models": self.get_available_models(),
            "model_status": self.get_model_status(),
            "config_path": str(sub_models_path),
            "python_version": sys.version,
            "platform": sys.platform
        }

# Global manager instance
_model_manager = None

def get_model_manager(config_path: Optional[str] = None) -> UnifiedModelManager:
    """Get global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = UnifiedModelManager(config_path)
    return _model_manager

# Convenient access functions
def get_model(model_name: str) -> Any:
    """Conveniently get model"""
    manager = get_model_manager()
    return manager.get_model(model_name)

def list_models() -> List[str]:
    """Conveniently list models"""
    manager = get_model_manager()
    return manager.get_available_models()

# Usage example
if __name__ == "__main__":
    # Initialize manager
    manager = get_model_manager()
    
    # Print system information
    print("=== AGI Brain Unified Model Management System ===")
    info = manager.get_system_info()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    
    # Test each model
    print("\n=== Testing Model Functions ===")
    
    # Test language model
    language_model = manager.get_model('language')
    if language_model:
        result = language_model.generate_response("你好，AGI系统")
        print(f"Language model test: {result}")
    
    # Test programming model
    programming_model = manager.get_model('programming')
    if programming_model:
        code = "def hello(): return 'Hello World'"
        analysis = programming_model.analyze_code(code)
        print(f"Programming model test: function count={len(analysis.functions)}")
    
    print("\n=== System Ready ===")
