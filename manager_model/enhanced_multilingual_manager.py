# -*- coding: utf-8 -*-
# Enhanced Multilingual Manager - AGI System with 5 Language Support
# Copyright 2025 Self Brain AGI System Authors
# Licensed under the Apache License, Version 2.0 (the "License")

import json
import os
import asyncio
from typing import Dict, List, Any, Optional, Callable
import logging
from datetime import datetime
from pathlib import Path

class EnhancedMultilingualManager:
    """Enhanced Multilingual Manager - Supports Chinese, English, German, Japanese, Russian
    """
    
    def __init__(self, default_language: str = 'zh'):
        """Initialize multilingual manager
        
        Parameters:
            default_language: Default language code (zh/en/de/ja/ru)
        """
        self.supported_languages = ['zh', 'en']
        self.language_names = {
            'zh': {'name': 'Chinese', 'native': '中文'},
            'en': {'name': 'English', 'native': 'English'}
        }
        
        self.language_resources = {}
        self.current_language = default_language
        self.language_change_callbacks = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Load language resources
        self._load_all_language_resources()
        
        self.logger.info(f"Multilingual manager initialized, current language: {self.current_language}")
    
    def _load_all_language_resources(self) -> None:
        """Load all language resource files"""
        # Try to load from language_resources in project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        resources_dir = os.path.join(project_root, 'language_resources')
        
        # Ensure resources directory exists
        os.makedirs(resources_dir, exist_ok=True)
        
        # Load resources for each supported language
        for lang in self.supported_languages:
            resource_file = os.path.join(resources_dir, f'{lang}.json')
            
            if os.path.exists(resource_file):
                try:
                    with open(resource_file, 'r', encoding='utf-8') as f:
                        self.language_resources[lang] = json.load(f)
                    self.logger.info(f"Loaded language resources: {lang}")
                except Exception as e:
                    self.logger.error(f"Failed to load language resource file {resource_file}: {str(e)}")
                    # Create default resources
                    self.language_resources[lang] = self._get_default_resources(lang)
            else:
                # Create default resources
                self.logger.warning(f"Language resource file not found: {resource_file}, using default")
                self.language_resources[lang] = self._get_default_resources(lang)
    
    def _get_default_resources(self, language_code: str) -> Dict[str, str]:
        """Get default language resources"""
        # Complete translation mapping
        translations = {
            "zh": {  # Chinese
                "system.initialization_complete": "系统初始化完成",
                "system.starting": "系统启动中...",
                "system.started": "系统已启动",
                "system.shutting_down": "系统关闭中...",
                "system.shut_down": "系统已关闭",
                "system.error": "系统错误",
                "emotion.neutral": "我明白了",
                "emotion.joy": "很高兴",
                "emotion.sadness": "感到难过",
                "emotion.anger": "感到生气",
                "emotion.surprise": "感到惊讶",
                "emotion.excitement": "感到兴奋",
                "emotion.confusion": "感到困惑",
                "emotion.anticipation": "期待",
                "intensity.high": "非常",
                "intensity.medium": "很",
                "intensity.low": "稍微",
                "ui.welcome": "欢迎使用Self Brain AGI系统",
                "ui.dashboard": "仪表盘",
                "ui.training": "模型训练",
                "ui.settings": "系统设置",
                "ui.model_management": "模型管理",
                "ui.help": "帮助",
                "ui.language": "语言",
                "ui.select_language": "选择语言",
                "ui.start_training": "开始训练",
                "ui.stop_training": "停止训练",
                "ui.training_progress": "训练进度",
                "ui.connected": "已连接",
                "ui.disconnected": "未连接",
                "ui.status": "状态",
                "ui.performance": "性能",
                "ui.knowledge": "知识库",
                "ui.import_knowledge": "导入知识",
                "ui.export_knowledge": "导出知识",
                "model.A": "管理模型",
                "model.B": "大语言模型",
                "model.C": "音频处理模型",
                "model.D": "图片视觉处理模型",
                "model.E": "视频流视觉处理模型",
                "model.F": "双目空间定位感知模型",
                "model.G": "传感器感知模型",
                "model.H": "计算机控制模型",
                "model.I": "知识库专家模型",
                "model.J": "运动和执行器控制模型",
                "model.K": "编程模型",
                "error.invalid_language": "无效的语言代码",
                "error.resource_not_found": "资源未找到",
                "error.connection_failed": "连接失败",
                "error.training_failed": "训练失败",
                "error.invalid_input": "无效输入",
                "success.language_changed": "语言切换成功",
                "success.training_started": "训练已开始",
                "success.training_completed": "训练已完成",
                "success.connection_established": "连接已建立",
                "success.knowledge_imported": "知识导入成功"
            },
            "en": {  # English
                "system.initialization_complete": "System initialization complete",
                "system.starting": "System starting...",
                "system.started": "System started",
                "system.shutting_down": "System shutting down...",
                "system.shut_down": "System shut down",
                "system.error": "System error",
                "emotion.neutral": "I understand",
                "emotion.joy": "Happy",
                "emotion.sadness": "Sad",
                "emotion.anger": "Angry",
                "emotion.surprise": "Surprised",
                "emotion.excitement": "Excited",
                "emotion.confusion": "Confused",
                "emotion.anticipation": "Anticipating",
                "intensity.high": "Very",
                "intensity.medium": "Quite",
                "intensity.low": "Slightly",
                "ui.welcome": "Welcome to Self Brain AGI System",
                "ui.dashboard": "Dashboard",
                "ui.training": "Model Training",
                "ui.settings": "System Settings",
                "ui.model_management": "Model Management",
                "ui.help": "Help",
                "ui.language": "Language",
                "ui.select_language": "Select Language",
                "ui.start_training": "Start Training",
                "ui.stop_training": "Stop Training",
                "ui.training_progress": "Training Progress",
                "ui.connected": "Connected",
                "ui.disconnected": "Disconnected",
                "ui.status": "Status",
                "ui.performance": "Performance",
                "ui.knowledge": "Knowledge Base",
                "ui.import_knowledge": "Import Knowledge",
                "ui.export_knowledge": "Export Knowledge",
                "model.A": "Management Model",
                "model.B": "Large Language Model",
                "model.C": "Audio Processing Model",
                "model.D": "Image Processing Model",
                "model.E": "Video Processing Model",
                "model.F": "Spatial Perception Model",
                "model.G": "Sensor Processing Model",
                "model.H": "Computer Control Model",
                "model.I": "Knowledge Base Model",
                "model.J": "Motion Control Model",
                "model.K": "Programming Model",
                "error.invalid_language": "Invalid language code",
                "error.resource_not_found": "Resource not found",
                "error.connection_failed": "Connection failed",
                "error.training_failed": "Training failed",
                "error.invalid_input": "Invalid input",
                "success.language_changed": "Language changed successfully",
                "success.training_started": "Training started",
                "success.training_completed": "Training completed",
                "success.connection_established": "Connection established",
                "success.knowledge_imported": "Knowledge imported successfully"
            }
        }
        
        return translations.get(language_code, translations["en"])
    
    def get_text(self, key: str, default: str = None, **kwargs) -> str:
        """Get localized text for specified key
        
        Parameters:
            key: Text key
            default: Default text if key not found
            **kwargs: Formatting parameters
        
        Returns:
            Localized text
        """
        try:
            # Get current language resources
            current_resources = self.language_resources.get(self.current_language, {})
            
            # Find text
            text = current_resources.get(key)
            
            # If not found in current language, try English
            if text is None and self.current_language != 'en':
                english_resources = self.language_resources.get('en', {})
                text = english_resources.get(key)
            
            # If still not found, use default value
            if text is None:
                text = default or key
            
            # Format text
            if kwargs:
                try:
                    text = text.format(**kwargs)
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"Text formatting failed {key}: {str(e)}")
            
            return text
            
        except Exception as e:
            self.logger.error(f"Failed to get text {key}: {str(e)}")
            return default or key
    
    async def switch_language(self, language: str) -> bool:
        """Switch system language
        
        Parameters:
            language: Target language code
        
        Returns:
            Whether switch was successful
        """
        if language not in self.supported_languages:
            self.logger.error(f"Unsupported language: {language}")
            return False
        
        old_language = self.current_language
        self.current_language = language
        
        # Notify all registered callback functions
        for callback in self.language_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(language, old_language)
                else:
                    callback(language, old_language)
            except Exception as e:
                self.logger.error(f"Language change callback error: {str(e)}")
        
        self.logger.info(f"Language switched from {old_language} to {language}")
        return True
    
    def register_language_callback(self, callback: Callable) -> None:
        """Register language change callback
        
        Parameters:
            callback: Callback function that receives (new_language, old_language) parameters
        """
        self.language_change_callbacks.append(callback)
        self.logger.info("Language change callback registered")
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages
        
        Returns:
            List of supported language codes
        """
        return self.supported_languages.copy()
    
    def get_language_info(self, language_code: str = None) -> Dict[str, Any]:
        """Get language information
        
        Parameters:
            language_code: Language code, returns current language info if None
        
        Returns:
            Language information dictionary
        """
        if language_code is None:
            language_code = self.current_language
        
        if language_code not in self.language_names:
            return {"error": "Unsupported language"}
        
        return {
            "code": language_code,
            "name": self.language_names[language_code]["name"],
            "native_name": self.language_names[language_code]["native"],
            "is_current": language_code == self.current_language,
            "resource_count": len(self.language_resources.get(language_code, {}))
        }
    
    def get_all_languages_info(self) -> List[Dict[str, Any]]:
        """Get information for all supported languages
        
        Returns:
            List of language information
        """
        return [self.get_language_info(lang) for lang in self.supported_languages]
    
    def reload_resources(self) -> bool:
        """Reload language resources
        
        Returns:
            Whether reload was successful
        """
        try:
            self._load_all_language_resources()
            self.logger.info("Language resources reloaded successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to reload language resources: {str(e)}")
            return False
    
    def add_custom_text(self, key: str, translations: Dict[str, str]) -> bool:
        """Add custom text
        
        Parameters:
            key: Text key
            translations: Dictionary of translations for each language
        
        Returns:
            Whether addition was successful
        """
        try:
            for lang_code, text in translations.items():
                if lang_code in self.supported_languages:
                    if lang_code not in self.language_resources:
                        self.language_resources[lang_code] = {}
                    self.language_resources[lang_code][key] = text
            
            self.logger.info(f"Added custom text: {key}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to add custom text {key}: {str(e)}")
            return False
    
    def get_current_language(self) -> str:
        """Get current language code
        
        Returns:
            Current language code
        """
        return self.current_language
    
    def export_language_resources(self, language_code: str, file_path: str) -> bool:
        """Export language resources to file
        
        Parameters:
            language_code: Language code
            file_path: Export file path
        
        Returns:
            Whether export was successful
        """
        try:
            if language_code not in self.language_resources:
                self.logger.error(f"Language resources not found: {language_code}")
                return False
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.language_resources[language_code], f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Language resources exported: {language_code} -> {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to export language resources: {str(e)}")
            return False

# Create global multilingual manager instance
_global_multilingual_manager = None

def get_multilingual_manager(default_language: str = 'zh') -> EnhancedMultilingualManager:
    """Get global multilingual manager instance
    
    Parameters:
        default_language: Default language code
    
    Returns:
        Multilingual manager instance
    """
    global _global_multilingual_manager
    if _global_multilingual_manager is None:
        _global_multilingual_manager = EnhancedMultilingualManager(default_language)
    return _global_multilingual_manager

# Convenience functions
def get_text(key: str, default: str = None, **kwargs) -> str:
    """Convenience function to get localized text"""
    manager = get_multilingual_manager()
    return manager.get_text(key, default, **kwargs)

async def switch_language(language: str) -> bool:
    """Convenience function to switch language"""
    manager = get_multilingual_manager()
    return await manager.switch_language(language)

# Example usage
async def demo_multilingual_manager():
    """Demonstrate multilingual manager functionality"""
    print("Initializing multilingual manager...")
    
    manager = EnhancedMultilingualManager()
    
    # Test text retrieval
    print("Testing text retrieval:")
    print(f"Chinese: {manager.get_text('ui.welcome')}")
    
    # Switch to English
    print("Switching to English...")
    await manager.switch_language('en')
    print(f"English: {manager.get_text('ui.welcome')}")
