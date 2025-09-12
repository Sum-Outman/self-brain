# -*- coding: utf-8 -*-
# 增强型多语言支持系统 - 支持汉语、英文、德语、日语、俄语
# Enhanced Multilingual Support System - Supports Chinese, English, German, Japanese, Russian
# Copyright 2025 Self Brain AGI System Authors
# Licensed under the Apache License, Version 2.0 (the "License")

import json
import os
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import asyncio

class EnhancedMultilingualManager:
    """增强型多语言管理器 - 支持汉语、英文、德语、日语、俄语
    Enhanced Multilingual Manager - Supports Chinese, English, German, Japanese, Russian
    """
    
    def __init__(self, default_language='zh'):
        """初始化增强型多语言管理器 | Initialize enhanced multilingual manager
        
        参数 Parameters:
            default_language: 默认语言代码 (zh/en/de/ja/ru) | Default language code (zh/en/de/ja/ru)
        """
        # 支持的五种语言
        self.supported_languages = {
            'zh': {
                'name': '中文',
                'native_name': '中文',
                'code': 'zh',
                'rtl': False,
                'locale': 'zh_CN'
            },
            'en': {
                'name': 'English',
                'native_name': 'English',
                'code': 'en',
                'rtl': False,
                'locale': 'en_US'
            }
        }
        
        self.current_language = default_language
        self.translations = {lang: {} for lang in self.supported_languages}
        self.language_change_callbacks = []
        
        # 设置日志
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # 初始化翻译资源
        self.initialize()
    
    def initialize(self) -> bool:
        """初始化多语言系统 | Initialize multilingual system
        
        返回 Returns:
            初始化是否成功 | Whether initialization was successful
        """
        try:
            # 加载核心翻译资源
            self._load_core_translations()
            
            # 加载模型特定翻译资源
            self._load_model_translations()
            
            # 加载界面翻译资源
            self._load_interface_translations()
            
            # 加载帮助文档翻译
            self._load_help_translations()
            
            self.logger.info("多语言系统初始化完成 | Multilingual system initialization completed")
            return True
            
        except Exception as e:
            self.logger.error(f"多语言系统初始化失败: {str(e)} | Multilingual system initialization failed: {str(e)}")
            return False
    
    def _load_core_translations(self) -> None:
        """加载核心翻译文件 | Load core translation files"""
        core_path = 'language_resources/core/'
        if not os.path.exists(core_path):
            os.makedirs(core_path, exist_ok=True)
            self.logger.warning(f"核心翻译目录不存在，已创建: {core_path}")
            return
        
        for lang_code in self.supported_languages:
            lang_file = os.path.join(core_path, f'core_{lang_code}.json')
            if os.path.exists(lang_file):
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        self.translations[lang_code].update(json.load(f))
                    self.logger.info(f"已加载核心翻译: {lang_code} | Loaded core translations: {lang_code}")
                except Exception as e:
                    self.logger.error(f"加载核心翻译文件错误 {lang_file}: {str(e)}")
            else:
                self.logger.warning(f"核心翻译文件不存在: {lang_file}")
    
    def _load_model_translations(self) -> None:
        """加载模型特定翻译 | Load model-specific translations"""
        models_path = 'language_resources/models/'
        if not os.path.exists(models_path):
            os.makedirs(models_path, exist_ok=True)
            self.logger.warning(f"模型翻译目录不存在，已创建: {models_path}")
            return
        
        # 加载所有模型的翻译
        for model_dir in os.listdir(models_path):
            model_path = os.path.join(models_path, model_dir)
            if os.path.isdir(model_path):
                for lang_code in self.supported_languages:
                    lang_file = os.path.join(model_path, f'{model_dir}_{lang_code}.json')
                    if os.path.exists(lang_file):
                        try:
                            with open(lang_file, 'r', encoding='utf-8') as f:
                                model_translations = json.load(f)
                                # 为模型翻译添加前缀
                                prefixed_translations = {
                                    f"model.{model_dir}.{key}": value 
                                    for key, value in model_translations.items()
                                }
                                self.translations[lang_code].update(prefixed_translations)
                            self.logger.info(f"已加载模型翻译: {model_dir}/{lang_code}")
                        except Exception as e:
                            self.logger.error(f"加载模型翻译文件错误 {lang_file}: {str(e)}")
    
    def _load_interface_translations(self) -> None:
        """加载界面翻译 | Load interface translations"""
        interface_path = 'language_resources/interface/'
        if not os.path.exists(interface_path):
            os.makedirs(interface_path, exist_ok=True)
            self.logger.warning(f"界面翻译目录不存在，已创建: {interface_path}")
            return
        
        for lang_code in self.supported_languages:
            lang_file = os.path.join(interface_path, f'interface_{lang_code}.json')
            if os.path.exists(lang_file):
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        interface_translations = json.load(f)
                        # 为界面翻译添加前缀
                        prefixed_translations = {
                            f"interface.{key}": value 
                            for key, value in interface_translations.items()
                        }
                        self.translations[lang_code].update(prefixed_translations)
                    self.logger.info(f"已加载界面翻译: {lang_code}")
                except Exception as e:
                    self.logger.error(f"加载界面翻译文件错误 {lang_file}: {str(e)}")
    
    def _load_help_translations(self) -> None:
        """加载帮助文档翻译 | Load help documentation translations"""
        help_path = 'language_resources/help/'
        if not os.path.exists(help_path):
            os.makedirs(help_path, exist_ok=True)
            self.logger.warning(f"帮助翻译目录不存在，已创建: {help_path}")
            return
        
        for lang_code in self.supported_languages:
            lang_file = os.path.join(help_path, f'help_{lang_code}.json')
            if os.path.exists(lang_file):
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        help_translations = json.load(f)
                        # 为帮助翻译添加前缀
                        prefixed_translations = {
                            f"help.{key}": value 
                            for key, value in help_translations.items()
                        }
                        self.translations[lang_code].update(prefixed_translations)
                    self.logger.info(f"已加载帮助翻译: {lang_code}")
                except Exception as e:
                    self.logger.error(f"加载帮助翻译文件错误 {lang_file}: {str(e)}")
    
    async def switch_language(self, new_language: str) -> bool:
        """异步切换系统语言 | Asynchronously switch system language
        
        参数 Parameters:
            new_language: 新语言代码 (zh/en/de/ja/ru) | New language code (zh/en/de/ja/ru)
            
        返回 Returns:
            是否切换成功 | Whether switching was successful
        """
        if new_language not in self.supported_languages:
            self.logger.error(f"不支持的语言: {new_language} | Unsupported language: {new_language}")
            return False
        
        if new_language == self.current_language:
            self.logger.info(f"语言已经是 {new_language} | Language is already {new_language}")
            return True
        
        # 更新当前语言
        old_language = self.current_language
        self.current_language = new_language
        
        # 异步通知所有注册的回调函数
        callback_tasks = []
        for callback in self.language_change_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    task = asyncio.create_task(callback(new_language, old_language))
                    callback_tasks.append(task)
                else:
                    # 对于同步回调，在线程池中执行
                    loop = asyncio.get_event_loop()
                    task = loop.run_in_executor(None, callback, new_language, old_language)
                    callback_tasks.append(task)
            except Exception as e:
                self.logger.error(f"语言切换回调错误: {str(e)}")
        
        # 等待所有回调完成
        if callback_tasks:
            await asyncio.gather(*callback_tasks, return_exceptions=True)
        
        self.logger.info(f"系统语言已从 {old_language} 切换到 {new_language} | System language switched from {old_language} to {new_language}")
        return True
    
    def register_language_callback(self, callback_func) -> None:
        """注册语言切换回调函数 | Register language change callback function
        
        参数 Parameters:
            callback_func: 回调函数，接受 (new_language, old_language) 参数
            | Callback function that accepts (new_language, old_language) parameters
        """
        self.language_change_callbacks.append(callback_func)
        self.logger.info(f"已注册语言切换回调函数 | Language change callback registered")
    
    def get_text(self, key: str, default: Optional[str] = None, **kwargs) -> str:
        """获取翻译文本 | Get translated text
        
        参数 Parameters:
            key: 文本键 | Text key
            default: 默认文本（如果键不存在） | Default text (if key not found)
            kwargs: 格式化参数 | Formatting parameters
            
        返回 Returns:
            翻译后的文本 | Translated text
        """
        # 获取当前语言的翻译
        translation = self.translations[self.current_language].get(key)
        
        if translation is None:
            # 如果当前语言没有翻译，尝试英文作为后备
            if self.current_language != 'en':
                translation = self.translations['en'].get(key)
            
            if translation is None:
                # 如果都没有，使用默认值或键本身
                translation = default if default is not None else key
                self.logger.warning(f"翻译键未找到: {key} | Translation key not found: {key}")
        
        # 格式化文本（如果有参数）
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                self.logger.error(f"文本格式化错误: {translation}, 参数: {kwargs}, 错误: {str(e)}")
        
        return translation
    
    def get_interface_text(self, component: str, element: str, **kwargs) -> str:
        """获取界面元素文本 | Get interface element text
        
        参数 Parameters:
            component: 组件名称 | Component name
            element: 元素名称 | Element name
            kwargs: 格式化参数 | Formatting parameters
            
        返回 Returns:
            翻译后的界面文本 | Translated interface text
        """
        key = f"interface.{component}.{element}"
        return self.get_text(key, **kwargs)
    
    def get_model_text(self, model_name: str, text_key: str, **kwargs) -> str:
        """获取模型相关文本 | Get model-related text
        
        参数 Parameters:
            model_name: 模型名称 | Model name
            text_key: 文本键 | Text key
            kwargs: 格式化参数 | Formatting parameters
            
        返回 Returns:
            翻译后的模型文本 | Translated model text
        """
        key = f"model.{model_name}.{text_key}"
        return self.get_text(key, **kwargs)
    
    def get_error_text(self, error_code: str, **kwargs) -> str:
        """获取错误信息文本 | Get error message text
        
        参数 Parameters:
            error_code: 错误代码 | Error code
            kwargs: 格式化参数 | Formatting parameters
            
        返回 Returns:
            翻译后的错误信息 | Translated error message
        """
        key = f"error.{error_code}"
        return self.get_text(key, **kwargs)
    
    def get_help_text(self, topic: str, subtopic: str = None, **kwargs) -> str:
        """获取帮助文本 | Get help text
        
        参数 Parameters:
            topic: 帮助主题 | Help topic
            subtopic: 帮助子主题 | Help subtopic
            kwargs: 格式化参数 | Formatting parameters
            
        返回 Returns:
            翻译后的帮助文本 | Translated help text
        """
        if subtopic:
            key = f"help.{topic}.{subtopic}"
        else:
            key = f"help.{topic}"
        return self.get_text(key, **kwargs)
    
    def create_language_resource_template(self) -> Dict[str, Any]:
        """创建语言资源模板 | Create language resource template
        
        返回 Returns:
            语言资源模板字典 | Language resource template dictionary
        """
        template = {
            "metadata": {
                "created_date": datetime.now().isoformat(),
                "version": "1.0.0",
                "description": "Self Brain AGI系统多语言资源模板 | Self Brain AGI System Multilingual Resource Template",
                "supported_languages": list(self.supported_languages.keys())
            },
            "core": {
                "welcome_message": {
                    "zh": "欢迎使用Self Brain AGI系统",
                    "en": "Welcome to Self Brain AGI System",
                    "de": "Willkommen beim Self Brain AGI-System",
                    "ja": "Self Brain AGIシステムへようこそ",
                    "ru": "Добро пожаловать в систему Self Brain AGI"
                },
                "language_selection": {
                    "zh": "语言选择",
                    "en": "Language Selection",
                    "de": "Sprachauswahl",
                    "ja": "言語選択",
                    "ru": "Выбор языка"
                }
            },
            "interface": {
                "main_menu": {
                    "training": {
                        "zh": "模型训练",
                        "en": "Model Training",
                        "de": "Modelltraining",
                        "ja": "モデル訓練",
                        "ru": "Обучение модели"
                    },
                    "settings": {
                        "zh": "系统设置",
                        "en": "System Settings",
                        "de": "Systemeinstellungen",
                        "ja": "システム設定",
                        "ru": "Системные настройки"
                    },
                    "model_management": {
                        "zh": "模型管理",
                        "en": "Model Management",
                        "de": "Modellverwaltung",
                        "ja": "モデル管理",
                        "ru": "Управление моделями"
                    },
                    "help": {
                        "zh": "帮助",
                        "en": "Help",
                        "de": "Hilfe",
                        "ja": "ヘルプ",
                        "ru": "Помощь"
                    }
                }
            },
            "models": {
                "A_manager": {
                    "name": {
                        "zh": "管理模型",
                        "en": "Manager Model",
                        "de": "Manager-Modell",
                        "ja": "管理モデル",
                        "ru": "Модель менеджера"
                    },
                    "description": {
                        "zh": "负责协调所有子模型的核心管理模型",
                        "en": "Core manager model responsible for coordinating all sub-models",
                        "de": "Kern-Manager-Modell für die Koordination aller Untermodelle",
                        "ja": "すべてのサブモデルを調整するコアマネージャーモデル",
                        "ru": "Основная модель менеджера, отвечающая за координацию всех подмоделей"
                    }
                }
            }
        }
        return template
    
    def export_language_resources(self, output_dir: str) -> bool:
        """导出语言资源文件 | Export language resource files
        
        参数 Parameters:
            output_dir: 输出目录 | Output directory
            
        返回 Returns:
            是否导出成功 | Whether export was successful
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 按语言组织资源
            for lang_code in self.supported_languages:
                lang_resources = {}
                
                # 收集所有该语言的翻译
                for key, value in self.translations[lang_code].items():
                    lang_resources[key] = value
                
                # 写入文件
                lang_file = os.path.join(output_dir, f'resources_{lang_code}.json')
                with open(lang_file, 'w', encoding='utf-8') as f:
                    json.dump(lang_resources, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"已导出语言资源: {lang_file} | Exported language resources: {lang_file}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"导出语言资源错误: {str(e)} | Error exporting language resources: {str(e)}")
            return False
    
    def get_current_language(self) -> str:
        """获取当前语言 | Get current language"""
        return self.current_language
    
    def get_supported_languages(self) -> List[str]:
        """获取支持的语言列表 | Get supported languages list"""
        return list(self.supported_languages.keys())
    
    def get_language_info(self, language_code: str) -> Optional[Dict[str, str]]:
        """获取语言详细信息 | Get language detailed information
        
        参数 Parameters:
            language_code: 语言代码 | Language code
            
        返回 Returns:
            语言信息字典或None | Language info dictionary or None
        """
        return self.supported_languages.get(language_code)
    
    def get_all_languages_info(self) -> Dict[str, Dict[str, str]]:
        """获取所有支持语言的详细信息 | Get detailed information for all supported languages"""
        return self.supported_languages.copy()

# 创建多语言系统实例的工厂函数
def create_enhanced_multilingual_manager(default_language='zh'):
    """创建增强型多语言管理器实例 | Create enhanced multilingual manager instance"""
    return EnhancedMultilingualManager(default_language)

# 示例使用
async def demo_multilingual_system():
    """演示多语言系统功能 | Demonstrate multilingual system functionality"""
    print("初始化增强型多语言支持系统... | Initializing enhanced multilingual support system...")
    
    manager = create_enhanced_multilingual_manager()
    
    # 显示支持的语言
    print("支持的语言: | Supported languages:")
    for lang_code in manager.get_supported_languages():
        info = manager.get_language_info(lang_code)
        print(f"  {lang_code}: {info['native_name']} ({info['name']})")
    
    # 测试语言切换
    print(f"\n当前语言: {manager.get_current_language()} | Current language: {manager.get_current_language()}")
    
    # 切换到英文
    await manager.switch_language('en')
    print(f"切换后语言: {manager.get_current_language()} | Language after switch: {manager.get_current_language()}")
    
    # 测试获取文本
    welcome_text = manager.get_text('welcome_message', default="Welcome to AGI System")
    print(f"欢迎文本: {welcome_text} | Welcome text: {welcome_text}")
    
    # 切换回中文
    await manager.switch_language('zh')
    print(f"最终语言: {manager.get_current_language()} | Final language: {manager.get_current_language()}")
    
    print("增强型多语言支持系统演示完成! | Enhanced multilingual support system demonstration completed!")

if __name__ == '__main__':
    # 运行演示
    asyncio.run(demo_multilingual_system())