# -*- coding: utf-8 -*-
# Unified Multilingual Manager - AGI System with 5 Language Support
# Copyright 2025 Self Brain AGI System Authors
# Licensed under the Apache License, Version 2.0 (the "License")
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import logging
from typing import Dict, List, Any, Optional, Callable

class UnifiedMultilingualManager:
    """Unified Multilingual Manager - Supports Chinese, English, German, Japanese, Russian
    """
    
    def __init__(self, default_language: str = 'zh', resource_dir: str = "language_resources"):
        """Initialize unified multilingual manager
        
        Parameters:
            default_language: Default language code (zh/en/de/ja/ru)
            resource_dir: Resource file directory
        """
        self.supported_languages = ['zh', 'en']
        self.language_names = {
            'zh': {'name': 'Chinese', 'native': '中文'},
            'en': {'name': 'English', 'native': 'English'}
        }
        
        self.language_resources = {}
        self.current_language = default_language
        self.resource_dir = resource_dir
        self.language_change_callbacks = []
        
        self.logger = logging.getLogger(__name__)
        
        # Ensure resource directory exists
        os.makedirs(resource_dir, exist_ok=True)
        
        # Load all language resources
        self._load_all_language_resources()
        
        self.logger.info(f"Unified multilingual manager initialized, current language: {self.current_language}")

    def _load_all_language_resources(self) -> None:
        """Load all language resource files"""
        for lang in self.supported_languages:
            resource_file = os.path.join(self.resource_dir, f'{lang}.json')
            
            if os.path.exists(resource_file):
                try:
                    with open(resource_file, 'r', encoding='utf-8') as f:
                        self.language_resources[lang] = json.load(f)
                    self.logger.info(f"Loaded language resources: {lang}")
                except Exception as e:
                    self.logger.error(f"Failed to load language resource file {resource_file}: {str(e)}")
                    self._create_default_resource_file(resource_file, lang)
            else:
                self.logger.warning(f"Language resource file does not exist: {resource_file}, creating default file")
                self._create_default_resource_file(resource_file, lang)

    def _create_default_resource_file(self, file_path: str, language_code: str) -> None:
        """Create default language resource file"""
        default_resources = self._get_default_resources(language_code)
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(default_resources, f, ensure_ascii=False, indent=2)
            self.language_resources[language_code] = default_resources
            self.logger.info(f"Created default language resource file: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to create language resource file {file_path}: {str(e)}")

    def _get_default_resources(self, language_code: str) -> Dict[str, str]:
        """Get default language resources"""
        base_resources = {
            "system.initialization_complete": self._get_translation("系统初始化完成", language_code),
            "system.error": self._get_translation("系统错误", language_code),
            "ui.welcome": self._get_translation("欢迎使用Self Brain AGI系统", language_code),
            "ui.dashboard": self._get_translation("仪表盘", language_code),
            "ui.training": self._get_translation("模型训练", language_code),
            "ui.settings": self._get_translation("系统设置", language_code),
            "ui.model_management": self._get_translation("模型管理", language_code),
            "ui.help": self._get_translation("帮助", language_code),
            "ui.language": self._get_translation("语言", language_code),
            "ui.select_language": self._get_translation("选择语言", language_code),
            "model.A": self._get_translation("管理模型", language_code),
            "model.B": self._get_translation("大语言模型", language_code),
            "model.C": self._get_translation("音频处理模型", language_code),
            "model.D": self._get_translation("图片视觉处理模型", language_code),
            "model.E": self._get_translation("视频流视觉处理模型", language_code),
            "model.F": self._get_translation("双目空间定位感知模型", language_code),
            "model.G": self._get_translation("传感器感知模型", language_code),
            "model.H": self._get_translation("计算机控制模型", language_code),
            "model.I": self._get_translation("知识库专家模型", language_code),
            "model.J": self._get_translation("运动和执行器控制模型", language_code),
            "model.K": self._get_translation("编程模型", language_code),
            "error.invalid_language": self._get_translation("无效的语言代码", language_code),
            "success.language_changed": self._get_translation("语言切换成功", language_code),
            "realtime.camera": self._get_translation("摄像头", language_code),
            "realtime.microphone": self._get_translation("麦克风", language_code),
            "training.individual": self._get_translation("单独训练", language_code),
            "training.joint": self._get_translation("联合训练", language_code),
            "dashboard.overview": self._get_translation("概览", language_code),
            "dashboard.performance": self._get_translation("性能监控", language_code)
        }
        
        return base_resources

    def _get_translation(self, text: str, target_language: str) -> str:
        """Get translation of text"""
        translation_map = {
            "zh": {
                "系统初始化完成": "系统初始化完成",
                "系统错误": "系统错误",
                "欢迎使用Self Brain AGI系统": "欢迎使用Self Brain AGI系统",
                "仪表盘": "仪表盘",
                "模型训练": "模型训练",
                "系统设置": "系统设置",
                "模型管理": "模型管理",
                "帮助": "帮助",
                "语言": "语言",
                "选择语言": "选择语言",
                "管理模型": "管理模型",
                "大语言模型": "大语言模型",
                "音频处理模型": "音频处理模型",
                "图片视觉处理模型": "图片视觉处理模型",
                "视频流视觉处理模型": "视频流视觉处理模型",
                "双目空间定位感知模型": "双目空间定位感知模型",
                "传感器感知模型": "传感器感知模型",
                "计算机控制模型": "计算机控制模型",
                "知识库专家模型": "知识库专家模型",
                "运动和执行器控制模型": "运动和执行器控制模型",
                "编程模型": "编程模型",
                "无效的语言代码": "无效的语言代码",
                "语言切换成功": "语言切换成功",
                "摄像头": "摄像头",
                "麦克风": "麦克风",
                "单独训练": "单独训练",
                "联合训练": "联合训练",
                "概览": "概览",
                "性能监控": "性能监控"
            },
            "en": {
                "系统初始化完成": "System initialization complete",
                "系统错误": "System error",
                "欢迎使用Self Brain AGI系统": "Welcome to Self Brain AGI System",
                "仪表盘": "Dashboard",
                "模型训练": "Model Training",
                "系统设置": "System Settings",
                "模型管理": "Model Management",
                "帮助": "Help",
                "语言": "Language",
                "选择语言": "Select Language",
                "管理模型": "Management Model",
                "大语言模型": "Large Language Model",
                "音频处理模型": "Audio Processing Model",
                "图片视觉处理模型": "Image Processing Model",
                "视频流视觉处理模型": "Video Processing Model",
                "双目空间定位感知模型": "Spatial Perception Model",
                "传感器感知模型": "Sensor Processing Model",
                "计算机控制模型": "Computer Control Model",
                "知识库专家模型": "Knowledge Base Model",
                "运动和执行器控制模型": "Motion Control Model",
                "编程模型": "Programming Model",
                "无效的语言代码": "Invalid language code",
                "语言切换成功": "Language changed successfully",
                "摄像头": "Camera",
                "麦克风": "Microphone",
                "单独训练": "Individual Training",
                "联合训练": "Joint Training",
                "概览": "Overview",
                "性能监控": "Performance Monitoring"
            },
            "de": {
                "系统初始化完成": "Systeminitialisierung abgeschlossen",
                "系统错误": "Systemfehler",
                "欢迎使用Self Brain AGI系统": "Willkommen beim Self Brain AGI System",
                "仪表盘": "Dashboard",
                "模型训练": "Modelltraining",
                "系统设置": "Systemeinstellungen",
                "模型管理": "Modellverwaltung",
                "帮助": "Hilfe",
                "语言": "Sprache",
                "选择语言": "Sprache auswählen",
                "管理模型": "Management-Modell",
                "大语言模型": "Großes Sprachmodell",
                "音频处理模型": "Audioverarbeitungsmodell",
                "图片视觉处理模型": "Bildverarbeitungsmodell",
                "视频流视觉处理模型": "Videoverarbeitungsmodell",
                "双目空间定位感知模型": "Räumliches Wahrnehmungsmodell",
                "传感器感知模型": "Sensorverarbeitungsmodell",
                "计算机控制模型": "Computersteuerungsmodell",
                "知识库专家模型": "Wissensdatenbank-Modell",
                "运动和执行器控制模型": "Bewegungssteuerungsmodell",
                "编程模型": "Programmiermodell",
                "无效的语言代码": "Ungültiger Sprachcode",
                "语言切换成功": "Sprache erfolgreich geändert",
                "摄像头": "Kamera",
                "麦克风": "Mikrofon",
                "单独训练": "Einzeltraining",
                "联合训练": "Gemeinsames Training",
                "概览": "Übersicht",
                "性能监控": "Leistungsüberwachung"
            },
            "ja": {
                "系统初始化完成": "システム初期化完了",
                "系统错误": "システムエラー",
                "欢迎使用Self Brain AGI系统": "Self Brain AGIシステムへようこそ",
                "仪表盘": "ダッシュボード",
                "模型训练": "モデル訓練",
                "系统设置": "システム設定",
                "模型管理": "モデル管理",
                "帮助": "ヘルプ",
                "语言": "言語",
                "选择语言": "言語選択",
                "管理模型": "管理モデル",
                "大语言模型": "大規模言語モデル",
                "音频处理模型": "音声処理モデル",
                "图片视觉处理模型": "画像処理モデル",
                "视频流视觉处理模型": "動画処理モデル",
                "双目空间定位感知模型": "空間知覚モデル",
                "传感器感知模型": "センサー処理モデル",
                "计算机控制模型": "コンピューター制御モデル",
                "知识库专家模型": "知識ベースモデル",
                "运动和执行器控制模型": "運動制御モデル",
                "编程模型": "プログラミングモデル",
                "无效的语言代码": "無効な言語コード",
                "语言切换成功": "言語変更成功",
                "摄像头": "カメラ",
                "麦克风": "マイク",
                "单独训练": "個別訓練",
                "联合训练": "共同訓練",
                "概览": "概要",
                "性能监控": "性能監視"
            },
            "ru": {
                "系统初始化完成": "Инициализация системы завершена",
                "系统错误": "Ошибка системы",
                "欢迎使用Self Brain AGI系统": "Добро пожаловать в систему Self Brain AGI",
                "仪表盘": "Приборная панель",
                "模型训练": "Обучение модели",
                "系统设置": "Системные настройки",
                "模型管理": "Управление моделью",
                "帮助": "Помощь",
                "语言": "Язык",
                "选择语言": "Выбрать язык",
                "管理模型": "Модель управления",
                "大语言模型": "Большая языковая модель",
                "音频处理模型": "Модель обработки аудио",
                "图片视觉处理模型": "Модель обработки изображений",
                "视频流视觉处理模型": "Модель обработки видео",
                "双目空间定位感知模型": "Модель пространственного восприятия",
                "传感器感知模型": "Модель обработки сенсоров",
                "计算机控制模型": "Модель управления компьютером",
                "知识库专家模型": "Модель экспертной базы знаний",
                "运动和执行器控制模型": "Модель управления движением",
                "编程模型": "Модель программирования",
                "无效的语言代码": "Неверный код языка",
                "语言切换成功": "Язык успешно изменен",
                "摄像头": "Камера",
                "麦克风": "Микрофон",
                "单独训练": "Индивидуальное обучение",
                "联合训练": "Совместное обучение",
                "概览": "Обзор",
                "性能监控": "Мониторинг производительности"
            }
        }
        
        return translation_map.get(target_language, {}).get(text, text)

    def get_string(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """Get localized string"""
        lang = language or self.current_language
        
        if lang not in self.language_resources:
            self._load_language_resources(lang)
            
        resource_dict = self.language_resources.get(lang, {})
        template = resource_dict.get(key, key)
        
        try:
            if kwargs:
                return template.format(**kwargs)
            return template
        except Exception:
            return template

    def _load_language_resources(self, language: str):
        """Load specified language resources"""
        if language not in self.supported_languages:
            self.logger.warning(f"Unsupported language: {language}")
            return False
            
        resource_file = os.path.join(self.resource_dir, f"{language}.json")
        
        try:
            if os.path.exists(resource_file):
                with open(resource_file, 'r', encoding='utf-8') as f:
                    self.language_resources[language] = json.load(f)
            else:
                self.language_resources[language] = self._get_default_resources(language)
                self._save_language_resources(language)
                
            return True
        except Exception as e:
            self.logger.error(f"Failed to load language resources {language}: {e}")
            return False

    def _save_language_resources(self, language: str):
        """Save language resources to file"""
        if language not in self.supported_languages:
            return False
            
        resource_file = os.path.join(self.resource_dir, f"{language}.json")
        
        try:
            with open(resource_file, 'w', encoding='utf-8') as f:
                json.dump(self.language_resources[language], f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to save language resources {language}: {e}")
            return False

    def switch_language(self, new_language: str):
        """Switch system language"""
        if new_language not in self.supported_languages:
            self.logger.warning(f"Unsupported language: {new_language}")
            return False
            
        if new_language != self.current_language:
            self.current_language = new_language
            self.logger.info(f"System language switched to: {new_language}")
            
            # Call all registered callbacks
            for callback in self.language_change_callbacks:
                try:
                    callback(new_language)
                except Exception as e:
                    self.logger.error(f"Language switch callback failed: {e}")
            
            return True
        return False

    def register_language_callback(self, callback):
        """Register language change callback function"""
        if callable(callback):
            self.language_change_callbacks.append(callback)
            return True
        return False

    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages"""
        return [{'code': lang, 'name': self.language_names.get(lang, {}).get('name', lang)} 
                for lang in self.supported_languages]

    def create_language_selector_html(self) -> str:
        """Create language selector HTML code"""
        languages = self.get_supported_languages()
        options = []
        
        for lang in languages:
            selected = 'selected' if lang['code'] == self.current_language else ''
            options.append(f'<option value="{lang["code"]}" {selected}>{lang["name"]}</option>')
        
        selector_html = f'''
        <select id="languageSelector" onchange="switchLanguage(this.value)" 
                style="padding: 5px; margin: 5px; border-radius: 4px; border: 1px solid #ccc;">
            {''.join(options)}
        </select>
        '''
        
        return selector_html

    def get_current_language(self) -> str:
        """Get current language code"""
        return self.current_language

    def get_language_name(self, language_code: str) -> str:
        """Get language name"""
        return self.language_names.get(language_code, {}).get('name', language_code)

    def add_resource(self, key: str, translations: Dict[str, str]):
        """Add or update resource item"""
        for lang, text in translations.items():
            if lang in self.supported_languages:
                if lang not in self.language_resources:
                    self.language_resources[lang] = {}
                self.language_resources[lang][key] = text
                
        # Save to files
        for lang in translations.keys():
            if lang in self.supported_languages:
                self._save_language_resources(lang)

    def export_resources(self, export_dir: str):
        """Export all resource files"""
        os.makedirs(export_dir, exist_ok=True)
        
        for lang in self.supported_languages:
            if lang in self.language_resources:
                export_file = os.path.join(export_dir, f"{lang}.json")
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(self.language_resources[lang], f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Resource files exported to: {export_dir}")

# Singleton instance
_multilingual_manager_instance = None

def get_multilingual_manager(resource_dir: str = "language_resources") -> UnifiedMultilingualManager:
    """Get multilingual manager instance"""
    global _multilingual_manager_instance
    if _multilingual_manager_instance is None:
        _multilingual_manager_instance = UnifiedMultilingualManager(resource_dir=resource_dir)
    return _multilingual_manager_instance

if __name__ == "__main__":
    # Test multilingual manager
    logging.basicConfig(level=logging.INFO)
    
    manager = UnifiedMultilingualManager()
    
    # Test getting strings
    print("Current language:", manager.get_current_language())
    print("Welcome message:", manager.get_string('ui.welcome'))
    print("Help text:", manager.get_string('ui.help'))
    
    # Test language switching
    manager.switch_language('en')
    print("Current language:", manager.get_current_language())
    print("Welcome message:", manager.get_string('ui.welcome'))
    print("Help text:", manager.get_string('ui.help'))
    
    
    # Return to Chinese
    manager.switch_language('zh')
    print("Final language:", manager.get_current_language())
