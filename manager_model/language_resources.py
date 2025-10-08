# Copyright 2025 Self Brain AGI System Authors
# Licensed under the Apache License, Version 2.0 (the "License");
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

# 多语言资源管理器 - 支持五种语言
# Multilingual Resource Manager - Supports five languages

import json
import os
from pathlib import Path

class InternationalizationManager:
    """全面的多语言管理系统 | Comprehensive internationalization management system"""
    
    def __init__(self, default_language='zh'):
        """
        初始化多语言管理器 | Initialize multilingual manager
        
        参数:
            default_language: 默认语言代码 (zh/en/de/ja/ru)
        Parameters:
            default_language: Default language code (zh/en/de/ja/ru)
        """
        self.supported_languages = ['zh', 'en']
        self.current_language = default_language
        self.translations = {}
        self.language_change_callbacks = []
        
        # 加载所有语言资源 | Load all language resources
        self._load_all_translations()
        
    def _load_all_translations(self):
        """加载所有翻译资源 | Load all translation resources"""
        base_path = Path(__file__).parent.parent / 'language_resources'
        
        for lang in self.supported_languages:
            lang_file = base_path / f'{lang}.json'
            try:
                if lang_file.exists():
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        self.translations[lang] = json.load(f)
                else:
                    # 如果文件不存在，创建默认翻译 | Create default translations if file doesn't exist
                    self.translations[lang] = self._create_default_translations(lang)
                    self._save_translations(lang, self.translations[lang])
            except Exception as e:
                print(f"加载语言 {lang} 失败: {e} | Failed to load language {lang}: {e}")
                self.translations[lang] = self._create_default_translations(lang)
    
    def _create_default_translations(self, language):
        """创建默认翻译内容 | Create default translation content"""
        defaults = {
            'zh': {
                'system_name': 'Self Brain (我心)',
                'welcome_message': '欢迎使用Self Brain AGI系统',
                'language_changed': '语言已切换为中文',
                'model_management': '模型管理',
                'training_control': '训练控制',
                'system_settings': '系统设置',
                'help': '帮助',
                'dashboard': '仪表盘',
                'real_time_monitoring': '实时监控',
                'voice_input': '语音输入',
                'text_input': '文本输入',
                'start_training': '开始训练',
                'stop_training': '停止训练',
                'model_status': '模型状态',
                'connected': '已连接',
                'disconnected': '未连接',
                'training_in_progress': '训练中',
                'training_completed': '训练完成',
                'error': '错误'
            },
            'en': {
                'system_name': 'Self Brain',
                'welcome_message': 'Welcome to Self Brain AGI System',
                'language_changed': 'Language changed to English',
                'model_management': 'Model Management',
                'training_control': 'Training Control',
                'system_settings': 'System Settings',
                'help': 'Help',
                'dashboard': 'Dashboard',
                'real_time_monitoring': 'Real-time Monitoring',
                'voice_input': 'Voice Input',
                'text_input': 'Text Input',
                'start_training': 'Start Training',
                'stop_training': 'Stop Training',
                'model_status': 'Model Status',
                'connected': 'Connected',
                'disconnected': 'Disconnected',
                'training_in_progress': 'Training in Progress',
                'training_completed': 'Training Completed',
                'error': 'Error'
            }
        }
        
        return defaults.get(language, defaults['en'])
    
    def _save_translations(self, language, translations):
        """保存翻译资源到文件 | Save translation resources to file"""
        base_path = Path(__file__).parent.parent / 'language_resources'
        base_path.mkdir(exist_ok=True)
        
        lang_file = base_path / f'{language}.json'
        with open(lang_file, 'w', encoding='utf-8') as f:
            json.dump(translations, f, ensure_ascii=False, indent=2)
    
    def get_translation(self, key, language=None):
        """
        获取翻译文本 | Get translated text
        
        参数:
            key: 翻译键值
            language: 目标语言 (默认为当前语言)
        Parameters:
            key: Translation key
            language: Target language (defaults to current language)
        """
        lang = language or self.current_language
        if lang not in self.supported_languages:
            lang = 'en'  # 默认英语 | Default to English
        
        return self.translations[lang].get(key, key)  # 找不到时返回key本身
    
    def switch_language(self, new_language):
        """
        切换系统语言 | Switch system language
        
        参数:
            new_language: 新的语言代码
        Parameters:
            new_language: New language code
        """
        if new_language in self.supported_languages:
            self.current_language = new_language
            # 通知所有注册的回调函数 | Notify all registered callbacks
            for callback in self.language_change_callbacks:
                callback(new_language)
            return True
        return False
    
    def register_language_callback(self, callback):
        """
        注册语言切换回调函数 | Register language change callback
        
        参数:
            callback: 回调函数，接受一个参数（新语言代码）
        Parameters:
            callback: Callback function that accepts one parameter (new language code)
        """
        self.language_change_callbacks.append(callback)
    
    def add_translation(self, key, translations):
        """
        添加或更新翻译项 | Add or update translation item
        
        参数:
            key: 翻译键值
            translations: 各语言翻译的字典 {lang: translation}
        Parameters:
            key: Translation key
            translations: Dictionary of translations by language {lang: translation}
        """
        for lang, text in translations.items():
            if lang in self.supported_languages:
                self.translations[lang][key] = text
                # 保存到文件 | Save to file
                self._save_translations(lang, self.translations[lang])
    
    def get_all_translations(self, language=None):
        """
        获取指定语言的所有翻译 | Get all translations for specified language
        
        参数:
            language: 目标语言 (默认为当前语言)
        Parameters:
            language: Target language (defaults to current language)
        """
        lang = language or self.current_language
        if lang not in self.supported_languages:
            lang = 'en'
        
        return self.translations[lang]
    
    def get_supported_languages(self):
        """
        获取支持的语言列表 | Get list of supported languages
        
        返回:
            支持的语言代码列表
        Returns:
            List of supported language codes
        """
        return self.supported_languages

# 全局实例 | Global instance
_language_manager = None

def get_language_manager():
    """
    获取全局语言管理器实例 | Get global language manager instance
    
    返回:
        InternationalizationManager 实例
    Returns:
        InternationalizationManager instance
    """
    global _language_manager
    if _language_manager is None:
        _language_manager = InternationalizationManager()
    return _language_manager

def get_string(key, language=None):
    """
    便捷函数：获取翻译文本 | Convenience function: Get translated text
    
    参数:
        key: 翻译键值
        language: 目标语言
    Parameters:
        key: Translation key
        language: Target language
    """
    manager = get_language_manager()
    return manager.get_translation(key, language)

def switch_language(new_language):
    """
    便捷函数：切换系统语言 | Convenience function: Switch system language
    
    参数:
        new_language: 新的语言代码
    Parameters:
        new_language: New language code
    """
    manager = get_language_manager()
    return manager.switch_language(new_language)

# 测试代码 | Test code
if __name__ == '__main__':
    # 测试多语言功能 | Test multilingual functionality
    manager = InternationalizationManager()
    
    print("当前语言:", manager.current_language)
    print("欢迎消息:", manager.get_translation('welcome_message'))
    
    # 切换语言测试 | Language switching test
    manager.switch_language('en')
    print("当前语言:", manager.current_language)
    print("欢迎消息:", manager.get_translation('welcome_message'))
    
    manager.switch_language('ja')
    print("当前语言:", manager.current_language)
    print("欢迎消息:", manager.get_translation('welcome_message'))
