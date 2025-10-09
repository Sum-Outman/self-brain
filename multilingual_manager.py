# 多语言资源管理器 | Multilingual Resource Manager
# Copyright 2025 Self Brain AGI System Team
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

import json
import os
import logging
from typing import Dict, List, Any, Optional

class MultilingualResourceManager:
    """多语言资源管理器 - 支持中文、英文、德语、日语、俄语
    Multilingual Resource Manager - Supports Chinese, English, German, Japanese, Russian
    """
    
    def __init__(self, resource_dir: str = "language_resources"):
        """
        初始化多语言资源管理器 | Initialize multilingual resource manager
        
        参数 | Parameters:
            resource_dir: 资源文件目录 | Resource file directory
        """
        self.supported_languages = ['zh', 'en']
        self.language_names = {
            'zh': '中文',
            'en': 'English', 
        }
        self.resources = {}
        self.current_language = 'zh'
        self.resource_dir = resource_dir
        self.logger = logging.getLogger(__name__)
        
        # 语言切换回调函数列表 | Language change callback list
        self.language_change_callbacks = []
        
        # 确保资源目录存在 | Ensure resource directory exists
        os.makedirs(resource_dir, exist_ok=True)
        
        self.logger.info("多语言资源管理器初始化 | Multilingual resource manager initialized")

    def load_all_resources(self):
        """加载所有语言资源 | Load all language resources"""
        try:
            for lang in self.supported_languages:
                self.load_language_resources(lang)
            self.logger.info("所有语言资源加载完成 | All language resources loaded")
        except Exception as e:
            self.logger.error(f"加载语言资源失败: {e} | Failed to load language resources: {e}")

    def load_language_resources(self, language: str):
        """
        加载指定语言资源 | Load specified language resources
        
        参数 | Parameters:
            language: 语言代码 (zh/en/de/ja/ru) | Language code
        """
        if language not in self.supported_languages:
            self.logger.warning(f"不支持的语言: {language} | Unsupported language: {language}")
            return False
            
        resource_file = os.path.join(self.resource_dir, f"{language}.json")
        
        try:
            if os.path.exists(resource_file):
                with open(resource_file, 'r', encoding='utf-8') as f:
                    self.resources[language] = json.load(f)
            else:
                # 创建默认资源文件 | Create default resource file
                self.resources[language] = self._create_default_resources(language)
                self._save_language_resources(language)
                
            self.logger.info(f"语言资源加载成功: {language} | Language resources loaded: {language}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载语言资源失败 {language}: {e} | Failed to load language resources {language}: {e}")
            return False

    def _create_default_resources(self, language: str) -> Dict[str, str]:
        """
        创建默认语言资源 | Create default language resources
        
        参数 | Parameters:
            language: 语言代码 | Language code
            
        返回 | Returns:
            默认资源字典 | Default resource dictionary
        """
        default_resources = {
            'zh': {
                'welcome': '欢迎使用Self Brain AGI系统',
                'model_training': '模型训练',
                'system_settings': '系统设置',
                'model_management': '模型管理',
                'help': '帮助',
                'language': '语言',
                'dashboard': '仪表盘',
                'training_control': '训练控制',
                'real_time_monitoring': '实时监控',
                'knowledge_base': '知识库',
                'api_configuration': 'API配置',
                'sensor_interface': '传感器接口',
                'video_interface': '视频接口',
                'audio_interface': '音频接口',
                'connect': '连接',
                'disconnect': '断开',
                'status': '状态',
                'configure': '配置',
                'start': '开始',
                'stop': '停止',
                'save': '保存',
                'cancel': '取消',
                'success': '成功',
                'error': '错误',
                'loading': '加载中...',
                'please_wait': '请稍候...'
            },
            'en': {
                'welcome': 'Welcome to Self Brain AGI System',
                'model_training': 'Model Training',
                'system_settings': 'System Settings',
                'model_management': 'Model Management',
                'help': 'Help',
                'language': 'Language',
                'dashboard': 'Dashboard',
                'training_control': 'Training Control',
                'real_time_monitoring': 'Real-time Monitoring',
                'knowledge_base': 'Knowledge Base',
                'api_configuration': 'API Configuration',
                'sensor_interface': 'Sensor Interface',
                'video_interface': 'Video Interface',
                'audio_interface': 'Audio Interface',
                'connect': 'Connect',
                'disconnect': 'Disconnect',
                'status': 'Status',
                'configure': 'Configure',
                'start': 'Start',
                'stop': 'Stop',
                'save': 'Save',
                'cancel': 'Cancel',
                'success': 'Success',
                'error': 'Error',
                'loading': 'Loading...',
                'please_wait': 'Please wait...'
            }
        }
        
        return default_resources.get(language, {})

    def _save_language_resources(self, language: str):
        """
        保存语言资源到文件 | Save language resources to file
        
        参数 | Parameters:
            language: 语言代码 | Language code
        """
        if language not in self.supported_languages:
            return False
            
        resource_file = os.path.join(self.resource_dir, f"{language}.json")
        
        try:
            with open(resource_file, 'w', encoding='utf-8') as f:
                json.dump(self.resources[language], f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"保存语言资源失败 {language}: {e} | Failed to save language resources {language}: {e}")
            return False

    def get_string(self, key: str, language: Optional[str] = None, **kwargs) -> str:
        """
        获取本地化字符串 | Get localized string
        
        参数 | Parameters:
            key: 资源键 | Resource key
            language: 语言代码 (可选) | Language code (optional)
            **kwargs: 格式化参数 | Formatting parameters
            
        返回 | Returns:
            本地化字符串 | Localized string
        """
        lang = language or self.current_language
        
        if lang not in self.resources:
            self.load_language_resources(lang)
            
        resource_dict = self.resources.get(lang, {})
        template = resource_dict.get(key, key)
        
        try:
            if kwargs:
                return template.format(**kwargs)
            return template
        except Exception:
            return template

    def switch_language(self, new_language: str):
        """
        切换系统语言 | Switch system language
        
        参数 | Parameters:
            new_language: 新语言代码 | New language code
        """
        if new_language not in self.supported_languages:
            self.logger.warning(f"不支持的语言: {new_language} | Unsupported language: {new_language}")
            return False
            
        if new_language != self.current_language:
            self.current_language = new_language
            self.logger.info(f"系统语言已切换至: {new_language} | System language switched to: {new_language}")
            
            # 调用所有注册的回调函数 | Call all registered callbacks
            for callback in self.language_change_callbacks:
                try:
                    callback(new_language)
                except Exception as e:
                    self.logger.error(f"语言切换回调失败: {e} | Language change callback failed: {e}")
            
            return True
        return False

    def register_language_callback(self, callback):
        """
        注册语言切换回调函数 | Register language change callback function
        
        参数 | Parameters:
            callback: 回调函数 | Callback function
        """
        if callable(callback):
            self.language_change_callbacks.append(callback)
            return True
        return False

    def get_supported_languages(self) -> List[Dict[str, str]]:
        """
        获取支持的语言列表 | Get list of supported languages
        
        返回 | Returns:
            支持的语言列表 | List of supported languages
        """
        return [{'code': lang, 'name': self.language_names.get(lang, lang)} 
                for lang in self.supported_languages]

    def create_language_selector_html(self) -> str:
        """
        创建语言选择器HTML代码 | Create language selector HTML code
        
        返回 | Returns:
            HTML代码字符串 | HTML code string
        """
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
        """
        获取当前语言代码 | Get current language code
        
        返回 | Returns:
            当前语言代码 | Current language code
        """
        return self.current_language

    def get_language_name(self, language_code: str) -> str:
        """
        获取语言名称 | Get language name
        
        参数 | Parameters:
            language_code: 语言代码 | Language code
            
        返回 | Returns:
            语言名称 | Language name
        """
        return self.language_names.get(language_code, language_code)

    def add_resource(self, key: str, translations: Dict[str, str]):
        """
        添加或更新资源项 | Add or update resource item
        
        参数 | Parameters:
            key: 资源键 | Resource key
            translations: 多语言翻译字典 | Multilingual translation dictionary
        """
        for lang, text in translations.items():
            if lang in self.supported_languages:
                if lang not in self.resources:
                    self.resources[lang] = {}
                self.resources[lang][key] = text
                
        # 保存到文件 | Save to files
        for lang in translations.keys():
            if lang in self.supported_languages:
                self._save_language_resources(lang)

    def export_resources(self, export_dir: str):
        """
        导出所有资源文件 | Export all resource files
        
        参数 | Parameters:
            export_dir: 导出目录 | Export directory
        """
        os.makedirs(export_dir, exist_ok=True)
        
        for lang in self.supported_languages:
            if lang in self.resources:
                export_file = os.path.join(export_dir, f"{lang}.json")
                with open(export_file, 'w', encoding='utf-8') as f:
                    json.dump(self.resources[lang], f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"资源文件已导出到: {export_dir} | Resource files exported to: {export_dir}")


# 单例实例 | Singleton instance
_multilingual_manager_instance = None

def get_multilingual_manager(resource_dir: str = "language_resources") -> MultilingualResourceManager:
    """
    获取多语言管理器实例 | Get multilingual manager instance
    
    参数 | Parameters:
        resource_dir: 资源文件目录 | Resource file directory
        
    返回 | Returns:
        多语言管理器实例 | Multilingual manager instance
    """
    global _multilingual_manager_instance
    if _multilingual_manager_instance is None:
        _multilingual_manager_instance = MultilingualResourceManager(resource_dir)
        _multilingual_manager_instance.load_all_resources()
    return _multilingual_manager_instance


if __name__ == "__main__":
    # 测试多语言管理器 | Test multilingual manager
    logging.basicConfig(level=logging.INFO)
    
    manager = MultilingualResourceManager()
    manager.load_all_resources()
    
    # 测试获取字符串 | Test getting strings
    print("当前语言:", manager.get_current_language())
    print("欢迎消息:", manager.get_string('welcome'))
    print("帮助文本:", manager.get_string('help'))
    
    # 测试语言切换 | Test language switching
    manager.switch_language('en')
    print("当前语言:", manager.get_current_language())
    print("欢迎消息:", manager.get_string('welcome'))
    print("帮助文本:", manager.get_string('help'))
    
    # 返回中文 | Return to Chinese
    manager.switch_language('zh')
    print("最终语言:", manager.get_current_language())