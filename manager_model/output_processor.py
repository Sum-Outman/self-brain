# Copyright 2025 AGI System Team
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

# 输出处理器 | Output Processor
import base64
import json
from typing import Dict, Any
from .language_resources import get_string

class OutputProcessor:
    def __init__(self, language='zh'):
        self.language = language
        # 输出风格映射 | Output style mapping
        self.output_styles = {
            "neutral": {"tone": "professional", "complexity": "medium"},
            "happy": {"tone": "friendly", "complexity": "simple"},
            "sad": {"tone": "compassionate", "complexity": "simple"},
            "angry": {"tone": "direct", "complexity": "simple"},
            "excited": {"tone": "enthusiastic", "complexity": "simple"},
            "calm": {"tone": "reassuring", "complexity": "medium"}
        }
        # 情感表情映射 | Emotion emoji mapping
        self.emotion_emojis = {
            "neutral": "😐",
            "happy": "😊",
            "sad": "😢",
            "angry": "😠",
            "excited": "🤩",
            "calm": "😌"
        }
    
    def set_language(self, lang: str):
        """设置处理器语言 | Set processor language"""
        if lang not in ['zh', 'en']:
            raise ValueError(get_string("unsupported_language", self.language).format(lang=lang))
        self.language = lang
    
    def process_output(self, output_data: Dict, system_emotion: str, output_emotion: str) -> Dict:
        """处理模型输出并生成最终响应 | Process model output and generate final response
        参数:
            output_data: 模型输出数据 | Model output data
            system_emotion: 系统当前情感状态 | Current system emotion state
            output_emotion: 输出内容的情感 | Emotion in output content
        返回:
            最终用户响应字典 | Final user response dictionary
        """
        # 获取输出风格 | Get output style
        style = self.output_styles.get(system_emotion, self.output_styles["neutral"])
        
        # 生成基础文本响应 | Generate base text response
        text_response = self._generate_text_response(output_data, style)
        
        # 添加情感表情 | Add emotion emoji
        emoji = self.emotion_emojis.get(output_emotion, "")
        if emoji:
            text_response = f"{emoji} {text_response}"
        
        # 构建响应对象 | Build response object
        response = {
            "text": text_response,
            "emotion": output_emotion,
            "system_emotion": system_emotion,
            "language": self.language,
            "timestamp": time.time()
        }
        
        # 添加多媒体输出（如果请求）| Add multimedia output if requested
        if "output_format" in output_data:
            if "audio" in output_data["output_format"]:
                response["audio"] = self._generate_audio_output(text_response, output_emotion)
            if "image" in output_data["output_format"]:
                response["image"] = self._generate_image_output(text_response, output_emotion)
        
        return response
    
    def _generate_text_response(self, output_data: Dict, style: Dict) -> str:
        """生成文本响应 | Generate text response
        参数:
            output_data: 模型输出数据 | Model output data
            style: 输出风格 | Output style
        返回:
            格式化文本 | Formatted text
        """
        # 获取基础内容 | Get base content
        content = output_data.get("content", "")
        
        # 根据风格调整内容 | Adjust content based on style
        if style["tone"] == "friendly":
            prefix = get_string("friendly_prefix", self.language)
            content = f"{prefix} {content}"
        elif style["tone"] == "compassionate":
            prefix = get_string("compassionate_prefix", self.language)
            content = f"{prefix} {content}"
        elif style["tone"] == "direct":
            content = f"❗ {content}"
        
        # 简化复杂内容（如果需要）| Simplify complex content if needed
        if style["complexity"] == "simple" and len(content.split()) > 20:
            # 在实际系统中，这里会使用摘要模型 | In real system, use summarization model
            content = content[:150] + "..." if len(content) > 150 else content
        
        return content
    
    def _generate_audio_output(self, text: str, emotion: str) -> Dict:
        """生成音频输出 | Generate audio output
        参数:
            text: 要转换为语音的文本 | Text to convert to speech
            emotion: 情感状态 | Emotion state
        返回:
            音频数据字典 | Audio data dictionary
        """
        # 在实际系统中，这里会调用音频处理模型 | In real system, call audio processing model
        # 返回占位符 | Return placeholder
        return {
            "format": "wav",
            "duration": len(text) * 0.1,  # 估计持续时间 | Estimated duration
            "emotion": emotion,
            "placeholder": True,
            "message": get_string("audio_placeholder", self.language)
        }
    
    def _generate_image_output(self, text: str, emotion: str) -> Dict:
        """生成图像输出 | Generate image output
        参数:
            text: 图像描述文本 | Image description text
            emotion: 情感状态 | Emotion state
        返回:
            图像数据字典 | Image data dictionary
        """
        # 在实际系统中，这里会调用图像生成模型 | In real system, call image generation model
        # 返回占位符 | Return placeholder
        return {
            "format": "png",
            "width": 256,
            "height": 256,
            "emotion": emotion,
            "placeholder": True,
            "message": get_string("image_placeholder", self.language)
        }
    
    def generate_error_response(self, error_message: str) -> Dict:
        """生成错误响应 | Generate error response
        参数:
            error_message: 错误消息 | Error message
        返回:
            错误响应字典 | Error response dictionary
        """
        return {
            "error": True,
            "message": error_message,
            "language": self.language,
            "suggestions": [
                get_string("error_suggestion_retry", self.language),
                get_string("error_suggestion_check_input", self.language),
                get_string("error_suggestion_contact_support", self.language)
            ],
            "timestamp": time.time()
        }
    
    def format_output(self, output_data: Dict, format_type: str = "text") -> Any:
        """格式化输出为指定类型 | Format output to specified type
        参数:
            output_data: 输出数据 | Output data
            format_type: 输出格式 (text/json/html) | Output format
        返回:
            格式化后的输出 | Formatted output
        """
        if format_type == "json":
            return json.dumps(output_data, ensure_ascii=False, indent=2)
        
        if format_type == "html":
            return self._format_html_output(output_data)
        
        # 默认返回文本 | Default to text
        return output_data.get("text", get_string("no_text_output", self.language))
    
    def _format_html_output(self, output_data: Dict) -> str:
        """将输出格式化为HTML | Format output as HTML
        参数:
            output_data: 输出数据 | Output data
        返回:
            HTML字符串 | HTML string
        """
        emotion = output_data.get("emotion", "neutral")
        emoji = self.emotion_emojis.get(emotion, "")
        
        html = f"""
        <div class="system-output" data-emotion="{emotion}">
            <div class="emotion-indicator">{emoji}</div>
            <div class="output-content">
                <p>{output_data.get('text', '')}</p>
        """
        
        if "audio" in output_data:
            html += f"""
                <div class="audio-output">
                    <p>{get_string("audio_output_available", self.language)}</p>
                </div>
            """
        
        if "image" in output_data:
            html += f"""
                <div class="image-output">
                    <p>{get_string("image_output_available", self.language)}</p>
                </div>
            """
        
        html += """
            </div>
        </div>
        """
        return html
