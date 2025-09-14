# Copyright 2025 The AI Management System Authors
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

# 音频处理模型API服务
# Audio Processing Model API Service

from flask import Flask, request, jsonify, send_file
from .model import AudioModel
import os
import uuid
import librosa
import soundfile as sf

app = Flask(__name__)
model = AudioModel()

@app.route('/recognize', methods=['POST'])
def recognize():
    """识别音频内容API | Recognize audio content API"""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    temp_path = f"/tmp/{uuid.uuid4()}.wav"
    audio_file.save(temp_path)
    
    try:
        result = model.recognize_audio(temp_path)
        return jsonify(result)
    finally:
        os.remove(temp_path)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """合成拟声语言API | Synthesize speech API"""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON format'}), 400
    
    data = request.json
    text = data.get('text')
    emotion = data.get('emotion', 'neutral')
    language = data.get('language', 'en')
    
    if not text:
        return jsonify({'error': 'Text is required | 需要提供文本'}), 400
    
    try:
        # 调用模型进行语音合成
        # Call model for speech synthesis
        result = model.synthesize_speech(text, emotion, language)
        
        if result['status'] == 'success':
            # 返回音频文件
            # Return audio file
            return send_file(result['file_path'], mimetype='audio/wav')
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/music', methods=['POST'])
def music():
    """音乐识别与合成API | Music recognition and synthesis API"""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON format'}), 400
    
    data = request.json
    operation = data.get('operation')  # 'recognize' or 'synthesize'
    genre = data.get('genre', 'pop')
    duration = data.get('duration', 30)
    
    if not operation:
        return jsonify({'error': 'Operation is required (recognize or synthesize) | 需要指定操作类型'}), 400
    
    try:
        if operation == 'recognize':
            if 'audio' not in request.files:
                return jsonify({'error': 'Audio file required for recognition | 识别需要音频文件'}), 400
            
            audio_file = request.files['audio']
            temp_path = f"/tmp/{uuid.uuid4()}.wav"
            audio_file.save(temp_path)
            
            result = model.process_music(temp_path, "recognize")
            os.remove(temp_path)
            
        elif operation == 'synthesize':
            result = model.process_music(None, "synthesize", genre, duration)
            
        else:
            return jsonify({'error': 'Invalid operation | 无效的操作类型'}), 400
            
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/effects', methods=['POST'])
def effects():
    """特效声音合成API | Sound effects synthesis API"""
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON format'}), 400
    
    data = request.json
    effect_type = data.get('type')  # 'apply' or 'generate'
    effect_name = data.get('effect')
    duration = data.get('duration', 3)
    
    if not effect_type or not effect_name:
        return jsonify({'error': 'Type and effect name are required | 需要指定类型和特效名称'}), 400
    
    try:
        if effect_type == 'apply':
            if 'audio' not in request.files:
                return jsonify({'error': 'Audio file required for effect application | 应用特效需要音频文件'}), 400
            
            audio_file = request.files['audio']
            temp_path = f"/tmp/{uuid.uuid4()}.wav"
            audio_file.save(temp_path)
            
            result = model.apply_effects(temp_path, [{'name': effect_name, 'value': 1.0}])
            os.remove(temp_path)
            
        elif effect_type == 'generate':
            # 生成特效声音
            # Generate special effect sound
            vocal_sounds = ['laugh', 'cry', 'sigh', 'gasp', 'moan', 'scream', 'whisper']
            if effect_name in vocal_sounds:
                import numpy as np
                import soundfile as sf
                
                audio_data = model.generate_vocal_sounds(effect_name, duration)
                output_path = f"/tmp/{uuid.uuid4()}.wav"
                sf.write(output_path, audio_data, 22050)
                
                result = {'status': 'success', 'file_path': output_path}
            else:
                return jsonify({'error': 'Unsupported effect for generation | 不支持生成此特效'}), 400
                
        else:
            return jsonify({'error': 'Invalid effect type | 无效的特效类型'}), 400
            
        if result['status'] == 'success':
            return send_file(result['file_path'], mimetype='audio/wav')
        else:
            return jsonify(result), 500
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
