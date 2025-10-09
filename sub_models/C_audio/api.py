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
import requests
import time

app = Flask(__name__)
model = AudioModel()

# 外部API配置与状态管理
EXTERNAL_API_CONFIG = {
    'enabled': False,
    'url': 'http://localhost:5015/external/audio',
    'api_key': '',
    'timeout': 10,
    'health_check_interval': 60,
    'last_health_check': 0,
    'health_status': False
}

# 本地模型状态
MODEL_STATUS = {
    'initialized': True,
    'active_tasks': 0,
    'last_request_time': time.time(),
    'total_requests': 0
}

# 外部API健康检查
def check_external_api_health():
    current_time = time.time()
    if current_time - EXTERNAL_API_CONFIG['last_health_check'] < EXTERNAL_API_CONFIG['health_check_interval']:
        return EXTERNAL_API_CONFIG['health_status']
    
    try:
        headers = {'Authorization': f'Bearer {EXTERNAL_API_CONFIG['api_key']}'} if EXTERNAL_API_CONFIG['api_key'] else {}
        response = requests.get(f'{EXTERNAL_API_CONFIG['url']}/health', headers=headers, timeout=5)
        EXTERNAL_API_CONFIG['health_status'] = response.status_code == 200
    except Exception:
        EXTERNAL_API_CONFIG['health_status'] = False
        
    EXTERNAL_API_CONFIG['last_health_check'] = current_time
    return EXTERNAL_API_CONFIG['health_status']

# 调用外部API
def call_external_api(endpoint, data=None, files=None):
    if not EXTERNAL_API_CONFIG['enabled'] or not check_external_api_health():
        return None, False
    
    try:
        url = f'{EXTERNAL_API_CONFIG['url']}{endpoint}'
        headers = {'Authorization': f'Bearer {EXTERNAL_API_CONFIG['api_key']}'} if EXTERNAL_API_CONFIG['api_key'] else {}
        
        if files:
            response = requests.post(url, data=data, files=files, headers=headers, timeout=EXTERNAL_API_CONFIG['timeout'])
        else:
            response = requests.post(url, json=data, headers=headers, timeout=EXTERNAL_API_CONFIG['timeout'])
            
        if response.status_code == 200:
            return response.json(), True
        return None, False
    except Exception as e:
        print(f'External API error: {str(e)}')
        return None, False

# 健康检查接口
@app.route('/health', methods=['GET'])
def health():
    """健康检查接口 | Health Check API"""
    external_health = check_external_api_health() if EXTERNAL_API_CONFIG['enabled'] else 'disabled'
    
    return jsonify({
        'status': 'healthy',
        'model': 'C_audio',
        'local_status': MODEL_STATUS,
        'external_api': {
            'enabled': EXTERNAL_API_CONFIG['enabled'],
            'health': external_health
        }
    })

# 状态查询接口
@app.route('/status', methods=['GET'])
def status():
    """获取模型运行状态 | Get Model Status"""
    return jsonify({
        'model': 'C_audio',
        'status': MODEL_STATUS,
        'config': EXTERNAL_API_CONFIG
    })

# 配置接口
@app.route('/config', methods=['POST'])
def config():
    """配置外部API设置 | Configure External API Settings"""
    try:
        data = request.json
        if data:
            if 'enabled' in data:
                EXTERNAL_API_CONFIG['enabled'] = bool(data['enabled'])
            if 'url' in data:
                EXTERNAL_API_CONFIG['url'] = data['url']
            if 'api_key' in data:
                EXTERNAL_API_CONFIG['api_key'] = data['api_key']
            if 'timeout' in data:
                EXTERNAL_API_CONFIG['timeout'] = int(data['timeout'])
            
            # 立即进行健康检查
            check_external_api_health()
            
        return jsonify({
            'status': 'success',
            'message': 'Configuration updated',
            'config': EXTERNAL_API_CONFIG
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/recognize', methods=['POST'])
def recognize():
    """识别音频内容API | Recognize audio content API"""
    MODEL_STATUS['total_requests'] += 1
    MODEL_STATUS['last_request_time'] = time.time()
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    temp_path = f"/tmp/{uuid.uuid4()}.wav"
    audio_file.save(temp_path)
    
    try:
        # 检查是否使用外部API
        if EXTERNAL_API_CONFIG['enabled']:
            # 尝试调用外部API
            files = {'audio': (audio_file.filename, open(temp_path, 'rb'), 'audio/wav')}
            result, success = call_external_api('/recognize', files=files)
            
            if success and result:
                return jsonify(result)
            
        # 如果外部API不可用或失败，使用本地模型
        result = model.recognize_audio(temp_path)
        return jsonify(result)
    finally:
        os.remove(temp_path)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    """合成拟声语言API | Synthesize speech API"""
    MODEL_STATUS['total_requests'] += 1
    MODEL_STATUS['last_request_time'] = time.time()
    
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON format'}), 400
    
    data = request.json
    text = data.get('text')
    emotion = data.get('emotion', 'neutral')
    language = data.get('language', 'en')
    
    if not text:
        return jsonify({'error': 'Text is required | 需要提供文本'}), 400
    
    try:
        # 检查是否使用外部API
        if EXTERNAL_API_CONFIG['enabled']:
            # 尝试调用外部API
            result, success = call_external_api('/synthesize', {
                'text': text,
                'emotion': emotion,
                'language': language
            })
            
            if success and result and result.get('status') == 'success':
                # 注意：这里简化处理，实际可能需要处理返回的音频数据
                # 由于外部API返回格式不确定，这里可能需要根据实际情况调整
                return jsonify(result)
        
        # 如果外部API不可用或失败，使用本地模型
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
    MODEL_STATUS['total_requests'] += 1
    MODEL_STATUS['last_request_time'] = time.time()
    
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON format'}), 400
    
    data = request.json
    operation = data.get('operation')  # 'recognize' or 'synthesize'
    genre = data.get('genre', 'pop')
    duration = data.get('duration', 30)
    
    if not operation:
        return jsonify({'error': 'Operation is required (recognize or synthesize) | 需要指定操作类型'}), 400
    
    try:
        # 检查是否使用外部API
        if EXTERNAL_API_CONFIG['enabled']:
            if operation == 'recognize':
                if 'audio' not in request.files:
                    return jsonify({'error': 'Audio file required for recognition | 识别需要音频文件'}), 400
                
                audio_file = request.files['audio']
                temp_path = f"/tmp/{uuid.uuid4()}.wav"
                audio_file.save(temp_path)
                
                try:
                    # 尝试调用外部API
                    files = {'audio': (audio_file.filename, open(temp_path, 'rb'), 'audio/wav')}
                    result, success = call_external_api('/music', {'operation': operation}, files=files)
                    
                    if success and result:
                        return jsonify(result)
                finally:
                    os.remove(temp_path)
            else:
                # 尝试调用外部API进行合成
                result, success = call_external_api('/music', {
                    'operation': operation,
                    'genre': genre,
                    'duration': duration
                })
                
                if success and result:
                    # 注意：这里简化处理，实际可能需要处理返回的音频数据
                    # 由于外部API返回格式不确定，这里可能需要根据实际情况调整
                    return jsonify(result)
        
        # 如果外部API不可用或失败，使用本地模型
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
    MODEL_STATUS['total_requests'] += 1
    MODEL_STATUS['last_request_time'] = time.time()
    
    if not request.is_json:
        return jsonify({'error': 'Request must be JSON format'}), 400
    
    data = request.json
    effect_type = data.get('type')  # 'apply' or 'generate'
    effect_name = data.get('effect')
    duration = data.get('duration', 3)
    
    if not effect_type or not effect_name:
        return jsonify({'error': 'Type and effect name are required | 需要指定类型和特效名称'}), 400
    
    try:
        # 检查是否使用外部API
        if EXTERNAL_API_CONFIG['enabled']:
            if effect_type == 'apply':
                if 'audio' not in request.files:
                    return jsonify({'error': 'Audio file required for effect application | 应用特效需要音频文件'}), 400
                
                audio_file = request.files['audio']
                temp_path = f"/tmp/{uuid.uuid4()}.wav"
                audio_file.save(temp_path)
                
                try:
                    # 尝试调用外部API
                    files = {'audio': (audio_file.filename, open(temp_path, 'rb'), 'audio/wav')}
                    result, success = call_external_api('/effects', {
                        'type': effect_type,
                        'effect': effect_name,
                        'duration': duration
                    }, files=files)
                    
                    if success and result and result.get('status') == 'success':
                        # 注意：这里简化处理，实际可能需要处理返回的音频数据
                        # 由于外部API返回格式不确定，这里可能需要根据实际情况调整
                        return jsonify(result)
                finally:
                    os.remove(temp_path)
            else:
                # 尝试调用外部API生成特效
                result, success = call_external_api('/effects', {
                    'type': effect_type,
                    'effect': effect_name,
                    'duration': duration
                })
                
                if success and result and result.get('status') == 'success':
                    # 注意：这里简化处理，实际可能需要处理返回的音频数据
                    # 由于外部API返回格式不确定，这里可能需要根据实际情况调整
                    return jsonify(result)
        
        # 如果外部API不可用或失败，使用本地模型
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

# 多波段音频分析接口
@app.route('/multiband', methods=['POST'])
def multiband_analysis():
    """多波段音频分析接口 | Multiband Audio Analysis API"""
    MODEL_STATUS['total_requests'] += 1
    MODEL_STATUS['last_request_time'] = time.time()
    
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    temp_path = f"/tmp/{uuid.uuid4()}.wav"
    audio_file.save(temp_path)
    
    try:
        # 检查是否使用外部API
        if EXTERNAL_API_CONFIG['enabled']:
            # 尝试调用外部API
            files = {'audio': (audio_file.filename, open(temp_path, 'rb'), 'audio/wav')}
            result, success = call_external_api('/multiband', files=files)
            
            if success and result:
                return jsonify(result)
        
        # 如果外部API不可用或失败，使用本地模型
        # 读取音频数据
        import librosa
        audio_data, sample_rate = librosa.load(temp_path, sr=None)
        
        # 执行多波段分析
        # 这里假设model有multiband_analysis方法，如果没有则需要实现
        if hasattr(model, 'multiband_analysis'):
            result = model.multiband_analysis(audio_data, sample_rate)
        else:
            # 简化的多波段分析实现
            n_fft = 2048
            hop_length = 512
            S = librosa.stft(audio_data, n_fft=n_fft, hop_length=hop_length)
            
            # 定义频段
            bands = [
                (0, 200, 'low'),
                (200, 2000, 'mid'),
                (2000, 8000, 'high'),
                (8000, sample_rate/2, 'ultra_high')
            ]
            
            # 计算每个频段的能量
            freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
            band_energies = {}
            
            for low, high, name in bands:
                indices = np.where((freq_bins >= low) & (freq_bins < high))[0]
                if len(indices) > 0:
                    band_energy = np.mean(np.abs(S[indices, :]))
                    band_energies[name] = float(band_energy)
            
            result = {
                'bands': band_energies,
                'sample_rate': sample_rate,
                'duration': len(audio_data) / sample_rate
            }
            
        return jsonify(result)
    finally:
        os.remove(temp_path)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5002))
    app.run(host='0.0.0.0', port=port, debug=True)
