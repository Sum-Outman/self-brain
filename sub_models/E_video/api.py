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

# 视频流视觉处理模型API服务
# Video Stream Visual Processing Model API Service

from flask import Flask, request, jsonify, send_file, make_response
from .model import VideoModel
import os
import uuid
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import requests
import logging
import time
from typing import Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("E_video_API")

app = Flask(__name__)
model = VideoModel()

# 外部API配置和状态管理
EXTERNAL_API_CONFIG = {
    'enabled': False,
    'url': 'http://localhost:5015/external/video',
    'api_key': '',
    'timeout': 30,
    'health_check_interval': 300,  # 5分钟检查一次健康状态
    'last_health_check': 0,
    'health_status': False
}

# 本地模型状态
def initialize_model_status():
    return {
        'status': 'initializing',
        'version': '1.0.0',
        'initialized_at': time.time(),
        'total_requests': 0,
        'total_recognize_requests': 0,
        'total_edit_requests': 0,
        'total_modify_requests': 0,
        'total_generate_requests': 0,
        'total_stream_requests': 0,
        'total_multi_stream_requests': 0,
        'last_request_time': None,
        'active_tasks': 0,
        'model_info': {
            'type': 'E_video',
            'name': 'Video Processing Model',
            'capabilities': ['recognition', 'editing', 'modification', 'generation', 'stream_processing', 'multi_stream_analysis']
        }
    }

MODEL_STATUS = initialize_model_status()

def check_external_api_health() -> bool:
    """检查外部API健康状态 | Check external API health status"""
    if not EXTERNAL_API_CONFIG['enabled'] or not EXTERNAL_API_CONFIG['url']:
        return False
    
    current_time = time.time()
    # 如果距离上次检查超过配置的间隔，重新检查
    if current_time - EXTERNAL_API_CONFIG['last_health_check'] > EXTERNAL_API_CONFIG['health_check_interval']:
        try:
            health_url = f"{EXTERNAL_API_CONFIG['url'].rstrip('/')}/health"
            response = requests.get(health_url, timeout=5)
            health_status = response.status_code == 200
            
            EXTERNAL_API_CONFIG['health_status'] = health_status
            EXTERNAL_API_CONFIG['last_health_check'] = current_time
            
            return health_status
        except Exception as e:
            logger.error(f"外部API健康检查失败: {str(e)}")
            EXTERNAL_API_CONFIG['health_status'] = False
            EXTERNAL_API_CONFIG['last_health_check'] = current_time
            return False
    
    return EXTERNAL_API_CONFIG['health_status']

@app.route('/health', methods=['GET'])
def health():
    """健康检查接口 | Health check endpoint"""
    external_health = check_external_api_health() if EXTERNAL_API_CONFIG['enabled'] else False
    return jsonify({
        'status': 'healthy',
        'local_model': True,
        'external_api': {
            'enabled': EXTERNAL_API_CONFIG['enabled'],
            'healthy': external_health
        },
        'timestamp': time.time()
    }), 200

@app.route('/status', methods=['GET'])
def status():
    """状态查询API | Status query API"""
    # 确保模型状态已初始化
    if MODEL_STATUS['status'] == 'initializing':
        MODEL_STATUS['status'] = 'running'
        
    # 获取外部API健康状态
    external_health = check_external_api_health() if EXTERNAL_API_CONFIG['enabled'] else False
    
    return jsonify({
        'model_id': 'E_video',
        'status': MODEL_STATUS['status'],
        'version': MODEL_STATUS['version'],
        'total_requests': MODEL_STATUS['total_requests'],
        'last_request_time': MODEL_STATUS['last_request_time'],
        'active_tasks': MODEL_STATUS['active_tasks'],
        'model_info': MODEL_STATUS['model_info'],
        'external_api': {
            'enabled': EXTERNAL_API_CONFIG['enabled'],
            'health': external_health,
            'last_health_check': EXTERNAL_API_CONFIG['last_health_check'],
            'health_check_interval': EXTERNAL_API_CONFIG['health_check_interval']
        },
        'request_stats': {
            'recognize': MODEL_STATUS['total_recognize_requests'],
            'edit': MODEL_STATUS['total_edit_requests'],
            'modify': MODEL_STATUS['total_modify_requests'],
            'generate': MODEL_STATUS['total_generate_requests'],
            'stream': MODEL_STATUS['total_stream_requests'],
            'multi_stream': MODEL_STATUS['total_multi_stream_requests']
        },
        'current_mode': 'external' if EXTERNAL_API_CONFIG['enabled'] and external_health else 'local'
    }), 200

@app.route('/config', methods=['GET', 'POST'])
def config():
    """配置API | Configuration API"""
    if request.method == 'GET':
        # 返回当前配置（不包含敏感信息）
        safe_config = {
            'enabled': EXTERNAL_API_CONFIG['enabled'],
            'url': EXTERNAL_API_CONFIG['url'] if EXTERNAL_API_CONFIG['enabled'] else '(hidden)',
            'timeout': EXTERNAL_API_CONFIG['timeout'],
            'health_check_interval': EXTERNAL_API_CONFIG['health_check_interval']
        }
        return jsonify({
            'status': 'success',
            'config': safe_config
        }), 200
    elif request.method == 'POST':
        try:
            data = request.json
            if not data:
                return jsonify({'error': 'No configuration data provided'}), 400
            
            # 更新外部API配置
            if 'enabled' in data:
                EXTERNAL_API_CONFIG['enabled'] = bool(data['enabled'])
            if 'url' in data:
                EXTERNAL_API_CONFIG['url'] = data['url']
            if 'api_key' in data:
                EXTERNAL_API_CONFIG['api_key'] = data['api_key']
            if 'timeout' in data:
                EXTERNAL_API_CONFIG['timeout'] = float(data['timeout'])
            if 'health_check_interval' in data:
                EXTERNAL_API_CONFIG['health_check_interval'] = float(data['health_check_interval'])
            
            # 立即检查健康状态
            EXTERNAL_API_CONFIG['last_health_check'] = 0
            check_external_api_health()
            
            logger.info("外部API配置已更新")
            return jsonify({
                'status': 'success',
                'message': 'External API configuration updated',
                'config': {
                    'enabled': EXTERNAL_API_CONFIG['enabled'],
                    'url': EXTERNAL_API_CONFIG['url'] if EXTERNAL_API_CONFIG['enabled'] else '(hidden)',
                    'timeout': EXTERNAL_API_CONFIG['timeout'],
                    'health_check_interval': EXTERNAL_API_CONFIG['health_check_interval']
                }
            }), 200
        except Exception as e:
            logger.error(f"配置更新失败: {str(e)}")
            return jsonify({'error': str(e)}), 500

def call_external_api(endpoint: str, method: str = 'post', **kwargs) -> Dict[str, Any]:
    """调用外部API | Call external API"""
    if not EXTERNAL_API_CONFIG['enabled'] or not check_external_api_health():
        raise Exception("External API is not enabled or not healthy")
    
    url = f"{EXTERNAL_API_CONFIG['url'].rstrip('/')}/{endpoint}"
    headers = {'Content-Type': 'application/json'}
    
    if EXTERNAL_API_CONFIG['api_key']:
        headers['Authorization'] = f"Bearer {EXTERNAL_API_CONFIG['api_key']}"
    
    try:
        if method.lower() == 'post':
            response = requests.post(url, headers=headers, timeout=EXTERNAL_API_CONFIG['timeout'], **kwargs)
        elif method.lower() == 'get':
            response = requests.get(url, headers=headers, timeout=EXTERNAL_API_CONFIG['timeout'], **kwargs)
        else:
            raise Exception(f"Unsupported method: {method}")
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"外部API调用失败: {str(e)}")
        # 更新外部API健康状态为失败
        EXTERNAL_API_CONFIG['health_status'] = False
        EXTERNAL_API_CONFIG['last_health_check'] = time.time()
        raise Exception(f"External API call failed: {str(e)}")

@app.route('/recognize', methods=['POST'])
def recognize():
    """识别视频内容API | Recognize video content API"""
    # 更新请求计数和状态
    with app.app_context():
        MODEL_STATUS['total_requests'] += 1
        MODEL_STATUS['total_recognize_requests'] += 1
        MODEL_STATUS['last_request_time'] = time.time()
        MODEL_STATUS['active_tasks'] += 1
        
    try:
        # 检查文件是否存在
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        # 检查是否应该使用外部API
        if EXTERNAL_API_CONFIG['enabled'] and check_external_api_health():
            try:
                # 准备调用外部API的数据
                files = {'video': request.files['video']} 
                data = request.form.to_dict() if request.form else None
                
                # 调用外部API
                result = call_external_api('recognize', files=files, data=data)
                return jsonify(result), 200
            except Exception as e:
                logger.warning(f"外部API调用失败，回退到本地处理: {str(e)}")
        
        # 本地处理
        video_file = request.files['video']
        temp_path = f"/tmp/{uuid.uuid4()}.mp4"
        
        try:
            video_file.save(temp_path)
            result = model.recognize_video(temp_path)
            return jsonify(result)
        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
    except Exception as e:
        logger.error(f"视频识别失败: {str(e)}")
        return jsonify({'error': f"Video recognition failed: {str(e)}"}), 500
    finally:
        # 减少活跃任务计数
        with app.app_context():
            MODEL_STATUS['active_tasks'] = max(0, MODEL_STATUS['active_tasks'] - 1)

@app.route('/edit', methods=['POST'])
def edit():
    """视频剪辑编辑API | Video editing API"""
    # 更新请求计数和状态
    with app.app_context():
        MODEL_STATUS['total_requests'] += 1
        MODEL_STATUS['total_edit_requests'] += 1
        MODEL_STATUS['last_request_time'] = time.time()
        MODEL_STATUS['active_tasks'] += 1
        
    try:
        # 检查文件是否存在
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        # 获取编辑参数
        start_time = float(request.form.get('start_time', 0))
        end_time = float(request.form.get('end_time', -1))
        output_format = request.form.get('format', 'mp4')
        
        # 检查是否应该使用外部API
        if EXTERNAL_API_CONFIG['enabled'] and check_external_api_health():
            try:
                # 准备调用外部API的数据
                files = {'video': request.files['video']} 
                data = {
                    'start_time': start_time,
                    'end_time': end_time,
                    'format': output_format
                }
                
                # 调用外部API
                result = call_external_api('edit', files=files, data=data)
                return jsonify(result), 200
            except Exception as e:
                logger.warning(f"外部API调用失败，回退到本地处理: {str(e)}")
        
        # 本地处理
        video_file = request.files['video']
        temp_input = f"/tmp/{uuid.uuid4()}.mp4"
        temp_output = f"/tmp/{uuid.uuid4()}.{output_format}"
        
        try:
            video_file.save(temp_input)
            
            # 使用moviepy进行视频剪辑
            clip = VideoFileClip(temp_input)
            total_duration = clip.duration
            
            if end_time < 0 or end_time > total_duration:
                end_time = total_duration
            
            # 剪辑视频
            edited_clip = clip.subclip(start_time, end_time)
            edited_clip.write_videofile(temp_output, codec="libx264")
            
            # 返回剪辑后的视频
            response = make_response(send_file(temp_output, as_attachment=True))
            response.headers["Content-Disposition"] = f"attachment; filename=edited_video.{output_format}"
            return response
        except Exception as e:
            logger.error(f"视频剪辑失败: {str(e)}")
            return jsonify({'error': f"Video editing failed: {str(e)}"}), 500
        finally:
            # 清理临时文件
            for f in [temp_input, temp_output]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
    except Exception as e:
        logger.error(f"视频编辑接口错误: {str(e)}")
        return jsonify({'error': f"Video edit endpoint error: {str(e)}"}), 500
    finally:
        # 减少活跃任务计数
        with app.app_context():
            MODEL_STATUS['active_tasks'] = max(0, MODEL_STATUS['active_tasks'] - 1)

@app.route('/modify', methods=['POST'])
def modify():
    """视频内容修改API | Video content modification API"""
    # 更新请求计数和状态
    with app.app_context():
        MODEL_STATUS['total_requests'] += 1
        MODEL_STATUS['total_modify_requests'] += 1
        MODEL_STATUS['last_request_time'] = time.time()
        MODEL_STATUS['active_tasks'] += 1
        
    try:
        # 检查文件是否存在
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        # 获取修改参数
        brightness = float(request.form.get('brightness', 0))  # -1.0 to 1.0
        contrast = float(request.form.get('contrast', 1))      # 0 to 3.0
        saturation = float(request.form.get('saturation', 1))  # 0 to 3.0
        speed = float(request.form.get('speed', 1))           # 0.1 to 10.0
        
        # 检查是否应该使用外部API
        if EXTERNAL_API_CONFIG['enabled'] and check_external_api_health():
            try:
                # 准备调用外部API的数据
                files = {'video': request.files['video']} 
                data = {
                    'brightness': brightness,
                    'contrast': contrast,
                    'saturation': saturation,
                    'speed': speed
                }
                
                # 调用外部API
                result = call_external_api('modify', files=files, data=data)
                return jsonify(result), 200
            except Exception as e:
                logger.warning(f"外部API调用失败，回退到本地处理: {str(e)}")
        
        # 本地处理
        video_file = request.files['video']
        temp_input = f"/tmp/{uuid.uuid4()}.mp4"
        temp_output = f"/tmp/{uuid.uuid4()}.mp4"
        
        try:
            video_file.save(temp_input)
            
            # 打开视频文件
            cap = cv2.VideoCapture(temp_input)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 创建输出视频
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps/speed, (width, height))
            
            # 处理每一帧
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 调整亮度和对比度
                adjusted_frame = cv2.convertScaleAbs(frame, alpha=contrast, beta=brightness*127)
                
                # 调整饱和度
                hsv = cv2.cvtColor(adjusted_frame, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], saturation)
                final_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                
                out.write(final_frame)
            
            cap.release()
            out.release()
            
            # 返回修改后的视频
            response = make_response(send_file(temp_output, as_attachment=True))
            response.headers["Content-Disposition"] = "attachment; filename=modified_video.mp4"
            return response
        except Exception as e:
            logger.error(f"视频修改失败: {str(e)}")
            return jsonify({'error': f"Video modification failed: {str(e)}"}), 500
        finally:
            # 清理临时文件
            for f in [temp_input, temp_output]:
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except:
                        pass
    except Exception as e:
        logger.error(f"视频修改接口错误: {str(e)}")
        return jsonify({'error': f"Video modify endpoint error: {str(e)}"}), 500
    finally:
        # 减少活跃任务计数
        with app.app_context():
            MODEL_STATUS['active_tasks'] = max(0, MODEL_STATUS['active_tasks'] - 1)

@app.route('/generate', methods=['POST'])
def generate():
    """生成视频API | Generate video API"""
    # 更新请求计数和状态
    with app.app_context():
        MODEL_STATUS['total_requests'] += 1
        MODEL_STATUS['total_generate_requests'] += 1
        MODEL_STATUS['last_request_time'] = time.time()
        MODEL_STATUS['active_tasks'] += 1
        
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        prompt = data.get('prompt')
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        emotion = data.get('emotion', 'neutral')
        duration = int(data.get('duration', 5))  # 默认5秒
        resolution = data.get('resolution', '720p')
        
        # 检查是否应该使用外部API
        if EXTERNAL_API_CONFIG['enabled'] and check_external_api_health():
            try:
                # 调用外部API
                result = call_external_api('generate', json=data)
                return jsonify(result), 200
            except Exception as e:
                logger.warning(f"外部API调用失败，回退到本地处理: {str(e)}")
        
        # 本地处理 - 生成模拟视频
        temp_output = f"/tmp/{uuid.uuid4()}.mp4"
        
        # 根据分辨率设置尺寸
        resolution_map = {
            '480p': (854, 480),
            '720p': (1280, 720),
            '1080p': (1920, 1080)
        }
        width, height = resolution_map.get(resolution, (1280, 720))
        fps = 30
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
            
            # 根据情绪选择颜色方案
            emotion_colors = {
                'happy': (0, 255, 255),    # 黄色
                'sad': (255, 0, 0),        # 蓝色
                'angry': (0, 0, 255),      # 红色
                'fear': (128, 0, 128),     # 紫色
                'surprise': (255, 255, 0), # 青色
                'neutral': (255, 255, 255) # 白色
            }
            color = emotion_colors.get(emotion.lower(), (255, 255, 255))
            
            # 生成指定时长的视频
            for _ in range(fps * duration):
                # 创建带有动态效果的帧
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # 添加基于prompt的简单可视化效果
                text = f"Generated: {prompt[:20]}..." if len(prompt) > 20 else f"Generated: {prompt}"
                cv2.putText(frame, text, (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # 添加动态效果
                t = _ / fps
                x = int(width // 2 + width // 4 * np.sin(t))
                y = int(height // 3 + height // 6 * np.cos(t))
                radius = 50 + 20 * np.sin(t * 2)
                cv2.circle(frame, (x, y), radius, color, -1)
                
                out.write(frame)
            
            out.release()
            
            # 返回生成的视频
            response = make_response(send_file(temp_output, as_attachment=True))
            response.headers["Content-Disposition"] = "attachment; filename=generated_video.mp4"
            return response
        except Exception as e:
            logger.error(f"视频生成失败: {str(e)}")
            return jsonify({'error': f"Video generation failed: {str(e)}"}), 500
        finally:
            # 清理临时文件
            if os.path.exists(temp_output):
                try:
                    os.remove(temp_output)
                except:
                    pass
    except Exception as e:
        logger.error(f"视频生成接口错误: {str(e)}")
        return jsonify({'error': f"Video generate endpoint error: {str(e)}"}), 500
    finally:
        # 减少活跃任务计数
        with app.app_context():
            MODEL_STATUS['active_tasks'] = max(0, MODEL_STATUS['active_tasks'] - 1)

@app.route('/stream', methods=['POST'])
def stream():
    """视频流处理API | Video stream processing API"""
    # 更新请求计数和状态
    with app.app_context():
        MODEL_STATUS['total_requests'] += 1
        MODEL_STATUS['total_stream_requests'] += 1
        MODEL_STATUS['last_request_time'] = time.time()
        MODEL_STATUS['active_tasks'] += 1
        
    try:
        # 检查是否应该使用外部API
        if EXTERNAL_API_CONFIG['enabled'] and check_external_api_health():
            try:
                # 准备调用外部API的数据
                data = request.json if request.is_json else {}
                if not data:
                    data = request.form.to_dict()
                
                # 调用外部API
                result = call_external_api('stream', json=data)
                return jsonify(result), 200
            except Exception as e:
                logger.warning(f"外部API调用失败，回退到本地处理: {str(e)}")
        
        # 本地处理
        data = request.json if request.is_json else {}
        if not data:
            data = request.form.to_dict()
        
        stream_url = data.get('stream_url')
        if not stream_url:
            return jsonify({'error': 'Stream URL is required'}), 400
        
        try:
            # 模拟视频流处理
            # 在实际应用中，这里应该实现真正的视频流处理逻辑
            result = {
                'status': 'processing',
                'stream_url': stream_url,
                'message': 'Video stream processing started',
                'timestamp': time.time(),
                'processing_method': 'local'
            }
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"视频流处理失败: {str(e)}")
            return jsonify({'error': f"Video stream processing failed: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"视频流处理接口错误: {str(e)}")
        return jsonify({'error': f"Video stream endpoint error: {str(e)}"}), 500
    finally:
        # 减少活跃任务计数
        with app.app_context():
            MODEL_STATUS['active_tasks'] = max(0, MODEL_STATUS['active_tasks'] - 1)

@app.route('/multi_stream', methods=['POST'])
def multi_stream():
    """多视频流分析API | Multi-video stream analysis API"""
    # 更新请求计数和状态
    with app.app_context():
        MODEL_STATUS['total_requests'] += 1
        MODEL_STATUS['total_multi_stream_requests'] += 1
        MODEL_STATUS['last_request_time'] = time.time()
        MODEL_STATUS['active_tasks'] += 1
        
    try:
        # 检查是否应该使用外部API
        if EXTERNAL_API_CONFIG['enabled'] and check_external_api_health():
            try:
                # 准备调用外部API的数据
                data = request.json if request.is_json else {}
                if not data:
                    data = request.form.to_dict()
                
                # 调用外部API
                result = call_external_api('multi_stream', json=data)
                return jsonify(result), 200
            except Exception as e:
                logger.warning(f"外部API调用失败，回退到本地处理: {str(e)}")
        
        # 本地处理
        data = request.json if request.is_json else {}
        if not data:
            data = request.form.to_dict()
        
        stream_urls = data.get('stream_urls', [])
        if not stream_urls or not isinstance(stream_urls, list):
            return jsonify({'error': 'Stream URLs list is required'}), 400
        
        try:
            # 模拟多视频流分析
            # 在实际应用中，这里应该实现真正的多视频流分析逻辑
            result = {
                'status': 'processing',
                'stream_count': len(stream_urls),
                'sample_streams': stream_urls[:3],  # 只返回前3个URL作为示例
                'message': 'Multi-video stream analysis started',
                'timestamp': time.time(),
                'processing_method': 'local'
            }
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"多视频流分析失败: {str(e)}")
            return jsonify({'error': f"Multi-video stream analysis failed: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"多视频流分析接口错误: {str(e)}")
        return jsonify({'error': f"Multi-video stream endpoint error: {str(e)}"}), 500
    finally:
        # 减少活跃任务计数
        with app.app_context():
            MODEL_STATUS['active_tasks'] = max(0, MODEL_STATUS['active_tasks'] - 1)

if __name__ == '__main__':
    # Start the Flask application
    app.run(host='0.0.0.0', port=5005, debug=True)
