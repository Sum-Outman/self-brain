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

# 图片视觉处理模型API服务
# Image Visual Processing Model API Service

from flask import Flask, request, jsonify, send_file
from .model import ImageModel
import os
import uuid
import json
import logging
import requests
import base64
from io import BytesIO
import cv2
import numpy as np
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("D_image_API")

app = Flask(__name__)
model = ImageModel()

# 外部API配置与状态管理
EXTERNAL_API_CONFIG = {
    'enabled': False,
    'base_url': 'http://localhost:5015/external/image',
    'api_key': os.environ.get('EXTERNAL_IMAGE_API_KEY', ''),
    'timeout': 30,
    'health_check_interval': 60,
    'last_health_check': 0,
    'health_status': False
}

# 初始化MODEL_STATUS
def initialize_model_status():
    return {
        'status': 'initializing',
        'version': '2.0.0',
        'initialized_at': time.time(),
        'total_requests': 0,
        'total_recognize_requests': 0,
        'total_modify_requests': 0,
        'total_adjust_requests': 0,
        'total_generate_requests': 0,
        'total_multiband_requests': 0,
        'total_style_transfer_requests': 0,
        'last_request_time': None,
        'active_tasks': 0,
        'model_info': {
            'type': 'D_image',
            'name': 'Image Processing Model',
            'capabilities': ['recognition', 'modification', 'generation', 'adjustment', 'multiband_analysis', 'style_transfer']
        }
    }

MODEL_STATUS = initialize_model_status()

# 检查外部API健康状态 | Check external API health
def check_external_api_health():
    current_time = time.time()
    if current_time - EXTERNAL_API_CONFIG['last_health_check'] < EXTERNAL_API_CONFIG['health_check_interval']:
        return EXTERNAL_API_CONFIG['health_status']
    
    try:
        headers = {'Authorization': f'Bearer {EXTERNAL_API_CONFIG['api_key']}'} if EXTERNAL_API_CONFIG['api_key'] else {}
        response = requests.get(f"{EXTERNAL_API_CONFIG['base_url']}/health", 
                              headers=headers, 
                              timeout=5)
        EXTERNAL_API_CONFIG['health_status'] = response.status_code == 200
    except Exception as e:
        logger.error(f"External API health check failed: {str(e)}")
        EXTERNAL_API_CONFIG['health_status'] = False
        
    EXTERNAL_API_CONFIG['last_health_check'] = current_time
    return EXTERNAL_API_CONFIG['health_status']

# 调用外部API | Call external API
def call_external_api(endpoint, method='POST', data=None, files=None):
    if not EXTERNAL_API_CONFIG['enabled'] or not check_external_api_health():
        return {'error': 'External API is not enabled or not healthy'}
    
    try:
        url = f"{EXTERNAL_API_CONFIG['base_url']}/{endpoint}"
        headers = {'Authorization': f'Bearer {EXTERNAL_API_CONFIG['api_key']}'} if EXTERNAL_API_CONFIG['api_key'] else {}
        
        if method == 'POST':
            if files:
                response = requests.post(url, headers=headers, files=files, timeout=EXTERNAL_API_CONFIG['timeout'])
            else:
                response = requests.post(url, headers=headers, json=data, timeout=EXTERNAL_API_CONFIG['timeout'])
        else:
            response = requests.get(url, headers=headers, params=data, timeout=EXTERNAL_API_CONFIG['timeout'])
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"External API error: {response.status_code}, {response.text}")
            return {'error': f'External API error: {response.status_code}'}
    except Exception as e:
        logger.error(f"External API request failed: {str(e)}")
        return {'error': str(e)}

@app.route('/config', methods=['POST'])
def configure_api():
    """配置外部API设置 | Configure External API Settings"""
    try:
        data = request.json
        if data:
            if 'enabled' in data:
                EXTERNAL_API_CONFIG['enabled'] = bool(data['enabled'])
            if 'base_url' in data:
                EXTERNAL_API_CONFIG['base_url'] = data['base_url']
            if 'api_key' in data:
                EXTERNAL_API_CONFIG['api_key'] = data['api_key']
            if 'timeout' in data:
                EXTERNAL_API_CONFIG['timeout'] = int(data['timeout'])
            if 'health_check_interval' in data:
                EXTERNAL_API_CONFIG['health_check_interval'] = int(data['health_check_interval'])
            
            # 立即进行健康检查
            check_external_api_health()
            
        return jsonify({
            'status': 'success',
            'message': 'Configuration updated',
            'config': EXTERNAL_API_CONFIG
        })
    except Exception as e:
        logger.error(f"Configuration error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/status', methods=['GET'])
def get_status():
    """获取模型运行状态 | Get Model Status"""
    external_health = check_external_api_health() if EXTERNAL_API_CONFIG['enabled'] else 'disabled'
    
    return jsonify({
        'model': 'D_image',
        'status': MODEL_STATUS,
        'external_api': {
            'enabled': EXTERNAL_API_CONFIG['enabled'],
            'health': external_health
        },
        'mode': 'external' if EXTERNAL_API_CONFIG['enabled'] else 'local'
    })

def check_external_api_health():
    """检查外部API健康状态 | Check external API health"""
    try:
        headers = {'Authorization': f'Bearer {EXTERNAL_API_CONFIG["api_key"]}'} if EXTERNAL_API_CONFIG['api_key'] else {}
        response = requests.get(f"{EXTERNAL_API_CONFIG['base_url']}/health", 
                              headers=headers, 
                              timeout=EXTERNAL_API_CONFIG['timeout'])
        return response.status_code == 200
    except:
        return False

def call_external_api(endpoint, method='POST', data=None, files=None):
    """调用外部API | Call external API"""
    try:
        url = f"{EXTERNAL_API_CONFIG['base_url']}/{endpoint}"
        headers = {'Authorization': f'Bearer {EXTERNAL_API_CONFIG["api_key"]}'} if EXTERNAL_API_CONFIG['api_key'] else {}
        
        if method == 'POST':
            if files:
                response = requests.post(url, headers=headers, files=files, timeout=EXTERNAL_API_CONFIG['timeout'])
            else:
                response = requests.post(url, headers=headers, json=data, timeout=EXTERNAL_API_CONFIG['timeout'])
        else:
            response = requests.get(url, headers=headers, params=data, timeout=EXTERNAL_API_CONFIG['timeout'])
        
        if response.status_code == 200:
            return response.json()
        else:
            logger.error(f"External API error: {response.status_code}, {response.text}")
            return {'error': f'External API error: {response.status_code}'}
    except Exception as e:
        logger.error(f"External API request failed: {str(e)}")
        return {'error': str(e)}

@app.route('/recognize', methods=['POST'])
def recognize():
    """识别图片内容API | Recognize image content API"""
    MODEL_STATUS['total_requests'] += 1
    MODEL_STATUS['last_request_time'] = time.time()
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    try:
        # 检查是否使用外部API
        if EXTERNAL_API_CONFIG['enabled'] and check_external_api_health():
            # 使用外部API
            image_file = request.files['image']
            files = {'image': (image_file.filename, image_file.stream, image_file.mimetype)}
            external_result = call_external_api('recognize', files=files)
            
            if 'error' not in external_result:
                return jsonify(external_result)
            else:
                logger.warning(f"External API call failed, falling back to local model: {external_result['error']}")
        
        # 使用本地模型
        image_file = request.files['image']
        temp_path = f"/tmp/{uuid.uuid4()}.jpg"
        image_file.save(temp_path)
        
        try:
            result = model.recognize_image(temp_path)
            return jsonify(result)
        finally:
            os.remove(temp_path)
    except Exception as e:
        logger.error(f"Image recognition error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/modify', methods=['POST'])
def modify():
    """修改图片内容API | Modify image content API"""
    MODEL_STATUS['total_requests'] += 1
    MODEL_STATUS['last_request_time'] = time.time()
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    modifications = request.form.get('modifications', '{}')
    
    try:
        modifications = json.loads(modifications)
    except:
        return jsonify({'error': 'Invalid modifications format'}), 400
    
    try:
        # 检查是否使用外部API
        if EXTERNAL_API_CONFIG['enabled'] and check_external_api_health():
            # 使用外部API
            files = {'image': (image_file.filename, image_file.stream, image_file.mimetype)}
            data = {'modifications': json.dumps(modifications)}
            external_result = call_external_api('modify', files=files, data=data)
            
            if 'error' not in external_result:
                # 注意：外部API可能返回JSON或直接返回图像，这里简化处理
                # 实际应用中可能需要根据外部API的返回格式进行调整
                return jsonify(external_result)
            else:
                logger.warning(f"External API call failed, falling back to local model: {external_result['error']}")
        
        # 使用本地模型
        temp_path = f"/tmp/{uuid.uuid4()}.jpg"
        output_path = f"/tmp/{uuid.uuid4()}.jpg"
        image_file.save(temp_path)
        
        try:
            # 加载图像
            img = cv2.imread(temp_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 应用修改
            if 'crop' in modifications:
                crop = modifications['crop']
                x1, y1, x2, y2 = crop.get('x1', 0), crop.get('y1', 0), \
                                 crop.get('x2', img.shape[1]), crop.get('y2', img.shape[0])
                img = img[y1:y2, x1:x2]
                
            if 'resize' in modifications:
                resize = modifications['resize']
                width = resize.get('width', img.shape[1])
                height = resize.get('height', img.shape[0])
                img = cv2.resize(img, (width, height))
                
            if 'rotate' in modifications:
                angle = modifications['rotate'].get('angle', 0)
                (h, w) = img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h))
                
            # 保存修改后的图像
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, img)
            
            # 返回修改后的图像
            return send_file(output_path, mimetype='image/jpeg', as_attachment=True, download_name='modified_image.jpg')
        except Exception as e:
            logger.error(f"Image modification error: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(output_path):
                os.remove(output_path)
    except Exception as e:
        logger.error(f"Image modification request error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/adjust', methods=['POST'])
def adjust():
    """调整图片API | Adjust image API"""
    MODEL_STATUS['total_requests'] += 1
    MODEL_STATUS['last_request_time'] = time.time()
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    adjustments = request.form.get('adjustments', '{}')
    
    try:
        adjustments = json.loads(adjustments)
    except:
        return jsonify({'error': 'Invalid adjustments format'}), 400
    
    try:
        # 检查是否使用外部API
        if EXTERNAL_API_CONFIG['enabled'] and check_external_api_health():
            # 使用外部API
            files = {'image': (image_file.filename, image_file.stream, image_file.mimetype)}
            data = {'adjustments': json.dumps(adjustments)}
            external_result = call_external_api('adjust', files=files, data=data)
            
            if 'error' not in external_result:
                # 注意：外部API可能返回JSON或直接返回图像，这里简化处理
                # 实际应用中可能需要根据外部API的返回格式进行调整
                return jsonify(external_result)
            else:
                logger.warning(f"External API call failed, falling back to local model: {external_result['error']}")
        
        # 使用本地模型
        temp_path = f"/tmp/{uuid.uuid4()}.jpg"
        output_path = f"/tmp/{uuid.uuid4()}.jpg"
        image_file.save(temp_path)
        
        try:
            # 加载图像
            img = cv2.imread(temp_path)
            
            # 应用调整
            if 'brightness' in adjustments:
                beta = adjustments['brightness']  # -100 to 100
                img = cv2.convertScaleAbs(img, beta=beta)
                
            if 'contrast' in adjustments:
                alpha = 1.0 + (adjustments['contrast'] / 100.0)  # 0 to 200
                img = cv2.convertScaleAbs(img, alpha=alpha)
                
            if 'saturation' in adjustments:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(img)
                s = cv2.convertScaleAbs(s, alpha=1.0 + (adjustments['saturation'] / 100.0))
                img = cv2.merge([h, s, v])
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
                
            if 'sharpness' in adjustments:
                kernel = np.array([[-1, -1, -1], [-1, 9 + adjustments['sharpness']/25, -1], [-1, -1, -1]])
                img = cv2.filter2D(img, -1, kernel)
                
            # 保存调整后的图像
            cv2.imwrite(output_path, img)
            
            # 返回调整后的图像
            return send_file(output_path, mimetype='image/jpeg', as_attachment=True, download_name='adjusted_image.jpg')
        except Exception as e:
            logger.error(f"Image adjustment error: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if os.path.exists(output_path):
                os.remove(output_path)
    except Exception as e:
        logger.error(f"Image adjustment request error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/generate', methods=['POST'])
def generate():
    """生成图片API | Generate image API"""
    MODEL_STATUS['total_requests'] += 1
    MODEL_STATUS['last_request_time'] = time.time()
    
    data = request.json
    prompt = data.get('prompt')
    emotion = data.get('emotion')
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    try:
        # 检查是否使用外部API
        if EXTERNAL_API_CONFIG['enabled'] and check_external_api_health():
            # 使用外部API
            payload = {'prompt': prompt}
            if emotion:
                payload['emotion'] = emotion
            external_result = call_external_api('generate', data=payload)
            
            if 'error' not in external_result:
                # 注意：外部API可能返回JSON或直接返回图像，这里简化处理
                # 实际应用中可能需要根据外部API的返回格式进行调整
                return jsonify(external_result)
            else:
                logger.warning(f"External API call failed, falling back to local model: {external_result['error']}")
        
        # 使用本地模型
        output_path = f"/tmp/{uuid.uuid4()}.jpg"
        
        try:
            # 生成模拟图像
            width = 512
            height = 512
            channels = 3
            
            # 根据prompt生成不同风格的随机图像
            if 'landscape' in prompt.lower() or 'nature' in prompt.lower():
                # 生成自然风景风格的图像
                img = np.zeros((height, width, channels), dtype=np.uint8)
                # 天空
                img[:height//2, :] = np.array([135, 206, 235], dtype=np.uint8)  # 天蓝色
                # 地面
                img[height//2:, :] = np.array([34, 139, 34], dtype=np.uint8)    # 森林绿
                # 简单的山
                for i in range(10):
                    y = height//2 + i*10
                    cv2.line(img, (0, y), (width, y), (0, 100, 0), 10)
                # 简单的云
                for i in range(3):
                    center = (100 + i*150, 100)
                    cv2.circle(img, center, 40, (255, 255, 255), -1)
            elif 'portrait' in prompt.lower() or 'person' in prompt.lower():
                # 生成人物肖像风格的图像
                img = np.ones((height, width, channels), dtype=np.uint8) * 255  # 白色背景
                # 简单的脸
                center = (width//2, height//3)
                cv2.circle(img, center, 80, (255, 222, 173), -1)  # 肤色
                # 简单的眼睛
                cv2.circle(img, (width//2 - 30, height//3 - 20), 10, (0, 0, 0), -1)
                cv2.circle(img, (width//2 + 30, height//3 - 20), 10, (0, 0, 0), -1)
                # 简单的嘴
                cv2.ellipse(img, (width//2, height//3 + 30), (30, 15), 0, 0, 180, (0, 0, 0), 2)
            else:
                # 生成抽象风格的图像
                img = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
            
            # 添加情绪相关的色彩调整
            if emotion == 'happy':
                # 明亮、鲜艳的色彩
                img = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
            elif emotion == 'sad':
                # 冷色调
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(img)
                h = (h - 30) % 180  # 蓝色调
                img = cv2.merge([h, s, v])
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            elif emotion == 'angry':
                # 暖色调
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(img)
                h = (h + 30) % 180  # 红色调
                img = cv2.merge([h, s, v])
                img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
            
            # 保存生成的图像
            cv2.imwrite(output_path, img)
            
            # 返回生成的图像
            return send_file(output_path, mimetype='image/jpeg', as_attachment=True, download_name='generated_image.jpg')
        except Exception as e:
            logger.error(f"Image generation error: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)
    except Exception as e:
        logger.error(f"Image generation request error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查 | Health check"""
    return jsonify({
        'status': 'healthy',
        'model': 'D_image',
        'version': '1.0.0'
    })

@app.route('/multiband', methods=['POST'])
def multiband_analysis():
    """多波段图像分析API | Multi-band image analysis API"""
    MODEL_STATUS['total_requests'] += 1
    MODEL_STATUS['total_multiband_requests'] += 1
    MODEL_STATUS['last_request_time'] = time.time()
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    
    try:
        # 检查是否使用外部API
        if EXTERNAL_API_CONFIG['enabled'] and check_external_api_health():
            # 使用外部API
            files = {'image': (image_file.filename, image_file.stream, image_file.mimetype)}
            external_result = call_external_api('multiband', files=files)
            
            if 'error' not in external_result:
                return jsonify(external_result)
            else:
                logger.warning(f"External API call failed, falling back to local model: {external_result['error']}")
        
        # 使用本地模型
        temp_path = f"/tmp/{uuid.uuid4()}.jpg"
        image_file.save(temp_path)
        
        try:
            # 加载图像
            img = cv2.imread(temp_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 获取图像尺寸
            height, width = img.shape[:2]
            
            # 多波段分析 - 将图像分为7个波段
            # 波段1: 低频率 - 整体色彩分布
            blurred_img = cv2.GaussianBlur(img, (101, 101), 0)
            band1 = blurred_img.mean(axis=(0, 1)).tolist()
            
            # 波段2-7: 中高频率分析
            # 计算不同尺度的特征
            band2 = cv2.GaussianBlur(img, (51, 51), 0).mean(axis=(0, 1)).tolist()
            band3 = cv2.GaussianBlur(img, (21, 21), 0).mean(axis=(0, 1)).tolist()
            band4 = cv2.GaussianBlur(img, (11, 11), 0).mean(axis=(0, 1)).tolist()
            band5 = cv2.GaussianBlur(img, (5, 5), 0).mean(axis=(0, 1)).tolist()
            
            # 边缘检测作为高频率信息
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            band6 = edges.mean().tolist()
            
            # 纹理分析
            gabor_kernel = cv2.getGaborKernel((21, 21), 5, 0, 10, 1, 0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)
            band7 = filtered.mean().tolist()
            
            # 计算色彩直方图特征
            hist_r = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten()
            hist_b = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten()
            
            # 计算图像统计特征
            mean = np.mean(img, axis=(0, 1)).tolist()
            std = np.std(img, axis=(0, 1)).tolist()
            min_val = np.min(img, axis=(0, 1)).tolist()
            max_val = np.max(img, axis=(0, 1)).tolist()
            
            # 构建多波段分析结果
            result = {
                'status': 'success',
                'image_size': {'width': width, 'height': height},
                'bands': {
                    'band1': {'name': 'Overall Color', 'values': band1},
                    'band2': {'name': 'Large Structures', 'values': band2},
                    'band3': {'name': 'Medium Structures', 'values': band3},
                    'band4': {'name': 'Small Structures', 'values': band4},
                    'band5': {'name': 'Fine Details', 'values': band5},
                    'band6': {'name': 'Edges', 'value': band6},
                    'band7': {'name': 'Texture', 'value': band7}
                },
                'statistics': {
                    'mean': mean,
                    'std': std,
                    'min': min_val,
                    'max': max_val
                },
                'color_histograms': {
                    'red': hist_r.tolist()[:50],  # 仅返回部分数据以减少响应大小
                    'green': hist_g.tolist()[:50],
                    'blue': hist_b.tolist()[:50]
                }
            }
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"Multiband analysis error: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    except Exception as e:
        logger.error(f"Multiband analysis request error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/style_transfer', methods=['POST'])
def style_transfer():
    """图像风格转换API | Image style transfer API"""
    MODEL_STATUS['total_requests'] += 1
    MODEL_STATUS['total_style_transfer_requests'] += 1
    MODEL_STATUS['last_request_time'] = time.time()
    
    if 'content_image' not in request.files or 'style_image' not in request.files:
        return jsonify({'error': 'Content image and style image are required'}), 400
    
    content_image = request.files['content_image']
    style_image = request.files['style_image']
    strength = request.form.get('strength', 1.0)
    
    try:
        strength = float(strength)
    except:
        return jsonify({'error': 'Invalid strength value'}), 400
    
    try:
        # 检查是否使用外部API
        if EXTERNAL_API_CONFIG['enabled'] and check_external_api_health():
            # 使用外部API
            files = {
                'content_image': (content_image.filename, content_image.stream, content_image.mimetype),
                'style_image': (style_image.filename, style_image.stream, style_image.mimetype)
            }
            data = {'strength': strength}
            external_result = call_external_api('style_transfer', files=files, data=data)
            
            if 'error' not in external_result:
                return jsonify(external_result)
            else:
                logger.warning(f"External API call failed, falling back to local model: {external_result['error']}")
        
        # 使用本地模型
        content_temp = f"/tmp/{uuid.uuid4()}.jpg"
        style_temp = f"/tmp/{uuid.uuid4()}.jpg"
        output_temp = f"/tmp/{uuid.uuid4()}.jpg"
        
        content_image.save(content_temp)
        style_image.save(style_temp)
        
        try:
            # 加载图像
            content_img = cv2.imread(content_temp)
            style_img = cv2.imread(style_temp)
            
            # 调整样式图像大小以匹配内容图像
            content_height, content_width = content_img.shape[:2]
            style_img = cv2.resize(style_img, (content_width, content_height))
            
            # 简单的风格混合（实际应用中应使用更复杂的神经风格迁移算法）
            # 混合比例基于strength参数
            blended_img = cv2.addWeighted(content_img, 1 - strength, style_img, strength, 0)
            
            # 保存结果
            cv2.imwrite(output_temp, blended_img)
            
            # 返回风格转换后的图像
            return send_file(output_temp, mimetype='image/jpeg', as_attachment=True, download_name='style_transferred_image.jpg')
        except Exception as e:
            logger.error(f"Style transfer error: {str(e)}")
            return jsonify({'error': str(e)}), 500
        finally:
            for path in [content_temp, style_temp, output_temp]:
                if os.path.exists(path):
                    os.remove(path)
    except Exception as e:
        logger.error(f"Style transfer request error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
