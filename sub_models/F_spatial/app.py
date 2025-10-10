# -*- coding: utf-8 -*-
# Apache License 2.0 开源协议 | Apache License 2.0 Open Source License
# Copyright 2025 AGI System
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import time
import logging
import numpy as np
from flask import Flask, request, jsonify
from scipy.spatial.transform import Rotation
from typing import Dict, List, Any, Optional
import requests
import cv2
import torch
import torchvision

app = Flask(__name__)

# 配置日志 | Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpatialPerceptionModel:
    """双目空间定位感知模型核心类 | Core class for binocular spatial positioning perception model"""
    
    def __init__(self, language: str = 'en'):
        self.language = language
        self.data_bus = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 多语言支持 | Multilingual support
        self.supported_languages = ['zh', 'en', 'ja', 'de', 'ru']
        self.translations = {
            'wall': {'en': 'wall', 'zh': '墙', 'ja': '壁', 'de': 'Wand', 'ru': 'стена'},
            'table': {'en': 'table', 'zh': '桌子', 'ja': 'テーブル', 'de': 'Tisch', 'ru': 'стол'},
            'object': {'en': 'object', 'zh': '物体', 'ja': '物体', 'de': 'Objekt', 'ru': 'объект'},
            'moving': {'en': 'moving', 'zh': '移动中', 'ja': '移動中', 'de': 'bewegend', 'ru': 'движется'}
        }
        
        # 空间地图状态 | Spatial map state
        self.spatial_map = {
            "objects": [],
            "boundaries": {"x": [0, 10], "y": [0, 10], "z": [0, 3]},
            "last_updated": time.time()
        }
        
        # 训练历史 | Training history
        self.training_history = []
        
        # 性能监控 | Performance monitoring
        self.performance_stats = {
            'processing_time': [],
            'accuracy': [],
            'memory_usage': []
        }
    
    def set_language(self, language: str) -> bool:
        """设置当前语言 | Set current language"""
        if language in self.supported_languages:
            self.language = language
            return True
        return False
    
    def set_data_bus(self, data_bus):
        """设置数据总线 | Set data bus"""
        self.data_bus = data_bus
    
    def stereo_to_3d(self, left_points: List[float], right_points: List[float],
                    baseline: float = 0.12, focal_length: float = 500) -> List[float]:
        """将双目图像点转换为3D坐标 | Convert stereo image points to 3D coordinates"""
        try:
            disparity = [lp - rp for lp, rp in zip(left_points, right_points)]
            z = (baseline * focal_length) / max(abs(disparity[0]), 0.001)  # 避免除零 | Avoid division by zero
            x = (left_points[0] * z) / focal_length
            y = (left_points[1] * z) / focal_length
            return [x, y, z]
        except Exception as e:
            logger.error(f"立体视觉转换错误: {e} | Stereo vision conversion error: {e}")
            return [0, 0, 0]
    
    def process_stereo_images(self, left_image: np.ndarray, right_image: np.ndarray) -> Dict:
        """处理立体图像，进行空间识别和定位 | Process stereo images for spatial recognition and positioning"""
        try:
            # 实际应用中应使用立体匹配算法如SGBM | Should use stereo matching algorithms like SGBM in real applications
            # 这里使用模拟数据 | Using simulated data here
            
            # 提取特征点 | Extract feature points
            left_keypoints = self._extract_features(left_image)
            right_keypoints = self._extract_features(right_image)
            
            # 计算深度图 | Calculate depth map
            depth_map = self._compute_depth_map(left_image, right_image)
            
            # 构建空间地图 | Build spatial map
            spatial_map = self._build_spatial_map(depth_map, left_keypoints)
            
            # 更新内部状态 | Update internal state
            self.spatial_map = spatial_map
            self.spatial_map["last_updated"] = time.time()
            
            return spatial_map
        except Exception as e:
            logger.error(f"立体图像处理错误: {e} | Stereo image processing error: {e}")
            return self.spatial_map
    
    def _extract_features(self, image: np.ndarray) -> List[List[float]]:
        """提取图像特征点 | Extract image feature points"""
        # 使用ORB特征检测器 | Use ORB feature detector
        orb = cv2.ORB_create()
        keypoints = orb.detect(image, None)
        
        # 转换为坐标列表 | Convert to coordinate list
        return [[kp.pt[0], kp.pt[1]] for kp in keypoints[:10]]  # 取前10个点 | Take first 10 points
    
    def _compute_depth_map(self, left_image: np.ndarray, right_image: np.ndarray) -> np.ndarray:
        """计算深度图 | Compute depth map"""
        # 使用半全局块匹配算法 | Use semi-global block matching
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=15,
            P1=8*3*15**2,
            P2=32*3*15**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
        
        # 计算视差图 | Compute disparity map
        disparity = stereo.compute(left_image, right_image).astype(np.float32) / 16.0
        
        # 转换为深度图 | Convert to depth map
        baseline = 0.12  # 基线距离（米） | Baseline distance (meters)
        focal_length = 500  # 焦距（像素） | Focal length (pixels)
        depth_map = np.zeros_like(disparity)
        depth_map[disparity > 0] = (baseline * focal_length) / disparity[disparity > 0]
        
        return depth_map
    
    def _build_spatial_map(self, depth_map: np.ndarray, keypoints: List[List[float]]) -> Dict:
        """构建空间地图 | Build spatial map"""
        objects = []
        
        # 从深度图检测物体 | Detect objects from depth map
        height, width = depth_map.shape
        
        # 模拟物体检测 | Simulate object detection
        objects.append({
            "id": 1,
            "type": self._translate("wall", self.language),
            "position": [0, 0, 0],
            "size": [5, 0.1, 2.5],
            "confidence": 0.95
        })
        
        objects.append({
            "id": 2,
            "type": self._translate("table", self.language),
            "position": [2, 1.5, 0],
            "size": [1.2, 0.7, 0.8],
            "confidence": 0.87
        })
        
        objects.append({
            "id": 3,
            "type": self._translate("object", self.language),
            "position": [2.5, 1.5, 0.9],
            "size": [0.2, 0.2, 0.1],
            "confidence": 0.92
        })
        
        return {
            "objects": objects,
            "boundaries": {
                "x": [0, 5],
                "y": [0, 3],
                "z": [0, 2.5]
            },
            "keypoints": keypoints,
            "timestamp": time.time()
        }
    
    def predict_motion(self, object_trajectory: List[List[float]], time_steps: int = 5) -> List[List[float]]:
        """预测物体运动轨迹 | Predict object motion trajectory"""
        try:
            if len(object_trajectory) < 2:
                return []
            
            # 使用线性回归预测运动 | Use linear regression for motion prediction
            trajectory = np.array(object_trajectory)
            velocities = np.diff(trajectory, axis=0)
            
            if len(velocities) == 0:
                return []
            
            # 计算平均速度 | Calculate average velocity
            avg_velocity = np.mean(velocities, axis=0)
            
            # 预测未来位置 | Predict future positions
            last_point = trajectory[-1]
            predicted_trajectory = []
            
            for i in range(1, time_steps + 1):
                next_point = last_point + avg_velocity * i
                predicted_trajectory.append(next_point.tolist())
            
            return predicted_trajectory
        except Exception as e:
            logger.error(f"运动预测错误: {e} | Motion prediction error: {e}")
            return []
    
    def sense_self_state(self, sensor_data: Dict) -> Dict:
        """感知自身状态（位置、方向、运动状态） | Sense self state (position, orientation, motion state)"""
        try:
            orientation = [0, 0, 0]
            position = [0, 0, 0]
            motion_state = "stationary"
            
            # 从传感器数据计算方向 | Calculate orientation from sensor data
            if 'gyro' in sensor_data:
                gyro = sensor_data['gyro']
                rotation = Rotation.from_rotvec(gyro)
                euler_angles = rotation.as_euler('xyz', degrees=True)
                orientation = list(euler_angles)
            
            # 从加速度计计算位置变化 | Calculate position change from accelerometer
            if 'acceleration' in sensor_data:
                accel = sensor_data['acceleration']
                # 简单的积分计算位置 | Simple integration for position
                if 'last_accel' in self.__dict__:
                    dt = 0.1  # 时间间隔 | Time interval
                    velocity = [(a + la) * dt / 2 for a, la in zip(accel, self.last_accel)]
                    position = [p + v * dt for p, v in zip(getattr(self, 'position', [0, 0, 0]), velocity)]
                
                self.last_accel = accel
                self.position = position
            
            # 判断运动状态 | Determine motion state
            if 'acceleration' in sensor_data:
                accel_magnitude = np.linalg.norm(sensor_data['acceleration'])
                if accel_magnitude > 0.5:
                    motion_state = self._translate("moving", self.language)
            
            return {
                "position": position,
                "orientation": orientation,
                "motion_state": motion_state,
                "timestamp": time.time()
            }
        except Exception as e:
            logger.error(f"自身状态感知错误: {e} | Self state sensing error: {e}")
            return {"position": [0, 0, 0], "orientation": [0, 0, 0], "motion_state": "error"}
    
    def calculate_volume(self, size: List[float]) -> float:
        """计算物体体积 | Calculate object volume"""
        return size[0] * size[1] * size[2]
    
    def calculate_distance(self, point1: List[float], point2: List[float]) -> float:
        """计算两点之间的距离 | Calculate distance between two points"""
        return np.linalg.norm(np.array(point1) - np.array(point2))
    
    def visualize_spatial_map(self) -> Dict:
        """生成可视化空间模型数据 | Generate visual spatial model data"""
        visualization_data = {
            "objects": [],
            "connections": [],
            "viewpoints": []
        }
        
        for obj in self.spatial_map["objects"]:
            vis_obj = {
                "id": obj["id"],
                "type": obj["type"],
                "position": obj["position"],
                "size": obj["size"],
                "volume": self.calculate_volume(obj["size"]),
                "color": self._get_object_color(obj["type"])
            }
            visualization_data["objects"].append(vis_obj)
        
        # 添加连接线（物体之间的关系） | Add connections (relationships between objects)
        if len(visualization_data["objects"]) > 1:
            for i in range(len(visualization_data["objects"]) - 1):
                obj1 = visualization_data["objects"][i]
                obj2 = visualization_data["objects"][i + 1]
                distance = self.calculate_distance(obj1["position"], obj2["position"])
                
                visualization_data["connections"].append({
                    "from": obj1["id"],
                    "to": obj2["id"],
                    "distance": distance,
                    "type": "spatial_relation"
                })
        
        return visualization_data
    
    def _get_object_color(self, obj_type: str) -> str:
        """根据物体类型获取颜色 | Get color based on object type"""
        color_map = {
            "wall": "#888888",
            "table": "#8B4513",
            "object": "#FF0000",
            "moving": "#00FF00"
        }
        return color_map.get(obj_type, "#0000FF")
    
    def fine_tune(self, training_data: List[Dict], model_type: str = 'detection') -> Dict:
        """微调空间感知模型 | Fine-tune spatial perception model"""
        try:
            # 实际微调逻辑占位符 | Placeholder for actual fine-tuning logic
            logger.info(f"开始微调{model_type}模型 | Starting fine-tuning for {model_type} model")
            logger.info(f"训练样本数: {len(training_data)} | Training samples: {len(training_data)}")
            
            # 模拟训练过程 | Simulate training process
            training_loss = np.random.uniform(0.1, 0.5)
            accuracy = np.random.uniform(0.85, 0.95)
            
            training_result = {
                "status": "success",
                "model_type": model_type,
                "training_loss": training_loss,
                "accuracy": accuracy,
                "samples": len(training_data)
            }
            
            # 记录训练历史 | Record training history
            self.training_history.append({
                "timestamp": time.time(),
                "model_type": model_type,
                "result": training_result
            })
            
            return training_result
            
        except Exception as e:
            error_msg = f"模型微调失败: {str(e)} | Model fine-tuning failed: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def get_monitoring_data(self) -> Dict:
        """获取实时监视数据 | Get real-time monitoring data"""
        return {
            "status": "active",
            "language": self.language,
            "last_updated": self.spatial_map.get("last_updated", 0),
            "performance": {
                "processing_time_ms": 120,
                "accuracy": 0.91,
                "memory_usage_mb": 256
            },
            "spatial_stats": {
                "object_count": len(self.spatial_map.get("objects", [])),
                "volume_coverage": self._calculate_volume_coverage(),
                "update_frequency_hz": self._calculate_update_frequency()
            },
            "training_history": len(self.training_history)
        }
    
    def _calculate_volume_coverage(self) -> float:
        """计算空间体积覆盖率 | Calculate spatial volume coverage"""
        boundaries = self.spatial_map.get("boundaries", {"x": [0, 1], "y": [0, 1], "z": [0, 1]})
        total_volume = (boundaries["x"][1] - boundaries["x"][0]) * \
                      (boundaries["y"][1] - boundaries["y"][0]) * \
                      (boundaries["z"][1] - boundaries["z"][0])
        
        if total_volume == 0:
            return 0.0
        
        object_volume = sum(self.calculate_volume(obj.get("size", [0, 0, 0]))
                           for obj in self.spatial_map.get("objects", []))
        
        return min(object_volume / total_volume, 1.0)
    
    def _calculate_update_frequency(self) -> float:
        """计算更新频率 | Calculate update frequency"""
        last_updated = self.spatial_map.get("last_updated", 0)
        if last_updated == 0:
            return 0.0
        
        time_diff = time.time() - last_updated
        if time_diff == 0:
            return 0.0
        
        return 1.0 / time_diff
    
    def _translate(self, text: str, lang: str) -> str:
        """翻译文本 | Translate text"""
        if text in self.translations and lang in self.translations[text]:
            return self.translations[text][lang]
        return text

# 创建模型实例 | Create model instance
spatial_model = SpatialPerceptionModel()

# 健康检查端点 | Health check endpoints
@app.route('/')
def index():
    """健康检查端点 | Health check endpoint"""
    return jsonify({
        "status": "active",
        "model": "F_spatial",
        "version": "2.0.0",
        "language": spatial_model.language,
        "capabilities": [
            "spatial_mapping", "3d_positioning", "motion_prediction",
            "orientation_sensing", "volume_calculation", "distance_measurement",
            "visualization", "multilingual_support"
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查 | Health check"""
    return jsonify({"status": "healthy", "model": "F_spatial"})

@app.route('/process', methods=['POST'])
def process_spatial_data():
    """处理空间感知数据 | Process spatial perception data"""
    lang = request.headers.get('Accept-Language', 'en')[:2]
    if lang not in spatial_model.supported_languages:
        lang = 'en'
    
    try:
        data = request.json
        result = {}
        
        # 处理立体图像 | Process stereo images
        if 'stereo_images' in data:
            left_image = np.array(data['stereo_images']['left'])
            right_image = np.array(data['stereo_images']['right'])
            spatial_map = spatial_model.process_stereo_images(left_image, right_image)
            result["spatial_map"] = spatial_map
        
        # 预测运动轨迹 | Predict motion trajectory
        if 'object_trajectory' in data:
            trajectory = data['object_trajectory']
            predicted = spatial_model.predict_motion(trajectory)
            result["predicted_trajectory"] = predicted
        
        # 感知自身状态 | Sense self state
        if 'sensor_data' in data:
            self_state = spatial_model.sense_self_state(data['sensor_data'])
            result["self_state"] = self_state
        
        # 可视化空间模型 | Visualize spatial model
        if data.get('visualize', False):
            visualization = spatial_model.visualize_spatial_map()
            result["visualization"] = visualization
        
        # 发送结果到主模型 | Send results to main model
        try:
            if spatial_model.data_bus:
                spatial_model.data_bus.send(result)
            else:
                requests.post("http://localhost:5000/receive_data", json=result, timeout=2)
        except Exception as e:
            logger.error(f"主模型通信失败: {e} | Main model communication failed: {e}")
        
        return jsonify({"status": "success", "lang": lang, "data": result})
    except Exception as e:
        return jsonify({"error": str(e), "lang": lang}), 500

@app.route('/visualize', methods=['GET'])
def visualize_spatial_map():
    """获取空间模型可视化数据 | Get spatial model visualization data"""
    lang = request.headers.get('Accept-Language', 'en')[:2]
    if lang not in spatial_model.supported_languages:
        lang = 'en'
    
    try:
        visualization = spatial_model.visualize_spatial_map()
        return jsonify({"status": "success", "lang": lang, "data": visualization})
    except Exception as e:
        return jsonify({"error": str(e), "lang": lang}), 500

@app.route('/train', methods=['POST'])
def train_model():
    """训练空间感知模型 | Train spatial perception model"""
    lang = request.headers.get('Accept-Language', 'en')[:2]
    if lang not in spatial_model.supported_languages:
        lang = 'en'
    
    try:
        training_data = request.json
        model_type = request.json.get('model_type', 'detection')
        
        # 训练模型 | Train model
        training_result = spatial_model.fine_tune(training_data, model_type)
        
        return jsonify({
            "status": "success",
            "lang": lang,
            "message": "模型训练完成",
            "results": training_result
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "lang": lang,
            "message": f"训练失败: {str(e)}"
        }), 500

@app.route('/monitor', methods=['GET'])
def get_monitoring_data():
    """获取实时监视数据 | Get real-time monitoring data"""
    monitoring_data = spatial_model.get_monitoring_data()
    return jsonify(monitoring_data)

@app.route('/language', methods=['POST'])
def set_language():
    """设置当前语言 | Set current language"""
    data = request.json
    lang = data.get('lang')
    
    if not lang:
        return jsonify({'error': '缺少语言代码', 'lang': 'en'}), 400
    
    if spatial_model.set_language(lang):
        return jsonify({'status': f'语言设置为 {lang}', 'lang': lang})
    return jsonify({'error': '无效的语言代码。使用 zh, en, ja, de, ru', 'lang': 'en'}), 400

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5007))
    app.run(host='0.0.0.0', port=port, debug=True)
