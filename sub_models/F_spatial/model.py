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

# 双目空间定位感知模型定义
# Binocular Spatial Localization Perception Model Definition

import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import os
import requests
from datetime import datetime
import open3d as o3d
from ultralytics import YOLO
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from collections import deque

class SpatialModel(nn.Module):
    def __init__(self, config_path="config/spatial_config.json"):
        """初始化空间定位感知模型 | Initialize spatial localization perception model"""
        super(SpatialModel, self).__init__()
        
        # 加载配置 | Load configuration
        self.config = self.load_config(config_path)
        self.model_type = self.config.get("model_type", "local")
        self.external_api_config = self.config.get("external_api", {})
        
        # 初始化YOLO物体检测模型 | Initialize YOLO object detection model
        try:
            self.detection_model = YOLO('yolov8n.pt')  # 使用YOLOv8 nano版本 | Use YOLOv8 nano version
            print("YOLO物体检测模型加载成功 | YOLO object detection model loaded successfully")
        except Exception as e:
            print(f"YOLO模型加载失败: {e} | YOLO model loading failed: {e}")
            self.detection_model = None
        
        # 初始化立体匹配 | Initialize stereo matching
        self.stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
        
        # 运动追踪相关变量 | Motion tracking related variables
        self.tracked_objects = {}
        self.track_id_counter = 0
        self.max_track_history = 50
        
        # 实时处理标志 | Real-time processing flag
        self.realtime_processing = False
        self.left_camera = None
        self.right_camera = None
        
        # 语言支持 | Language support
        self.current_lang = self.config.get("default_language", "en")
        self.language_resources = self.load_language_resources()
    
    def load_config(self, config_path):
        """加载配置文件 | Load configuration file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {
                "model_type": "local",
                "default_language": "en",
                "camera_calibration": {
                    "focal_length": 1000,
                    "baseline": 0.12,
                    "image_width": 640,
                    "image_height": 480
                }
            }
    
    def load_language_resources(self):
        """加载语言资源 | Load language resources"""
        try:
            with open("config/language_resources.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            # 默认语言资源 | Default language resources
            return {
                "en": {
                    "object_detected": "Object detected",
                    "distance": "Distance",
                    "volume": "Volume",
                    "movement_detected": "Movement detected"
                },
                "zh": {
                    "object_detected": "检测到物体",
                    "distance": "距离",
                    "volume": "体积",
                    "movement_detected": "检测到运动"
                }
            }
    
    def localize_objects(self, left_img_path, right_img_path):
        """空间定位 | Spatial localization"""
        try:
            # 读取双目图像 | Read binocular images
            left_img = cv2.imread(left_img_path)
            right_img = cv2.imread(right_img_path)
            
            if left_img is None or right_img is None:
                return {'status': 'error', 'message': '无法读取图像文件 | Cannot read image files'}
            
            # 转换为灰度图 | Convert to grayscale
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # 计算视差图 | Compute disparity map
            disparity = self.stereo.compute(left_gray, right_gray)
            
            # 计算深度图 | Compute depth map
            depth = self.disparity_to_depth(disparity)
            
            # 检测物体 | Detect objects
            detection_results = self.detect_objects(left_img)
            
            # 为每个检测到的物体计算3D位置 | Calculate 3D position for each detected object
            objects_3d = []
            for detection in detection_results:
                obj_3d = self.calculate_object_3d_properties(detection, depth, left_img)
                if obj_3d:
                    objects_3d.append(obj_3d)
            
            return {
                'status': 'success',
                'objects': objects_3d,
                'depth_map': depth.tolist() if depth is not None else [],
                'disparity_map': disparity.tolist() if disparity is not None else []
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def disparity_to_depth(self, disparity):
        """将视差图转换为深度图 | Convert disparity map to depth map"""
        if disparity is None:
            return None
        
        # 避免除以零 | Avoid division by zero
        disparity = np.where(disparity == 0, 0.1, disparity)
        
        # 使用相机标定参数计算深度 | Calculate depth using camera calibration parameters
        focal_length = self.config["camera_calibration"]["focal_length"]
        baseline = self.config["camera_calibration"]["baseline"]
        
        depth = (focal_length * baseline) / disparity
        return depth
    
    def detect_objects(self, image):
        """检测图像中的物体 | Detect objects in image"""
        if self.detection_model is None:
            # 回退到简单检测 | Fallback to simple detection
            return self.simple_object_detection(image)
        
        try:
            # 使用YOLO进行物体检测 | Use YOLO for object detection
            results = self.detection_model(image)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = box.cls[0].cpu().numpy()
                        class_name = self.detection_model.names[int(class_id)]
                        
                        detections.append({
                            'label': class_name,
                            'confidence': float(confidence),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'center': (float((x1 + x2) / 2), float((y1 + y2) / 2))
                        })
            
            return detections
            
        except Exception as e:
            print(f"物体检测失败: {e} | Object detection failed: {e}")
            return self.simple_object_detection(image)
    
    def simple_object_detection(self, image):
        """简单物体检测（备用方法） | Simple object detection (fallback method)"""
        # 使用OpenCV的轮廓检测 | Use OpenCV contour detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 500:  # 只处理足够大的轮廓 | Only process large enough contours
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'label': f'object_{i}',
                    'confidence': 0.7,
                    'bbox': [x, y, x + w, y + h],
                    'center': (x + w // 2, y + h // 2)
                })
        
        return detections
    
    def calculate_object_3d_properties(self, detection, depth_map, image):
        """计算物体的3D属性 | Calculate 3D properties of object"""
        try:
            x1, y1, x2, y2 = detection['bbox']
            center_x, center_y = detection['center']
            
            # 计算物体区域的深度值 | Calculate depth values in object region
            object_region = depth_map[int(y1):int(y2), int(x1):int(x2)]
            valid_depths = object_region[object_region > 0]
            
            if len(valid_depths) == 0:
                return None
            
            # 使用中值深度减少噪声影响 | Use median depth to reduce noise impact
            median_depth = np.median(valid_depths)
            
            # 计算3D位置 | Calculate 3D position
            fx = self.config["camera_calibration"]["focal_length"]
            cx = self.config["camera_calibration"]["image_width"] / 2
            cy = self.config["camera_calibration"]["image_height"] / 2
            
            # 将2D坐标转换为3D坐标 | Convert 2D coordinates to 3D coordinates
            Z = median_depth
            X = (center_x - cx) * Z / fx
            Y = (center_y - cy) * Z / fx
            
            # 计算物体尺寸 | Calculate object dimensions
            width_pixels = x2 - x1
            height_pixels = y2 - y1
            width_real = (width_pixels * Z) / fx
            height_real = (height_pixels * Z) / fx
            
            # 估算体积（假设为长方体） | Estimate volume (assuming cuboid)
            depth_real = np.std(valid_depths) if len(valid_depths) > 1 else width_real * 0.5
            volume = width_real * height_real * depth_real
            
            return {
                'label': detection['label'],
                'confidence': detection['confidence'],
                'position_3d': {'x': float(X), 'y': float(Y), 'z': float(Z)},
                'dimensions': {'width': float(width_real), 'height': float(height_real), 'depth': float(depth_real)},
                'volume': float(volume),
                'bbox_2d': detection['bbox'],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"3D属性计算错误: {e} | 3D property calculation error: {e}")
            return None
    
    def create_spatial_model(self, left_img_path, right_img_path, output_format="pointcloud"):
        """创建空间模型 | Create spatial model"""
        try:
            # 读取双目图像 | Read binocular images
            left_img = cv2.imread(left_img_path)
            right_img = cv2.imread(right_img_path)
            
            if left_img is None or right_img is None:
                return {'status': 'error', 'message': '无法读取图像文件 | Cannot read image files'}
            
            # 转换为灰度图 | Convert to grayscale
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # 计算视差图 | Compute disparity map
            disparity = self.stereo.compute(left_gray, right_gray)
            
            # 计算深度图 | Compute depth map
            depth = self.disparity_to_depth(disparity)
            
            if output_format == "pointcloud":
                # 生成点云 | Generate point cloud
                point_cloud = self.generate_point_cloud(left_img, depth)
                return {'status': 'success', 'point_cloud': point_cloud}
            
            elif output_format == "mesh":
                # 生成网格模型 | Generate mesh model
                mesh = self.generate_mesh_model(left_img, depth)
                return {'status': 'success', 'mesh': mesh}
            
            else:
                return {'status': 'error', 'message': '不支持的输出格式 | Unsupported output format'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def generate_point_cloud(self, image, depth_map):
        """生成点云 | Generate point cloud"""
        try:
            height, width = depth_map.shape
            points = []
            colors = []
            
            fx = self.config["camera_calibration"]["focal_length"]
            cx = self.config["camera_calibration"]["image_width"] / 2
            cy = self.config["camera_calibration"]["image_height"] / 2
            
            # 采样点以减少数据量 | Sample points to reduce data volume
            step = 2
            for y in range(0, height, step):
                for x in range(0, width, step):
                    Z = depth_map[y, x]
                    if Z > 0 and Z < 100:  # 有效的深度范围 | Valid depth range
                        # 计算3D坐标 | Calculate 3D coordinates
                        X = (x - cx) * Z / fx
                        Y = (y - cy) * Z / fx
                        
                        points.append([X, Y, Z])
                        colors.append(image[y, x] / 255.0)  # 归一化颜色 | Normalize color
            
            # 创建Open3D点云对象 | Create Open3D point cloud object
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # 降采样和去噪 | Downsample and denoise
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            
            # 转换为可序列化的格式 | Convert to serializable format
            point_cloud_data = {
                'points': np.asarray(pcd.points).tolist(),
                'colors': np.asarray(pcd.colors).tolist()
            }
            
            return point_cloud_data
            
        except Exception as e:
            print(f"点云生成错误: {e} | Point cloud generation error: {e}")
            return {'points': [], 'colors': []}
    
    def generate_mesh_model(self, image, depth_map):
        """生成网格模型 | Generate mesh model"""
        try:
            # 先生成点云 | First generate point cloud
            point_cloud_data = self.generate_point_cloud(image, depth_map)
            
            if len(point_cloud_data['points']) == 0:
                return {'status': 'error', 'message': '无法生成点云 | Cannot generate point cloud'}
            
            # 创建Open3D点云 | Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(point_cloud_data['points'])
            pcd.colors = o3d.utility.Vector3dVector(point_cloud_data['colors'])
            
            # 估计法线 | Estimate normals
            pcd.estimate_normals()
            
            # 使用泊松重建创建网格 | Use Poisson reconstruction to create mesh
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            
            # 简化网格 | Simplify mesh
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=1000)
            
            # 转换为可序列化的格式 | Convert to serializable format
            mesh_data = {
                'vertices': np.asarray(mesh.vertices).tolist(),
                'triangles': np.asarray(mesh.triangles).tolist(),
                'vertex_colors': np.asarray(mesh.vertex_colors).tolist() if mesh.vertex_colors else []
            }
            
            return mesh_data
            
        except Exception as e:
            print(f"网格生成错误: {e} | Mesh generation error: {e}")
            return {'vertices': [], 'triangles': [], 'vertex_colors': []}
    
    def track_moving_objects(self, left_img, right_img, prev_detections=None):
        """追踪运动物体 | Track moving objects"""
        try:
            # 检测当前帧中的物体 | Detect objects in current frame
            current_detections = self.detect_objects(left_img)
            
            # 计算深度信息 | Calculate depth information
            left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            disparity = self.stereo.compute(left_gray, right_gray)
            depth = self.disparity_to_depth(disparity)
            
            # 为每个检测计算3D位置 | Calculate 3D position for each detection
            current_objects = []
            for detection in current_detections:
                obj_3d = self.calculate_object_3d_properties(detection, depth, left_img)
                if obj_3d:
                    current_objects.append(obj_3d)
            
            # 如果没有先前的检测，初始化追踪 | If no previous detections, initialize tracking
            if prev_detections is None:
                for obj in current_objects:
                    self.track_id_counter += 1
                    obj['track_id'] = self.track_id_counter
                    obj['track_history'] = deque([obj['position_3d']], maxlen=self.max_track_history)
                    self.tracked_objects[self.track_id_counter] = obj
                
                return {'status': 'success', 'tracked_objects': list(self.tracked_objects.values())}
            
            # 匹配当前检测与先前追踪的物体 | Match current detections with previously tracked objects
            matched_objects = self.match_objects(prev_detections, current_objects)
            
            # 更新追踪状态 | Update tracking status
            updated_objects = []
            for track_id, current_obj in matched_objects.items():
                if track_id in self.tracked_objects:
                    # 更新现有追踪 | Update existing track
                    tracked_obj = self.tracked_objects[track_id]
                    tracked_obj.update(current_obj)
                    tracked_obj['track_history'].append(current_obj['position_3d'])
                    
                    # 计算运动向量 | Calculate movement vector
                    if len(tracked_obj['track_history']) > 1:
                        prev_pos = tracked_obj['track_history'][-2]
                        curr_pos = tracked_obj['track_history'][-1]
                        movement = {
                            'dx': curr_pos['x'] - prev_pos['x'],
                            'dy': curr_pos['y'] - prev_pos['y'],
                            'dz': curr_pos['z'] - prev_pos['z'],
                            'speed': np.sqrt((curr_pos['x'] - prev_pos['x'])**2 +
                                           (curr_pos['y'] - prev_pos['y'])**2 +
                                           (curr_pos['z'] - prev_pos['z'])**2)
                        }
                        tracked_obj['movement'] = movement
                    
                    updated_objects.append(tracked_obj)
                else:
                    # 新追踪 | New track
                    self.track_id_counter += 1
                    current_obj['track_id'] = self.track_id_counter
                    current_obj['track_history'] = deque([current_obj['position_3d']], maxlen=self.max_track_history)
                    self.tracked_objects[self.track_id_counter] = current_obj
                    updated_objects.append(current_obj)
            
            # 移除丢失的追踪 | Remove lost tracks
            lost_tracks = set(self.tracked_objects.keys()) - set(matched_objects.keys())
            for track_id in lost_tracks:
                if track_id in self.tracked_objects:
                    del self.tracked_objects[track_id]
            
            return {'status': 'success', 'tracked_objects': updated_objects}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def match_objects(self, prev_objects, current_objects):
        """匹配物体检测 | Match object detections"""
        matched = {}
        
        if not prev_objects or not current_objects:
            return matched
        
        # 创建位置KD树进行快速最近邻搜索 | Create position KD-tree for fast nearest neighbor search
        prev_positions = np.array([[obj['position_3d']['x'], obj['position_3d']['y'], obj['position_3d']['z']]
                                 for obj in prev_objects])
        current_positions = np.array([[obj['position_3d']['x'], obj['position_3d']['y'], obj['position_3d']['z']]
                                    for obj in current_objects])
        
        if len(prev_positions) > 0 and len(current_positions) > 0:
            tree = KDTree(prev_positions)
            distances, indices = tree.query(current_positions, k=1)
            
            for i, (dist, idx) in enumerate(zip(distances, indices)):
                if dist < 1.0:  # 匹配阈值（米） | Matching threshold (meters)
                    matched[prev_objects[idx]['track_id']] = current_objects[i]
        
        return matched
    
    def start_realtime_processing(self, left_camera_index=0, right_camera_index=1):
        """启动实时空间感知处理 | Start real-time spatial perception processing"""
        try:
            self.left_camera = cv2.VideoCapture(left_camera_index)
            self.right_camera = cv2.VideoCapture(right_camera_index)
            
            if not self.left_camera.isOpened() or not self.right_camera.isOpened():
                return {'status': 'error', 'message': '无法打开摄像头 | Cannot open cameras'}
            
            self.realtime_processing = True
            print("实时空间感知处理已启动 | Real-time spatial perception processing started")
            
            # 启动处理线程 | Start processing thread
            import threading
            self.processing_thread = threading.Thread(target=self._realtime_processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            return {'status': 'success'}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def stop_realtime_processing(self):
        """停止实时处理 | Stop real-time processing"""
        self.realtime_processing = False
        if self.left_camera:
            self.left_camera.release()
        if self.right_camera:
            self.right_camera.release()
        return {'status': 'success'}
    
    def _realtime_processing_loop(self):
        """实时处理循环 | Real-time processing loop"""
        prev_detections = None
        
        while self.realtime_processing:
            # 读取双目帧 | Read stereo frames
            ret_left, frame_left = self.left_camera.read()
            ret_right, frame_right = self.right_camera.read()
            
            if not ret_left or not ret_right:
                continue
            
            # 追踪运动物体 | Track moving objects
            result = self.track_moving_objects(frame_left, frame_right, prev_detections)
            
            if result['status'] == 'success':
                prev_detections = result['tracked_objects']
                
                # 发送到主模型 | Send to main model
                self.send_to_main_model({
                    'type': 'spatial_update',
                    'timestamp': datetime.now().isoformat(),
                    'tracked_objects': prev_detections
                })
            
            # 控制处理频率 | Control processing frequency
            cv2.waitKey(30)
    
    def send_to_main_model(self, data):
        """发送数据到主模型 | Send data to main model"""
        try:
            # 在实际系统中，这里应该通过数据总线或API发送数据
            # In actual system, should send data via data bus or API
            print(f"发送空间数据到主模型: {len(data['tracked_objects'])} 个物体 | Sending spatial data to main model: {len(data['tracked_objects'])} objects")
            return {'status': 'success'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def switch_language(self, lang):
        """切换界面语言 | Switch interface language"""
        if lang in self.language_resources:
            self.current_lang = lang
            return {'status': 'success', 'language': lang}
        return {'status': 'error', 'message': '不支持的语言 | Unsupported language'}

if __name__ == '__main__':
    # 测试模型
    # Test model
    model = SpatialModel()
    print("空间定位感知模型初始化成功 | Spatial localization perception model initialized successfully")
    
    # 测试空间定位 | Test spatial localization
    print("测试空间定位: ", model.localize_objects("left_image.jpg", "right_image.jpg"))
    
    # 测试空间模型创建 | Test spatial model creation
    print("测试点云生成: ", model.create_spatial_model("left_image.jpg", "right_image.jpg", "pointcloud"))
    
    # 测试运动追踪 | Test motion tracking
    left_img = cv2.imread("left_image.jpg")
    right_img = cv2.imread("right_image.jpg")
    if left_img is not None and right_img is not None:
        print("测试运动追踪: ", model.track_moving_objects(left_img, right_img))
    
    # 测试实时处理 | Test real-time processing
    print("启动实时处理: ", model.start_realtime_processing())
    print("停止实时处理: ", model.stop_realtime_processing())
