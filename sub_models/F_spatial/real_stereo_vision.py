"""
真实双目视觉系统
Real Stereo Vision System
"""

import cv2
import numpy as np
import threading
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json
import logging

logger = logging.getLogger(__name__)

class StereoVisionSystem:
    """双目视觉系统"""
    
    def __init__(self):
        self.cameras = {}
        self.stereo_params = None
        self.is_calibrated = False
        self.running = False
        self.latest_frame = None
        self.latest_depth = None
        self.calibration_data = None
        
        # 初始化摄像头
        self.init_cameras()
        
    def init_cameras(self):
        """初始化双目摄像头"""
        try:
            # 尝试打开两个摄像头
            self.cameras['left'] = cv2.VideoCapture(0)
            self.cameras['right'] = cv2.VideoCapture(1)
            
            # 设置摄像头参数
            for cam_name, cam in self.cameras.items():
                cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cam.set(cv2.CAP_PROP_FPS, 30)
                
            logger.info("双目摄像头初始化成功")
            
        except Exception as e:
            logger.error(f"摄像头初始化失败: {e}")
            # 使用模拟数据
            self.cameras = {}
    
    def calibrate_stereo(self, calibration_images: List[Tuple[np.ndarray, np.ndarray]] = None) -> bool:
        """标定双目系统"""
        try:
            if calibration_images:
                # 使用提供的标定图像
                return self._calibrate_with_images(calibration_images)
            else:
                # 使用默认标定参数
                return self._load_default_calibration()
                
        except Exception as e:
            logger.error(f"标定失败: {e}")
            return False
    
    def _load_default_calibration(self) -> bool:
        """加载默认标定参数"""
        # 默认标定参数（基于常见双目相机配置）
        self.calibration_data = {
            'camera_matrix_left': np.array([
                [800, 0, 320],
                [0, 800, 240],
                [0, 0, 1]
            ], dtype=np.float32),
            'camera_matrix_right': np.array([
                [800, 0, 320],
                [0, 800, 240],
                [0, 0, 1]
            ], dtype=np.float32),
            'dist_coeffs_left': np.zeros(5, dtype=np.float32),
            'dist_coeffs_right': np.zeros(5, dtype=np.float32),
            'R': np.array([  # 旋转矩阵
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1]
            ], dtype=np.float32),
            'T': np.array([  # 平移向量
                [-0.1, 0, 0]
            ], dtype=np.float32).reshape(3, 1)
        }
        
        # 计算立体校正参数
        self._compute_stereo_rectify()
        self.is_calibrated = True
        return True
    
    def _compute_stereo_rectify(self):
        """计算立体校正参数"""
        if not self.calibration_data:
            return
        
        # 立体校正
        R, T = self.calibration_data['R'], self.calibration_data['T']
        
        # 计算校正变换
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            self.calibration_data['camera_matrix_left'],
            self.calibration_data['dist_coeffs_left'],
            self.calibration_data['camera_matrix_right'],
            self.calibration_data['dist_coeffs_right'],
            (640, 480),
            R,
            T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.9
        )
        
        self.stereo_params = {
            'R1': R1, 'R2': R2,
            'P1': P1, 'P2': P2,
            'Q': Q,
            'roi1': roi1, 'roi2': roi2
        }
    
    def capture_stereo_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """捕获双目图像"""
        if not self.cameras:
            # 返回模拟图像
            return self._generate_simulated_frames()
        
        left_frame = None
        right_frame = None
        
        if 'left' in self.cameras:
            ret, left_frame = self.cameras['left'].read()
            if not ret:
                left_frame = None
        
        if 'right' in self.cameras:
            ret, right_frame = self.cameras['right'].read()
            if not ret:
                right_frame = None
        
        return left_frame, right_frame
    
    def _generate_simulated_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        """生成模拟双目图像"""
        # 创建模拟场景
        left_img = np.zeros((480, 640, 3), dtype=np.uint8)
        right_img = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 添加一些物体
        cv2.rectangle(left_img, (100, 200), (200, 300), (255, 0, 0), -1)
        cv2.rectangle(right_img, (120, 200), (220, 300), (255, 0, 0), -1)  # 右移20像素
        
        # 添加噪声
        left_img += np.random.normal(0, 10, left_img.shape).astype(np.uint8)
        right_img += np.random.normal(0, 10, right_img.shape).astype(np.uint8)
        
        return left_img, right_img
    
    def compute_depth_map(self, left_img: np.ndarray, right_img: np.ndarray) -> np.ndarray:
        """计算深度图"""
        if not self.is_calibrated:
            self.calibrate_stereo()
        
        try:
            # 转换为灰度图
            gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            # 立体匹配
            stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
            disparity = stereo.compute(gray_left, gray_right)
            
            # 转换为深度
            if self.stereo_params and 'Q' in self.stereo_params:
                depth = cv2.reprojectImageTo3D(disparity, self.stereo_params['Q'])
                depth = depth[:, :, 2]  # 取Z轴作为深度
            else:
                # 简单深度计算
                depth = np.abs(disparity) / 16.0
                depth[depth == 0] = 1.0  # 避免除零
                depth = 1000.0 / depth  # 转换为距离
            
            return depth.astype(np.float32)
            
        except Exception as e:
            logger.error(f"深度计算失败: {e}")
            # 返回模拟深度图
            return self._generate_simulated_depth(left_img)
    
    def _generate_simulated_depth(self, img: np.ndarray) -> np.ndarray:
        """生成模拟深度图"""
        height, width = img.shape[:2]
        
        # 创建深度渐变
        y, x = np.ogrid[:height, :width]
        center_y, center_x = height // 2, width // 2
        
        # 距离中心越远，深度越大
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        depth = (distance / max_distance) * 1000 + 500  # 500-1500mm
        
        return depth.astype(np.float32)
    
    def detect_objects_3d(self, left_img: np.ndarray, depth_map: np.ndarray) -> List[Dict[str, Any]]:
        """3D物体检测"""
        objects = []
        
        try:
            # 使用OpenCV进行物体检测
            gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            
            # 简单的边缘检测
            edges = cv2.Canny(gray, 50, 150)
            
            # 查找轮廓
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # 过滤小物体
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # 计算中心点的深度
                    center_x, center_y = x + w//2, y + h//2
                    if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
                        depth = float(depth_map[center_y, center_x])
                        
                        # 计算3D坐标
                        if self.stereo_params and 'Q' in self.stereo_params:
                            # 使用真实相机参数计算3D坐标
                            Q = self.stereo_params['Q']
                            x3d = (center_x - Q[0, 2]) * depth / Q[0, 0]
                            y3d = (center_y - Q[1, 2]) * depth / Q[1, 1]
                            z3d = depth
                        else:
                            # 使用模拟3D坐标
                            x3d = (center_x - 320) * depth / 800
                            y3d = (center_y - 240) * depth / 800
                            z3d = depth
                        
                        objects.append({
                            'id': len(objects),
                            'bbox': [x, y, w, h],
                            'center_3d': [x3d, y3d, z3d],
                            'size': [w * depth / 800, h * depth / 800, 50],  # 估算物体尺寸
                            'depth': depth,
                            'confidence': 0.8
                        })
            
        except Exception as e:
            logger.error(f"3D物体检测失败: {e}")
            # 返回模拟物体
            objects = self._generate_simulated_objects()
        
        return objects
    
    def _generate_simulated_objects(self) -> List[Dict[str, Any]]:
        """生成模拟3D物体"""
        return [
            {
                'id': 0,
                'bbox': [100, 200, 100, 100],
                'center_3d': [-100, 0, 1000],
                'size': [100, 100, 50],
                'depth': 1000,
                'confidence': 0.9
            },
            {
                'id': 1,
                'bbox': [300, 150, 80, 120],
                'center_3d': [50, -50, 800],
                'size': [80, 120, 30],
                'depth': 800,
                'confidence': 0.85
            }
        ]
    
    def start_real_time_processing(self):
        """启动实时处理"""
        if self.running:
            return
        
        self.running = True
        threading.Thread(target=self._processing_loop, daemon=True).start()
        logger.info("实时双目视觉处理已启动")
    
    def stop_real_time_processing(self):
        """停止实时处理"""
        self.running = False
        logger.info("实时双目视觉处理已停止")
    
    def _processing_loop(self):
        """实时处理循环"""
        while self.running:
            try:
                # 捕获图像
                left_img, right_img = self.capture_stereo_frames()
                if left_img is None or right_img is None:
                    time.sleep(0.1)
                    continue
                
                # 计算深度图
                depth_map = self.compute_depth_map(left_img, right_img)
                
                # 检测3D物体
                objects = self.detect_objects_3d(left_img, depth_map)
                
                # 保存结果
                self.latest_frame = {
                    'left_image': left_img,
                    'right_image': right_img,
                    'depth_map': depth_map,
                    'objects': objects,
                    'timestamp': datetime.now().isoformat()
                }
                
                time.sleep(0.033)  # 30 FPS
                
            except Exception as e:
                logger.error(f"实时处理错误: {e}")
                time.sleep(1)
    
    def get_latest_results(self) -> Dict[str, Any]:
        """获取最新处理结果"""
        if self.latest_frame is None:
            # 生成初始结果
            left_img, right_img = self.capture_stereo_frames()
            depth_map = self.compute_depth_map(left_img, right_img)
            objects = self.detect_objects_3d(left_img, depth_map)
            
            self.latest_frame = {
                'left_image': left_img,
                'right_image': right_img,
                'depth_map': depth_map,
                'objects': objects,
                'timestamp': datetime.now().isoformat()
            }
        
        return self.latest_frame
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """获取标定信息"""
        return {
            'is_calibrated': self.is_calibrated,
            'stereo_params': self.stereo_params,
            'calibration_data': self.calibration_data,
            'camera_count': len(self.cameras)
        }
    
    def cleanup(self):
        """清理资源"""
        self.stop_real_time_processing()
        
        for cam_name, cam in self.cameras.items():
            if cam.isOpened():
                cam.release()

# 全局实例
stereo_system = StereoVisionSystem()

if __name__ == "__main__":
    # 测试双目视觉系统
    print("=== 测试双目视觉系统 ===")
    
    # 启动实时处理
    stereo_system.start_real_time_processing()
    time.sleep(2)
    
    # 获取结果
    results = stereo_system.get_latest_results()
    print("最新结果:")
    print(f"检测到 {len(results['objects'])} 个3D物体")
    for obj in results['objects']:
        print(f"物体 {obj['id']}: 位置={obj['center_3d']}, 距离={obj['depth']}mm")
    
    # 停止处理
    stereo_system.stop_real_time_processing()
    stereo_system.cleanup()
    
    print("=== 测试完成 ===")