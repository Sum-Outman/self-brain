# -*- coding: utf-8 -*-
"""
多摄像头管理模块
Camera Management Module

负责管理多个摄像头设备，提供摄像头控制、视频流获取、快照拍摄等功能
"""

import cv2
import threading
import time
import logging
import base64
import os
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime

# 设置日志 | Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CameraManager")

# 是否使用模拟摄像头 (可通过环境变量控制)
MOCK_CAMERAS = os.environ.get('MOCK_CAMERAS', 'False').lower() == 'true'

class CameraManager:
    """多摄像头管理器
    Multi-camera Manager
    
    管理多个摄像头设备，提供统一的接口进行摄像头控制和数据获取
    """
    def __init__(self):
        """初始化摄像头管理器"""
        self.cameras = {}
        self.camera_locks = {}
        self.active_cameras = set()
        self.global_lock = threading.Lock()
        self.settings = {}
        # 模拟摄像头配置
        self.mock_cameras_config = {
            0: {"name": "Mock Camera 1", "width": 640, "height": 480, "fps": 30},
            1: {"name": "Mock Camera 2", "width": 1280, "height": 720, "fps": 30}
        }
        logger.info(f"摄像头管理器已初始化 | Camera Manager initialized (Mock mode: {MOCK_CAMERAS})")
    
    def list_available_cameras(self, max_devices: int = 10) -> List[Dict[str, Any]]:
        """列出所有可用的摄像头设备
        List all available camera devices
        
        参数:
            max_devices: 最大检查的设备数量
        
        返回:
            可用摄像头设备列表
        """
        available_cameras = []
        
        # 如果启用了模拟摄像头或没有实际摄像头可用，则返回模拟摄像头
        if MOCK_CAMERAS:
            logger.info("使用模拟摄像头 | Using mock cameras")
            for cam_id, cam_config in self.mock_cameras_config.items():
                available_cameras.append({
                    "id": cam_id,
                    "name": cam_config["name"],
                    "width": cam_config["width"],
                    "height": cam_config["height"],
                    "fps": cam_config["fps"]
                })
            return available_cameras
        
        # 尝试打开摄像头设备以检查是否可用
        for i in range(max_devices):
            try:
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # 使用DSHOW后端以避免某些兼容性问题
                if cap.isOpened():
                    # 获取摄像头基本信息
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    camera_info = {
                        "id": i,
                        "name": f"Camera {i}",
                        "width": width,
                        "height": height,
                        "fps": fps if fps > 0 else 30  # 默认30fps如果无法获取
                    }
                    available_cameras.append(camera_info)
                    cap.release()
                    logger.debug(f"检测到可用摄像头: {camera_info}")
            except Exception as e:
                logger.warning(f"检查摄像头 {i} 时出错: {str(e)}")
            
            # 短暂暂停以避免资源竞争
            time.sleep(0.1)
        
        # 如果没有找到实际摄像头，返回模拟摄像头
        if not available_cameras:
            logger.info("未找到实际摄像头，使用模拟摄像头 | No real cameras found, using mock cameras")
            for cam_id, cam_config in self.mock_cameras_config.items():
                available_cameras.append({
                    "id": cam_id,
                    "name": cam_config["name"],
                    "width": cam_config["width"],
                    "height": cam_config["height"],
                    "fps": cam_config["fps"]
                })
        
        logger.info(f"找到 {len(available_cameras)} 个可用摄像头 | Found {len(available_cameras)} available cameras")
        return available_cameras
    
    def start_camera(self, camera_id: int, params: Dict[str, Any] = None) -> bool:
        """启动指定的摄像头
        Start specified camera
        
        参数:
            camera_id: 摄像头ID
            params: 摄像头参数 (分辨率、帧率等)
        
        返回:
            启动成功返回True，否则返回False
        """
        with self.global_lock:
            if camera_id in self.active_cameras:
                logger.warning(f"摄像头 {camera_id} 已经在运行 | Camera {camera_id} is already running")
                return True
            
            try:
                # 创建摄像头锁
                if camera_id not in self.camera_locks:
                    self.camera_locks[camera_id] = threading.Lock()
                
                with self.camera_locks[camera_id]:
                    # 检查是否使用模拟摄像头
                    is_mock_camera = MOCK_CAMERAS or camera_id in self.mock_cameras_config
                    
                    if is_mock_camera:
                        logger.info(f"启动模拟摄像头 {camera_id} | Starting mock camera {camera_id}")
                        
                        # 获取模拟摄像头配置
                        cam_config = self.mock_cameras_config.get(camera_id, {
                            "width": 640,
                            "height": 480,
                            "fps": 30
                        })
                        
                        # 应用参数（如果有）
                        width = params.get("width", cam_config["width"]) if params else cam_config["width"]
                        height = params.get("height", cam_config["height"]) if params else cam_config["height"]
                        fps = params.get("fps", cam_config["fps"]) if params else cam_config["fps"]
                        
                        # 保存设置
                        self.settings[camera_id] = {
                            "width": width,
                            "height": height,
                            "fps": fps,
                            "brightness": 0.5,
                            "contrast": 0.5,
                            "saturation": 0.5
                        }
                        
                        # 创建模拟摄像头数据
                        self.cameras[camera_id] = {
                            "capture": "mock",  # 标记为模拟摄像头
                            "last_frame": None,
                            "last_error": None,
                            "is_running": True,
                            "start_time": datetime.now().isoformat(),
                            "params": params or {},
                            "mock_config": {
                                "width": width,
                                "height": height,
                                "frame_count": 0
                            }
                        }
                        
                    else:
                        # 初始化真实摄像头捕获对象
                        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
                        
                        if not cap.isOpened():
                            logger.error(f"无法打开摄像头 {camera_id} | Failed to open camera {camera_id}")
                            return False
                        
                        # 应用摄像头参数
                        if params:
                            if "width" in params and "height" in params:
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, params["width"])
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, params["height"])
                            if "fps" in params:
                                cap.set(cv2.CAP_PROP_FPS, params["fps"])
                            if "brightness" in params:
                                cap.set(cv2.CAP_PROP_BRIGHTNESS, params["brightness"])
                            if "contrast" in params:
                                cap.set(cv2.CAP_PROP_CONTRAST, params["contrast"])
                            if "saturation" in params:
                                cap.set(cv2.CAP_PROP_SATURATION, params["saturation"])
                        
                        # 保存当前设置
                        self.settings[camera_id] = {
                            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            "fps": cap.get(cv2.CAP_PROP_FPS),
                            "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
                            "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
                            "saturation": cap.get(cv2.CAP_PROP_SATURATION)
                        }
                        
                        # 创建摄像头数据
                        self.cameras[camera_id] = {
                            "capture": cap,
                            "last_frame": None,
                            "last_error": None,
                            "is_running": True,
                            "start_time": datetime.now().isoformat(),
                            "params": params or {}
                        }
                    
                    self.active_cameras.add(camera_id)
                    logger.info(f"摄像头 {camera_id} 已启动 | Camera {camera_id} started successfully")
                    return True
                    
            except Exception as e:
                logger.error(f"启动摄像头 {camera_id} 时出错: {str(e)} | Error starting camera {camera_id}: {str(e)}")
                if camera_id in self.cameras:
                    self.cameras[camera_id]["last_error"] = str(e)
                return False
    
    def stop_camera(self, camera_id: int) -> bool:
        """停止指定的摄像头
        Stop specified camera
        
        参数:
            camera_id: 摄像头ID
        
        返回:
            停止成功返回True，否则返回False
        """
        with self.global_lock:
            if camera_id not in self.active_cameras:
                logger.warning(f"摄像头 {camera_id} 未运行 | Camera {camera_id} is not running")
                return True
            
            try:
                if camera_id in self.camera_locks:
                    with self.camera_locks[camera_id]:
                        if camera_id in self.cameras:
                            cap = self.cameras[camera_id]["capture"]
                            if cap.isOpened():
                                cap.release()
                                
                            # 清理资源
                            del self.cameras[camera_id]
                            
                    # 移除锁
                    del self.camera_locks[camera_id]
                
                # 从活动摄像头集合中移除
                self.active_cameras.remove(camera_id)
                
                # 保存最后设置
                if camera_id in self.settings:
                    logger.info(f"摄像头 {camera_id} 设置已保存 | Camera {camera_id} settings saved")
                else:
                    logger.warning(f"未找到摄像头 {camera_id} 的设置 | No settings found for camera {camera_id}")
                
                logger.info(f"摄像头 {camera_id} 已停止 | Camera {camera_id} stopped successfully")
                return True
                
            except Exception as e:
                logger.error(f"停止摄像头 {camera_id} 时出错: {str(e)} | Error stopping camera {camera_id}: {str(e)}")
                return False
    
    def get_camera_frame(self, camera_id: int) -> Optional[Dict[str, Any]]:
        """获取摄像头的当前帧
        Get current frame from camera
        
        参数:
            camera_id: 摄像头ID
        
        返回:
            包含帧数据和时间戳的字典，失败返回None
        """
        if camera_id not in self.active_cameras:
            logger.warning(f"无法获取帧: 摄像头 {camera_id} 未运行 | Cannot get frame: camera {camera_id} is not running")
            return None
        
        try:
            if camera_id in self.camera_locks:
                with self.camera_locks[camera_id]:
                    if camera_id in self.cameras and self.cameras[camera_id]["is_running"]:
                        camera = self.cameras[camera_id]
                        
                        # 检查是否是模拟摄像头
                        if camera["capture"] == "mock":
                            # 生成模拟帧
                            mock_config = camera.get("mock_config", {"width": 640, "height": 480, "frame_count": 0})
                            width = mock_config["width"]
                            height = mock_config["height"]
                            frame_count = mock_config["frame_count"]
                            
                            # 创建一个简单的测试图像（渐变背景+移动方块）
                            frame = np.zeros((height, width, 3), dtype=np.uint8)
                            
                            # 渐变背景
                            for i in range(height):
                                for j in range(width):
                                    r = int((i / height) * 100 + 50)
                                    g = int((j / width) * 100 + 50)
                                    b = 100
                                    frame[i, j] = [b, g, r]  # OpenCV uses BGR format
                            
                            # 移动方块
                            block_size = 50
                            x = (frame_count * 5) % (width - block_size)
                            y = (frame_count * 3) % (height - block_size)
                            frame[y:y+block_size, x:x+block_size] = [0, 255, 0]  # Green block
                            
                            # 添加文本
                            cv2.putText(frame, f"Mock Camera {camera_id}", (10, 30), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(frame, f"Frame: {frame_count}", (10, 70), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            
                            # 更新帧计数器
                            mock_config["frame_count"] += 1
                            
                            # 转换为base64编码以便网络传输
                            _, buffer = cv2.imencode('.jpg', frame)
                            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                            
                            return {
                                "camera_id": camera_id,
                                "frame": jpg_as_text,
                                "timestamp": datetime.now().isoformat(),
                                "error": None,
                                "is_mock": True
                            }
                        else:
                            # 真实摄像头处理
                            cap = camera["capture"]
                            ret, frame = cap.read()
                            
                            if not ret:
                                logger.error(f"无法从摄像头 {camera_id} 获取帧 | Failed to read frame from camera {camera_id}")
                                camera["last_error"] = "Failed to read frame"
                                return None
                            
                            # 保存最后一帧
                            camera["last_frame"] = frame
                            
                            # 转换为base64编码以便网络传输
                            _, buffer = cv2.imencode('.jpg', frame)
                            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
                            
                            return {
                                "camera_id": camera_id,
                                "frame": jpg_as_text,
                                "timestamp": datetime.now().isoformat(),
                                "error": None
                            }
            
            return None
            
        except Exception as e:
            logger.error(f"获取摄像头 {camera_id} 帧时出错: {str(e)} | Error getting frame from camera {camera_id}: {str(e)}")
            if camera_id in self.cameras:
                self.cameras[camera_id]["last_error"] = str(e)
            return None
    
    def take_snapshot(self, camera_id: int) -> Optional[Dict[str, Any]]:
        """从摄像头拍摄快照
        Take snapshot from camera
        
        参数:
            camera_id: 摄像头ID
        
        返回:
            包含快照数据和时间戳的字典，失败返回None
        """
        frame_data = self.get_camera_frame(camera_id)
        
        if frame_data:
            # 添加快照特有信息
            snapshot_info = {
                **frame_data,
                "snapshot_id": f"snap_{camera_id}_{int(time.time())}",
                "type": "snapshot"
            }
            
            logger.info(f"从摄像头 {camera_id} 拍摄快照 | Took snapshot from camera {camera_id}")
            return snapshot_info
        
        logger.error(f"无法从摄像头 {camera_id} 拍摄快照 | Failed to take snapshot from camera {camera_id}")
        return None
    
    def get_camera_settings(self, camera_id: int) -> Optional[Dict[str, Any]]:
        """获取摄像头的当前设置
        Get current camera settings
        
        参数:
            camera_id: 摄像头ID
        
        返回:
            摄像头设置字典，失败返回None
        """
        with self.global_lock:
            if camera_id in self.settings:
                return self.settings[camera_id].copy()
            
            logger.warning(f"未找到摄像头 {camera_id} 的设置 | No settings found for camera {camera_id}")
            return None
    
    def update_camera_settings(self, camera_id: int, settings: Dict[str, Any]) -> bool:
        """更新摄像头设置
        Update camera settings
        
        参数:
            camera_id: 摄像头ID
            settings: 要更新的设置
        
        返回:
            更新成功返回True，否则返回False
        """
        if camera_id not in self.active_cameras:
            logger.warning(f"无法更新设置: 摄像头 {camera_id} 未运行 | Cannot update settings: camera {camera_id} is not running")
            return False
        
        try:
            if camera_id in self.camera_locks:
                with self.camera_locks[camera_id]:
                    if camera_id in self.cameras and self.cameras[camera_id]["is_running"]:
                        cap = self.cameras[camera_id]["capture"]
                        
                        # 应用设置
                        for key, value in settings.items():
                            if key.lower() == "width":
                                cap.set(cv2.CAP_PROP_FRAME_WIDTH, value)
                            elif key.lower() == "height":
                                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, value)
                            elif key.lower() == "fps":
                                cap.set(cv2.CAP_PROP_FPS, value)
                            elif key.lower() == "brightness":
                                cap.set(cv2.CAP_PROP_BRIGHTNESS, value)
                            elif key.lower() == "contrast":
                                cap.set(cv2.CAP_PROP_CONTRAST, value)
                            elif key.lower() == "saturation":
                                cap.set(cv2.CAP_PROP_SATURATION, value)
                            elif key.lower() == "exposure":
                                cap.set(cv2.CAP_PROP_EXPOSURE, value)
                            elif key.lower() == "gain":
                                cap.set(cv2.CAP_PROP_GAIN, value)
                        
                        # 更新设置缓存
                        self.settings[camera_id] = {
                            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                            "fps": cap.get(cv2.CAP_PROP_FPS),
                            "brightness": cap.get(cv2.CAP_PROP_BRIGHTNESS),
                            "contrast": cap.get(cv2.CAP_PROP_CONTRAST),
                            "saturation": cap.get(cv2.CAP_PROP_SATURATION),
                            "exposure": cap.get(cv2.CAP_PROP_EXPOSURE),
                            "gain": cap.get(cv2.CAP_PROP_GAIN)
                        }
                        
                        logger.info(f"摄像头 {camera_id} 设置已更新 | Camera {camera_id} settings updated")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"更新摄像头 {camera_id} 设置时出错: {str(e)} | Error updating camera {camera_id} settings: {str(e)}")
            return False
    
    def get_active_camera_ids(self) -> List[int]:
        """获取所有活动摄像头的ID
        Get IDs of all active cameras
        
        返回:
            活动摄像头ID列表
        """
        with self.global_lock:
            return list(self.active_cameras)
    
    def stop_all_cameras(self) -> bool:
        """停止所有活动摄像头
        Stop all active cameras
        
        返回:
            所有摄像头停止成功返回True，部分失败返回False
        """
        all_success = True
        active_ids = self.get_active_camera_ids()
        
        for camera_id in active_ids:
            if not self.stop_camera(camera_id):
                all_success = False
        
        logger.info(f"尝试停止所有摄像头 ({len(active_ids)} 个) | Attempted to stop all cameras ({len(active_ids)})")
        return all_success
    
    def get_camera_status(self, camera_id: int) -> Dict[str, Any]:
        """获取摄像头的当前状态
        Get current status of camera
        
        参数:
            camera_id: 摄像头ID
        
        返回:
            包含摄像头状态信息的字典
        """
        status = {
            "camera_id": camera_id,
            "is_active": camera_id in self.active_cameras,
            "has_error": False,
            "error_message": None,
            "settings": None,
            "start_time": None
        }
        
        if camera_id in self.active_cameras and camera_id in self.cameras:
            camera = self.cameras[camera_id]
            status["has_error"] = camera["last_error"] is not None
            status["error_message"] = camera["last_error"]
            status["start_time"] = camera["start_time"]
            
        if camera_id in self.settings:
            status["settings"] = self.settings[camera_id].copy()
        
        return status

# 创建全局摄像头管理器实例
global_camera_manager = CameraManager()

# 工具函数
def get_camera_manager() -> CameraManager:
    """获取全局摄像头管理器实例
    Get global camera manager instance
    """
    return global_camera_manager

if __name__ == "__main__":
    # 测试摄像头管理器
    print("测试摄像头管理器...")
    
    # 创建管理器实例
    manager = CameraManager()
    
    # 列出可用摄像头
    print("列出可用摄像头:")
    cameras = manager.list_available_cameras()
    for cam in cameras:
        print(f"- ID: {cam['id']}, 名称: {cam['name']}, 分辨率: {cam['width']}x{cam['height']}, FPS: {cam['fps']}")
    
    # 测试启动第一个可用摄像头
    if cameras:
        first_camera_id = cameras[0]['id']
        print(f"\n启动摄像头 {first_camera_id}...")
        success = manager.start_camera(first_camera_id)
        
        if success:
            print(f"摄像头 {first_camera_id} 启动成功!")
            
            # 获取摄像头设置
            settings = manager.get_camera_settings(first_camera_id)
            print(f"当前设置: {settings}")
            
            # 获取一帧数据
            print("获取摄像头帧...")
            frame_data = manager.get_camera_frame(first_camera_id)
            if frame_data:
                print(f"成功获取帧，大小: {len(frame_data['frame'])} 字节")
            
            # 拍摄快照
            print("拍摄快照...")
            snapshot = manager.take_snapshot(first_camera_id)
            if snapshot:
                print(f"成功拍摄快照: {snapshot['snapshot_id']}")
            
            # 停止摄像头
            print(f"停止摄像头 {first_camera_id}...")
            manager.stop_camera(first_camera_id)
            print("摄像头已停止")
        else:
            print(f"无法启动摄像头 {first_camera_id}")
    else:
        print("未找到可用摄像头")
    
    print("\n摄像头管理器测试完成")