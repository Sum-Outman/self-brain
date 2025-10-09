# Copyright 2025 The AGI System Authors
#
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

import threading
import time
import logging
from typing import Dict, Any, Optional

# 导入摄像头管理器 | Import camera manager
from camera_manager import get_camera_manager

# 设置日志 | Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RealtimeInput")

class RealtimeInputInterface:
    def __init__(self):
        """实时输入接口 | Realtime Input Interface
        
        负责管理摄像头、麦克风、网络流等实时输入源
        Manages real-time input sources like cameras, microphones, and network streams
        """
        self.camera_sources = {}  # 摄像头源 | Camera sources
        self.microphone_sources = {}  # 麦克风源 | Microphone sources
        self.network_streams = {}  # 网络流 | Network streams
        self.sensor_data = {}  # 传感器数据 | Sensor data
        self.active_connections = {}  # 活动连接 | Active connections
        self.data_lock = threading.Lock()  # 数据锁 | Data lock
        
        # 获取摄像头管理器实例 | Get camera manager instance
        self.camera_manager = get_camera_manager()
        
        # 启动数据更新线程 | Start data update thread
        self.update_thread = threading.Thread(target=self._update_data_loop, daemon=True)
        self.update_thread.start()
        logger.info("实时输入接口已初始化 | Realtime input interface initialized")

    def connect_camera(self, camera_id: str, source: str = "local", resolution: str = "1080p"):
        """连接摄像头 | Connect to camera
        
        参数:
            camera_id: 摄像头唯一标识 | Unique camera identifier
            source: 来源类型 (local, network) | Source type (local, network)
            resolution: 分辨率 (720p, 1080p, 4K) | Resolution (720p, 1080p, 4K)
        """
        try:
            # 将camera_id转换为整数 | Convert camera_id to integer
            cam_id = int(camera_id)
            
            # 设置分辨率参数 | Set resolution parameters
            params = {}
            if resolution.lower() == "720p":
                params["width"] = 1280
                params["height"] = 720
            elif resolution.lower() == "1080p":
                params["width"] = 1920
                params["height"] = 1080
            elif resolution.lower() == "4k":
                params["width"] = 3840
                params["height"] = 2160
            
            # 使用CameraManager启动摄像头 | Start camera using CameraManager
            success = self.camera_manager.start_camera(cam_id, params)
            
            with self.data_lock:
                self.camera_sources[camera_id] = {
                    "type": "camera",
                    "source": source,
                    "resolution": resolution,
                    "status": "connected" if success else "failed",
                    "last_update": time.time()
                }
                self.active_connections[camera_id] = success
            
            if success:
                logger.info(f"摄像头已连接: {camera_id} | Camera connected: {camera_id}")
            else:
                logger.error(f"摄像头连接失败: {camera_id} | Failed to connect camera: {camera_id}")
            
        except ValueError:
            logger.error(f"无效的摄像头ID: {camera_id} | Invalid camera ID: {camera_id}")
            with self.data_lock:
                self.camera_sources[camera_id] = {
                    "type": "camera",
                    "source": source,
                    "resolution": resolution,
                    "status": "failed",
                    "last_update": time.time()
                }
                self.active_connections[camera_id] = False

    def connect_microphone(self, mic_id: str, source: str = "local", sample_rate: int = 44100):
        """连接麦克风 | Connect to microphone
        
        参数:
            mic_id: 麦克风唯一标识 | Unique microphone identifier
            source: 来源类型 (local, network) | Source type (local, network)
            sample_rate: 采样率 | Sample rate
        """
        with self.data_lock:
            self.microphone_sources[mic_id] = {
                "type": "microphone",
                "source": source,
                "sample_rate": sample_rate,
                "status": "connected",
                "last_update": time.time()
            }
            self.active_connections[mic_id] = True
        logger.info(f"麦克风已连接: {mic_id} | Microphone connected: {mic_id}")

    def connect_network_stream(self, stream_id: str, url: str, stream_type: str):
        """连接网络流 | Connect to network stream
        
        参数:
            stream_id: 流唯一标识 | Unique stream identifier
            url: 流URL | Stream URL
            stream_type: 流类型 (video, audio) | Stream type (video, audio)
        """
        with self.data_lock:
            self.network_streams[stream_id] = {
                "type": stream_type,
                "url": url,
                "status": "connected",
                "last_update": time.time()
            }
            self.active_connections[stream_id] = True
        logger.info(f"网络流已连接: {stream_id} | Network stream connected: {stream_id}")

    def update_sensor_data(self, sensor_type: str, data: Dict[str, Any]):
        """更新传感器数据 | Update sensor data
        
        参数:
            sensor_type: 传感器类型 | Sensor type
            data: 传感器数据 | Sensor data
        """
        with self.data_lock:
            self.sensor_data[sensor_type] = {
                "data": data,
                "timestamp": time.time()
            }
        logger.debug(f"传感器数据更新: {sensor_type} | Sensor data updated: {sensor_type}")

    def get_sensor_data(self) -> Dict[str, Any]:
        """获取传感器数据 | Get sensor data
        
        返回:
            当前所有传感器数据的字典 | Dictionary of current sensor data
        """
        with self.data_lock:
            # 返回传感器数据的副本 | Return a copy of sensor data
            return {k: v.copy() for k, v in self.sensor_data.items()}

    def get_camera_feed(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """获取摄像头数据 | Get camera feed
        
        参数:
            camera_id: 摄像头ID | Camera ID
        
        返回:
            摄像头数据或None | Camera data or None
        """
        try:
            # 将camera_id转换为整数 | Convert camera_id to integer
            cam_id = int(camera_id)
            
            # 使用CameraManager获取实际的摄像头帧数据 | Get actual camera frame using CameraManager
            frame_data = self.camera_manager.get_camera_frame(cam_id)
            
            if frame_data:
                # 更新最后更新时间 | Update last update time
                with self.data_lock:
                    if camera_id in self.camera_sources:
                        self.camera_sources[camera_id]["last_update"] = time.time()
                
                # 返回帧数据 | Return frame data
                return {
                    "camera_id": camera_id,
                    "frame": frame_data["frame"],
                    "timestamp": frame_data["timestamp"]
                }
            
            return None
        except ValueError:
            logger.error(f"无效的摄像头ID: {camera_id} | Invalid camera ID: {camera_id}")
            return None

    def get_audio_stream(self, mic_id: str) -> Optional[Dict[str, Any]]:
        """获取音频流 | Get audio stream
        
        参数:
            mic_id: 麦克风ID | Microphone ID
        
        返回:
            音频数据或None | Audio data or None
        """
        with self.data_lock:
            if mic_id in self.microphone_sources:
                # 模拟返回音频数据 | Simulate returning audio data
                return {
                    "mic_id": mic_id,
                    "audio": "模拟音频数据 | Simulated audio data",
                    "timestamp": time.time()
                }
            return None

    def get_network_stream(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """获取网络流 | Get network stream
        
        参数:
            stream_id: 流ID | Stream ID
        
        返回:
            网络流数据或None | Network stream data or None
        """
        with self.data_lock:
            if stream_id in self.network_streams:
                # 模拟返回网络流数据 | Simulate returning network stream data
                return {
                    "stream_id": stream_id,
                    "data": "模拟网络流数据 | Simulated network stream data",
                    "timestamp": time.time()
                }
            return None

    def disconnect_source(self, source_id: str):
        """断开连接 | Disconnect source
        
        参数:
            source_id: 源ID | Source ID
        """
        with self.data_lock:
            if source_id in self.active_connections:
                self.active_connections[source_id] = False
                
                # 如果是摄像头源，使用CameraManager停止摄像头 | If it's a camera source, stop it using CameraManager
                if source_id in self.camera_sources:
                    try:
                        cam_id = int(source_id)
                        self.camera_manager.stop_camera(cam_id)
                    except ValueError:
                        # 如果无法转换为整数，则忽略 | Ignore if cannot convert to integer
                        pass
                
                logger.info(f"源已断开: {source_id} | Source disconnected: {source_id}")

    def _update_data_loop(self):
        """数据更新循环 | Data update loop"""
        while True:
            try:
                # 更新所有活动连接的数据 | Update data for all active connections
                with self.data_lock:
                    # 更新摄像头数据 | Update camera data
                    for cam_id, cam_info in self.camera_sources.items():
                        if self.active_connections.get(cam_id, False):
                            # 这里可以添加实际的数据采集逻辑 | Add actual data collection logic here
                            cam_info["last_update"] = time.time()
                    
                    # 更新麦克风数据 | Update microphone data
                    for mic_id, mic_info in self.microphone_sources.items():
                        if self.active_connections.get(mic_id, False):
                            # 这里可以添加实际的数据采集逻辑 | Add actual data collection logic here
                            mic_info["last_update"] = time.time()
                    
                    # 更新网络流数据 | Update network stream data
                    for stream_id, stream_info in self.network_streams.items():
                        if self.active_connections.get(stream_id, False):
                            # 这里可以添加实际的数据采集逻辑 | Add actual data collection logic here
                            stream_info["last_update"] = time.time()
                
                # 每0.5秒更新一次 | Update every 0.5 seconds
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"数据更新循环出错: {str(e)} | Error in data update loop: {str(e)}")
                time.sleep(1)
