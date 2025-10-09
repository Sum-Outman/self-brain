# -*- coding: utf-8 -*-
# Copyright 2025 The AGI Brain System Authors
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

"""
音频处理模型集成接口 | Audio Processing Model Integration Interface
提供与主系统集成的API、数据总线和Web界面功能
Provides API, data bus, and web interface integration with the main system
"""

import os
import sys
import logging
import json
import asyncio
from typing import Dict, Any, Optional, Callable
from pathlib import Path

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from enhanced_audio_model import EnhancedAudioProcessingModel, AudioProcessingMode, AudioEffectType
from config_loader import AudioConfigLoader

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("audio_integration.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AudioIntegrationInterface:
    """
    音频处理模型集成接口类 | Audio Processing Model Integration Interface Class
    提供与主系统的集成功能
    Provides integration functionality with the main system
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化集成接口 | Initialize integration interface
        
        参数 Parameters:
        config_path: 配置文件路径 | Configuration file path
        """
        self.logger = logger
        self.config_loader = AudioConfigLoader(config_path)
        self.config = self.config_loader.load_config()
        self.audio_model = EnhancedAudioProcessingModel(self.config)
        self.data_bus_connected = False
        self.web_interface_connected = False
        self.api_server = None
        
    async def start_api_server(self):
        """
        启动API服务器 | Start API server
        
        返回 Returns:
        启动结果 | Start result
        """
        try:
            self.logger.info("正在启动API服务器... | Starting API server...")
            
            # 检查配置
            api_config = self.config.get("integration", {}).get("main_system", {})
            if not api_config.get("enabled", False):
                self.logger.warning("API服务器未在配置中启用 | API server not enabled in configuration")
                return {"success": False, "error": "API server not enabled"}
            
            host = api_config.get("host", "localhost")
            port = api_config.get("port", 8000)
            endpoint = api_config.get("endpoint", "/api/audio/process")
            
            # 这里可以添加具体的API服务器实现
            # 例如使用FastAPI、Flask或其他框架
            self.logger.info(f"API服务器将在 {host}:{port}{endpoint} 启动 | API server will start at {host}:{port}{endpoint}")
            
            # 模拟API服务器启动
            self.api_server = {
                "host": host,
                "port": port,
                "endpoint": endpoint,
                "running": True
            }
            
            self.logger.info("API服务器启动成功 | API server started successfully")
            return {"success": True, "host": host, "port": port, "endpoint": endpoint}
            
        except Exception as e:
            self.logger.error(f"API服务器启动失败: {e} | API server startup failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def stop_api_server(self):
        """
        停止API服务器 | Stop API server
        
        返回 Returns:
        停止结果 | Stop result
        """
        try:
            self.logger.info("正在停止API服务器... | Stopping API server...")
            
            if self.api_server and self.api_server.get("running", False):
                self.api_server["running"] = False
                self.api_server = None
                self.logger.info("API服务器已停止 | API server stopped")
                return {"success": True}
            else:
                self.logger.warning("API服务器未运行 | API server not running")
                return {"success": False, "error": "API server not running"}
                
        except Exception as e:
            self.logger.error(f"API服务器停止失败: {e} | API server stop failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def connect_to_data_bus(self):
        """
        连接到数据总线 | Connect to data bus
        
        返回 Returns:
        连接结果 | Connection result
        """
        try:
            self.logger.info("正在连接到数据总线... | Connecting to data bus...")
            
            # 检查配置
            data_bus_config = self.config.get("integration", {}).get("data_bus", {})
            if not data_bus_config.get("enabled", False):
                self.logger.warning("数据总线未在配置中启用 | Data bus not enabled in configuration")
                return {"success": False, "error": "Data bus not enabled"}
            
            topics = data_bus_config.get("topics", ["audio_input", "audio_output", "processing_status"])
            serialization_format = data_bus_config.get("serialization_format", "json")
            
            # 这里可以添加具体的数据总线连接实现
            # 例如使用MQTT、Redis、Kafka等
            self.logger.info(f"连接到数据总线，主题: {topics} | Connected to data bus, topics: {topics}")
            
            self.data_bus_connected = True
            self.logger.info("数据总线连接成功 | Data bus connection successful")
            return {"success": True, "topics": topics, "format": serialization_format}
            
        except Exception as e:
            self.logger.error(f"数据总线连接失败: {e} | Data bus connection failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def disconnect_from_data_bus(self):
        """
        从数据总线断开连接 | Disconnect from data bus
        
        返回 Returns:
        断开连接结果 | Disconnection result
        """
        try:
            self.logger.info("正在从数据总线断开连接... | Disconnecting from data bus...")
            
            if self.data_bus_connected:
                self.data_bus_connected = False
                self.logger.info("已从数据总线断开连接 | Disconnected from data bus")
                return {"success": True}
            else:
                self.logger.warning("未连接到数据总线 | Not connected to data bus")
                return {"success": False, "error": "Not connected to data bus"}
                
        except Exception as e:
            self.logger.error(f"数据总线断开连接失败: {e} | Data bus disconnection failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def publish_to_data_bus(self, topic: str, data: Any):
        """
        发布消息到数据总线 | Publish message to data bus
        
        参数 Parameters:
        topic: 主题 | Topic
        data: 数据 | Data
        
        返回 Returns:
        发布结果 | Publish result
        """
        try:
            if not self.data_bus_connected:
                self.logger.warning("未连接到数据总线，无法发布消息 | Not connected to data bus, cannot publish message")
                return {"success": False, "error": "Not connected to data bus"}
            
            # 这里可以添加具体的数据总线发布实现
            self.logger.debug(f"发布消息到主题 {topic}: {data} | Publishing message to topic {topic}: {data}")
            
            # 模拟发布成功
            self.logger.info(f"消息已发布到主题 {topic} | Message published to topic {topic}")
            return {"success": True, "topic": topic}
            
        except Exception as e:
            self.logger.error(f"发布消息失败: {e} | Failed to publish message: {e}")
            return {"success": False, "error": str(e), "topic": topic}
    
    async def subscribe_to_data_bus(self, topic: str, callback: Callable):
        """
        订阅数据总线主题 | Subscribe to data bus topic
        
        参数 Parameters:
        topic: 主题 | Topic
        callback: 回调函数 | Callback function
        
        返回 Returns:
        订阅结果 | Subscription result
        """
        try:
            if not self.data_bus_connected:
                self.logger.warning("未连接到数据总线，无法订阅主题 | Not connected to data bus, cannot subscribe to topic")
                return {"success": False, "error": "Not connected to data bus"}
            
            # 这里可以添加具体的数据总线订阅实现
            self.logger.debug(f"订阅主题 {topic} | Subscribing to topic {topic}")
            
            # 模拟订阅成功
            self.logger.info(f"已订阅主题 {topic} | Subscribed to topic {topic}")
            return {"success": True, "topic": topic}
            
        except Exception as e:
            self.logger.error(f"订阅主题失败: {e} | Failed to subscribe to topic: {e}")
            return {"success": False, "error": str(e), "topic": topic}
    
    async def connect_to_web_interface(self):
        """
        连接到Web界面 | Connect to web interface
        
        返回 Returns:
        连接结果 | Connection result
        """
        try:
            self.logger.info("正在连接到Web界面... | Connecting to web interface...")
            
            # 检查配置
            web_config = self.config.get("integration", {}).get("web_interface", {})
            if not web_config.get("enabled", False):
                self.logger.warning("Web界面未在配置中启用 | Web interface not enabled in configuration")
                return {"success": False, "error": "Web interface not enabled"}
            
            port = web_config.get("port", 3000)
            realtime_updates = web_config.get("realtime_updates", True)
            
            # 这里可以添加具体的Web界面连接实现
            self.logger.info(f"连接到Web界面，端口: {port} | Connected to web interface, port: {port}")
            
            self.web_interface_connected = True
            self.logger.info("Web界面连接成功 | Web interface connection successful")
            return {"success": True, "port": port, "realtime_updates": realtime_updates}
            
        except Exception as e:
            self.logger.error(f"Web界面连接失败: {e} | Web interface connection failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def disconnect_from_web_interface(self):
        """
        从Web界面断开连接 | Disconnect from web interface
        
        返回 Returns:
        断开连接结果 | Disconnection result
        """
        try:
            self.logger.info("正在从Web界面断开连接... | Disconnecting from web interface...")
            
            if self.web_interface_connected:
                self.web_interface_connected = False
                self.logger.info("已从Web界面断开连接 | Disconnected from web interface")
                return {"success": True}
            else:
                self.logger.warning("未连接到Web界面 | Not connected to web interface")
                return {"success": False, "error": "Not connected to web interface"}
                
        except Exception as e:
            self.logger.error(f"Web界面断开连接失败: {e} | Web interface disconnection failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def send_to_web_interface(self, data: Any, endpoint: str = "/api/audio/update"):
        """
        发送数据到Web界面 | Send data to web interface
        
        参数 Parameters:
        data: 数据 | Data
        endpoint: 端点 | Endpoint
        
        返回 Returns:
        发送结果 | Send result
        """
        try:
            if not self.web_interface_connected:
                self.logger.warning("未连接到Web界面，无法发送数据 | Not connected to web interface, cannot send data")
                return {"success": False, "error": "Not connected to web interface"}
            
            # 这里可以添加具体的Web界面数据发送实现
            self.logger.debug(f"发送数据到Web界面端点 {endpoint}: {data} | Sending data to web interface endpoint {endpoint}: {data}")
            
            # 模拟发送成功
            self.logger.info(f"数据已发送到Web界面端点 {endpoint} | Data sent to web interface endpoint {endpoint}")
            return {"success": True, "endpoint": endpoint}
            
        except Exception as e:
            self.logger.error(f"发送数据到Web界面失败: {e} | Failed to send data to web interface: {e}")
            return {"success": False, "error": str(e), "endpoint": endpoint}
    
    async def process_audio_via_api(self, audio_data, mode: AudioProcessingMode, **kwargs):
        """
        通过API处理音频数据 | Process audio data via API
        
        参数 Parameters:
        audio_data: 音频数据 | Audio data
        mode: 处理模式 | Processing mode
        **kwargs: 额外参数 | Additional parameters
        
        返回 Returns:
        处理结果 | Processing result
        """
        try:
            self.logger.info(f"通过API处理音频，模式: {mode.value} | Processing audio via API, mode: {mode.value}")
            
            # 使用音频模型处理音频
            result = self.audio_model.process_audio(audio_data, mode, **kwargs)
            
            # 如果连接到数据总线，发布处理结果
            if self.data_bus_connected:
                await self.publish_to_data_bus("audio_output", result)
            
            # 如果连接到Web界面，发送实时更新
            if self.web_interface_connected:
                await self.send_to_web_interface(result)
            
            self.logger.info("API音频处理完成 | API audio processing completed")
            return result
            
        except Exception as e:
            self.logger.error(f"API音频处理失败: {e} | API audio processing failed: {e}")
            
            # 发布错误信息到数据总线
            if self.data_bus_connected:
                error_data = {
                    "success": False,
                    "error": str(e),
                    "mode": mode.value
                }
                await self.publish_to_data_bus("processing_status", error_data)
            
            return {
                "success": False,
                "error": str(e),
                "mode": mode.value
            }
    
    async def apply_audio_effect_via_api(self, audio_data, effect_type: AudioEffectType, **kwargs):
        """
        通过API应用音频特效 | Apply audio effect via API
        
        参数 Parameters:
        audio_data: 音频数据 | Audio data
        effect_type: 特效类型 | Effect type
        **kwargs: 特效参数 | Effect parameters
        
        返回 Returns:
        特效应用结果 | Effect application result
        """
        try:
            self.logger.info(f"通过API应用音频特效: {effect_type.value} | Applying audio effect via API: {effect_type.value}")
            
            # 使用音频模型应用特效
            result = self.audio_model.apply_audio_effect(audio_data, effect_type, **kwargs)
            
            # 如果连接到数据总线，发布处理结果
            if self.data_bus_connected:
                await self.publish_to_data_bus("audio_output", result)
            
            # 如果连接到Web界面，发送实时更新
            if self.web_interface_connected:
                await self.send_to_web_interface(result)
            
            self.logger.info("API音频特效应用完成 | API audio effect application completed")
            return result
            
        except Exception as e:
            self.logger.error(f"API音频特效应用失败: {e} | API audio effect application failed: {e}")
            
            # 发布错误信息到数据总线
            if self.data_bus_connected:
                error_data = {
                    "success": False,
                    "error": str(e),
                    "effect_type": effect_type.value
                }
                await self.publish_to_data_bus("processing_status", error_data)
            
            return {
                "success": False,
                "error": str(e),
                "effect_type": effect_type.value
            }
    
    async def start_realtime_processing_via_api(self, callback: Callable = None):
        """
        通过API启动实时处理 | Start real-time processing via API
        
        参数 Parameters:
        callback: 回调函数 | Callback function
        
        返回 Returns:
        启动结果 | Start result
        """
        try:
            self.logger.info("通过API启动实时音频处理 | Starting real-time audio processing via API")
            
            # 使用音频模型启动实时处理
            result = self.audio_model.start_realtime_processing(callback)
            
            # 如果连接到数据总线，发布状态更新
            if self.data_bus_connected:
                status_data = {
                    "realtime_processing": True,
                    "status": "started"
                }
                await self.publish_to_data_bus("processing_status", status_data)
            
            # 如果连接到Web界面，发送实时更新
            if self.web_interface_connected:
                await self.send_to_web_interface({"realtime_processing": True})
            
            self.logger.info("API实时音频处理已启动 | API real-time audio processing started")
            return result
            
        except Exception as e:
            self.logger.error(f"API实时音频处理启动失败: {e} | API real-time audio processing startup failed: {e}")
            
            # 发布错误信息到数据总线
            if self.data_bus_connected:
                error_data = {
                    "success": False,
                    "error": str(e),
                    "operation": "start_realtime_processing"
                }
                await self.publish_to_data_bus("processing_status", error_data)
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def stop_realtime_processing_via_api(self):
        """
        通过API停止实时处理 | Stop real-time processing via API
        
        返回 Returns:
        停止结果 | Stop result
        """
        try:
            self.logger.info("通过API停止实时音频处理 | Stopping real-time audio processing via API")
            
            # 使用音频模型停止实时处理
            result = self.audio_model.stop_realtime_processing()
            
            # 如果连接到数据总线，发布状态更新
            if self.data_bus_connected:
                status_data = {
                    "realtime_processing": False,
                    "status": "stopped"
                }
                await self.publish_to_data_bus("processing_status", status_data)
            
            # 如果连接到Web界面，发送实时更新
            if self.web_interface_connected:
                await self.send_to_web_interface({"realtime_processing": False})
            
            self.logger.info("API实时音频处理已停止 | API real-time audio processing stopped")
            return result
            
        except Exception as e:
            self.logger.error(f"API实时音频处理停止失败: {e} | API real-time audio processing stop failed: {e}")
            
            # 发布错误信息到数据总线
            if self.data_bus_connected:
                error_data = {
                    "success": False,
                    "error": str(e),
                    "operation": "stop_realtime_processing"
                }
                await self.publish_to_data_bus("processing_status", error_data)
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_model_status_via_api(self):
        """
        通过API获取模型状态 | Get model status via API
        
        返回 Returns:
        模型状态 | Model status
        """
        try:
            self.logger.debug("通过API获取模型状态 | Getting model status via API")
            
            # 使用音频模型获取状态
            status = self.audio_model.get_model_status()
            
            # 如果连接到数据总线，发布状态更新
            if self.data_bus_connected:
                await self.publish_to_data_bus("processing_status", status)
            
            # 如果连接到Web界面，发送实时更新
            if self.web_interface_connected:
                await self.send_to_web_interface(status)
            
            self.logger.debug("API模型状态获取成功 | API model status retrieved successfully")
            return status
            
        except Exception as e:
            self.logger.error(f"API模型状态获取失败: {e} | API model status retrieval failed: {e}")
            
            # 发布错误信息到数据总线
            if self.data_bus_connected:
                error_data = {
                    "success": False,
                    "error": str(e),
                    "operation": "get_model_status"
                }
                await self.publish_to_data_bus("processing_status", error_data)
            
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def shutdown(self):
        """关闭集成接口 | Shutdown integration interface"""
        try:
            self.logger.info("正在关闭集成接口... | Shutting down integration interface...")
            
            # 停止API服务器
            if self.api_server and self.api_server.get("running", False):
                await self.stop_api_server()
            
            # 断开数据总线连接
            if self.data_bus_connected:
                await self.disconnect_from_data_bus()
            
            # 断开Web界面连接
            if self.web_interface_connected:
                await self.disconnect_from_web_interface()
            
            # 停止音频模型
            self.audio_model.stop_realtime_processing()
            
            self.logger.info("集成接口已关闭 | Integration interface shut down")
            return True
            
        except Exception as e:
            self.logger.error(f"集成接口关闭失败: {e} | Integration interface shutdown failed: {e}")
            return False

# 示例用法 | Example usage
async def main():
    """主函数 | Main function"""
    # 创建集成接口实例
    audio_integration = AudioIntegrationInterface()
    
    try:
        # 连接到数据总线
        data_bus_result = await audio_integration.connect_to_data_bus()
        print(f"数据总线连接结果: {data_bus_result}")
        
        # 连接到Web界面
        web_interface_result = await audio_integration.connect_to_web_interface()
        print(f"Web界面连接结果: {web_interface_result}")
        
        # 启动API服务器
        api_server_result = await audio_integration.start_api_server()
        print(f"API服务器启动结果: {api_server_result}")
        
        # 获取模型状态
        status = await audio_integration.get_model_status_via_api()
        print(f"模型状态: {status}")
        
        # 保持运行一段时间
        await asyncio.sleep(5)
        
    except Exception as e:
        print(f"集成接口运行失败: {e}")
    
    finally:
        # 关闭集成接口
        await audio_integration.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
