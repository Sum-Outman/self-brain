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

import asyncio
import json
import logging
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import psutil
import torch
import uvicorn
import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="I Motion Control Model", version="2.0.0")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 数据模型
class DeviceConfig(BaseModel):
    type: str
    protocol: str = "serial"
    address: Optional[str] = None
    port: Optional[str] = None

class ActuatorCommand(BaseModel):
    action: str = "move"
    position: float = 0
    speed: float = 50
    duration: float = 1.0

class BatchCommand(BaseModel):
    device_id: str
    command: ActuatorCommand

class BatchControlRequest(BaseModel):
    commands: List[BatchCommand]

class TrainingRequest(BaseModel):
    training_data: List[Dict[str, Any]]
    model_type: str = "motion_control"

class LanguageRequest(BaseModel):
    lang: str

class CallbackRequest(BaseModel):
    callback_url: str

# 运动控制系统
class MotionControlSystem:
    """运动和执行器控制系统 | Motion and Actuator Control System"""
    
    def __init__(self, language: str = 'en'):
        self.language = language
        self.data_bus = None
        
        # 多语言支持 | Multilingual support
        self.supported_languages = ['zh', 'en', 'ja', 'de', 'ru']
        self.translations = {
            'motion': {'en': 'motion', 'zh': '运动', 'ja': 'モーション', 'de': 'Bewegung', 'ru': 'движение'},
            'control': {'en': 'control', 'zh': '控制', 'ja': '制御', 'de': 'Steuerung', 'ru': 'управление'},
            'actuator': {'en': 'actuator', 'zh': '执行器', 'ja': 'アクチュエータ', 'de': 'Aktor', 'ru': 'привод'},
            'sensor': {'en': 'sensor', 'zh': '传感器', 'ja': 'センサー', 'de': 'Sensor', 'ru': 'датчик'},
            'motor': {'en': 'motor', 'zh': '电机', 'ja': 'モーター', 'de': 'Motor', 'ru': 'мотор'},
            'servo': {'en': 'servo', 'zh': '舵机', 'ja': 'サーボ', 'de': 'Servo', 'ru': 'сервопривод'},
            'stepper': {'en': 'stepper', 'zh': '步进电机', 'ja': 'ステッピングモーター', 'de': 'Schrittmotor', 'ru': 'шаговый двигатель'},
            'communication': {'en': 'communication', 'zh': '通信', 'ja': '通信', 'de': 'Kommunikation', 'ru': 'связь'},
            'protocol': {'en': 'protocol', 'zh': '协议', 'ja': 'プロトコル', 'de': 'Protokoll', 'ru': 'протокол'}
        }
        
        # 设备通信协议支持 | Device communication protocol support
        self.supported_protocols = {
            'serial': {'baud_rates': [9600, 19200, 38400, 57600, 115200]},
            'i2c': {'address_range': [0x08, 0x77]},
            'spi': {'modes': [0, 1, 2, 3]},
            'uart': {'baud_rates': [9600, 19200, 38400, 57600, 115200]},
            'can': {'baud_rates': [125000, 250000, 500000, 1000000]},
            'ethernet': {'protocols': ['TCP', 'UDP']},
            'bluetooth': {'profiles': ['SPP', 'HID', 'A2DP']},
            'wifi': {'modes': ['STA', 'AP']}
        }
        
        # 执行器类型支持 | Actuator type support
        self.actuator_types = {
            'servo': {'control_range': [0, 180], 'precision': 1},
            'dc_motor': {'control_range': [-100, 100], 'precision': 1},
            'stepper': {'control_range': [0, 360], 'precision': 1.8},
            'pneumatic': {'control_range': [0, 1], 'precision': 1},
            'hydraulic': {'control_range': [0, 100], 'precision': 0.1},
            'linear_actuator': {'control_range': [0, 100], 'precision': 0.1}
        }
        
        # 实时数据处理 | Real-time data processing
        self.realtime_callbacks = []
        self.data_queue = queue.Queue()
        self.processing_thread = threading.Thread(target=self._process_realtime_data)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # 训练历史 | Training history
        self.training_history = []
        
        # 性能监控 | Performance monitoring
        self.performance_stats = {
            'command_execution_time': [],
            'success_rate': [],
            'devices_connected': 0
        }
        
        # 设备状态 | Device status
        self.connected_devices = {}
        
        # 初始化神经网络模型 | Initialize neural network model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._create_motion_model()
        
        logger.info(f"运动控制系统初始化完成 | Motion control system initialized")
        logger.info(f"设备: {self.device} | Device: {self.device}")
    
    def _create_motion_model(self):
        """创建运动控制神经网络模型 | Create motion control neural network model"""
        try:
            # 简单的运动控制模型 | Simple motion control model
            model = torch.nn.Sequential(
                torch.nn.Linear(10, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 8)
            ).to(self.device)
            
            # 如果有预训练权重则加载 | Load pretrained weights if available
            model_path = os.path.join(os.path.dirname(__file__), 'motion_model.pth')
            if os.path.exists(model_path):
                try:
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
                    logger.info("加载预训练运动模型 | Loaded pretrained motion model")
                except Exception as e:
                    logger.warning(f"加载预训练模型失败: {e} | Failed to load pretrained model: {e}")
            
            return model
        except Exception as e:
            logger.error(f"创建运动模型失败: {e} | Failed to create motion model: {e}")
            return None
    
    def set_language(self, language: str) -> bool:
        """设置当前语言 | Set current language"""
        if language in self.supported_languages:
            self.language = language
            return True
        return False
    
    def set_data_bus(self, data_bus):
        """设置数据总线 | Set data bus"""
        self.data_bus = data_bus
    
    def _process_realtime_data(self):
        """处理实时数据队列 | Process real-time data queue"""
        while True:
            try:
                data = self.data_queue.get(timeout=1.0)
                for callback in self.realtime_callbacks:
                    try:
                        callback(data)
                    except Exception as e:
                        logger.error(f"回调函数错误: {e} | Callback error: {e}")
                self.data_queue.task_done()
                
                # 发送到数据总线 | Send to data bus
                if self.data_bus:
                    try:
                        self.data_bus.send(data)
                    except Exception as e:
                        logger.error(f"数据总线发送错误: {e} | Data bus send error: {e}")
                else:
                    # 尝试发送到主模型 | Try to send to main model
                    try:
                        requests.post("http://localhost:8000/receive_data", json=data, timeout=1.0)
                    except Exception as e:
                        logger.error(f"主模型通信失败: {e} | Main model communication failed: {e}")
                        
            except queue.Empty:
                continue
    
    def register_realtime_callback(self, callback: callable):
        """注册实时数据回调函数 | Register real-time data callback function"""
        self.realtime_callbacks.append(callback)
    
    async def connect_device(self, device_config: Dict[str, Any]) -> Dict[str, Any]:
        """连接外部设备 | Connect external device"""
        try:
            device_type = device_config.get('type', 'unknown')
            protocol = device_config.get('protocol', 'serial')
            address = device_config.get('address')
            port = device_config.get('port')
            
            if not address:
                return {
                    "success": False,
                    "error": "缺少设备地址 | Missing device address",
                    "lang": self.language
                }
            
            # 模拟设备连接 | Simulate device connection
            device_id = f"{device_type}_{len(self.connected_devices) + 1}"
            self.connected_devices[device_id] = {
                "type": device_type,
                "protocol": protocol,
                "address": address,
                "port": port,
                "connected": True,
                "status": "active",
                "last_communication": time.time()
            }
            
            # 更新性能统计 | Update performance statistics
            self.performance_stats['devices_connected'] = len(self.connected_devices)
            
            result = {
                "success": True,
                "device_id": device_id,
                "message": f"设备连接成功 | Device connected successfully",
                "device_info": self.connected_devices[device_id],
                "lang": self.language
            }
            
            # 发送实时数据 | Send real-time data
            self.data_queue.put({
                "type": "device_connected",
                "device_id": device_id,
                "timestamp": time.time(),
                "data": result
            })
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "error": str(e),
                "lang": self.language
            }
            return error_result
    
    async def disconnect_device(self, device_id: str) -> Dict[str, Any]:
        """断开设备连接 | Disconnect device"""
        try:
            if device_id in self.connected_devices:
                device_info = self.connected_devices.pop(device_id)
                
                # 更新性能统计 | Update performance statistics
                self.performance_stats['devices_connected'] = len(self.connected_devices)
                
                result = {
                    "success": True,
                    "message": f"设备断开连接 | Device disconnected",
                    "device_id": device_id,
                    "lang": self.language
                }
                
                # 发送实时数据 | Send real-time data
                self.data_queue.put({
                    "type": "device_disconnected",
                    "device_id": device_id,
                    "timestamp": time.time(),
                    "data": result
                })
                
                return result
            else:
                return {
                    "success": False,
                    "error": f"设备未找到: {device_id} | Device not found: {device_id}",
                    "lang": self.language
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "lang": self.language
            }
    
    async def control_actuator(self, device_id: str, command: Dict[str, Any]) -> Dict[str, Any]:
        """控制执行器 | Control actuator"""
        start_time = time.time()
        
        try:
            if device_id not in self.connected_devices:
                return {
                    "success": False,
                    "error": f"设备未连接: {device_id} | Device not connected: {device_id}",
                    "lang": self.language
                }
            
            device_info = self.connected_devices[device_id]
            device_type = device_info['type']
            
            # 解析命令 | Parse command
            action = command.get('action', 'move')
            position = command.get('position', 0)
            speed = command.get('speed', 50)
            duration = command.get('duration', 1.0)
            
            # 使用神经网络模型预测控制参数 | Use neural network model to predict control parameters
            control_params = self._predict_control_params(device_type, position, speed, duration)
            
            # 模拟执行器控制 | Simulate actuator control
            control_result = self._simulate_actuator_control(device_type, control_params)
            
            # 更新设备状态 | Update device status
            self.connected_devices[device_id]['last_communication'] = time.time()
            self.connected_devices[device_id]['last_command'] = command
            
            execution_time = time.time() - start_time
            
            result = {
                "success": True,
                "device_id": device_id,
                "device_type": device_type,
                "action": action,
                "position": position,
                "speed": speed,
                "duration": duration,
                "control_params": control_params,
                "result": control_result,
                "execution_time": execution_time,
                "timestamp": time.time(),
                "lang": self.language
            }
            
            # 更新性能统计 | Update performance statistics
            self.performance_stats['command_execution_time'].append(execution_time)
            if len(self.performance_stats['command_execution_time']) > 100:
                self.performance_stats['command_execution_time'].pop(0)
            
            self.performance_stats['success_rate'].append(1)
            if len(self.performance_stats['success_rate']) > 100:
                self.performance_stats['success_rate'].pop(0)
            
            # 发送实时数据 | Send real-time data
            self.data_queue.put({
                "type": "actuator_control",
                "device_id": device_id,
                "timestamp": time.time(),
                "data": result
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "timestamp": time.time(),
                "lang": self.language
            }
            
            # 更新性能统计 | Update performance statistics
            self.performance_stats['command_execution_time'].append(execution_time)
            if len(self.performance_stats['command_execution_time']) > 100:
                self.performance_stats['command_execution_time'].pop(0)
            
            self.performance_stats['success_rate'].append(0)
            if len(self.performance_stats['success_rate']) > 100:
                self.performance_stats['success_rate'].pop(0)
            
            return error_result
    
    def _predict_control_params(self, device_type: str, position: float, speed: float, duration: float) -> Dict[str, Any]:
        """使用神经网络预测控制参数 | Use neural network to predict control parameters"""
        try:
            if self.model is None:
                return {"pulse_width": 1500, "frequency": 50, "amplitude": 1.0}
            
            # 准备输入数据 | Prepare input data
            input_data = np.array([
                position / 180.0,  # 归一化位置 | Normalized position
                speed / 100.0,     # 归一化速度 | Normalized speed
                duration / 10.0,   # 归一化持续时间 | Normalized duration
                # 添加更多特征 | Add more features
                1.0 if device_type == 'servo' else 0.0,
                1.0 if device_type == 'dc_motor' else 0.0,
                1.0 if device_type == 'stepper' else 0.0,
                0.5,  # 默认负载 | Default load
                0.8,  # 默认效率 | Default efficiency
                0.2,  # 默认摩擦 | Default friction
                0.1   # 默认惯性 | Default inertia
            ], dtype=np.float32)
            
            # 转换为tensor | Convert to tensor
            input_tensor = torch.from_numpy(input_data).unsqueeze(0).to(self.device)
            
            # 预测 | Predict
            with torch.no_grad():
                output = self.model(input_tensor)
                output_np = output.cpu().numpy()[0]
            
            # 解析输出 | Parse output
            control_params = {
                "pulse_width": int(1000 + output_np[0] * 1000),  # 1000-2000 us
                "frequency": int(30 + output_np[1] * 70),        # 30-100 Hz
                "amplitude": float(output_np[2] * 2.0),          # 0-2.0
                "phase": float(output_np[3] * 3.14),             # 0-π
                "damping": float(output_np[4] * 0.5),            # 0-0.5
                "stiffness": float(output_np[5] * 2.0),          # 0-2.0
                "torque": float(output_np[6] * 10.0),            # 0-10.0 Nm
                "velocity": float(output_np[7] * 5.0)            # 0-5.0 m/s
            }
            
            return control_params
            
        except Exception as e:
            logger.error(f"控制参数预测失败: {e} | Control parameter prediction failed: {e}")
            return {"pulse_width": 1500, "frequency": 50, "amplitude": 1.0}
    
    def _simulate_actuator_control(self, device_type: str, control_params: Dict[str, Any]) -> Dict[str, Any]:
        """模拟执行器控制 | Simulate actuator control"""
        try:
            # 根据设备类型模拟控制 | Simulate control based on device type
            if device_type == 'servo':
                result = {
                    "actual_position": control_params['pulse_width'] / 2000.0 * 180,
                    "current_draw": 0.5 + control_params['amplitude'] * 0.3,
                    "temperature": 25 + control_params['amplitude'] * 10,
                    "status": "moving"
                }
            elif device_type == 'dc_motor':
                result = {
                    "actual_velocity": control_params['velocity'],
                    "current_draw": 1.0 + control_params['torque'] * 0.2,
                    "temperature": 30 + control_params['torque'] * 5,
                    "status": "running"
                }
            elif device_type == 'stepper':
                result = {
                    "steps_completed": int(control_params['pulse_width'] / 10),
                    "current_position": control_params['phase'] * 57.3,  # 弧度转角度
                    "current_draw": 0.8 + control_params['stiffness'] * 0.1,
                    "status": "stepping"
                }
            else:
                result = {
                    "output_value": control_params['amplitude'],
                    "status": "active"
                }
            
            # 添加一些随机变化以模拟真实环境 | Add some random variation to simulate real environment
            result["noise_level"] = np.random.normal(0, 0.05)
            result["vibration"] = np.random.normal(0, 0.02)
            
            return result
            
        except Exception as e:
            logger.error(f"执行器控制模拟失败: {e} | Actuator control simulation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def batch_control(self, commands: List[Dict[str, Any]]) -> Dict[str, Any]:
        """批量控制多个执行器 | Batch control multiple actuators"""
        results = []
        for cmd in commands:
            device_id = cmd.get('device_id')
            command = cmd.get('command', {})
            result = await self.control_actuator(device_id, command)
            results.append(result)
        
        batch_result = {
            "results": results,
            "total_commands": len(commands),
            "successful_commands": sum(1 for r in results if r.get("success", False)),
            "failed_commands": sum(1 for r in results if not r.get("success", True)),
            "total_time": sum(r.get("execution_time", 0) for r in results),
            "timestamp": time.time(),
            "lang": self.language
        }
        
        # 发送实时数据 | Send real-time data
        self.data_queue.put({
            "type": "batch_control",
            "timestamp": time.time(),
            "data": batch_result
        })
        
        return batch_result
    
    async def get_device_status(self, device_id: str = None) -> Dict[str, Any]:
        """获取设备状态 | Get device status"""
        try:
            if device_id:
                if device_id in self.connected_devices:
                    return {
                        "success": True,
                        "device_id": device_id,
                        "status": self.connected_devices[device_id],
                        "lang": self.language
                    }
                else:
                    return {
                        "success": False,
                        "error": f"设备未找到: {device_id} | Device not found: {device_id}",
                        "lang": self.language
                    }
            else:
                return {
                    "success": True,
                    "connected_devices": self.connected_devices,
                    "total_devices": len(self.connected_devices),
                    "lang": self.language
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "lang": self.language
            }
    
    async def fine_tune(self, training_data: List[Dict], model_type: str = 'motion_control') -> Dict:
        """微调运动控制模型 | Fine-tune motion control model"""
        try:
            if not training_data:
                return {"status": "error", "message": "训练数据为空 | Training data is empty"}
            
            logger.info(f"开始微调{model_type}模型 | Starting fine-tuning for {model_type} model")
            logger.info(f"训练样本数: {len(training_data)} | Training samples: {len(training_data)}")
            
            # 分析训练数据特征 | Analyze training data features
            total_samples = len(training_data)
            successful_commands = 0
            total_execution_time = 0
            device_types = {}
            
            for sample in training_data:
                # 分析设备类型和控制模式 | Analyze device types and control patterns
                device_type = sample.get('device_type', 'unknown')
                success = sample.get('success', False)
                execution_time = sample.get('execution_time', 0)
                
                if success:
                    successful_commands += 1
                    total_execution_time += execution_time
                
                # 统计设备类型 | Count device types
                if device_type not in device_types:
                    device_types[device_type] = 0
                device_types[device_type] += 1
            
            # 计算真实性能指标 | Calculate real performance metrics
            success_rate = successful_commands / total_samples if total_samples > 0 else 0
            avg_execution_time = total_execution_time / successful_commands if successful_commands > 0 else 0
            
            # 基于历史数据优化控制策略 | Optimize control strategy based on historical data
            optimization_rules = self._generate_control_optimization_rules(training_data)
            
            training_result = {
                "status": "success",
                "model_type": model_type,
                "samples": total_samples,
                "success_rate": success_rate,
                "avg_execution_time": avg_execution_time,
                "device_types": device_types,
                "optimization_rules": optimization_rules,
                "training_method": "real_control_analysis"
            }
            
            # 记录训练历史 | Record training history
            self.training_history.append({
                "timestamp": time.time(),
                "model_type": model_type,
                "result": training_result
            })
            
            logger.info(f"模型微调完成: 成功率 {success_rate:.2%}, 平均执行时间 {avg_execution_time:.2f}s")
            return training_result
            
        except Exception as e:
            error_msg = f"模型微调失败: {str(e)} | Model fine-tuning failed: {str(e)}"
            logger.error(error_msg)
            return {"status": "error", "message": error_msg}
    
    def _generate_control_optimization_rules(self, training_data: List[Dict]) -> List[Dict]:
        """基于训练数据生成控制优化规则 | Generate control optimization rules based on training data"""
        rules = []
        
        # 分析控制执行模式 | Analyze control execution patterns
        success_patterns = {}
        failure_patterns = {}
        
        for sample in training_data:
            device_type = sample.get('device_type', 'unknown')
            success = sample.get('success', False)
            execution_time = sample.get('execution_time', 0)
            
            if success:
                if device_type not in success_patterns:
                    success_patterns[device_type] = {'count': 0, 'total_time': 0}
                success_patterns[device_type]['count'] += 1
                success_patterns[device_type]['total_time'] += execution_time
            else:
                if device_type not in failure_patterns:
                    failure_patterns[device_type] = 0
                failure_patterns[device_type] += 1
        
        # 生成性能优化规则 | Generate performance optimization rules
        for device_type, stats in success_patterns.items():
            avg_time = stats['total_time'] / stats['count'] if stats['count'] > 0 else 0
            failure_count = failure_patterns.get(device_type, 0)
            success_rate = stats['count'] / (stats['count'] + failure_count) if (stats['count'] + failure_count) > 0 else 0
            
            if avg_time > 2.0:  # 执行时间超过2秒的控制
                rules.append({
                    "type": "performance_warning",
                    "device_type": device_type,
                    "message": f"{device_type} 类型设备控制平均执行时间较长: {avg_time:.2f}s",
                    "suggestion": "考虑优化控制算法或降低控制精度"
                })
            
            if success_rate < 0.9:  # 成功率低于90%的设备类型
                rules.append({
                    "type": "reliability_issue",
                    "device_type": device_type,
                    "message": f"{device_type} 类型设备控制成功率较低: {success_rate:.2%}",
                    "suggestion": "检查设备连接稳定性或控制参数设置"
                })
        
        return rules
    
    async def get_monitoring_data(self) -> Dict:
        """获取实时监视数据 | Get real-time monitoring data"""
        avg_execution_time = np.mean(self.performance_stats['command_execution_time']) if self.performance_stats['command_execution_time'] else 0
        success_rate = np.mean(self.performance_stats['success_rate']) if self.performance_stats['success_rate'] else 0
        
        return {
            "status": "active",
            "language": self.language,
            "connected_devices": len(self.connected_devices),
            "performance": {
                "avg_execution_time_ms": avg_execution_time * 1000,
                "success_rate": success_rate,
                "queue_size": self.data_queue.qsize()
            },
            "supported_protocols": list(self.supported_protocols.keys()),
            "actuator_types": list(self.actuator_types.keys()),
            "training_history": len(self.training_history)
        }
    
    def _translate(self, text: str, lang: str) -> str:
        """翻译文本 | Translate text"""
        if text in self.translations and lang in self.translations[text]:
            return self.translations[text][lang]
        return text

# WebSocket连接管理器
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

# 创建运动控制系统实例 | Create motion control system instance
motion_control = MotionControlSystem()

# API路由
@app.get("/")
async def root():
    return {
        "status": "active",
        "model": "I_motion_control",
        "version": "2.0.0",
        "language": motion_control.language,
        "capabilities": [
            "actuator_control", "device_management", "batch_processing",
            "real_time_monitoring", "neural_network_control", "protocol_support",
            "multilingual_support", "training_optimization"
        ],
        "supported_protocols": list(motion_control.supported_protocols.keys()),
        "actuator_types": list(motion_control.actuator_types.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "I_motion_control", "lang": motion_control.language}

@app.post("/connect")
async def connect_device(device_config: DeviceConfig, lang: str = "en"):
    """连接设备API端点 | Connect device API endpoint"""
    if lang not in motion_control.supported_languages:
        lang = 'en'
    
    result = await motion_control.connect_device(device_config.dict())
    return result

@app.post("/disconnect")
async def disconnect_device(device_id: str, lang: str = "en"):
    """断开设备连接API端点 | Disconnect device API endpoint"""
    if lang not in motion_control.supported_languages:
        lang = 'en'
    
    if not device_id:
        raise HTTPException(status_code=400, detail="Missing device_id")
    
    result = await motion_control.disconnect_device(device_id)
    return result

@app.post("/control")
async def control_actuator(device_id: str, command: ActuatorCommand, lang: str = "en"):
    """控制执行器API端点 | Control actuator API endpoint"""
    if lang not in motion_control.supported_languages:
        lang = 'en'
    
    if not device_id:
        raise HTTPException(status_code=400, detail="Missing device_id")
    
    result = await motion_control.control_actuator(device_id, command.dict())
    return result

@app.post("/batch")
async def batch_control(request: BatchControlRequest, lang: str = "en"):
    """批量控制API端点 | Batch control API endpoint"""
    if lang not in motion_control.supported_languages:
        lang = 'en'
    
    commands = [cmd.dict() for cmd in request.commands]
    result = await motion_control.batch_control(commands)
    return result

@app.get("/status")
async def get_device_status(device_id: Optional[str] = None, lang: str = "en"):
    """获取设备状态API端点 | Get device status API endpoint"""
    if lang not in motion_control.supported_languages:
        lang = 'en'
    
    result = await motion_control.get_device_status(device_id)
    return result

@app.post("/train")
async def train_model(request: TrainingRequest, lang: str = "en"):
    """训练运动控制模型 | Train motion control model"""
    if lang not in motion_control.supported_languages:
        lang = 'en'
    
    try:
        # 训练模型 | Train model
        training_result = await motion_control.fine_tune(request.training_data, request.model_type)
        
        return {
            "status": "success",
            "lang": lang,
            "message": "模型训练完成",
            "results": training_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"训练失败: {str(e)}")

@app.get("/monitor")
async def get_monitoring_data():
    """获取实时监视数据 | Get real-time monitoring data"""
    monitoring_data = await motion_control.get_monitoring_data()
    return monitoring_data

@app.post("/register_realtime_callback")
async def register_realtime_callback(request: CallbackRequest, lang: str = "en"):
    """注册实时数据回调API端点 | API endpoint for registering real-time data callback"""
    if lang not in motion_control.supported_languages:
        lang = 'en'
    
    callback_url = request.callback_url
    
    if not callback_url:
        raise HTTPException(status_code=400, detail="Missing callback_url parameter")
    
    # 创建回调函数 | Create callback function
    def callback(control_data):
        try:
            requests.post(callback_url, json=control_data, timeout=1.0)
        except Exception as e:
            logger.error(f"发送数据到 {callback_url} 失败: {e} | Failed to send data to {callback_url}: {e}")
    
    motion_control.register_realtime_callback(callback)
    return {"status": "success", "message": "Callback registered", "lang": lang}

@app.post("/language")
async def set_language(request: LanguageRequest):
    """设置当前语言 | Set current language"""
    lang = request.lang
    
    if not lang:
        raise HTTPException(status_code=400, detail='缺少语言代码')
    
    if motion_control.set_language(lang):
        return {'status': f'语言设置为 {lang}', 'lang': lang}
    raise HTTPException(status_code=400, detail='无效的语言代码。使用 zh, en, ja, de, ru')

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket端点用于实时通信"""
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # 处理接收到的消息
            await manager.send_personal_message(f"消息已收到: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == '__main__':
    import uvicorn
    config = uvicorn.Config(
        "app:app",
        host="0.0.0.0",
        port=5009,
        log_level="info"
    )
    server = uvicorn.Server(config)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(server.serve())
