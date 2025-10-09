# -*- coding: utf-8 -*-
# Copyright 2025 The AGI Brain System Authors
# Licensed under the Apache License, Version 2.0

"""
统一传感器处理模型 | Unified Sensor Processing Model
整合标准模式和增强模式功能
"""

import numpy as np
import threading
import time
from typing import Dict, List, Any, Optional, Callable
import logging
import json
from dataclasses import dataclass
from enum import Enum

class SensorType(Enum):
    """传感器类型枚举"""
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    MOTION = "motion"
    LIGHT = "light"
    SOUND = "sound"
    PROXIMITY = "proximity"
    ACCELEROMETER = "accelerometer"
    GYROSCOPE = "gyroscope"
    MAGNETOMETER = "magnetometer"

@dataclass
class SensorData:
    """传感器数据结构"""
    sensor_type: SensorType
    value: float
    timestamp: float
    unit: str
    metadata: Dict[str, Any] = None

class UnifiedSensorModel:
    """
    统一传感器处理模型
    支持多种传感器数据采集、处理和分析
    """
    
    def __init__(self, mode: str = "standard", config: Optional[Dict] = None):
        self.mode = mode
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 传感器配置
        self.sensors = {}
        self.data_buffer = []
        self.callbacks = {}
        
        # 实时处理
        self.is_monitoring = False
        self.monitor_thread = None
        self._lock = threading.Lock()
        
        # 初始化传感器
        self._initialize_sensors()
        
    def _initialize_sensors(self):
        """初始化所有传感器"""
        default_sensors = {
            SensorType.TEMPERATURE: {"min": -40, "max": 85, "unit": "°C"},
            SensorType.HUMIDITY: {"min": 0, "max": 100, "unit": "%"},
            SensorType.PRESSURE: {"min": 300, "max": 1100, "unit": "hPa"},
            SensorType.LIGHT: {"min": 0, "max": 65535, "unit": "lux"},
            SensorType.MOTION: {"threshold": 50, "unit": "boolean"},
            SensorType.ACCELEROMETER: {"range": 16, "unit": "g"},
            SensorType.GYROSCOPE: {"range": 2000, "unit": "°/s"},
        }
        
        for sensor_type, config in default_sensors.items():
            self.sensors[sensor_type] = {
                "config": config,
                "enabled": True,
                "last_reading": None
            }
    
    def read_sensor(self, sensor_type: SensorType) -> Optional[SensorData]:
        """读取单个传感器数据"""
        if sensor_type not in self.sensors or not self.sensors[sensor_type]["enabled"]:
            return None
        
        try:
            # 模拟传感器读取（实际应用中连接到真实传感器）
            config = self.sensors[sensor_type]["config"]
            if "min" in config and "max" in config:
                value = np.random.uniform(config["min"], config["max"])
            elif "threshold" in config:
                value = 1 if np.random.random() > 0.8 else 0
            else:
                value = np.random.uniform(0, 100)
            
            data = SensorData(
                sensor_type=sensor_type,
                value=value,
                timestamp=time.time(),
                unit=config["unit"]
            )
            
            self.sensors[sensor_type]["last_reading"] = data
            
            # 增强模式下的额外处理
            if self.mode == "enhanced":
                data = self._enhanced_processing(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"读取传感器 {sensor_type} 失败: {e}")
            return None
    
    def read_all_sensors(self) -> Dict[str, SensorData]:
        """读取所有启用传感器的数据"""
        results = {}
        for sensor_type in self.sensors:
            data = self.read_sensor(sensor_type)
            if data:
                results[sensor_type.value] = data
        return results
    
    def start_monitoring(self, interval: float = 1.0):
        """开始实时监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,)
        )
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("传感器监控已启动")
    
    def stop_monitoring(self):
        """停止实时监控"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        self.logger.info("传感器监控已停止")
    
    def _monitoring_loop(self, interval: float):
        """监控循环"""
        while self.is_monitoring:
            try:
                data_batch = self.read_all_sensors()
                
                with self._lock:
                    self.data_buffer.append({
                        "timestamp": time.time(),
                        "data": data_batch
                    })
                    
                    # 保持缓冲区大小
                    if len(self.data_buffer) > 1000:
                        self.data_buffer.pop(0)
                
                # 触发回调
                self._trigger_callbacks(data_batch)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
                time.sleep(interval)
    
    def _enhanced_processing(self, data: SensorData) -> SensorData:
        """增强模式下的额外处理"""
        # 添加噪声过滤
        if abs(data.value) > 1000:  # 异常值过滤
            data.metadata = data.metadata or {}
            data.metadata["filtered"] = True
            data.value = np.clip(data.value, -1000, 1000)
        
        # 添加趋势分析
        recent_data = [
            d for d in self.data_buffer[-10:] 
            if data.sensor_type.value in d["data"]
        ]
        
        if recent_data:
            values = [d["data"][data.sensor_type.value].value for d in recent_data]
            trend = "increasing" if values[-1] > values[0] else "decreasing"
            data.metadata = data.metadata or {}
            data.metadata["trend"] = trend
        
        return data
    
    def register_callback(self, sensor_type: SensorType, callback: Callable):
        """注册传感器数据回调"""
        if sensor_type not in self.callbacks:
            self.callbacks[sensor_type] = []
        self.callbacks[sensor_type].append(callback)
    
    def _trigger_callbacks(self, data_batch: Dict[str, SensorData]):
        """触发回调函数"""
        for sensor_type_str, data in data_batch.items():
            sensor_type = SensorType(sensor_type_str)
            if sensor_type in self.callbacks:
                for callback in self.callbacks[sensor_type]:
                    try:
                        callback(data)
                    except Exception as e:
                        self.logger.error(f"回调错误: {e}")
    
    def get_statistics(self, sensor_type: SensorType = None) -> Dict[str, Any]:
        """获取传感器统计信息"""
        with self._lock:
            if not self.data_buffer:
                return {}
            
            if sensor_type:
                # 特定传感器统计
                values = [
                    d["data"][sensor_type.value].value 
                    for d in self.data_buffer 
                    if sensor_type.value in d["data"]
                ]
                
                if values:
                    return {
                        "sensor": sensor_type.value,
                        "count": len(values),
                        "mean": np.mean(values),
                        "std": np.std(values),
                        "min": np.min(values),
                        "max": np.max(values),
                        "mode": self.mode
                    }
            else:
                # 所有传感器统计
                stats = {}
                for sensor_type in self.sensors:
                    sensor_data = self.get_statistics(sensor_type)
                    if sensor_data:
                        stats[sensor_type.value] = sensor_data
                return stats