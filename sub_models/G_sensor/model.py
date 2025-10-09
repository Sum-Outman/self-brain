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

<<<<<<< HEAD
=======
# 传感器感知模型定义
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
# Sensor Perception Model Definition

import torch
import torch.nn as nn
import numpy as np
import json
import os
import requests
from datetime import datetime
from collections import deque
import threading
import time
import cv2
import pyaudio
import wave
import serial
import socket
import struct
import asyncio
import websockets
import logging
from typing import Dict, List, Any, Optional, Union
import queue
import select
import subprocess
import platform
import psutil
import GPUtil
from enum import Enum
import RPi.GPIO as GPIO
import smbus2
import spidev
import can
import modbus_tk
import modbus_tk.defines as cst
from modbus_tk import modbus_tcp
import minimalmodbus
import pymodbus
from pymodbus.client.sync import ModbusTcpClient
import paho.mqtt.client as mqtt
import zmq
import grpc
import avro
import msgpack
import orjson
import ujson
import simplejson
import bson
import cbor2
import pickle
import dill
import marshal
import shelve
import sqlite3
import redis
import pymongo
import influxdb
import prometheus_client
from prometheus_client import Counter, Gauge, Histogram, Summary
import statsd
import datadog
import newrelic
import sentry_sdk
import elasticapm
import opentelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.semconv.resource import ResourceAttributes
import jaeger_client
import zipkin
import lightstep
import honeycomb
import signalfx
import wavefront
import datadog_agent
import newrelic_agent
import appdynamics
import dynatrace
import splunk_hec
import fluentd
import logstash
import rsyslog
import syslog_ng
import graylog
import loki
import tempo
import otel
import otlp
import gelf
import capnp
import flatbuffers
import thrift
import orc
import parquet
import arrow
import feather
import hdf5
import netcdf4
import zarr
import xarray
import dask
import ray
import modin
import vaex
import cuDF
import tensorflow as tf
import jax
import flax
import haiku
import elegy
import trax
import sonnet
import keras
import mxnet
import cntk
import theano
import caffe
import caffe2
import paddlepaddle
import mindspore
import oneflow
import megengine
import jittor
import darknet
import dlib
import skimage
from skimage import io, color, filters
import PIL as pillow
import wand
import imageio
import ffmpeg
import gstreamer
import openal
import portaudio
import alsa
import pulseaudio
import jack
import coreaudio
import wasapi
import asio
import opensl
import oboe
import aaudio
import audiounit
import sounddevice
import soundfile
import librosa
import pydub
import audioread
import sox
import rubberband
import samplerate
import resampy
import pyrubberband
import pyts
import tsfresh
import tslearn
import sktime
import pmdarima
import prophet
import statsmodels
import arch
import pyflux
import gluonts
import orbit
import neuralprophet
import darts
import kats
import merlion
import greykite
import mlforecast
import timemachines
import river
import creme
import skmultiflow
import moa
import matrixprofile
import stumpy
import seglearn
import tsfel
import tsfractal
import pycatch22
import hctsa
import cesium
import antropy
import nolds
import pyentrp
import entropy
import sampen
import apen
import fuzzyen
import multiscale_entropy
import permutation_entropy
import weighted_permutation_entropy
import composite_multiscale_entropy
import refined_composite_multiscale_entropy
import multiscale_permutation_entropy
import multiscale_fuzzy_entropy
import multiscale_sample_entropy
import multiscale_approximate_entropy
import multiscale_renyi_entropy
import multiscale_tsallis_entropy
import torch.nn.functional as F
from torch.distributions import Normal, Categorical, Bernoulli, Poisson, Gamma, Beta, Dirichlet
from torch.distributions import MultivariateNormal, Laplace, StudentT, Cauchy, Uniform, LogNormal
from torch.distributions import Exponential, Weibull, Pareto, Gumbel, Frechet, Rayleigh
from torch.distributions import Chi2, FisherSnedecor, NegativeBinomial, Geometric, Hypergeometric
from torch.distributions import VonMises, WrappedCauchy, Kent, Bingham, MatrixNormal, Wishart
from torch.distributions import InverseWishart, LKJCholesky, LowRankMultivariateNormal, MixtureSameFamily
from torch.distributions import TransformedDistribution, Independent, Censored, TruncatedNormal
from torch.distributions import RelaxedBernoulli, RelaxedOneHotCategorical, Kumaraswamy, LogitNormal
from torch.distributions import LogisticNormal
from torch.distributions.transforms import AffineTransform, ExpTransform, PowerTransform, SigmoidTransform
from torch.distributions.transforms import SoftmaxTransform, StickBreakingTransform, LowerCholeskyTransform
from torch.distributions.transforms import CatTransform, StackTransform, ComposeTransform, ReshapeTransform

class SensorModel(nn.Module):
    def __init__(self, config_path="config/sensor_config.json"):
<<<<<<< HEAD
        """Initialize sensor perception model"""
        super(SensorModel, self).__init__()
        
        # Load configuration
=======
        """初始化传感器感知模型 | Initialize sensor perception model"""
        super(SensorModel, self).__init__()
        
        # 加载配置 | Load configuration
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        self.config = self.load_config(config_path)
        self.model_type = self.config.get("model_type", "local")
        self.external_api_config = self.config.get("external_api", {})
        
<<<<<<< HEAD
        # Sensor type mapping
=======
        # 传感器类型映射 | Sensor type mapping
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        self.sensor_types = {
            "temperature": {"unit": "°C", "range": [-40, 125]},
            "humidity": {"unit": "%", "range": [0, 100]},
            "acceleration": {"unit": "m/s²", "range": [-16, 16]},
            "velocity": {"unit": "m/s", "range": [0, 100]},
            "displacement": {"unit": "m", "range": [0, 100]},
            "gyroscope": {"unit": "rad/s", "range": [-2000, 2000]},
            "pressure": {"unit": "hPa", "range": [300, 1100]},
            "barometric": {"unit": "hPa", "range": [300, 1100]},
            "distance": {"unit": "m", "range": [0, 100]},
            "infrared": {"unit": "lux", "range": [0, 10000]},
            "gas": {"unit": "ppm", "range": [0, 5000]},
            "smoke": {"unit": "ppm", "range": [0, 1000]},
            "light": {"unit": "lux", "range": [0, 100000]},
<<<<<<< HEAD
            "taste": {"unit": "intensity", "range": [0, 10]}  # Taste sensor simulation
        }
        
        # Neural network architecture
        self.input_dim = len(self.sensor_types) * 3  # Each sensor has value, min, max
=======
            "taste": {"unit": "intensity", "range": [0, 10]}  # 味觉传感器模拟值 | Taste sensor simulation
        }
        
        # 神经网络架构 | Neural network architecture
        self.input_dim = len(self.sensor_types) * 3  # 每个传感器有值、最小值、最大值 | Each sensor has value, min, max
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
<<<<<<< HEAD
        self.fc6 = nn.Linear(32, len(self.sensor_types))  # Output status for each sensor
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Real-time data processing
        self.realtime_data = {}
        self.data_history = deque(maxlen=1000)  # Save historical data
        self.realtime_thread = None
        self.realtime_active = False
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            "temperature": 2.0,  # Temperature change threshold
            "humidity": 10.0,    # Humidity change threshold
            "acceleration": 5.0, # Acceleration change threshold
            "default": 3.0       # Default threshold
        }
        
        # Language support
        self.current_lang = self.config.get("default_language", "en")
        self.language_resources = self.load_language_resources()
        
        # Real-time input interfaces
=======
        self.fc6 = nn.Linear(32, len(self.sensor_types))  # 输出每个传感器的状态 | Output status for each sensor
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # 实时数据处理 | Real-time data processing
        self.realtime_data = {}
        self.data_history = deque(maxlen=1000)  # 保存历史数据 | Save historical data
        self.realtime_thread = None
        self.realtime_active = False
        
        # 异常检测阈值 | Anomaly detection thresholds
        self.anomaly_thresholds = {
            "temperature": 2.0,  # 温度变化阈值 | Temperature change threshold
            "humidity": 10.0,    # 湿度变化阈值 | Humidity change threshold
            "acceleration": 5.0, # 加速度变化阈值 | Acceleration change threshold
            "default": 3.0       # 默认阈值 | Default threshold
        }
        
        # 语言支持 | Language support
        self.current_lang = self.config.get("default_language", "en")
        self.language_resources = self.load_language_resources()
        
        # 实时输入接口 | Real-time input interfaces
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        self.camera_stream = None
        self.microphone_stream = None
        self.network_streams = {}
        self.serial_ports = {}
        
<<<<<<< HEAD
        # Connect to main model
=======
        # 连接到主模型 | Connect to main model
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        self.data_bus = None
        self.connect_to_main_model()
    
    def load_config(self, config_path):
<<<<<<< HEAD
        """Load configuration file"""
=======
        """加载配置文件 | Load configuration file"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {
                "model_type": "local",
                "default_language": "en",
                "data_bus_host": "localhost",
                "data_bus_port": 6379,
                "realtime_interfaces": {
                    "camera": {"enabled": True, "device_index": 0},
                    "microphone": {"enabled": True, "device_index": 0},
                    "network_streams": {"enabled": True},
                    "serial_ports": {"enabled": True}
                }
            }
    
    def load_language_resources(self):
<<<<<<< HEAD
        """Load language resources"""
=======
        """加载语言资源 | Load language resources"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        try:
            with open("config/language_resources.json", 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
<<<<<<< HEAD
            # Default language resources
=======
            # 默认语言资源 | Default language resources
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            return {
                "en": {
                    "sensor_status": "Sensor Status",
                    "anomaly_detected": "Anomaly detected",
                    "normal_operation": "Normal operation",
                    "camera_connected": "Camera connected",
                    "microphone_connected": "Microphone connected",
                    "network_stream_connected": "Network stream connected",
                    "serial_port_connected": "Serial port connected"
                },
                "zh": {
                    "sensor_status": "传感器状态",
                    "anomaly_detected": "检测到异常",
                    "normal_operation": "正常运行",
                    "camera_connected": "摄像头已连接",
                    "microphone_connected": "麦克风已连接",
                    "network_stream_connected": "网络流已连接",
                    "serial_port_connected": "串口已连接"
                }
            }
    
    def forward(self, x):
<<<<<<< HEAD
        """Forward pass"""
=======
        """前向传播 | Forward pass"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        x = self.relu(self.fc4(x))
        x = self.dropout(x)
        x = self.relu(self.fc5(x))
        x = self.fc6(x)
        return x
    
    def process_sensor_data(self, sensor_data):
<<<<<<< HEAD
        """Process sensor data"""
        try:
            # Normalize sensor data
=======
        """处理传感器数据 | Process sensor data"""
        try:
            # 标准化传感器数据 | Normalize sensor data
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            normalized_data = self.normalize_sensor_data(sensor_data)
            
            # 转换为张量 | Convert to tensor
            input_tensor = torch.tensor(normalized_data, dtype=torch.float32).unsqueeze(0)
            
            # 使用模型处理 | Process with model
            with torch.no_grad():
                output = self.forward(input_tensor)
            
            # 解释输出 | Interpret output
            result = self.interpret_output(output.squeeze(0).numpy(), sensor_data)
            
            # 保存到历史数据 | Save to historical data
            self.data_history.append({
                "timestamp": datetime.now().isoformat(),
                "raw_data": sensor_data,
                "processed_result": result
            })
            
            # 发送到主模型 | Send to main model
            self.send_to_main_model(result)
            
            return result
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def normalize_sensor_data(self, sensor_data):
        """标准化传感器数据 | Normalize sensor data"""
        normalized = []
        for sensor_type, value in sensor_data.items():
            if sensor_type in self.sensor_types:
                sensor_range = self.sensor_types[sensor_type]["range"]
                # 归一化到[0,1]范围 | Normalize to [0,1] range
                normalized_value = (value - sensor_range[0]) / (sensor_range[1] - sensor_range[0])
                normalized.append(normalized_value)
<<<<<<< HEAD
                # Add min and max values
                normalized.append(sensor_range[0])
                normalized.append(sensor_range[1])
            else:
                # Unknown sensor type, use default values
                normalized.extend([0.5, 0, 1])
        
        # Ensure consistent length
=======
                # 添加最小值和最大值 | Add min and max values
                normalized.append(sensor_range[0])
                normalized.append(sensor_range[1])
            else:
                # 未知传感器类型，使用默认值 | Unknown sensor type, use default values
                normalized.extend([0.5, 0, 1])
        
        # 确保长度一致 | Ensure consistent length
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        while len(normalized) < self.input_dim:
            normalized.append(0.0)
        
        return normalized[:self.input_dim]
    
    def interpret_output(self, output, sensor_data):
<<<<<<< HEAD
        """Interpret model output"""
=======
        """解释模型输出 | Interpret model output"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        results = {}
        sensor_types = list(self.sensor_types.keys())
        
        for i, sensor_type in enumerate(sensor_types):
            if i < len(output):
                confidence = output[i]
                status = "normal" if confidence > 0.5 else "anomaly"
                
                results[sensor_type] = {
                    "value": sensor_data.get(sensor_type, 0),
                    "status": status,
                    "confidence": float(confidence),
                    "unit": self.sensor_types[sensor_type]["unit"],
                    "timestamp": datetime.now().isoformat()
                }
        
<<<<<<< HEAD
        # Overall status
=======
        # 整体状态 | Overall status
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        overall_status = "normal"
        if any(result["status"] == "anomaly" for result in results.values()):
            overall_status = "anomaly"
        
        return {
            "status": "success",
            "overall_status": overall_status,
            "sensor_results": results,
            "timestamp": datetime.now().isoformat()
        }
    
    def detect_anomalies(self, sensor_data, historical_data=None):
<<<<<<< HEAD
        """Detect sensor anomalies"""
=======
        """检测传感器异常 | Detect sensor anomalies"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        try:
            anomalies = []
            
            for sensor_type, value in sensor_data.items():
                if sensor_type in self.sensor_types:
<<<<<<< HEAD
                    # Check if value is within reasonable range
=======
                    # 检查值是否在合理范围内 | Check if value is within reasonable range
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
                    sensor_range = self.sensor_types[sensor_type]["range"]
                    if value < sensor_range[0] or value > sensor_range[1]:
                        anomalies.append({
                            "sensor_type": sensor_type,
                            "value": value,
                            "reason": f"Value out of range ({sensor_range[0]} to {sensor_range[1]})",
                            "severity": "high"
                        })
                        continue
                    
<<<<<<< HEAD
                    # Check for sudden changes
=======
                    # 检查突变异常 | Check for sudden changes
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
                    if historical_data:
                        threshold = self.anomaly_thresholds.get(sensor_type, self.anomaly_thresholds["default"])
                        recent_values = [d["raw_data"].get(sensor_type, 0) for d in historical_data[-10:] if d]
                        
                        if len(recent_values) > 1:
                            avg_change = np.mean(np.abs(np.diff(recent_values)))
                            if avg_change > threshold:
                                anomalies.append({
                                    "sensor_type": sensor_type,
                                    "value": value,
                                    "reason": f"Sudden change detected (average change: {avg_change:.2f})",
                                    "severity": "medium"
                                })
            
            return {
                'status': 'success',
                'anomalies': anomalies,
                'anomaly_count': len(anomalies),
<<<<<<< HEAD
                'confidence': 1.0 - (len(anomalies) * 0.1)  # Confidence based on number of anomalies
=======
                'confidence': 1.0 - (len(anomalies) * 0.1)  # 基于异常数量的置信度 | Confidence based on number of anomalies
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def fuse_sensor_data(self, sensor_data_list):
<<<<<<< HEAD
        """Fuse multi-sensor data"""
=======
        """融合多传感器数据 | Fuse multi-sensor data"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        try:
            if not sensor_data_list:
                return {'status': 'error', 'message': 'No sensor data provided'}
            
<<<<<<< HEAD
            # Use weighted average to fuse data
=======
            # 使用加权平均融合数据 | Use weighted average to fuse data
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            fused_data = {}
            weights = {}
            
            for sensor_data in sensor_data_list:
                for sensor_type, value in sensor_data.items():
                    if sensor_type in self.sensor_types:
                        if sensor_type not in fused_data:
                            fused_data[sensor_type] = 0.0
                            weights[sensor_type] = 0.0
                        
<<<<<<< HEAD
                        # Confidence-based weighting
                        confidence = 0.8  # Default confidence
=======
                        # 基于置信度的权重 | Confidence-based weighting
                        confidence = 0.8  # 默认置信度 | Default confidence
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
                        if 'confidence' in sensor_data:
                            confidence = sensor_data['confidence']
                        
                        fused_data[sensor_type] += value * confidence
                        weights[sensor_type] += confidence
            
<<<<<<< HEAD
            # Calculate weighted average
=======
            # 计算加权平均值 | Calculate weighted average
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            for sensor_type in fused_data:
                if weights[sensor_type] > 0:
                    fused_data[sensor_type] /= weights[sensor_type]
            
            return {
                'status': 'success',
                'fused_data': fused_data,
                'weights': weights,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def connect_to_main_model(self):
<<<<<<< HEAD
        """Connect to main model data bus"""
=======
        """连接到主模型数据总线 | Connect to main model data bus"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        try:
            import redis
            self.data_bus = redis.Redis(
                host=self.config.get('data_bus_host', 'localhost'),
                port=self.config.get('data_bus_port', 6379),
                db=0
            )
            self.data_bus.ping()
<<<<<<< HEAD
            print(f"Connected to main model data bus")
            return True
        except Exception as e:
            print(f"Connection failed: {str(e)}")
=======
            print(f"{self.get_text('connected_to_main_model')} | 已连接到主模型数据总线")
            return True
        except Exception as e:
            print(f"{self.get_text('connection_failed')}: {str(e)} | 连接主模型失败: {str(e)}")
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            self.data_bus = None
            return False
    
    def send_to_main_model(self, data):
<<<<<<< HEAD
        """Send data to main model"""
=======
        """发送数据到主模型 | Send data to main model"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        try:
            if self.data_bus:
                self.data_bus.publish('sensor_data', json.dumps(data))
                return True
            return False
        except Exception as e:
<<<<<<< HEAD
            print(f"Send failed: {str(e)}")
            return False
    
    def get_text(self, key):
        """Get multilingual text"""
        return self.language_resources.get(self.current_lang, {}).get(key, key)
    
    def switch_language(self, language):
        """Switch language"""
=======
            print(f"{self.get_text('send_failed')}: {str(e)} | 发送数据失败: {str(e)}")
            return False
    
    def get_text(self, key):
        """获取多语言文本 | Get multilingual text"""
        return self.language_resources.get(self.current_lang, {}).get(key, key)
    
    def switch_language(self, language):
        """切换语言 | Switch language"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        if language in self.language_resources:
            self.current_lang = language
            return True
        return False
    
    # ========== 实时输入接口功能 ========== #
    # ========== Real-time Input Interface Functions ========== #
    
    def start_camera_stream(self, device_index=0):
        """启动摄像头实时流 | Start camera real-time stream"""
        try:
            self.camera_stream = cv2.VideoCapture(device_index)
            if not self.camera_stream.isOpened():
                return {'status': 'error', 'message': '无法打开摄像头 | Cannot open camera'}
            
            print(f"{self.get_text('camera_connected')} (设备 {device_index})")
            return {'status': 'success', 'message': '摄像头流已启动 | Camera stream started'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def stop_camera_stream(self):
        """停止摄像头流 | Stop camera stream"""
        if self.camera_stream:
            self.camera_stream.release()
            self.camera_stream = None
    
    def capture_frame(self):
        """捕获摄像头帧 | Capture camera frame"""
        if self.camera_stream and self.camera_stream.isOpened():
            ret, frame = self.camera_stream.read()
            if ret:
                return {'status': 'success', 'frame': frame}
            return {'status': 'error', 'message': '无法读取帧 | Cannot read frame'}
        return {'status': 'error', 'message': '摄像头未启动 | Camera not started'}
    
    def start_microphone_stream(self, device_index=0, sample_rate=44100, channels=1):
        """启动麦克风实时流 | Start microphone real-time stream"""
        try:
            self.audio = pyaudio.PyAudio()
            self.microphone_stream = self.audio.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=1024
            )
<<<<<<< HEAD
            print(f"Microphone connected (device {device_index})")
            return {'status': 'success', 'message': 'Microphone stream started'}
=======
            print(f"{self.get_text('microphone_connected')} (设备 {device_index})")
            return {'status': 'success', 'message': '麦克风流已启动 | Microphone stream started'}
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def stop_microphone_stream(self):
<<<<<<< HEAD
        """Stop microphone stream"""
=======
        """停止麦克风流 | Stop microphone stream"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        if self.microphone_stream:
            self.microphone_stream.stop_stream()
            self.microphone_stream.close()
            self.microphone_stream = None
        if self.audio:
            self.audio.terminate()
    
    def read_audio_data(self, num_frames=1024):
<<<<<<< HEAD
        """Read audio data"""
=======
        """读取音频数据 | Read audio data"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        if self.microphone_stream:
            try:
                data = self.microphone_stream.read(num_frames, exception_on_overflow=False)
                return {'status': 'success', 'audio_data': data}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}
<<<<<<< HEAD
        return {'status': 'error', 'message': 'Microphone not started'}
    
    def connect_network_stream(self, stream_type, url, config=None):
        """Connect network stream"""
        try:
            if stream_type == 'video':
                # Connect network video stream
=======
        return {'status': 'error', 'message': '麦克风未启动 | Microphone not started'}
    
    def connect_network_stream(self, stream_type, url, config=None):
        """连接网络流 | Connect network stream"""
        try:
            if stream_type == 'video':
                # 连接网络视频流 | Connect network video stream
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
                cap = cv2.VideoCapture(url)
                if cap.isOpened():
                    self.network_streams[url] = {
                        'type': 'video',
                        'capture': cap,
                        'config': config or {}
                    }
<<<<<<< HEAD
                    print(f"Network stream connected: {url}")
                    return {'status': 'success', 'message': f'Network video stream connected: {url}'}
            
            elif stream_type == 'audio':
                # Connect network audio stream
                # Need to implement based on specific protocol
=======
                    print(f"{self.get_text('network_stream_connected')}: {url}")
                    return {'status': 'success', 'message': f'网络视频流已连接: {url}'}
            
            elif stream_type == 'audio':
                # 连接网络音频流 | Connect network audio stream
                # 这里需要根据具体协议实现 | Need to implement based on specific protocol
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
                self.network_streams[url] = {
                    'type': 'audio',
                    'config': config or {}
                }
<<<<<<< HEAD
                print(f"Network stream connected: {url}")
                return {'status': 'success', 'message': f'Network audio stream connected: {url}'}
            
            return {'status': 'error', 'message': 'Unsupported stream type'}
=======
                print(f"{self.get_text('network_stream_connected')}: {url}")
                return {'status': 'success', 'message': f'网络音频流已连接: {url}'}
            
            return {'status': 'error', 'message': '不支持的流类型 | Unsupported stream type'}
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def disconnect_network_stream(self, url):
<<<<<<< HEAD
        """Disconnect network stream"""
=======
        """断开网络流 | Disconnect network stream"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        if url in self.network_streams:
            stream = self.network_streams[url]
            if stream['type'] == 'video' and 'capture' in stream:
                stream['capture'].release()
            del self.network_streams[url]
    
    def open_serial_port(self, port, baudrate=9600, timeout=1):
<<<<<<< HEAD
        """Open serial port"""
        try:
            ser = serial.Serial(port, baudrate, timeout=timeout)
            self.serial_ports[port] = ser
            print(f"Serial port connected: {port}")
            return {'status': 'success', 'message': f'Serial port opened: {port}'}
=======
        """打开串口 | Open serial port"""
        try:
            ser = serial.Serial(port, baudrate, timeout=timeout)
            self.serial_ports[port] = ser
            print(f"{self.get_text('serial_port_connected')}: {port}")
            return {'status': 'success', 'message': f'串口已打开: {port}'}
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def close_serial_port(self, port):
<<<<<<< HEAD
        """Close serial port"""
=======
        """关闭串口 | Close serial port"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        if port in self.serial_ports:
            self.serial_ports[port].close()
            del self.serial_ports[port]
    
    def read_serial_data(self, port):
<<<<<<< HEAD
        """Read serial data"""
=======
        """读取串口数据 | Read serial data"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        if port in self.serial_ports:
            try:
                data = self.serial_ports[port].readline().decode('utf-8').strip()
                if data:
                    return {'status': 'success', 'data': data}
                return {'status': 'success', 'data': ''}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}
<<<<<<< HEAD
        return {'status': 'error', 'message': f'Serial port not open: {port}'}
    
    def write_serial_data(self, port, data):
        """Write serial data"""
        if port in self.serial_ports:
            try:
                self.serial_ports[port].write(data.encode('utf-8'))
                return {'status': 'success', 'message': 'Data sent'}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}
        return {'status': 'error', 'message': f'Serial port not open: {port}'}
    
    def start_realtime_processing(self):
        """Start real-time data processing"""
        if self.realtime_active:
            return {'status': 'error', 'message': 'Real-time processing already running'}
=======
        return {'status': 'error', 'message': f'串口未打开: {port}'}
    
    def write_serial_data(self, port, data):
        """写入串口数据 | Write serial data"""
        if port in self.serial_ports:
            try:
                self.serial_ports[port].write(data.encode('utf-8'))
                return {'status': 'success', 'message': '数据已发送 | Data sent'}
            except Exception as e:
                return {'status': 'error', 'message': str(e)}
        return {'status': 'error', 'message': f'串口未打开: {port}'}
    
    def start_realtime_processing(self):
        """启动实时数据处理 | Start real-time data processing"""
        if self.realtime_active:
            return {'status': 'error', 'message': '实时处理已在运行 | Real-time processing already running'}
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        
        self.realtime_active = True
        self.realtime_thread = threading.Thread(target=self._realtime_processing_loop)
        self.realtime_thread.daemon = True
        self.realtime_thread.start()
        
<<<<<<< HEAD
        return {'status': 'success', 'message': 'Real-time processing started'}
    
    def stop_realtime_processing(self):
        """Stop real-time data processing"""
=======
        return {'status': 'success', 'message': '实时处理已启动 | Real-time processing started'}
    
    def stop_realtime_processing(self):
        """停止实时数据处理 | Stop real-time data processing"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        self.realtime_active = False
        if self.realtime_thread:
            self.realtime_thread.join(timeout=2.0)
        self.realtime_thread = None
    
    def _realtime_processing_loop(self):
<<<<<<< HEAD
        """Real-time processing loop"""
        while self.realtime_active:
            try:
                # Process camera data
=======
        """实时处理循环 | Real-time processing loop"""
        while self.realtime_active:
            try:
                # 处理摄像头数据 | Process camera data
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
                if self.camera_stream and self.camera_stream.isOpened():
                    frame_result = self.capture_frame()
                    if frame_result['status'] == 'success':
                        # 分析帧数据 | Analyze frame data
                        analysis = self.analyze_frame(frame_result['frame'])
                        self.send_to_main_model({
                            'type': 'camera_frame',
                            'analysis': analysis,
                            'timestamp': datetime.now().isoformat()
                        })
                
                # 处理音频数据 | Process audio data
                if self.microphone_stream:
                    audio_result = self.read_audio_data()
                    if audio_result['status'] == 'success':
                        # 分析音频数据 | Analyze audio data
                        analysis = self.analyze_audio(audio_result['audio_data'])
                        self.send_to_main_model({
                            'type': 'audio_data',
                            'analysis': analysis,
                            'timestamp': datetime.now().isoformat()
                        })
                
                # 处理网络流数据 | Process network stream data
                for url, stream in list(self.network_streams.items()):
                    if stream['type'] == 'video' and 'capture' in stream:
                        ret, frame = stream['capture'].read()
                        if ret:
                            analysis = self.analyze_frame(frame)
                            self.send_to_main_model({
                                'type': 'network_video',
                                'url': url,
                                'analysis': analysis,
                                'timestamp': datetime.now().isoformat()
                            })
                
                # 处理串口数据 | Process serial data
                for port in list(self.serial_ports.keys()):
                    serial_result = self.read_serial_data(port)
                    if serial_result['status'] == 'success' and serial_result['data']:
                        # 解析传感器数据 | Parse sensor data
                        sensor_data = self.parse_serial_data(serial_result['data'])
                        if sensor_data:
                            processed = self.process_sensor_data(sensor_data)
                            self.send_to_main_model({
                                'type': 'serial_data',
                                'port': port,
                                'raw_data': serial_result['data'],
                                'processed_data': processed,
                                'timestamp': datetime.now().isoformat()
                            })
                
<<<<<<< HEAD
                time.sleep(0.1)  # 100ms interval
=======
                time.sleep(0.1)  # 100ms间隔 | 100ms interval
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
                
            except Exception as e:
                print(f"实时处理错误: {str(e)} | Real-time processing error: {str(e)}")
                time.sleep(1)
    
    def analyze_frame(self, frame):
        """分析视频帧 | Analyze video frame"""
        try:
            # 转换为灰度图 | Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 计算基本统计信息 | Calculate basic statistics
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # 边缘检测 | Edge detection
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size
            
            return {
                'brightness': float(brightness),
                'contrast': float(contrast),
                'edge_density': float(edge_density),
                'resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                'channels': frame.shape[2] if len(frame.shape) > 2 else 1
            }
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_audio(self, audio_data):
        """分析音频数据 | Analyze audio data"""
        try:
            # 转换为numpy数组 | Convert to numpy array
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # 计算基本统计信息 | Calculate basic statistics
            volume = np.sqrt(np.mean(audio_array**2))
            max_amplitude = np.max(np.abs(audio_array))
            
            return {
                'volume': float(volume),
                'max_amplitude': float(max_amplitude),
                'samples': len(audio_array),
<<<<<<< HEAD
                'sample_rate': 44100  # Assuming sample rate
=======
                'sample_rate': 44100  # 假设采样率 | Assuming sample rate
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            }
        except Exception as e:
            return {'error': str(e)}
    
    def parse_serial_data(self, data):
<<<<<<< HEAD
        """Parse serial data"""
        try:
            # Simple CSV format parsing
=======
        """解析串口数据 | Parse serial data"""
        try:
            # 简单的CSV格式解析 | Simple CSV format parsing
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            if ',' in data:
                parts = data.split(',')
                sensor_data = {}
                for part in parts:
                    if ':' in part:
                        key, value = part.split(':', 1)
                        try:
                            sensor_data[key.strip()] = float(value.strip())
                        except ValueError:
                            sensor_data[key.strip()] = value.strip()
                return sensor_data
            
<<<<<<< HEAD
            # JSON format parsing
=======
            # JSON格式解析 | JSON format parsing
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            if data.startswith('{') and data.endswith('}'):
                return json.loads(data)
            
            return None
        except:
            return None
    
    def get_realtime_interfaces_status(self):
<<<<<<< HEAD
        """Get real-time interfaces status"""
=======
        """获取实时接口状态 | Get real-time interfaces status"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        status = {
            'camera': {
                'active': self.camera_stream is not None and self.camera_stream.isOpened(),
                'device_index': getattr(self.camera_stream, 'get', lambda x: 0)('CAP_PROP_POS_FRAMES') if self.camera_stream else 0
            },
            'microphone': {
                'active': self.microphone_stream is not None,
<<<<<<< HEAD
                'device_index': 0  # Need actual implementation
=======
                'device_index': 0  # 需要实际实现 | Need actual implementation
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            },
            'network_streams': {
                'count': len(self.network_streams),
                'urls': list(self.network_streams.keys())
            },
            'serial_ports': {
                'count': len(self.serial_ports),
                'ports': list(self.serial_ports.keys())
            },
            'realtime_processing': {
                'active': self.realtime_active
            }
        }
        return status
    
    def cleanup(self):
<<<<<<< HEAD
        """Cleanup resources"""
=======
        """清理资源 | Cleanup resources"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        self.stop_realtime_processing()
        self.stop_camera_stream()
        self.stop_microphone_stream()
        
        for url in list(self.network_streams.keys()):
            self.disconnect_network_stream(url)
        
        for port in list(self.serial_ports.keys()):
            self.close_serial_port(port)
    
    def __del__(self):
<<<<<<< HEAD
        """Destructor"""
        self.cleanup()

# External API integration functions
def connect_external_api(api_config):
    """Connect to external API"""
=======
        """析构函数 | Destructor"""
        self.cleanup()

# 外部API集成功能 | External API integration functions
def connect_external_api(api_config):
    """连接外部API | Connect to external API"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
    try:
        api_type = api_config.get('type')
        api_key = api_config.get('api_key')
        endpoint = api_config.get('endpoint')
        
        if api_type == 'openai':
            import openai
            openai.api_key = api_key
            return {'status': 'success', 'client': openai}
        
        elif api_type == 'azure':
            from azure.cognitiveservices.vision.computervision import ComputerVisionClient
            from msrest.authentication import CognitiveServicesCredentials
            client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(api_key))
            return {'status': 'success', 'client': client}
        
        elif api_type == 'google_cloud':
            from google.cloud import vision
            client = vision.ImageAnnotatorClient.from_service_account_json(api_key)
            return {'status': 'success', 'client': client}
        
        elif api_type == 'aws':
            import boto3
            client = boto3.client('rekognition', 
                                aws_access_key_id=api_config.get('access_key'),
                                aws_secret_access_key=api_config.get('secret_key'),
                                region_name=api_config.get('region', 'us-east-1'))
            return {'status': 'success', 'client': client}
        
        else:
<<<<<<< HEAD
            return {'status': 'error', 'message': f'Unsupported API type: {api_type}'}
=======
            return {'status': 'error', 'message': f'不支持的API类型: {api_type} | Unsupported API type: {api_type}'}
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def process_with_external_api(api_client, data, api_type):
<<<<<<< HEAD
    """Process data with external API"""
    try:
        if api_type == 'openai':
            # OpenAI API processing
=======
    """使用外部API处理数据 | Process data with external API"""
    try:
        if api_type == 'openai':
            # OpenAI API处理 | OpenAI API processing
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            response = api_client.Completion.create(
                engine="text-davinci-003",
                prompt=str(data),
                max_tokens=100
            )
            return {'status': 'success', 'result': response.choices[0].text.strip()}
        
        elif api_type == 'azure':
<<<<<<< HEAD
            # Azure Computer Vision API processing
            if isinstance(data, np.ndarray):
                # Process image data
=======
            # Azure计算机视觉API处理 | Azure Computer Vision API processing
            if isinstance(data, np.ndarray):
                # 处理图像数据 | Process image data
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
                _, img_encoded = cv2.imencode('.jpg', data)
                analysis = api_client.analyze_image_in_stream(img_encoded.tobytes(), visual_features=['Categories', 'Description'])
                return {'status': 'success', 'result': analysis.as_dict()}
        
        elif api_type == 'google_cloud':
<<<<<<< HEAD
            # Google Cloud Vision API processing
            if isinstance(data, np.ndarray):
                # Process image data
=======
            # Google Cloud Vision API处理 | Google Cloud Vision API processing
            if isinstance(data, np.ndarray):
                # 处理图像数据 | Process image data
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
                _, img_encoded = cv2.imencode('.jpg', data)
                image = vision.Image(content=img_encoded.tobytes())
                response = api_client.label_detection(image=image)
                return {'status': 'success', 'result': [label.description for label in response.label_annotations]}
        
        elif api_type == 'aws':
<<<<<<< HEAD
            # AWS Rekognition API processing
            if isinstance(data, np.ndarray):
                # Process image data
=======
            # AWS Rekognition API处理 | AWS Rekognition API processing
            if isinstance(data, np.ndarray):
                # 处理图像数据 | Process image data
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
                _, img_encoded = cv2.imencode('.jpg', data)
                response = api_client.detect_labels(Image={'Bytes': img_encoded.tobytes()})
                return {'status': 'success', 'result': [label['Name'] for label in response['Labels']]}
        
<<<<<<< HEAD
        return {'status': 'error', 'message': 'Unsupported API operation'}
=======
        return {'status': 'error', 'message': '不支持的API操作 | Unsupported API operation'}
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

<<<<<<< HEAD
# Model training functions
def train_sensor_model(model, training_data, epochs=10, learning_rate=0.001):
    """Train sensor model"""
=======
# 模型训练功能 | Model training functions
def train_sensor_model(model, training_data, epochs=10, learning_rate=0.001):
    """训练传感器模型 | Train sensor model"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            for data in training_data:
                inputs = torch.tensor(data['input'], dtype=torch.float32)
                targets = torch.tensor(data['target'], dtype=torch.float32)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(training_data)
            losses.append(avg_loss)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        
        return {'status': 'success', 'losses': losses, 'final_loss': losses[-1]}
        
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

def create_training_data_from_history(history_data, sensor_types):
<<<<<<< HEAD
    """Create training data from history"""
=======
    """从历史数据创建训练数据 | Create training data from history"""
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
    try:
        training_data = []
        for data_point in history_data:
            if 'raw_data' in data_point and 'processed_result' in data_point:
<<<<<<< HEAD
                # Create input features
=======
                # 创建输入特征 | Create input features
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
                input_features = []
                for sensor_type in sensor_types:
                    value = data_point['raw_data'].get(sensor_type, 0)
                    sensor_range = sensor_types[sensor_type]["range"]
                    normalized_value = (value - sensor_range[0]) / (sensor_range[1] - sensor_range[0])
                    input_features.extend([normalized_value, sensor_range[0], sensor_range[1]])
                
<<<<<<< HEAD
                # Create target output
=======
                # 创建目标输出 | Create target output
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
                target_output = []
                for sensor_type in sensor_types:
                    confidence = data_point['processed_result']['sensor_results'].get(sensor_type, {}).get('confidence', 0.5)
                    target_output.append(confidence)
                
                training_data.append({
                    'input': input_features,
                    'target': target_output
                })
        
        return training_data
        
    except Exception as e:
        return []

<<<<<<< HEAD
# Main function and test code
if __name__ == "__main__":
    # Create sensor model instance
    sensor_model = SensorModel()
    
    # Test sensor data processing
=======
# 主函数和测试代码 | Main function and test code
if __name__ == "__main__":
    # 创建传感器模型实例 | Create sensor model instance
    sensor_model = SensorModel()
    
    # 测试传感器数据处理 | Test sensor data processing
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
    test_data = {
        'temperature': 25.5,
        'humidity': 60.0,
        'acceleration': 0.5
    }
    
    result = sensor_model.process_sensor_data(test_data)
<<<<<<< HEAD
    print("Sensor processing result:", result)
    
    # Test anomaly detection
    anomalies = sensor_model.detect_anomalies(test_data)
    print("Anomaly detection result:", anomalies)
    
    # Test multi-sensor data fusion
=======
    print("传感器处理结果:", result)
    
    # 测试异常检测 | Test anomaly detection
    anomalies = sensor_model.detect_anomalies(test_data)
    print("异常检测结果:", anomalies)
    
    # 测试多传感器数据融合 | Test multi-sensor data fusion
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
    sensor_data_list = [
        {'temperature': 25.0, 'confidence': 0.9},
        {'temperature': 26.0, 'confidence': 0.8},
        {'temperature': 24.5, 'confidence': 0.7}
    ]
    fused = sensor_model.fuse_sensor_data(sensor_data_list)
<<<<<<< HEAD
    print("Data fusion result:", fused)
    
    print("Sensor model test completed")
=======
    print("数据融合结果:", fused)
    
    print("传感器模型测试完成 | Sensor model test completed")
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
