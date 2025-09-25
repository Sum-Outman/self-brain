#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
传感器API客户端示例
Sensor API Client Example
用于演示如何调用传感器集成模块提供的各种API
"""

import os
import sys
import json
import time
import requests
from datetime import datetime

# 配置日志
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 传感器API基础URL
SENSOR_API_BASE_URL = "http://localhost:5006/api/sensor"

class SensorApiClient:
    """传感器API客户端"""
    
    def __init__(self, base_url=SENSOR_API_BASE_URL):
        """初始化客户端"""
        self.base_url = base_url
    
    def process_sensor_data(self, sensor_data):
        """处理传感器数据"""
        url = f"{self.base_url}/process"
        try:
            response = requests.post(url, json=sensor_data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"处理传感器数据失败: {e}")
            return None
    
    def detect_anomalies(self, sensor_data, use_history=False):
        """检测传感器异常"""
        url = f"{self.base_url}/detect_anomalies"
        try:
            data = sensor_data.copy()
            data['use_history'] = use_history
            response = requests.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"检测异常失败: {e}")
            return None
    
    def fuse_sensor_data(self, sensor_data_list):
        """融合多传感器数据"""
        url = f"{self.base_url}/fuse"
        try:
            response = requests.post(url, json=sensor_data_list)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"数据融合失败: {e}")
            return None
    
    def get_sensor_status(self):
        """获取传感器状态"""
        url = f"{self.base_url}/status"
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"获取状态失败: {e}")
            return None
    
    def process_with_external_api(self, api_name, data):
        """使用外部API处理数据"""
        url = f"{self.base_url}/external_api/process"
        try:
            payload = {
                'api_name': api_name,
                'data': data
            }
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"外部API处理失败: {e}")
            return None

# 测试函数
def test_sensor_api():
    """测试传感器API功能"""
    client = SensorApiClient()
    
    print("===== 测试传感器API =====")
    
    # 1. 获取传感器状态
    print("\n1. 获取传感器状态")
    status = client.get_sensor_status()
    if status and status['status'] == 'success':
        print(f"可用传感器: {status['config']['available_sensors']}")
        print(f"外部API数量: {len(status['config']['external_apis'])}")
    else:
        print("无法获取传感器状态")
    
    # 2. 处理传感器数据
    print("\n2. 处理传感器数据")
    test_data = {
        'temperature': 25.5,
        'humidity': 60.0,
        'acceleration': 0.5,
        'light': 500.0,
        'pressure': 1013.25
    }
    print(f"输入数据: {test_data}")
    
    result = client.process_sensor_data(test_data)
    if result and result['status'] == 'success':
        print(f"整体状态: {result['overall_status']}")
        print("传感器结果:")
        for sensor_type, sensor_result in result['sensor_results'].items():
            print(f"  - {sensor_type}: {sensor_result['value']} {sensor_result['unit']}, "
                  f"状态: {sensor_result['status']}, 置信度: {sensor_result['confidence']:.2f}")
    else:
        print(f"处理失败: {result}")
    
    # 3. 检测异常
    print("\n3. 检测异常")
    # 创建一个包含异常值的数据
    anomaly_data = test_data.copy()
    anomaly_data['temperature'] = 150.0  # 超出正常范围的温度
    print(f"异常检测输入: {anomaly_data}")
    
    anomaly_result = client.detect_anomalies(anomaly_data)
    if anomaly_result and anomaly_result['status'] == 'success':
        print(f"检测到 {anomaly_result['anomaly_count']} 个异常")
        for anomaly in anomaly_result['anomalies']:
            print(f"  - {anomaly['sensor_type']}: {anomaly['reason']}, 严重程度: {anomaly['severity']}")
    else:
        print(f"异常检测失败: {anomaly_result}")
    
    # 4. 数据融合
    print("\n4. 多传感器数据融合")
    sensor_data_list = [
        {'temperature': 25.0, 'humidity': 59.0, 'confidence': 0.9},
        {'temperature': 26.0, 'humidity': 61.0, 'confidence': 0.8},
        {'temperature': 24.5, 'humidity': 60.5, 'confidence': 0.7}
    ]
    
    fused_result = client.fuse_sensor_data(sensor_data_list)
    if fused_result and fused_result['status'] == 'success':
        print("融合结果:")
        for sensor_type, value in fused_result['fused_data'].items():
            print(f"  - {sensor_type}: {value:.2f} (权重: {fused_result['weights'][sensor_type]:.2f})")
    else:
        print(f"数据融合失败: {fused_result}")
    
    # 5. 连续数据采集示例
    print("\n5. 连续数据采集示例 (5次采样)")
    for i in range(5):
        # 生成模拟数据
        simulated_data = {
            'temperature': 25.0 + (i * 0.2),
            'humidity': 60.0 - (i * 0.5),
            'acceleration': 0.5 + (i * 0.1)
        }
        
        result = client.process_sensor_data(simulated_data)
        if result and result['status'] == 'success':
            timestamp = result['timestamp']
            temp_value = result['sensor_results'].get('temperature', {}).get('value', 0)
            humidity_value = result['sensor_results'].get('humidity', {}).get('value', 0)
            print(f"  采样 {i+1} ({timestamp}): 温度={temp_value:.2f}°C, 湿度={humidity_value:.2f}%")
        
        # 等待1秒
        time.sleep(1)
    
    print("\n===== 测试完成 =====")

# 实时数据监控类
class SensorDataMonitor:
    """传感器数据监控器"""
    
    def __init__(self, client, interval=2, duration=30):
        """初始化监控器"""
        self.client = client
        self.interval = interval  # 采样间隔(秒)
        self.duration = duration  # 监控持续时间(秒)
        self.running = False
    
    def start_monitoring(self):
        """开始监控"""
        self.running = True
        start_time = time.time()
        samples = []
        
        print(f"\n开始监控传感器数据 (持续 {self.duration} 秒, 间隔 {self.interval} 秒)")
        
        try:
            while self.running and (time.time() - start_time) < self.duration:
                # 生成模拟数据
                current_time = datetime.now().isoformat()
                simulated_data = {
                    'temperature': 25.0 + (time.time() * 0.1 % 2),  # 小幅度波动
                    'humidity': 60.0 + (time.time() * 0.2 % 5),
                    'acceleration': 0.5 + (time.time() * 0.05 % 1),
                    'timestamp': current_time
                }
                
                # 处理数据
                result = self.client.process_sensor_data(simulated_data)
                if result and result['status'] == 'success':
                    samples.append(result)
                    print(f"监控点: {current_time}")
                    print(f"  温度: {result['sensor_results'].get('temperature', {}).get('value', 0):.2f}°C")
                    print(f"  湿度: {result['sensor_results'].get('humidity', {}).get('value', 0):.2f}%")
                    print(f"  状态: {result['overall_status']}")
                
                # 等待下一次采样
                time.sleep(self.interval)
                
        except KeyboardInterrupt:
            print("监控已停止")
        finally:
            self.running = False
            print(f"监控结束，共采集 {len(samples)} 个样本")
            
            # 简单统计分析
            if samples:
                self.analyze_samples(samples)
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
    
    def analyze_samples(self, samples):
        """分析采集的样本"""
        print("\n样本统计分析:")
        
        # 提取温度数据
        temp_values = []
        humidity_values = []
        
        for sample in samples:
            if sample['status'] == 'success':
                temp = sample['sensor_results'].get('temperature', {}).get('value', None)
                humidity = sample['sensor_results'].get('humidity', {}).get('value', None)
                if temp is not None:
                    temp_values.append(temp)
                if humidity is not None:
                    humidity_values.append(humidity)
        
        if temp_values:
            print(f"温度统计: 平均={sum(temp_values)/len(temp_values):.2f}°C, "
                  f"最小={min(temp_values):.2f}°C, 最大={max(temp_values):.2f}°C")
        
        if humidity_values:
            print(f"湿度统计: 平均={sum(humidity_values)/len(humidity_values):.2f}%, "
                  f"最小={min(humidity_values):.2f}%, 最大={max(humidity_values):.2f}%")

# 主函数
def main():
    """主函数"""
    print("传感器API客户端示例")
    print(f"连接到: {SENSOR_API_BASE_URL}")
    
    # 测试基本API功能
    test_sensor_api()
    
    # 创建客户端
    client = SensorApiClient()
    
    # 创建并启动监控器
    monitor = SensorDataMonitor(client, interval=3, duration=15)
    monitor.start_monitoring()

if __name__ == '__main__':
    main()