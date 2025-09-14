"""
离线版本的改进系统
Offline Improved System
不需要外部依赖
"""

import sys
import os
import json
import random
import time
from datetime import datetime
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class OfflineEmotionAnalyzer:
    """离线情感分析器"""
    
    EMOTIONS = {
        'happy': ['joy', 'excitement', 'delight', 'cheerful'],
        'sad': ['sorrow', 'grief', 'melancholy', 'down'],
        'angry': ['fury', 'rage', 'irritation', 'annoyance'],
        'fear': ['anxiety', 'worry', 'dread', 'panic'],
        'surprise': ['amazement', 'astonishment', 'wonder', 'shock'],
        'neutral': ['calm', 'peaceful', 'balanced', 'steady']
    }
    
    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """分析文本情感"""
        text_lower = text.lower()
        
        # 简单的关键词匹配
        emotion_scores = {}
        
        for emotion, keywords in self.EMOTIONS.items():
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1
            emotion_scores[emotion] = score
        
        # 归一化分数
        total = sum(emotion_scores.values()) or 1
        normalized_scores = {k: v/total for k, v in emotion_scores.items()}
        
        # 获取主要情感
        primary_emotion = max(normalized_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'primary_emotion': primary_emotion,
            'scores': normalized_scores,
            'confidence': normalized_scores[primary_emotion],
            'timestamp': datetime.now().isoformat()
        }

class OfflineTrainingSystem:
    """离线训练系统"""
    
    def __init__(self):
        self.jobs = {}
        self.job_id = 1
    
    def start_training(self, model_type: str, dataset_path: str, config: Dict) -> str:
        """启动训练"""
        job_id = f"job_{self.job_id}"
        self.job_id += 1
        
        self.jobs[job_id] = {
            'job_id': job_id,
            'model_type': model_type,
            'dataset_path': dataset_path,
            'config': config,
            'status': 'running',
            'progress': 0,
            'start_time': datetime.now().isoformat(),
            'estimated_completion': None,
            'metrics': {}
        }
        
        # 模拟训练进度
        threading.Thread(target=self._simulate_training, args=(job_id,), daemon=True).start()
        
        return job_id
    
    def _simulate_training(self, job_id: str):
        """模拟训练过程"""
        import threading
        
        for progress in range(0, 101, 10):
            if job_id in self.jobs:
                self.jobs[job_id]['progress'] = progress
                self.jobs[job_id]['metrics'] = {
                    'loss': round(random.uniform(0.1, 2.0), 3),
                    'accuracy': round(random.uniform(0.7, 0.99), 3),
                    'epoch': progress // 10
                }
                time.sleep(1)
        
        if job_id in self.jobs:
            self.jobs[job_id]['status'] = 'completed'
            self.jobs[job_id]['completion_time'] = datetime.now().isoformat()
    
    def get_training_status(self, job_id: str) -> Dict[str, Any]:
        """获取训练状态"""
        if job_id not in self.jobs:
            return {'error': 'Job not found'}
        
        return self.jobs[job_id]

class OfflineSensorSystem:
    """离线传感器系统"""
    
    def __init__(self):
        self.devices = {}
        self.data_history = []
        self.running = False
        self._add_mock_devices()
    
    def _add_mock_devices(self):
        """添加模拟设备"""
        mock_devices = [
            {'id': 'temp_001', 'type': 'temperature', 'location': 'room1'},
            {'id': 'humidity_001', 'type': 'humidity', 'location': 'room1'},
            {'id': 'light_001', 'type': 'light', 'location': 'room2'},
            {'id': 'motion_001', 'type': 'motion', 'location': 'entrance'}
        ]
        
        for device in mock_devices:
            self.devices[device['id']] = device
    
    def discover_devices(self) -> Dict[str, List[str]]:
        """发现设备"""
        return {
            'temperature': ['temp_001'],
            'humidity': ['humidity_001'],
            'light': ['light_001'],
            'motion': ['motion_001']
        }
    
    def start_data_collection(self):
        """开始数据收集"""
        self.running = True
        threading.Thread(target=self._collect_data, daemon=True).start()
    
    def _collect_data(self):
        """收集数据"""
        import threading
        
        while self.running:
            for device_id, device in self.devices.items():
                if device['type'] == 'temperature':
                    value = round(random.uniform(18, 25), 1)
                elif device['type'] == 'humidity':
                    value = round(random.uniform(30, 70), 1)
                elif device['type'] == 'light':
                    value = round(random.uniform(0, 1000), 1)
                elif device['type'] == 'motion':
                    value = random.choice([0, 1])
                
                data_point = {
                    'device_id': device_id,
                    'type': device['type'],
                    'value': value,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.data_history.append(data_point)
                
                # 只保留最近100条记录
                if len(self.data_history) > 100:
                    self.data_history = self.data_history[-100:]
            
            time.sleep(2)
    
    def get_real_time_data(self) -> Dict[str, Any]:
        """获取实时数据"""
        latest_data = {}
        for device_id in self.devices:
            device_data = [d for d in self.data_history if d['device_id'] == device_id]
            if device_data:
                latest_data[device_id] = device_data[-1]
        
        return {
            'devices': latest_data,
            'count': len(latest_data)
        }
    
    def get_device_status(self) -> Dict[str, Any]:
        """获取设备状态"""
        status = {}
        for device_id, device in self.devices.items():
            status[device_id] = {
                'online': True,
                'last_reading': datetime.now().isoformat(),
                'type': device['type']
            }
        return status

class OfflineStereoVision:
    """离线双目视觉系统"""
    
    def __init__(self):
        self.running = False
        self.calibrated = False
        self.latest_results = {}
    
    def calibrate_stereo(self):
        """标定相机"""
        self.calibrated = True
        print("Stereo cameras calibrated successfully")
    
    def start_real_time_processing(self):
        """开始实时处理"""
        self.running = True
        threading.Thread(target=self._process_frames, daemon=True).start()
    
    def _process_frames(self):
        """处理视频帧"""
        import threading
        
        while self.running:
            # 模拟物体检测
            objects = []
            for i in range(random.randint(0, 3)):
                objects.append({
                    'id': f'obj_{i}',
                    'type': random.choice(['person', 'chair', 'table', 'laptop']),
                    'distance': round(random.uniform(0.5, 5.0), 2),
                    'x': random.randint(0, 640),
                    'y': random.randint(0, 480),
                    'confidence': round(random.uniform(0.7, 1.0), 2)
                })
            
            self.latest_results = {
                'objects': objects,
                'timestamp': datetime.now().isoformat(),
                'frame_count': len(objects)
            }
            
            time.sleep(1)
    
    def get_latest_results(self) -> Dict[str, Any]:
        """获取最新结果"""
        return self.latest_results

class OfflineProgrammingSystem:
    """离线编程系统"""
    
    def __init__(self):
        self.learning_history = []
    
    def generate_and_optimize(self, requirements: str) -> Dict[str, Any]:
        """生成和优化代码"""
        # 简单的代码生成逻辑
        code_templates = {
            'fibonacci': '''
def fibonacci(n):
    """Generate fibonacci sequence up to n terms"""
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    
    sequence = [0, 1]
    for i in range(2, n):
        sequence.append(sequence[i-1] + sequence[i-2])
    return sequence

# Example usage
result = fibonacci(10)
print(result)
''',
            'sort': '''
def quick_sort(arr):
    """Quick sort implementation"""
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quick_sort(left) + middle + quick_sort(right)

# Example usage
numbers = [3, 1, 4, 1, 5, 9, 2, 6]
sorted_numbers = quick_sort(numbers)
print(sorted_numbers)
''',
            'search': '''
def binary_search(arr, target):
    """Binary search implementation"""
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Example usage
numbers = [1, 3, 5, 7, 9, 11, 13]
index = binary_search(numbers, 7)
print(f"Found at index: {index}")
'''
        }
        
        # 根据需求选择模板
        requirements_lower = requirements.lower()
        selected_template = 'fibonacci'  # 默认模板
        
        if 'sort' in requirements_lower:
            selected_template = 'sort'
        elif 'search' in requirements_lower:
            selected_template = 'search'
        
        result = {
            'code': code_templates[selected_template].strip(),
            'language': 'python',
            'optimization_suggestions': [
                'Add input validation',
                'Consider edge cases',
                'Add type hints for better documentation'
            ],
            'performance_score': round(random.uniform(0.7, 1.0), 2),
            'generated_at': datetime.now().isoformat()
        }
        
        self.learning_history.append({
            'requirements': requirements,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        return result
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """获取学习摘要"""
        return {
            'total_optimizations': len(self.learning_history),
            'recent_optimizations': self.learning_history[-5:] if self.learning_history else []
        }

class OfflineDataBus:
    """离线数据总线"""
    
    def __init__(self):
        self.data_queue = {}
    
    def publish(self, channel: str, data: Any):
        """发布数据"""
        if channel not in self.data_queue:
            self.data_queue[channel] = []
        
        self.data_queue[channel].append({
            'data': data,
            'timestamp': datetime.now().isoformat()
        })
        
        # 限制每个频道的消息数量
        if len(self.data_queue[channel]) > 100:
            self.data_queue[channel] = self.data_queue[channel][-100:]
    
    def get_data(self, channel: str) -> List[Dict[str, Any]]:
        """获取频道数据"""
        return self.data_queue.get(channel, [])

# 全局实例
import threading

offline_emotion = OfflineEmotionAnalyzer()
offline_training = OfflineTrainingSystem()
offline_sensors = OfflineSensorSystem()
offline_stereo = OfflineStereoVision()
offline_programming = OfflineProgrammingSystem()
offline_data_bus = OfflineDataBus()

if __name__ == "__main__":
    print("=== 测试离线改进系统 ===\n")
    
    # 1. 测试情感分析
    print("1. 情感分析测试:")
    emotion_result = offline_emotion.analyze_emotion("I love this new system!")
    print(f"   结果: {emotion_result}")
    
    # 2. 测试传感器系统
    print("\n2. 传感器系统测试:")
    offline_sensors.start_data_collection()
    time.sleep(3)
    sensor_data = offline_sensors.get_real_time_data()
    print(f"   数据: {sensor_data}")
    
    # 3. 测试双目视觉
    print("\n3. 双目视觉测试:")
    offline_stereo.calibrate_stereo()
    offline_stereo.start_real_time_processing()
    time.sleep(3)
    vision_data = offline_stereo.get_latest_results()
    print(f"   结果: {vision_data}")
    
    # 4. 测试编程系统
    print("\n4. 编程系统测试:")
    programming_result = offline_programming.generate_and_optimize("Create a fibonacci function")
    print(f"   代码: {programming_result['code']}")
    
    # 5. 测试训练系统
    print("\n5. 训练系统测试:")
    job_id = offline_training.start_training('neural_network', 'test_data', {'epochs': 5})
    print(f"   任务ID: {job_id}")
    time.sleep(2)
    training_status = offline_training.get_training_status(job_id)
    print(f"   状态: {training_status}")
    
    # 6. 测试数据总线
    print("\n6. 数据总线测试:")
    offline_data_bus.publish('test', {'message': 'Hello World'})
    bus_data = offline_data_bus.get_data('test')
    print(f"   数据: {bus_data}")
    
    print("\n=== 所有离线系统测试完成 ===")