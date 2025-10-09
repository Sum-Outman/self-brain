# Copyright 2025 AGI System Team
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

import json
import requests
import time
import threading
from queue import Queue
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ModelCommunication")

class ModelCommunicator:
    def __init__(self):
        # 模型服务端点配置
        self.model_endpoints = {
            'B_language': 'http://localhost:5001',
            'C_audio': 'http://localhost:5002',
            'D_image': 'http://localhost:5003',
            'E_video': 'http://localhost:5004',
            'F_spatial': 'http://localhost:5005',
            'G_sensor': 'http://localhost:5010',
            'H_computer': 'http://localhost:5011',
            'I_motion': 'http://localhost:5012',
            'I_knowledge': 'http://localhost:5009',
            'K_programming': 'http://localhost:5013'
        }
        
        # 请求队列和响应缓存
        self.request_queues = {model_id: Queue() for model_id in self.model_endpoints}
        self.response_cache = {model_id: {} for model_id in self.model_endpoints}
        self.request_counter = {model_id: 0 for model_id in self.model_endpoints}
        
        # 启动异步请求处理线程
        self.threads = {}
        self.running = True
        for model_id in self.model_endpoints:
            self.threads[model_id] = threading.Thread(target=self._process_request_queue, args=(model_id,))
            self.threads[model_id].daemon = True
            self.threads[model_id].start()
        
        # 健康检查线程
        self.health_check_thread = threading.Thread(target=self._health_check)
        self.health_check_thread.daemon = True
        self.health_check_thread.start()
        
        # 模型健康状态
        self.model_health = {model_id: True for model_id in self.model_endpoints}
    
    def send_request(self, model_id, endpoint, data, timeout=30, async_mode=False, retries=3):
        """向指定模型发送请求，带重试机制"""
        if model_id not in self.model_endpoints:
            return {"error": f"Unknown model: {model_id}"}
        
        if not self.model_health.get(model_id, False):
            return {"error": f"Model {model_id} is not healthy"}
        
        url = f"{self.model_endpoints[model_id]}/{endpoint}"
        
        if async_mode:
            # 异步模式，将请求加入队列
            request_id = self.request_counter[model_id]
            self.request_counter[model_id] += 1
            
            self.request_queues[model_id].put({
                'request_id': request_id,
                'url': url,
                'data': data,
                'timeout': timeout,
                'retries': retries
            })
            
            return {"request_id": request_id, "status": "queued"}
        else:
            # 同步模式，直接发送请求，带重试
            for attempt in range(retries + 1):
                try:
                    start_time = time.time()
                    response = requests.post(url, json=data, timeout=timeout)
                    end_time = time.time()
                    
                    if response.status_code == 200:
                        return {
                            "status": "success",
                            "data": response.json(),
                            "response_time": end_time - start_time,
                            "attempts": attempt + 1
                        }
                    else:
                        if attempt < retries:
                            logger.warning(f"Request attempt {attempt+1} failed, retrying...")
                            time.sleep(1)  # 简单退避策略
                        else:
                            return {
                                "status": "error",
                                "code": response.status_code,
                                "message": response.text,
                                "attempts": attempt + 1
                            }
                except requests.exceptions.RequestException as e:
                    if attempt < retries:
                        logger.warning(f"Request attempt {attempt+1} failed with exception: {str(e)}, retrying...")
                        time.sleep(1)
                    else:
                        logger.error(f"Request to {model_id} failed after {retries+1} attempts: {str(e)}")
                        return {"status": "error", "message": str(e), "attempts": attempt + 1}
    
    def _process_request_queue(self, model_id):
        """处理异步请求队列"""
        while self.running:
            try:
                request_data = self.request_queues[model_id].get(timeout=1)
                request_id = request_data['request_id']
                url = request_data['url']
                data = request_data['data']
                timeout = request_data['timeout']
                
                try:
                    response = requests.post(url, json=data, timeout=timeout)
                    if response.status_code == 200:
                        self.response_cache[model_id][request_id] = {
                            "status": "success",
                            "data": response.json()
                        }
                    else:
                        self.response_cache[model_id][request_id] = {
                            "status": "error",
                            "code": response.status_code,
                            "message": response.text
                        }
                except requests.exceptions.RequestException as e:
                    logger.error(f"Async request to {model_id} failed: {str(e)}")
                    self.response_cache[model_id][request_id] = {
                        "status": "error",
                        "message": str(e)
                    }
                finally:
                    self.request_queues[model_id].task_done()
            except:
                pass
    
    def get_response(self, model_id, request_id, delete_after=True):
        """获取异步请求的响应"""
        if model_id not in self.response_cache or request_id not in self.response_cache[model_id]:
            return {"status": "pending"}
        
        response = self.response_cache[model_id][request_id]
        
        if delete_after:
            del self.response_cache[model_id][request_id]
        
        return response
    
    def _health_check(self):
        """定期检查模型服务健康状态"""
        while self.running:
            for model_id, endpoint in self.model_endpoints.items():
                try:
                    response = requests.get(f"{endpoint}/status", timeout=5)
                    self.model_health[model_id] = (response.status_code == 200)
                except:
                    self.model_health[model_id] = False
                    logger.warning(f"Health check failed for {model_id}")
            
            # 每30秒检查一次
            time.sleep(30)
    
    def get_model_health(self):
        """获取所有模型的健康状态"""
        return self.model_health
    
    def stop(self):
        """停止所有线程"""
        self.running = False
        for thread in self.threads.values():
            thread.join(timeout=2)
        self.health_check_thread.join(timeout=2)

# 单例模式
model_communicator = ModelCommunicator()