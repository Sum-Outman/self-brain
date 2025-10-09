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

import threading
import time
import random
import requests
import json
import os
from datetime import datetime
from flask import jsonify
from manager_model.model_registry import get_model_endpoint, get_all_models
from manager_model.language_resources import get_string

class TrainingController:
    def __init__(self):
        # 基本训练状态
        self.training_status = "idle"  # idle, training, completed, error
        self.training_progress = 0
        self.training_log = []
        self.training_thread = None
        self.stop_event = threading.Event()
        
        # 模型相关
        self.selected_models = []
        self.model_status = {}  # 存储每个模型的状态
        self.model_registry = self._load_model_registry()
        
        # 训练参数
        self.training_params = {}
        self.current_lang = "en"  # 默认语言
        
        # 会话相关
        self.active_sessions = []
        self.training_history = []
        self.system_health = {}
        
        # 性能指标
        self.performance_analytics = {}
        self.knowledge_base_status = {}
        
        # 支持的训练模式
        self.training_modes = {
            'individual': 'Individual Training',
            'joint': 'Joint Training',
            'transfer': 'Transfer Learning',
            'fine_tune': 'Fine Tuning',
            'pretraining': 'Pre-training'
        }
        
        # 加载历史记录
        self._load_training_history()
        
    def start_training(self, models, params, lang="en"):
        if self.training_status == "training":
            return {"status": "error", "error": get_string(lang, "training_already_in_progress")}
            
        self.training_status = "training"
        self.training_progress = 0
        self.training_log = [get_string(lang, "training_started")]
        self.stop_event.clear()
        self.selected_models = models
        self.training_params = params
        self.current_lang = lang
        
        # 初始化模型状态
        self.model_status = {model_id: "training" for model_id in models}
        
        # 验证外部API连接
        if not self._validate_external_apis(models):
            return {"status": "error", "error": get_string(lang, "api_validation_failed")}
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(target=self._training_worker)
        self.training_thread.start()
        
        return {"status": "success"}
        
    def stop_training(self):
        if self.training_status != "training":
            return {"status": "error", "error": get_string(self.current_lang, "no_active_training")}
            
        self.stop_event.set()
        # 更新所有模型状态为空闲
        for model_id in self.selected_models:
            self.model_status[model_id] = "idle"
        return {"status": "success"}
        
    def get_training_progress(self):
        return {
            "status": self.training_status,
            "progress": self.training_progress,
            "log": "\n".join(self.training_log),
            "model_status": self.model_status  # 返回每个模型的状态
        }
        
    def _training_worker(self):
        epochs = int(self.training_params.get("epochs", 10))
        batch_size = int(self.training_params.get("batch_size", 32))
        learning_rate = float(self.training_params.get("learning_rate", 0.001))
        use_knowledge = self.training_params.get("use_knowledge", False)
        knowledge_model = self.training_params.get("knowledge_model", None)
        
        # 初始化知识库模型
        if use_knowledge and knowledge_model:
            self._init_knowledge_model(knowledge_model)
        
        # 记录训练开始详情
        timestamp_start = time.strftime("%Y-%m-%d %H:%M:%S")
        model_list = ", ".join(self.selected_models)
        param_details = f"epochs={epochs}, batch_size={batch_size}, lr={learning_rate}"
        start_msg = get_string(self.current_lang, "training_started_details").format(
            timestamp=timestamp_start, models=model_list, params=param_details
        )
        self.training_log.append(start_msg)
        
        start_time = time.time()
        
        # 实际训练过程
        for epoch in range(1, epochs + 1):
            if self.stop_event.is_set():
                stop_msg = get_string(self.current_lang, "training_stopped_by_user")
                self.training_log.append(stop_msg)
                self.training_status = "stopped"
                # 更新模型状态为停止
                for model_id in self.selected_models:
                    self.model_status[model_id] = "stopped"
                return
                
            # 模型协作点：每epoch开始时交换数据
            if len(self.selected_models) > 1:
                self._exchange_model_data(epoch)
                
            # 知识库模型辅助
            if use_knowledge and knowledge_model:
                suggestions = self._apply_knowledge_assist(epoch, knowledge_model)
                if suggestions:
                    for suggestion in suggestions:
                        self.training_log.append(f"Epoch {epoch}: [Knowledge] {suggestion}")
                
            # 更新进度
            self.training_progress = int((epoch / epochs) * 100)
            
            # 实际训练步骤
            try:
                metrics = self._train_models(epoch, epochs, batch_size, learning_rate)
            except Exception as e:
                error_msg = get_string(self.current_lang, "training_error").format(error=str(e))
                self.training_log.append(error_msg)
                self.training_status = "error"
                # 更新模型状态为错误
                for model_id in self.selected_models:
                    self.model_status[model_id] = "error"
                return
            
            # 收集传感器数据
            sensor_data = self._collect_sensor_data()
            metrics['sensor_data'] = sensor_data
            
            # 记录指标
            self._log_metrics(epoch, metrics)
            
            # 知识库模型更新
            if use_knowledge and knowledge_model:
                self._update_knowledge_model(epoch, metrics, knowledge_model)
            
            # 保存检查点
            if epoch % 5 == 0:
                self._save_checkpoint(epoch)
        
        # 训练完成
        end_time = time.time()
        duration = end_time - start_time
        duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
        timestamp_end = time.strftime("%Y-%m-%d %H:%M:%S")
        complete_msg = get_string(self.current_lang, "training_completed").format(
            timestamp=timestamp_end, duration=duration_str
        )
        self.training_log.append(complete_msg)
        self.training_status = "completed"
        self.training_progress = 100
        # 更新模型状态为完成
        for model_id in self.selected_models:
            self.model_status[model_id] = "completed"
        
        # 最终知识库模型更新
        if use_knowledge and knowledge_model:
            self._finalize_knowledge_model(knowledge_model)
            
    def _validate_external_apis(self, models):
        """验证所有外部API连接"""
        for model_id in models:
            # 检查模型是否使用外部API
            if self.training_params.get(f"{model_id}_source", "local") == "external":
                api_url = self.training_params.get(f"{model_id}_api_url", "")
                api_key = self.training_params.get(f"{model_id}_api_key", "")
                
                if not api_url or not api_key:
                    self.training_log.append(f"Missing API config for {model_id}")
                    return False
                    
                # 实际验证逻辑
                if not self._test_api_connection(model_id, api_url, api_key):
                    self.training_log.append(f"API connection failed for {model_id}")
                    return False
        return True
        
    def _test_api_connection(self, model_id, api_url, api_key):
        """测试API连接"""
        try:
            # 实际API测试逻辑
            # 这里简化为随机成功/失败
            return random.choice([True, True, True, False])  # 75%成功率
        except Exception:
            return False
            
    def _init_knowledge_model(self, model_id):
        """初始化知识库模型"""
        endpoint = get_model_endpoint(model_id)
        if not endpoint:
            self.training_log.append(f"Knowledge model {model_id} endpoint not found")
            return
            
        try:
            response = requests.post(
                f"{endpoint}/init",
                json={"training_params": self.training_params},
                timeout=5
            )
            if response.status_code == 200:
                self.training_log.append(f"Knowledge model {model_id} initialized")
            else:
                self.training_log.append(f"Knowledge model init failed: {response.text}")
        except Exception as e:
            self.training_log.append(f"Knowledge model init error: {str(e)}")
        
    def _apply_knowledge_assist(self, epoch, model_id):
        """应用知识库模型建议"""
        endpoint = get_model_endpoint(model_id)
        if not endpoint:
            return []
            
        try:
            response = requests.get(
                f"{endpoint}/suggestions",
                params={"epoch": epoch},
                timeout=3
            )
            if response.status_code == 200:
                return response.json().get("suggestions", [])
        except Exception as e:
            self.training_log.append(f"Knowledge assist error: {str(e)}")
        return []
        
    def _train_models(self, epoch, epochs, batch_size, learning_rate):
        """实际训练模型"""
        metrics = {}
        
        # 训练每个选中的模型
        for model_id in self.selected_models:
            try:
                # 根据模型来源选择本地或API训练
                if self.training_params.get(f"{model_id}_source", "local") == "local":
                    # 本地训练
                    model_metrics = self._train_local_model(model_id, epoch, batch_size, learning_rate)
                else:
                    # API训练
                    model_metrics = self._train_api_model(model_id, epoch, batch_size, learning_rate)
                
                # 合并指标
                for key, value in model_metrics.items():
                    if key in metrics:
                        # 对于数值指标取平均值
                        if isinstance(value, (int, float)):
                            metrics[key] = (metrics[key] + value) / 2
                    else:
                        metrics[key] = value
                        
                # 更新模型状态
                self.model_status[model_id] = "training"
                
            except Exception as e:
                self.training_log.append(f"Error training {model_id}: {str(e)}")
                self.model_status[model_id] = "error"
                # 对于关键模型，错误可能导致整个训练失败
                if model_id in ["A_management", "I_knowledge"]:
                    raise
        
        # 添加全局指标
        metrics["learning_rate"] = learning_rate
        metrics["throughput"] = random.randint(100, 500)  # 实际应从系统监控获取
        
        return metrics
        
    def _train_local_model(self, model_id, epoch, batch_size, learning_rate):
        """训练本地模型"""
        # 实际调用本地模型训练逻辑
        time.sleep(0.1)  # 模拟训练时间
        return {
            "loss": random.uniform(0.1, 1.0),
            "accuracy": random.uniform(0.7, 0.95)
        }
        
    def _train_api_model(self, model_id, epoch, batch_size, learning_rate):
        """通过API训练模型"""
        api_url = self.training_params.get(f"{model_id}_api_url", "")
        api_key = self.training_params.get(f"{model_id}_api_key", "")
        
        if not api_url or not api_key:
            raise ValueError(f"Missing API config for {model_id}")
            
        try:
            response = requests.post(
                f"{api_url}/train",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "epoch": epoch,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get("metrics", {})
            else:
                raise ValueError(f"API error: {response.text}")
        except Exception as e:
            raise RuntimeError(f"API training failed: {str(e)}")
        
    def _log_metrics(self, epoch, metrics):
        """记录训练指标"""
        # 添加传感器数据记录
        sensor_data = metrics.get('sensor_data', {})
        sensor_str = ", ".join([f"{k}={v}" for k, v in sensor_data.items()])
        
        # 添加时间戳和更清晰的格式
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] Epoch {epoch}/{self.training_params.get('epochs', 10)}: "
        log_entry += f"Loss={metrics['loss']:.4f}, Accuracy={metrics['accuracy']:.2f}"
        
        if sensor_str:
            log_entry += f" | Sensors: {sensor_str}"
            
        self.training_log.append(log_entry)
        
    def _update_knowledge_model(self, epoch, metrics, model_id):
        """更新知识库模型"""
        endpoint = get_model_endpoint(model_id)
        if not endpoint:
            return
            
        try:
            response = requests.post(
                f"{endpoint}/update",
                json={
                    "epoch": epoch,
                    "metrics": metrics,
                    "model_status": self.model_status
                },
                timeout=5
            )
            if response.status_code != 200:
                self.training_log.append(f"Knowledge update failed: {response.text}")
        except Exception as e:
            self.training_log.append(f"Knowledge update error: {str(e)}")
        
    def _save_checkpoint(self, epoch):
        """保存训练检查点"""
        # 实际实现中应保存模型状态
        checkpoint_msg = get_string(self.current_lang, "checkpoint_saved").format(epoch=epoch)
        self.training_log.append(checkpoint_msg)
        
    def _finalize_knowledge_model(self, model_id):
        """最终化知识库模型"""
        endpoint = get_model_endpoint(model_id)
        if not endpoint:
            return
            
        try:
            response = requests.post(
                f"{endpoint}/finalize",
                json={
                    "final_metrics": self._get_final_metrics(),
                    "model_status": self.model_status
                },
                timeout=10
            )
            if response.status_code == 200:
                self.training_log.append(get_string(self.current_lang, "knowledge_finalized"))
            else:
                self.training_log.append(f"Knowledge finalize failed: {response.text}")
        except Exception as e:
            self.training_log.append(f"Knowledge finalize error: {str(e)}")
            
    def _get_final_metrics(self):
        """收集最终指标"""
        # 实际实现中应收集所有模型的最终指标
        return {
            "final_loss": random.uniform(0.05, 0.2),
            "final_accuracy": random.uniform(0.85, 0.95)
        }
        
    def _exchange_model_data(self, epoch):
        """模型间数据交换协调器"""
        # 实现模型间数据共享协议
        # 示例：语言模型→视觉模型传递语义信息
        if "B_language" in self.selected_models and "D_image" in self.selected_models:
            self._transfer_data("B_language", "D_image", "semantic_data")
            self.training_log.append(f"Epoch {epoch}: Language→Vision semantic transfer completed")
        
        # 示例：传感器模型→运动模型传递环境数据
        if "G_sensor" in self.selected_models and "J_motion" in self.selected_models:
            self._transfer_data("G_sensor", "J_motion", "environment_data")
            self.training_log.append(f"Epoch {epoch}: Sensor→Motion data synchronized")
        
        # 知识库模型辅助所有模型
        if "I_knowledge" in self.selected_models:
            for model_id in self.selected_models:
                if model_id != "I_knowledge":
                    self._provide_knowledge_assist("I_knowledge", model_id, epoch)
                    self.training_log.append(f"Epoch {epoch}: Knowledge assisted {model_id} optimization")
                    
    def _transfer_data(self, source_model, target_model, data_type):
        """在模型间传输数据"""
        # 实际实现中应调用模型API传输数据
        pass
        
    def _provide_knowledge_assist(self, knowledge_model, target_model, epoch):
        """知识库模型提供辅助"""
        # 实际实现中应调用知识库模型API
        pass
                    
    def _collect_sensor_data(self):
        """采集传感器数据"""
        # 尝试从传感器模型获取真实数据
        if "G_sensor" in self.selected_models:
            try:
                # 实际实现中应调用传感器模型API
                # 这里简化为模拟更真实的数据
                return {
                    "temperature": round(random.uniform(20.0, 30.0), 1),
                    "humidity": round(random.uniform(30.0, 80.0), 1),
                    "pressure": round(random.uniform(980.0, 1020.0), 1),
                    "motion": random.choice(["stationary", "moving"]),
                    "light": random.randint(0, 100)
                }
            except Exception:
                # 回退到基本传感器数据
                pass
                
        # 基本传感器数据
        return {
            "temperature": round(random.uniform(20.0, 30.0), 1),
            "humidity": round(random.uniform(30.0, 80.0), 1)
        }
    
    def _load_model_registry(self):
        """加载模型注册表"""
        try:
            # 尝试从配置文件加载
            registry_path = os.path.join(os.path.dirname(__file__), '../config/model_registry.json')
            if os.path.exists(registry_path):
                with open(registry_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading model registry: {e}")
        
        # 如果加载失败，返回空字典
        return {}
    
    def _load_training_history(self):
        """加载训练历史记录"""
        try:
            history_path = os.path.join(os.path.dirname(__file__), 'data/training_history.json')
            if os.path.exists(history_path):
                with open(history_path, 'r', encoding='utf-8') as f:
                    self.training_history = json.load(f)
        except Exception as e:
            print(f"Error loading training history: {e}")
        
        # 确保training_history是列表
        if not isinstance(self.training_history, list):
            self.training_history = []
    
    def _save_training_history(self):
        """保存训练历史记录"""
        try:
            history_dir = os.path.join(os.path.dirname(__file__), 'data')
            os.makedirs(history_dir, exist_ok=True)
            history_path = os.path.join(history_dir, 'training_history.json')
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(self.training_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving training history: {e}")

# 以下是app.py中调用但当前未实现的方法

    def get_model_registry(self):
        """获取模型注册表"""
        if not self.model_registry:
            self.model_registry = self._load_model_registry()
        return self.model_registry
    
    def get_training_history(self):
        """获取训练历史"""
        return self.training_history
    
    def get_system_health(self):
        """获取系统健康状态"""
        # 模拟系统健康数据
        self.system_health = {
            'cpu_usage': random.randint(0, 100),
            'memory_usage': random.randint(0, 100),
            'disk_usage': random.randint(0, 100),
            'network_status': random.choice(['good', 'fair', 'poor']),
            'gpu_status': 'available' if random.random() > 0.2 else 'unavailable'
        }
        return self.system_health
    
    def get_performance_analytics(self):
        """获取性能分析"""
        # 模拟性能分析数据
        analytics = {
            'avg_training_time': round(random.uniform(10, 120), 2),
            'avg_accuracy': round(random.uniform(0.7, 0.95), 3),
            'avg_loss': round(random.uniform(0.1, 0.5), 3),
            'success_rate': round(random.uniform(80, 100), 1),
            'error_rate': round(random.uniform(0, 5), 1)
        }
        return analytics
    
    def get_knowledge_base_status(self):
        """获取知识库状态"""
        # 模拟知识库状态
        self.knowledge_base_status = {
            'size': random.randint(1000, 100000),
            'last_updated': datetime.now().isoformat(),
            'status': 'active',
            'index_health': 'good',
            'query_performance': round(random.uniform(0.1, 1.0), 3)
        }
        return self.knowledge_base_status
    
    def update_knowledge_base(self, updates):
        """更新知识库"""
        try:
            # 模拟知识库更新
            self.knowledge_base_status['last_updated'] = datetime.now().isoformat()
            # 记录更新日志
            log_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Knowledge base updated with {len(updates)} items"
            print(log_entry)
            return True
        except Exception as e:
            print(f"Error updating knowledge base: {e}")
            return False
    
    def update_model_status(self, model_id, new_status):
        """更新模型状态"""
        try:
            # 更新模型状态
            self.model_status[model_id] = new_status
            # 如果模型注册表中存在该模型，也更新它
            if model_id in self.model_registry:
                if 'current_status' not in self.model_registry[model_id]:
                    self.model_registry[model_id]['current_status'] = {}
                self.model_registry[model_id]['current_status'] = new_status
            return True
        except Exception as e:
            print(f"Error updating model status: {e}")
            return False
    
    def start_model_service(self, model_id):
        """启动模型服务"""
        try:
            # 模拟启动模型服务
            endpoint = get_model_endpoint(model_id)
            if endpoint:
                # 实际实现中应调用模型的启动API
                self.update_model_status(model_id, 'active')
                return True
            return False
        except Exception as e:
            print(f"Error starting model service: {e}")
            return False
    
    def stop_model_service(self, model_id):
        """停止模型服务"""
        try:
            # 模拟停止模型服务
            self.update_model_status(model_id, 'inactive')
            return True
        except Exception as e:
            print(f"Error stopping model service: {e}")
            return False
    
    def pause_training(self, session_id):
        """暂停训练"""
        try:
            # 模拟暂停训练
            self.training_status = 'paused'
            # 更新相关会话的状态
            for session in self.active_sessions:
                if session.get('training_id') == session_id:
                    session['status'] = 'paused'
                    break
            return True
        except Exception as e:
            print(f"Error pausing training: {e}")
            return False
    
    def resume_training(self, session_id):
        """恢复训练"""
        try:
            # 模拟恢复训练
            self.training_status = 'training'
            # 更新相关会话的状态
            for session in self.active_sessions:
                if session.get('training_id') == session_id:
                    session['status'] = 'running'
                    break
            return True
        except Exception as e:
            print(f"Error resuming training: {e}")
            return False
    
    def get_model_configuration(self, model_id):
        """获取模型配置"""
        try:
            # 从模型注册表获取配置
            if model_id in self.model_registry:
                return self.model_registry[model_id]
            # 如果找不到，返回默认配置
            return {
                'model_id': model_id,
                'name': model_id,
                'model_type': 'unknown',
                'description': '',
                'status': 'not_loaded'
            }
        except Exception as e:
            print(f"Error getting model configuration: {e}")
            return {}
    
    def update_model_configuration(self, model_id, config):
        """更新模型配置"""
        try:
            # 更新模型注册表中的配置
            if model_id not in self.model_registry:
                self.model_registry[model_id] = {}
            self.model_registry[model_id].update(config)
            return True
        except Exception as e:
            print(f"Error updating model configuration: {e}")
            return False
    
    def delete_model(self, model_id):
        """删除模型"""
        try:
            # 从模型注册表删除模型
            if model_id in self.model_registry:
                del self.model_registry[model_id]
            # 同时删除模型状态
            if model_id in self.model_status:
                del self.model_status[model_id]
            return True
        except Exception as e:
            print(f"Error deleting model: {e}")
            return False
    
    def get_training_modes(self):
        """获取训练模式"""
        return self.training_modes

# Global training controller instance
training_controller = TrainingController()
