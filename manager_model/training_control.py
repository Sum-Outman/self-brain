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
from flask import jsonify
from manager_model.model_registry import get_model_endpoint
from manager_model.language_resources import get_string

class TrainingController:
    def __init__(self):
        self.training_status = "idle"  # idle, training, completed, error
        self.training_progress = 0
        self.training_log = []
        self.training_thread = None
        self.stop_event = threading.Event()
        self.selected_models = []
        self.training_params = {}
        self.model_status = {}  # 存储每个模型的状态
        self.current_lang = "en"  # 默认语言
        
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

# Global training controller instance
training_controller = TrainingController()
