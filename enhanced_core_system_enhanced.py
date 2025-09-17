import os
import json
import time
import uuid
import logging
from datetime import datetime
import threading
import psutil
import gc
import numpy as np
import torch
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from collections import deque

# 设置日志 | Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AGIBrainCore")

class SubModelManager:
    """子模型管理器 | Sub-model manager"""
    def __init__(self):
        self.models = {}
        self.model_configs = {}
        self.logger = logging.getLogger("SubModelManager")
    
    def load_model(self, model_name: str, config: Dict[str, Any]) -> bool:
        """加载子模型 | Load sub-model"""
        try:
            # 模拟模型加载 | Simulate model loading
            self.models[model_name] = {"status": "loaded", "config": config}
            self.model_configs[model_name] = config
            self.logger.info(f"模型 {model_name} 已加载 | Model {model_name} loaded")
            return True
        except Exception as e:
            self.logger.error(f"加载模型 {model_name} 失败: {e} | Failed to load model {model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """卸载子模型 | Unload sub-model"""
        try:
            if model_name in self.models:
                del self.models[model_name]
                self.logger.info(f"模型 {model_name} 已卸载 | Model {model_name} unloaded")
                return True
            return False
        except Exception as e:
            self.logger.error(f"卸载模型 {model_name} 失败: {e} | Failed to unload model {model_name}: {e}")
            return False
    
    def get_model(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取子模型 | Get sub-model"""
        return self.models.get(model_name)
    
    def get_loaded_models(self) -> List[str]:
        """获取所有已加载的模型 | Get all loaded models"""
        return list(self.models.keys())
    
    def shutdown_all(self):
        """关闭所有模型 | Shutdown all models"""
        for model_name in list(self.models.keys()):
            self.unload_model(model_name)

class TaskExecutor:
    """任务执行器 | Task executor"""
    def __init__(self, max_workers: int = 5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks = {}
        self.logger = logging.getLogger("TaskExecutor")
    
    def submit(self, task_id: str, task_func: Callable, *args, **kwargs):
        """提交任务 | Submit task"""
        try:
            future = self.executor.submit(task_func, *args, **kwargs)
            self.tasks[task_id] = future
            return future
        except Exception as e:
            self.logger.error(f"提交任务 {task_id} 失败: {e} | Failed to submit task {task_id}: {e}")
            return None
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态 | Get task status"""
        if task_id not in self.tasks:
            return {"status": "not_found"}
        
        future = self.tasks[task_id]
        if future.done():
            try:
                result = future.result()
                return {"status": "completed", "result": result}
            except Exception as e:
                return {"status": "failed", "error": str(e)}
        else:
            return {"status": "running"}
    
    def shutdown(self, wait: bool = True):
        """关闭执行器 | Shutdown executor"""
        self.executor.shutdown(wait=wait)
        self.logger.info("任务执行器已关闭 | Task executor shut down")

class AGIBrainCore:
    """AGI大脑核心系统 | AGI Brain Core System"""
    def __init__(self, config_path=None):
        # 初始化默认配置 | Initialize default configuration
        self.default_config = {
            "submodels": {
                "text_processor": {"enabled": True}, 
                "image_analyzer": {"enabled": True}, 
                "audio_recognizer": {"enabled": True}
            },
            "optimization": {
                "self_optimization": True,
                "resource_threshold": 80,
                "learning_rate": 0.01,
                "batch_size": 32,
                "epochs": 5,
                "regular_optimization_interval": 100,
                "adjustment_factor": 1.1
            },
            "data_storage": {
                "enabled": True,
                "db_path": "agi_brain.db",
                "data_dir": "agi_data",
                "backup_interval": 3600,
                "max_history_entries": 10000,
                "max_learning_data_entries": 5000
            },
            "performance_monitoring": {
                "enabled": True,
                "check_interval": 10,
                "success_rate_threshold": 80,
                "processing_time_threshold": 2.0,
                "memory_usage_threshold": 85,
                "cpu_usage_threshold": 85
            },
            "knowledge_extraction": {
                "enabled": True,
                "min_confidence": 0.7,
                "keyword_extraction_enabled": True,
                "max_keywords": 5
            },
            "model_optimization": {
                "enabled": True,
                "hyperparameter_tuning": True,
                "architecture_optimization": True,
                "memory_optimization": True,
                "cpu_optimization": True
            },
            "logging": {
                "level": "INFO",
                "file_path": "agi_brain.log",
                "max_file_size_mb": 10,
                "backup_count": 5
            }
        }
        
        # 读取配置文件 | Read configuration file
        self.config = self._load_config(config_path)
        
        # 更新日志配置 | Update logging configuration
        self._configure_logging()
        
        # 初始化系统状态 | Initialize system status
        self.system_status = {
            "total_operations": 0,
            "successful_operations": 0,
            "failed_operations": 0,
            "start_time": datetime.now().isoformat(),
            "learning_data": [],
            "task_history": [],
            "optimization_history": [],
            "performance_metrics": {}
        }
        
        # 初始化组件 | Initialize components
        self.submodel_manager = SubModelManager()
        self.task_executor = TaskExecutor()
        
        # 初始化数据存储 | Initialize data storage
        self._init_data_storage()
        
        # 启动监控线程 | Start monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        # 初始化备份线程 | Initialize backup thread
        self.backup_thread = threading.Thread(target=self._backup_system)
        self.backup_thread.daemon = True
        self.backup_thread.start()
        
        logger.info("AGI大脑核心系统已初始化 | AGI Brain Core System initialized")
    
    def _load_config(self, config_path=None):
        """加载配置文件 | Load configuration file"""
        # 默认配置文件路径 | Default configuration file path
        default_config_path = Path("config") / "enhanced_core_config.json"
        
        # 使用指定路径或默认路径 | Use specified path or default path
        if config_path is None:
            config_path = default_config_path
        else:
            config_path = Path(config_path)
        
        try:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
                
                # 合并默认配置和加载的配置 | Merge default configuration with loaded configuration
                merged_config = self.default_config.copy()
                merged_config.update(loaded_config)
                
                # 递归合并嵌套配置 | Recursively merge nested configurations
                for key, value in loaded_config.items():
                    if key in merged_config and isinstance(value, dict) and isinstance(merged_config[key], dict):
                        merged_config[key].update(value)
                
                logger.info(f"配置文件已加载: {config_path} | Configuration file loaded: {config_path}")
                return merged_config
            else:
                logger.warning(f"配置文件不存在，使用默认配置 | Configuration file not found, using default configuration")
                return self.default_config
                
        except Exception as e:
            logger.error(f"加载配置文件失败: {e} | Failed to load configuration file: {e}")
            return self.default_config
            
    def _configure_logging(self):
        """配置日志系统 | Configure logging system"""
        try:
            logging_config = self.config.get("logging", {})
            log_level = getattr(logging, logging_config.get("level", "INFO"), logging.INFO)
            log_file = logging_config.get("file_path", "agi_brain.log")
            
            # 创建日志器 | Create logger
            logger = logging.getLogger("AGIBrainCore")
            logger.setLevel(log_level)
            
            # 清除现有处理器 | Clear existing handlers
            if logger.handlers:
                logger.handlers.clear()
                
            # 添加控制台处理器 | Add console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
            # 添加文件处理器 | Add file handler
            if log_file:
                max_bytes = logging_config.get("max_file_size_mb", 10) * 1024 * 1024
                backup_count = logging_config.get("backup_count", 5)
                
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
                )
                file_handler.setLevel(log_level)
                file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                file_handler.setFormatter(file_formatter)
                logger.addHandler(file_handler)
                
        except Exception as e:
            print(f"配置日志系统失败: {e} | Failed to configure logging system: {e}")
        
    def _init_data_storage(self):
        """初始化数据存储 | Initialize data storage"""
        if not self.config["data_storage"]["enabled"]:
            return
        
        # 创建数据目录 | Create data directory
        data_dir = self.config["data_storage"]["data_dir"]
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # 初始化数据库 | Initialize database
        db_path = self.config["data_storage"]["db_path"]
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 创建任务历史表 | Create task history table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS task_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT,
                    timestamp TEXT,
                    description TEXT,
                    result TEXT,
                    status TEXT
                )
            ''')
            
            # 创建性能指标表 | Create performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    processing_time REAL,
                    memory_usage REAL,
                    cpu_usage REAL,
                    successful_operations INTEGER,
                    failed_operations INTEGER
                )
            ''')
            
            # 创建模型参数表 | Create model parameters table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_parameters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT,
                    param_name TEXT,
                    param_value TEXT,
                    timestamp TEXT,
                    optimization_round INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info("数据存储初始化完成 | Data storage initialized")
            
        except Exception as e:
            logger.error(f"数据存储初始化失败: {e} | Failed to initialize data storage: {e}")
        
        # 创建学习数据目录 | Create learning data directory
        self.learning_data_dir = os.path.join(data_dir, "learning_data")
        if not os.path.exists(self.learning_data_dir):
            os.makedirs(self.learning_data_dir)
        
    def _backup_system(self):
        """系统备份 | System backup"""
        while True:
            try:
                if self.config["data_storage"]["enabled"]:
                    backup_interval = self.config["data_storage"]["backup_interval"]
                    time.sleep(backup_interval)
                    
                    # 执行备份 | Execute backup
                    self._backup_data()
                    
            except Exception as e:
                logger.error(f"系统备份失败: {e} | System backup failed: {e}")
                time.sleep(60)  # 出错后等待60秒重试
    
    def _backup_data(self):
        """备份数据 | Backup data"""
        try:
            # 创建备份目录 | Create backup directory
            backup_dir = os.path.join(self.config["data_storage"]["data_dir"], "backups")
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)
            
            # 备份数据库 | Backup database
            backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            db_path = self.config["data_storage"]["db_path"]
            backup_db_path = os.path.join(backup_dir, f"agi_brain_backup_{backup_time}.db")
            
            if os.path.exists(db_path):
                # 复制数据库文件 | Copy database file
                import shutil
                shutil.copy2(db_path, backup_db_path)
                logger.info(f"数据库已备份到 {backup_db_path} | Database backed up to {backup_db_path}")
            
            # 清理旧备份 | Clean up old backups
            self._clean_old_backups(backup_dir)
            
        except Exception as e:
            logger.error(f"数据备份失败: {e} | Data backup failed: {e}")
            
    def _clean_old_backups(self, backup_dir: str):
        """清理旧备份 | Clean up old backups"""
        try:
            # 获取所有备份文件 | Get all backup files
            backups = []
            for filename in os.listdir(backup_dir):
                if filename.startswith("agi_brain_backup_") and filename.endswith(".db"):
                    filepath = os.path.join(backup_dir, filename)
                    file_time = os.path.getmtime(filepath)
                    backups.append((filepath, file_time))
            
            # 按时间排序 | Sort by time
            backups.sort(key=lambda x: x[1])
            
            # 保留最近10个备份 | Keep last 10 backups
            while len(backups) > 10:
                old_backup, _ = backups.pop(0)
                os.remove(old_backup)
                logger.info(f"已删除旧备份: {old_backup} | Removed old backup: {old_backup}")
                
        except Exception as e:
            logger.error(f"清理旧备份失败: {e} | Failed to clean old backups: {e}")
    
    def process_task(self, task_description: str) -> Dict[str, Any]:
        """处理任务 | Process task"""
        start_time = time.time()
        task_id = f"task_{uuid.uuid4().hex[:8]}"
        
        try:
            # 构建多模态输入 | Build multimodal input
            multimodal_input = {
                "text": task_description,
                "timestamp": datetime.now().isoformat(),
                "task_id": task_id
            }
            
            # 执行任务 | Execute task
            executive_output = self._execute_task(multimodal_input)
            
            # 收集自主学习数据 | Collect self-learning data
            self._collect_self_learning_data(multimodal_input, executive_output)
            
            # 更新系统状态 | Update system status
            processing_time = time.time() - start_time
            self._update_system_status(executive_output, processing_time)
            
            # 记录任务历史 | Record task history
            self._record_task_history(task_id, task_description, executive_output)
            
            return executive_output
            
        except Exception as e:
            logger.error(f"处理任务 {task_id} 失败: {e} | Failed to process task {task_id}: {e}")
            self.system_status["failed_operations"] += 1
            
            # 生成备用响应 | Generate fallback response
            fallback_response = self._generate_fallback_response({"text": task_description}, e)
            
            # 记录任务历史 | Record task history
            self._record_task_history(task_id, task_description, fallback_response)
            
            return fallback_response
            
    def _execute_task(self, multimodal_input: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务 | Execute task"""
        # 这里应该实现实际的任务执行逻辑 | Should implement actual task execution logic
        # 模拟任务执行 | Simulate task execution
        task_id = multimodal_input.get("task_id", "unknown")
        task_description = multimodal_input.get("text", "")
        
        logger.info(f"执行任务 {task_id}: {task_description} | Executing task {task_id}: {task_description}")
        
        # 模拟处理时间 | Simulate processing time
        time.sleep(0.5)
        
        # 模拟结果 | Simulate result
        result = {
            "status": "success",
            "task_id": task_id,
            "result": f"已处理任务: {task_description}",
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.95
        }
        
        return result
        
    def _record_task_history(self, task_id: str, task_description: str, result: Dict[str, Any]):
        """记录任务历史 | Record task history"""
        # 创建任务记录 | Create task record
        task_record = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "description": task_description,
            "result": json.dumps(result),
            "status": result.get("status", "completed")
        }
        
        # 记录到内存中 | Record in memory
        if "task_history" not in self.system_status:
            max_history_entries = self.config["data_storage"].get("max_history_entries", 10000)
            self.system_status["task_history"] = deque(maxlen=max_history_entries)
        
        self.system_status["task_history"].append(task_record)
        
        # 持久化到数据库 | Persist to database
        if self.config["data_storage"]["enabled"]:
            try:
                db_path = self.config["data_storage"]["db_path"]
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute(
                    "INSERT INTO task_history (task_id, timestamp, description, result, status) VALUES (?, ?, ?, ?, ?)",
                    (task_record["task_id"], task_record["timestamp"], task_record["description"], 
                     task_record["result"], task_record["status"])
                )
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logger.error(f"记录任务历史到数据库失败: {e} | Failed to record task history to database: {e}")
                
        # 如果任务历史记录超过限制，清理最旧的记录 | Clean up oldest records if history exceeds limit
        max_history_entries = self.config["data_storage"].get("max_history_entries", 10000)
        if len(self.system_status["task_history"]) > max_history_entries:
            self.system_status["task_history"] = deque(list(self.system_status["task_history"])[-max_history_entries:], maxlen=max_history_entries)
            
    def _update_system_status(self, output: Dict[str, Any], processing_time: float):
        """更新系统状态 | Update system status"""
        self.system_status["total_operations"] += 1
        
        # 根据输出状态更新成功/失败计数 | Update success/failure count based on output status
        if output.get("status") == "success":
            self.system_status["successful_operations"] += 1
        else:
            self.system_status["failed_operations"] += 1
        
        # 更新性能指标 | Update performance metrics
        if "performance_metrics" not in self.system_status:
            self.system_status["performance_metrics"] = {}
        
        metrics = self.system_status["performance_metrics"]
        metrics["last_processing_time"] = processing_time
        metrics["average_processing_time"] = (
            metrics.get("average_processing_time", 0) * (self.system_status["total_operations"] - 1) + processing_time
        ) / self.system_status["total_operations"]
        
        # 计算成功率 | Calculate success rate
        metrics["success_rate"] = (
            self.system_status["successful_operations"] / self.system_status["total_operations"]
        ) * 100 if self.system_status["total_operations"] > 0 else 0
        
        # 更新内存使用情况 | Update memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        self.system_status["memory_usage"] = {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent()
        }
        
        # 更新CPU使用率 | Update CPU usage
        self.system_status["cpu_usage"] = psutil.cpu_percent(interval=0.1)
        
        # 持久化性能指标 | Persist performance metrics
        if self.config["data_storage"]["enabled"]:
            try:
                db_path = self.config["data_storage"]["db_path"]
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                
                cursor.execute(
                    "INSERT INTO performance_metrics (timestamp, processing_time, memory_usage, cpu_usage, successful_operations, failed_operations) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        datetime.now().isoformat(),
                        processing_time,
                        self.system_status["memory_usage"]["percent"],
                        self.system_status["cpu_usage"],
                        self.system_status["successful_operations"],
                        self.system_status["failed_operations"]
                    )
                )
                
                conn.commit()
                conn.close()
                
            except Exception as e:
                logger.error(f"记录性能指标到数据库失败: {e} | Failed to record performance metrics to database: {e}")
                
    def _collect_self_learning_data(self, input_data: Dict[str, Any], output_data: Dict[str, Any]):
        """收集自主学习数据 | Collect self-learning data"""
        # 创建学习数据 | Create learning data
        learning_data = {
            "timestamp": datetime.now().isoformat(),
            "input": input_data,
            "output": output_data,
            "performance_metrics": self.system_status["performance_metrics"].copy(),
            "memory_usage": self.system_status["memory_usage"],
            "cpu_usage": self.system_status["cpu_usage"]
        }
        
        # 记录到内存中 | Record in memory
        if "learning_data" not in self.system_status:
            max_learning_data_entries = self.config["data_storage"].get("max_learning_data_entries", 5000)
            self.system_status["learning_data"] = deque(maxlen=max_learning_data_entries)
        
        self.system_status["learning_data"].append(learning_data)
        
        # 持久化到文件 | Persist to file
        if self.config["data_storage"]["enabled"]:
            try:
                # 创建唯一的文件名 | Create unique file name
                file_id = uuid.uuid4().hex[:12]
                file_name = f"learning_data_{file_id}.json"
                file_path = os.path.join(self.learning_data_dir, file_name)
                
                # 写入文件 | Write to file
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(learning_data, f, ensure_ascii=False, indent=2)
                    
                # 执行简单的知识提取和学习 | Perform simple knowledge extraction and learning
                if self.config["knowledge_extraction"].get("enabled", True):
                    self._extract_knowledge(learning_data)
                
                if self.config["model_optimization"].get("enabled", True):
                    self._update_model_parameters_based_on_learning(learning_data)
                
            except Exception as e:
                logger.error(f"持久化学习数据失败: {e} | Failed to persist learning data: {e}")
                
    def _extract_knowledge(self, learning_data: Dict[str, Any]):
        """从学习数据中提取知识 | Extract knowledge from learning data"""
        try:
            # 检查置信度是否达到阈值 | Check if confidence reaches threshold
            confidence = learning_data["output"].get("confidence", 0.5)
            min_confidence = self.config["knowledge_extraction"].get("min_confidence", 0.7)
            
            if confidence < min_confidence:
                logger.debug(f"跳过知识提取，置信度 ({confidence}) 低于阈值 ({min_confidence}) | Skipping knowledge extraction, confidence ({confidence}) below threshold ({min_confidence})")
                return
            
            # 简单的知识提取逻辑 | Simple knowledge extraction logic
            input_text = learning_data["input"].get("text", "")
            output_result = learning_data["output"].get("result", "")
            
            if input_text and output_result:
                # 提取关键词 | Extract keywords
                keywords = []
                if self.config["knowledge_extraction"].get("keyword_extraction_enabled", True):
                    keywords = self._extract_keywords(input_text)
                
                # 存储知识映射 | Store knowledge mappings
                knowledge_mapping = {
                    "input_text": input_text,
                    "output_result": output_result,
                    "keywords": keywords,
                    "timestamp": datetime.now().isoformat(),
                    "confidence": confidence
                }
                
                # 保存到知识文件 | Save to knowledge file
                knowledge_file = os.path.join(self.config["data_storage"]["data_dir"], "knowledge_mappings.jsonl")
                with open(knowledge_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(knowledge_mapping, ensure_ascii=False) + '\n')
                    
                logger.debug(f"已提取知识，关键词: {keywords} | Knowledge extracted, keywords: {keywords}")
                    
        except Exception as e:
            logger.error(f"知识提取失败: {e} | Knowledge extraction failed: {e}")
            
    def _extract_keywords(self, text: str) -> List[str]:
        """从文本中提取关键词 | Extract keywords from text"""
        # 简单的关键词提取逻辑 | Simple keyword extraction logic
        # 在实际系统中，应该使用更复杂的NLP算法 | In a real system, should use more complex NLP algorithms
        import re
        
        # 去除标点符号和数字 | Remove punctuation and numbers
        text = re.sub(r'[^\w\s]|\d', ' ', text)
        
        # 转为小写 | Convert to lowercase
        text = text.lower()
        
        # 分词 | Split into words
        words = text.split()
        
        # 过滤停用词 | Filter stop words
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'of', 'for', 'with', 'by'])
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # 返回指定数量的关键词 | Return specified number of keywords
        max_keywords = self.config["knowledge_extraction"].get("max_keywords", 5)
        return keywords[:max_keywords]
        
    def _update_model_parameters_based_on_learning(self, learning_data: Dict[str, Any]):
        """基于学习数据更新模型参数 | Update model parameters based on learning data"""
        try:
            # 检查是否启用了超参数调整 | Check if hyperparameter tuning is enabled
            if not self.config["model_optimization"].get("hyperparameter_tuning", True):
                logger.debug("超参数调整已禁用 | Hyperparameter tuning is disabled")
                return
            
            # 简单的模型参数更新逻辑 | Simple model parameter update logic
            # 这里模拟根据性能指标调整参数 | Here we simulate adjusting parameters based on performance metrics
            metrics = learning_data["performance_metrics"]
            success_rate = metrics.get("success_rate", 0)
            processing_time = metrics.get("last_processing_time", 0)
            
            # 定义需要调整的参数 | Define parameters to adjust
            params_to_adjust = {}
            
            # 从配置中获取阈值 | Get thresholds from configuration
            success_rate_threshold = self.config["performance_monitoring"].get("success_rate_threshold", 80)
            processing_time_threshold = self.config["performance_monitoring"].get("processing_time_threshold", 2.0)
            adjustment_factor = self.config["optimization"].get("adjustment_factor", 1.1)
            
            # 如果成功率低于阈值，调整学习率 | If success rate is below threshold, adjust learning rate
            if success_rate < success_rate_threshold:
                current_lr = self.config["optimization"]["learning_rate"]
                new_lr = current_lr * adjustment_factor  # 根据配置调整
                max_lr = 0.1  # 最大学习率上限
                new_lr = min(max_lr, new_lr)
                params_to_adjust["learning_rate"] = new_lr
                
            # 如果处理时间过长，调整批处理大小 | If processing time is too long, adjust batch size
            if processing_time > processing_time_threshold:
                current_batch_size = self.config["optimization"]["batch_size"]
                new_batch_size = max(8, current_batch_size // 2)  # 减半，但不低于8
                params_to_adjust["batch_size"] = new_batch_size
                
            # 应用参数调整 | Apply parameter adjustments
            for param_name, param_value in params_to_adjust.items():
                old_value = self.config["optimization"][param_name]
                self.config["optimization"][param_name] = param_value
                
                logger.info(f"已调整参数 {param_name}: {old_value} -> {param_value} | Adjusted parameter {param_name}: {old_value} -> {param_value}")
                
                # 持久化参数调整 | Persist parameter adjustments
                if self.config["data_storage"]["enabled"]:
                    try:
                        db_path = self.config["data_storage"]["db_path"]
                        conn = sqlite3.connect(db_path)
                        cursor = conn.cursor()
                        
                        # 获取当前优化轮次 | Get current optimization round
                        cursor.execute("SELECT MAX(optimization_round) FROM model_parameters")
                        result = cursor.fetchone()
                        current_round = result[0] + 1 if result[0] is not None else 1
                        
                        cursor.execute(
                            "INSERT INTO model_parameters (model_name, param_name, param_value, timestamp, optimization_round) VALUES (?, ?, ?, ?, ?)",
                            ("global", param_name, str(param_value), datetime.now().isoformat(), current_round)
                        )
                        
                        conn.commit()
                        conn.close()
                        
                    except Exception as e:
                        logger.error(f"持久化参数调整失败: {e} | Failed to persist parameter adjustment: {e}")
                        
        except Exception as e:
            logger.error(f"更新模型参数失败: {e} | Failed to update model parameters: {e}")
            
    def _generate_fallback_response(self, input_data: Dict[str, Any], error: Exception) -> Dict[str, Any]:
        """生成备用响应 | Generate fallback response"""
        return {
            "status": "error",
            "message": f"系统处理失败: {str(error)} | System processing failed: {str(error)}",
            "fallback_response": "抱歉，系统暂时无法处理您的请求。请稍后再试或联系管理员。",
            "timestamp": datetime.now().isoformat()
        }
        
    def _monitor_system(self):
        """监控系统状态 | Monitor system status"""
        while True:
            try:
                # 检查系统资源 | Check system resources
                cpu_percent = psutil.cpu_percent()
                memory_info = psutil.virtual_memory()
                
                # 记录资源使用情况 | Record resource usage
                self.system_status["resource_usage"] = {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory_info.percent,
                    "memory_available_gb": memory_info.available / 1024 / 1024 / 1024,
                    "timestamp": datetime.now().isoformat()
                }
                
                # 检查是否需要优化 | Check if optimization is needed
                resource_threshold = self.config["optimization"]["resource_threshold"]
                if (cpu_percent > resource_threshold or memory_info.percent > resource_threshold) and self.config["optimization"]["self_optimization"]:
                    logger.warning(f"系统资源使用过高 (CPU: {cpu_percent}%, Memory: {memory_info.percent}%), 触发优化 | High system resource usage (CPU: {cpu_percent}%, Memory: {memory_info.percent}%), triggering optimization")
                    self.optimize_system()
                    
                # 定期优化 | Regular optimization
                regular_interval = self.config["optimization"].get("regular_optimization_interval", 100)
                if self.system_status["total_operations"] % regular_interval == 0 and self.system_status["total_operations"] > 0:
                    logger.info(f"已处理 {self.system_status['total_operations']} 个操作，触发定期优化 | Processed {self.system_status['total_operations']} operations, triggering regular optimization")
                    self.optimize_system()
                    
                # 检查子模型状态 | Check submodel status
                self._update_submodel_status()
                
                # 休眠一段时间 | Sleep for a while
                check_interval = self.config["performance_monitoring"].get("check_interval", 10)
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"系统监控错误: {e} | System monitoring error: {e}")
                time.sleep(30)
                
    def _update_submodel_status(self):
        """更新子模型状态 | Update submodel status"""
        try:
            submodel_status = {}
            for model_name in self.config["submodels"]:
                if self.config["submodels"][model_name]["enabled"]:
                    submodel = self.submodel_manager.get_model(model_name)
                    if submodel:
                        submodel_status[model_name] = {
                            "status": "active",
                            "last_activity": datetime.now().isoformat()
                        }
                    else:
                        submodel_status[model_name] = {
                            "status": "inactive",
                            "reason": "Model not loaded"
                        }
                else:
                    submodel_status[model_name] = {
                        "status": "disabled",
                        "reason": "Model disabled in config"
                    }
                
            self.system_status["submodel_status"] = submodel_status
            
        except Exception as e:
            logger.error(f"更新子模型状态失败: {e} | Failed to update submodel status: {e}")
            
    def optimize_system(self):
        """优化系统性能 | Optimize system performance"""
        logger.info("开始系统优化 | Starting system optimization")
        
        # 记录优化开始前的状态 | Record status before optimization
        pre_optimization_state = {
            "memory_usage": self.system_status.get("memory_usage", {}),
            "cpu_usage": self.system_status.get("cpu_usage", 0),
            "performance_metrics": self.system_status.get("performance_metrics", {})
        }
        
        try:
            # 清理内存 | Clean up memory
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 分析系统性能 | Analyze system performance
            performance_analysis = self._analyze_performance()
            
            # 调整超参数 | Adjust hyperparameters
            self._adjust_hyperparameters(performance_analysis)
            
            # 优化子模型 | Optimize submodels
            self._optimize_submodels()
            
            # 优化架构 | Optimize architecture
            self._optimize_architecture(performance_analysis)
            
            # 记录优化后状态 | Record status after optimization
            post_optimization_state = {
                "memory_usage": {
                    "rss_mb": psutil.Process().memory_info().rss / 1024 / 1024,
                    "vms_mb": psutil.Process().memory_info().vms / 1024 / 1024,
                    "percent": psutil.Process().memory_percent()
                },
                "cpu_usage": psutil.cpu_percent(interval=0.1),
                "performance_metrics": self.system_status.get("performance_metrics", {})
            }
            
            # 记录优化历史 | Record optimization history
            optimization_record = {
                "timestamp": datetime.now().isoformat(),
                "pre_optimization": pre_optimization_state,
                "post_optimization": post_optimization_state,
                "performance_analysis": performance_analysis,
                "status": "success"
            }
            
            if "optimization_history" not in self.system_status:
                self.system_status["optimization_history"] = []
            
            self.system_status["optimization_history"].append(optimization_record)
            
            # 限制优化历史大小 | Limit optimization history size
            if len(self.system_status["optimization_history"]) > 100:
                self.system_status["optimization_history"] = self.system_status["optimization_history"][-100:]
                
            logger.info("系统优化完成 | System optimization completed")
            
        except Exception as e:
            logger.error(f"系统优化失败: {e} | System optimization failed: {e}")
            
            optimization_record = {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "status": "failed"
            }
            
            if "optimization_history" not in self.system_status:
                self.system_status["optimization_history"] = []
            
            self.system_status["optimization_history"].append(optimization_record)
            
    def _analyze_performance(self) -> Dict[str, Any]:
        """分析系统性能 | Analyze system performance"""
        try:
            # 获取性能指标 | Get performance metrics
            metrics = self.system_status.get("performance_metrics", {})
            
            # 计算关键性能指标 | Calculate key performance indicators
            success_rate = metrics.get("success_rate", 0)
            avg_processing_time = metrics.get("average_processing_time", 0)
            
            # 获取配置的阈值 | Get configured thresholds
            monitoring_config = self.config["performance_monitoring"]
            success_rate_threshold = monitoring_config.get("success_rate_threshold", 80)
            processing_time_threshold = monitoring_config.get("processing_time_threshold", 2.0)
            memory_threshold = monitoring_config.get("memory_usage_threshold", 85)
            cpu_threshold = monitoring_config.get("cpu_usage_threshold", 85)
            
            # 确定性能瓶颈 | Identify performance bottlenecks
            bottlenecks = []
            
            # 成功率低 | Low success rate
            if success_rate < (success_rate_threshold - 10):
                bottlenecks.append({"type": "success_rate", "severity": "high", "value": success_rate})
            elif success_rate < success_rate_threshold:
                bottlenecks.append({"type": "success_rate", "severity": "medium", "value": success_rate})
                
            # 处理时间长 | Long processing time
            if avg_processing_time > (processing_time_threshold * 2.5):
                bottlenecks.append({"type": "processing_time", "severity": "high", "value": avg_processing_time})
            elif avg_processing_time > processing_time_threshold:
                bottlenecks.append({"type": "processing_time", "severity": "medium", "value": avg_processing_time})
                
            # 内存使用率高 | High memory usage
            memory_percent = self.system_status.get("memory_usage", {}).get("percent", 0)
            if memory_percent > memory_threshold:
                bottlenecks.append({"type": "memory_usage", "severity": "high", "value": memory_percent})
            elif memory_percent > (memory_threshold - 15):
                bottlenecks.append({"type": "memory_usage", "severity": "medium", "value": memory_percent})
                
            # CPU使用率高 | High CPU usage
            cpu_percent = self.system_status.get("cpu_usage", 0)
            if cpu_percent > cpu_threshold:
                bottlenecks.append({"type": "cpu_usage", "severity": "high", "value": cpu_percent})
            elif cpu_percent > (cpu_threshold - 15):
                bottlenecks.append({"type": "cpu_usage", "severity": "medium", "value": cpu_percent})
                
            # 构建性能分析结果 | Build performance analysis result
            analysis_result = {
                "success_rate": success_rate,
                "avg_processing_time": avg_processing_time,
                "memory_percent": memory_percent,
                "cpu_percent": cpu_percent,
                "bottlenecks": bottlenecks,
                "timestamp": datetime.now().isoformat()
            }
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"性能分析失败: {e} | Performance analysis failed: {e}")
            return {"error": str(e), "status": "failed"}
            
    def _adjust_hyperparameters(self, performance_analysis: Dict[str, Any]):
        """调整超参数 | Adjust hyperparameters"""
        try:
            # 检查是否启用了超参数调整 | Check if hyperparameter tuning is enabled
            if not self.config["model_optimization"].get("hyperparameter_tuning", True):
                logger.debug("超参数调整已禁用 | Hyperparameter tuning is disabled")
                return
            
            # 获取当前优化配置 | Get current optimization config
            optimization_config = self.config["optimization"]
            
            # 获取性能瓶颈 | Get performance bottlenecks
            bottlenecks = performance_analysis.get("bottlenecks", [])
            
            # 根据瓶颈调整超参数 | Adjust hyperparameters based on bottlenecks
            for bottleneck in bottlenecks:
                bottleneck_type = bottleneck["type"]
                severity = bottleneck["severity"]
                
                if bottleneck_type == "success_rate" and severity == "high":
                    # 成功率低，增加学习率和训练轮数 | Low success rate, increase learning rate and epochs
                    current_lr = optimization_config["learning_rate"]
                    new_lr = min(0.1, current_lr * 1.5)  # 增加50%，但不超过0.1
                    optimization_config["learning_rate"] = new_lr
                    
                    current_epochs = optimization_config["epochs"]
                    new_epochs = min(20, current_epochs * 2)  # 增加1倍，最多20轮
                    optimization_config["epochs"] = new_epochs
                    
                    logger.info(f"调整超参数: 学习率={new_lr}, 训练轮数={new_epochs} | Adjusted hyperparameters: learning_rate={new_lr}, epochs={new_epochs}")
                    
                elif bottleneck_type == "processing_time" and severity == "high":
                    # 处理时间长，减少批处理大小 | Long processing time, decrease batch size
                    current_batch_size = optimization_config["batch_size"]
                    new_batch_size = max(8, current_batch_size // 2)  # 减半，最小8
                    optimization_config["batch_size"] = new_batch_size
                    
                    logger.info(f"调整超参数: 批处理大小={new_batch_size} | Adjusted hyperparameters: batch_size={new_batch_size}")
                    
                elif bottleneck_type == "memory_usage" and severity == "high":
                    # 内存使用率高，减少批处理大小 | High memory usage, decrease batch size
                    current_batch_size = optimization_config["batch_size"]
                    new_batch_size = max(4, current_batch_size // 2)  # 减半，最小4
                    optimization_config["batch_size"] = new_batch_size
                    
                    logger.info(f"调整超参数: 批处理大小={new_batch_size} | Adjusted hyperparameters: batch_size={new_batch_size}")
                    
        except Exception as e:
            logger.error(f"调整超参数失败: {e} | Failed to adjust hyperparameters: {e}")
            
    def _optimize_submodels(self):
        """优化子模型 | Optimize submodels"""
        try:
            for model_name in self.submodel_manager.get_loaded_models():
                try:
                    submodel = self.submodel_manager.get_model(model_name)
                    if hasattr(submodel, 'optimize'):
                        submodel.optimize()
                        logger.info(f"子模型 {model_name} 优化完成 | Submodel {model_name} optimization completed")
                        
                except Exception as e:
                    logger.warning(f"优化子模型 {model_name} 失败: {e} | Failed to optimize submodel {model_name}: {e}")
                    
        except Exception as e:
            logger.error(f"优化子模型失败: {e} | Failed to optimize submodels: {e}")
            
    def _optimize_architecture(self, performance_analysis: Dict[str, Any]):
        """优化系统架构 | Optimize system architecture"""
        try:
            # 检查是否启用了架构优化 | Check if architecture optimization is enabled
            if not self.config["model_optimization"].get("architecture_optimization", True):
                logger.debug("架构优化已禁用 | Architecture optimization is disabled")
                return
            
            # 基于性能分析结果优化架构 | Optimize architecture based on performance analysis
            bottlenecks = performance_analysis.get("bottlenecks", [])
            
            # 检查是否有严重的CPU或内存瓶颈 | Check for severe CPU or memory bottlenecks
            has_severe_resource_bottleneck = any(
                b["severity"] == "high" and b["type"] in ("cpu_usage", "memory_usage")
                for b in bottlenecks
            )
            
            if has_severe_resource_bottleneck:
                # 尝试卸载不常用的子模型 | Try to unload less frequently used submodels
                # 这里简单模拟，实际系统中应该根据使用频率和重要性决定 | Simple simulation here, in real system should decide based on usage frequency and importance
                loaded_models = self.submodel_manager.get_loaded_models()
                if len(loaded_models) > 1:
                    # 选择一个模型卸载 | Choose a model to unload
                    model_to_unload = loaded_models[-1]  # 简单选择最后一个加载的模型
                    if self.submodel_manager.unload_model(model_to_unload):
                        logger.info(f"为优化系统资源，已卸载子模型 {model_to_unload} | Unloaded submodel {model_to_unload} for resource optimization")
                        
        except Exception as e:
            logger.error(f"优化系统架构失败: {e} | Failed to optimize system architecture: {e}")
            
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态 | Get system status"""
        return self.system_status
        
    def shutdown(self):
        """关闭系统 | Shutdown system"""
        logger.info("正在关闭AGI大脑核心系统 | Shutting down AGI Brain Core System")
        
        # 关闭子模型 | Shutdown submodels
        self.submodel_manager.shutdown_all()
        
        # 关闭任务执行器 | Shutdown task executor
        self.task_executor.shutdown(wait=False)
        
        # 记录关闭时间 | Record shutdown time
        self.system_status["shutdown_time"] = datetime.now().isoformat()
        
        logger.info("AGI大脑核心系统已关闭 | AGI Brain Core System shut down")

# 主程序入口
# Main program entry
if __name__ == "__main__":
    # 初始化AGI大脑核心系统
    # Initialize AGI Brain Core System
    agi_brain = AGIBrainCore()
    
    try:
        # 示例任务处理
        # Example task processing
        task_result = agi_brain.process_task("请分析这张图片中的内容并生成描述")
        print(f"任务结果: {task_result}")
        
        # 获取系统状态
        status = agi_brain.get_system_status()
        print(f"系统状态: {json.dumps(status, indent=2, ensure_ascii=False)}")
        
        # 保持系统运行
        # Keep system running
        import time
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在关闭系统 | Received interrupt signal, shutting down system")
    finally:
        agi_brain.shutdown()