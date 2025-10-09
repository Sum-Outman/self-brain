<<<<<<< HEAD
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data Bus Module
This module implements a central data bus for communication between different components of the Self Brain AGI system.
"""

import logging
import threading
from typing import Dict, List, Any, Callable, Optional
from queue import Queue
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DataBus")

class DataBus:
    """Central data bus for inter-component communication
    
    Implements a publish-subscribe pattern for message passing between different components
    of the Self Brain AGI system.
    """
    
    def __init__(self):
        """Initialize the data bus"""
        # Dictionary to store subscribers for each topic
        self._subscribers: Dict[str, List[Dict[str, Any]]] = {}
        # Lock for thread safety
        self._lock = threading.RLock()
        # Counter for subscription IDs
        self._subscription_counter = 0
        # Internal message queue for processing
        self._message_queue = Queue()
        # Flag to control the message processing thread
        self._running = True
        # Start the message processing thread
        self._start_processing_thread()
        logger.info("DataBus initialized successfully")
    
    def _start_processing_thread(self):
        """Start the message processing thread"""
        self._processing_thread = threading.Thread(target=self._process_messages, daemon=True)
        self._processing_thread.start()
    
    def _process_messages(self):
        """Process messages from the queue"""
        while self._running:
            try:
                # Get message from queue (blocking)
                topic, data, timestamp = self._message_queue.get(timeout=1)
                
                # Process the message
                self._deliver_message(topic, data, timestamp)
                
                # Mark task as done
                self._message_queue.task_done()
            except Exception as e:
                # Ignore queue timeout exceptions
                if not isinstance(e, TimeoutError):
                    logger.error(f"Error processing message: {str(e)}")
    
    def subscribe(self, topic: str, callback: Callable) -> int:
        """Subscribe to a topic
        
        Args:
            topic: Topic to subscribe to
            callback: Function to call when a message is published to the topic
            
        Returns:
            Subscription ID that can be used to unsubscribe
        """
        with self._lock:
            # Generate subscription ID
            subscription_id = self._subscription_counter
            self._subscription_counter += 1
            
            # Create topic if it doesn't exist
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            
            # Add subscriber
            self._subscribers[topic].append({
                "id": subscription_id,
                "callback": callback
            })
            
            logger.debug(f"New subscription to topic '{topic}': ID {subscription_id}")
            return subscription_id
    
    def unsubscribe(self, subscription_id: int) -> bool:
        """Unsubscribe from a topic using subscription ID
        
        Args:
            subscription_id: ID returned by subscribe()
            
        Returns:
            True if unsubscribe was successful, False otherwise
        """
        with self._lock:
            for topic, subscribers in self._subscribers.items():
                for i, subscriber in enumerate(subscribers):
                    if subscriber["id"] == subscription_id:
                        subscribers.pop(i)
                        logger.debug(f"Unsubscribed from topic '{topic}': ID {subscription_id}")
                        # Clean up empty topics
                        if not subscribers:
                            del self._subscribers[topic]
                        return True
            logger.warning(f"Failed to unsubscribe: Subscription ID {subscription_id} not found")
            return False
    
    def publish(self, topic: str, data: Any) -> None:
        """Publish data to a topic
        
        Args:
            topic: Topic to publish to
            data: Data to publish
        """
        timestamp = datetime.now().isoformat()
        
        # Add message to queue for async processing
        self._message_queue.put((topic, data, timestamp))
        logger.debug(f"Published message to topic '{topic}'")
    
    def _deliver_message(self, topic: str, data: Any, timestamp: str) -> None:
        """Deliver message to all subscribers of the topic
        
        Args:
            topic: Topic to deliver to
            data: Data to deliver
            timestamp: Timestamp of the message
        """
        with self._lock:
            # Check if topic exists
            if topic not in self._subscribers:
                logger.debug(f"No subscribers for topic '{topic}', message ignored")
                return
            
            # Make a copy of subscribers to avoid issues if list changes during iteration
            subscribers = self._subscribers[topic].copy()
        
        # Deliver message to each subscriber
        for subscriber in subscribers:
            try:
                # Call subscriber callback with data and metadata
                subscriber["callback"]({
                    "data": data,
                    "topic": topic,
                    "timestamp": timestamp,
                    "subscription_id": subscriber["id"]
                })
            except Exception as e:
                logger.error(f"Error delivering message to subscriber {subscriber['id']}: {str(e)}")
    
    def get_topics(self) -> List[str]:
        """Get list of all available topics
        
        Returns:
            List of topic names
        """
        with self._lock:
            return list(self._subscribers.keys())
    
    def get_subscriber_count(self, topic: Optional[str] = None) -> int:
        """Get the number of subscribers for a specific topic or all topics
        
        Args:
            topic: Topic name or None for all topics
            
        Returns:
            Number of subscribers
        """
        with self._lock:
            if topic is not None:
                if topic not in self._subscribers:
                    return 0
                return len(self._subscribers[topic])
            else:
                return sum(len(subscribers) for subscribers in self._subscribers.values())
    
    def shutdown(self):
        """Shut down the data bus and stop message processing"""
        self._running = False
        if hasattr(self, '_processing_thread'):
            self._processing_thread.join(timeout=5)
        logger.info("DataBus shut down successfully")

# Singleton instance of DataBus
_data_bus_instance = None
_data_bus_lock = threading.Lock()

def get_data_bus() -> DataBus:
    """Get the singleton instance of DataBus
    
    Returns:
        The singleton DataBus instance
    """
    global _data_bus_instance
    with _data_bus_lock:
        if _data_bus_instance is None:
            _data_bus_instance = DataBus()
    return _data_bus_instance

# Example usage
if __name__ == "__main__":
    # Create a test subscriber callback
    def test_callback(data):
        print(f"Received data: {data}")
    
    # Get data bus instance
    bus = get_data_bus()
    
    # Subscribe to a test topic
    subscription_id = bus.subscribe("test_topic", test_callback)
    
    # Publish a test message
    bus.publish("test_topic", {"message": "Hello, World!"})
    
    # Wait for message processing
    import time
    time.sleep(1)
    
    # Unsubscribe
    bus.unsubscribe(subscription_id)
    
    # Shutdown
    bus.shutdown()
=======
# 模型间通信总线 | Inter-model Communication Bus
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

import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
import re
import time
import json
import os
import threading  # 用于直接通道的线程安全 | For thread safety in direct channels
import subprocess  # 用于编程模型命令执行 | For programming model command execution
import shlex  # 用于安全拆分命令行参数 | For safely splitting command line arguments

# DataBus singleton instance
_data_bus_instance = None
_data_bus_lock = threading.RLock()

class DataBus:
    def __init__(self, storage_path: str = "data_bus_storage"):
        self.channels = {}  # 通信通道字典 | Communication channels dictionary
        self.direct_channels = {}  # 直接通信通道字典 | Direct communication channels dictionary
        self.message_routing = {}  # 消息路由表 | Message routing table
        self.knowledge_channels = set()  # 知识库专用通道 | Knowledge base specific channels
        self.subscriptions = {}  # 订阅者字典 | Subscribers dictionary
        self.storage_path = storage_path
        self.logger = logging.getLogger("DataBus")
        
        # 确保存储目录存在 | Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
    
    def create_channel(self, channel_id: str, capacity: int = 10, persistent: bool = False, priority: int = 1):
        """创建新的通信通道 | Create new communication channel
        参数:
            channel_id: 通道ID (格式: 任务ID-源模型-目标模型) | Channel ID (format: taskID-sourceModel-targetModel)
            capacity: 通道容量 | Channel capacity
            persistent: 是否持久化消息 | Whether to persist messages
            priority: 通道优先级 (1-5, 1最高) | Channel priority (1-5, 1 highest)
        """
        if channel_id in self.channels:
            self.logger.warning(f"通道已存在: {channel_id} | Channel already exists: {channel_id}")
            return
            
        self.channels[channel_id] = {
            "messages": [],
            "capacity": capacity,
            "throughput": 0,
            "persistent": persistent,
            "priority": max(1, min(5, priority)),  # 限制优先级在1-5范围 | Limit priority to 1-5 range
            "created_at": time.time()
        }
        self.logger.info(f"创建通道: {channel_id} | Channel created: {channel_id}")
        
        # 如果是知识库通道则特殊标记 | Special mark for knowledge base channels
        if channel_id.endswith("-I"):
            self.knowledge_channels.add(channel_id)
            self.logger.info(f"注册知识库通道: {channel_id} | Registered knowledge channel: {channel_id}")
    
    def register_message_route(self, message_type: str, target_model: str, pattern_match: bool = False):
        """注册消息路由 | Register message route
        参数:
            message_type: 消息类型或模式 (支持正则表达式) | Message type or pattern (supports regex)
            target_model: 目标模型ID | Target model ID
            pattern_match: 是否使用模式匹配 | Whether to use pattern matching
        """
        self.message_routing[message_type] = {
            "target": target_model,
            "pattern": pattern_match
        }
        self.logger.info(f"注册消息路由: {message_type} -> {target_model} | Message route registered: {message_type} -> {target_model}")
        
    def register_knowledge_query(self, query_type: str):
        """注册知识查询类型 | Register knowledge query type
        参数:
            query_type: 知识查询类型 | Knowledge query type
        """
        self.register_message_route(query_type, "I")
        self.logger.info(f"注册知识查询: {query_type} -> I | Knowledge query registered: {query_type} -> I")
    
    def boost_channel(self, channel_pattern: str, factor: float = 1.5):
        """提升匹配通道的容量 | Boost capacity of matching channels
        参数:
            channel_pattern: 通道ID模式 (支持*通配符) | Channel ID pattern (supports * wildcard)
            factor: 容量提升因子 | Capacity boost factor
        """
        boosted = 0
        for channel_id in self.channels:
            if self._match_pattern(channel_id, channel_pattern):
                self.channels[channel_id]["capacity"] = int(self.channels[channel_id]["capacity"] * factor)
                boosted += 1
                
        self.logger.info(f"提升 {boosted} 个通道容量 (模式: {channel_pattern}) | Boosted {boosted} channels (pattern: {channel_pattern})")
    
    def send_message(self, channel_id: str, message: Dict, ttl: int = 60) -> bool:
        """通过通道发送消息 | Send message through channel
        参数:
            channel_id: 通道ID | Channel ID
            message: 消息内容 | Message content
            ttl: 消息生存时间(秒) | Message time-to-live (seconds)
        返回:
            是否发送成功 | Whether send was successful
        """
        if channel_id not in self.channels:
            self.logger.error(f"通道不存在: {channel_id} | Channel not found: {channel_id}")
            return False
            
        channel = self.channels[channel_id]
        if len(channel["messages"]) >= channel["capacity"]:
            self.logger.warning(f"通道已满: {channel_id} | Channel full: {channel_id}")
            return False
            
        # 添加元数据 | Add metadata
        message["_meta"] = {
            "timestamp": time.time(),
            "ttl": ttl,
            "priority": channel["priority"]
        }
        
        channel["messages"].append(message)
        channel["throughput"] += 1
        
        # 持久化消息 | Persist message if required
        if channel["persistent"]:
            self._persist_message(channel_id, message)
            
        return True
        
    def broadcast_to_knowledge(self, message: Dict, source_model: str, ttl: int = 60) -> int:
        """广播消息到所有知识库通道 | Broadcast message to all knowledge channels
        参数:
            message: 消息内容 | Message content
            source_model: 源模型ID | Source model ID
            ttl: 消息生存时间(秒) | Message time-to-live (seconds)
        返回:
            成功发送的消息数量 | Number of successful sends
        """
        count = 0
        for channel_id in self.knowledge_channels:
            if source_model in channel_id:  # 避免发送给自己 | Avoid sending to self
                continue
                
            if self.send_message(channel_id, message, ttl):
                count += 1
                
        self.logger.info(f"知识库广播: 发送 {count} 条消息 | Knowledge broadcast: sent {count} messages")
        return count
        
    def push_from_knowledge(self, target_model: str, message: Dict, ttl: int = 60) -> bool:
        """知识库主动推送信息到指定模型 | Knowledge base actively pushes information to specified model
        参数:
            target_model: 目标模型ID | Target model ID
            message: 消息内容 | Message content
            ttl: 消息生存时间(秒) | Message time-to-live (seconds)
        返回:
            是否推送成功 | Whether push was successful
        """
        # 创建专用推送通道 | Create dedicated push channel
        channel_id = f"knowledge_push-I-{target_model}"
        if channel_id not in self.channels:
            self.create_channel(channel_id, capacity=20, priority=1)
            
        # 标记为知识库通道 | Mark as knowledge channel
        if channel_id not in self.knowledge_channels:
            self.knowledge_channels.add(channel_id)
            
        return self.send_message(channel_id, message, ttl)
    
    def route_message(self, message: Dict) -> Optional[str]:
        """智能路由消息到目标模型"""
        if "message_type" not in message:
            self.logger.error("消息缺少'message_type'字段 | Message missing 'message_type' field")
            return None
            
        message_type = message["message_type"]
        source_model = message.get("source_model", "unknown")
        task_id = message.get("task_id", "global")
        
        # 1. 检查精确匹配
        if message_type in self.message_routing:
            target_model = self.message_routing[message_type]["target"]
            return f"{task_id}-{source_model}-{target_model}"
            
        # 2. 检查模式匹配
        for pattern, route_info in self.message_routing.items():
            if route_info["pattern"] and re.match(pattern, message_type):
                target_model = route_info["target"]
                return f"{task_id}-{source_model}-{target_model}"
                
        # 3. 新增：基于内容相关性的智能路由
        if "content" in message:
            best_match = self._find_best_content_match(message)
            if best_match:
                return f"{task_id}-{source_model}-{best_match}"
                
        self.logger.warning(f"未注册的消息类型: {message_type} | Unregistered message type: {message_type}")
        return None
        
    def _find_best_content_match(self, message: Dict) -> Optional[str]:
        """基于内容相关性找到最佳匹配模型
        增强版: 使用更复杂的关键词匹配和权重系统
        """
        content = message["content"]
        best_match = None
        highest_score = 0.0
        
        # 增强的关键词匹配系统，包含权重
        keyword_model_mapping = {
            "text": {"models": ["B"], "weight": 0.8},
            "image": {"models": ["D"], "weight": 0.9},
            "audio": {"models": ["C"], "weight": 0.8},
            "video": {"models": ["E"], "weight": 0.9},
            "knowledge": {"models": ["I"], "weight": 0.7},
            "code": {"models": ["K"], "weight": 0.8},
            "spatial": {"models": ["F"], "weight": 0.7},
            "sensor": {"models": ["G"], "weight": 0.6},
            "motion": {"models": ["J"], "weight": 0.6},
            "control": {"models": ["H"], "weight": 0.7}
        }
        
        for keyword, config in keyword_model_mapping.items():
            if keyword in str(content).lower():
                # 计算匹配得分
                # 基础分 + 关键词权重 * 匹配频率
                frequency = str(content).lower().count(keyword)
                score = 0.3 + (config["weight"] * min(frequency, 3) / 3)
                
                for model in config["models"]:
                    if score > highest_score:
                        highest_score = score
                        best_match = model
        
        # 如果找到最佳匹配，记录路由决策
        if best_match:
            self.logger.info(f"智能路由决策: 消息类型 '{message.get('message_type', 'unknown')}' 通过内容匹配路由到 {best_match} (得分: {highest_score:.2f})")
        
        return best_match
    
    def receive_message(self, channel_id: str) -> Optional[Dict]:
        """从通道接收消息 | Receive message from channel
        参数:
            channel_id: 通道ID | Channel ID
        返回:
            消息内容或None | Message content or None
        """
        if channel_id not in self.channels or not self.channels[channel_id]["messages"]:
            # 尝试从持久化存储恢复 | Try to recover from persistent storage
            return self._recover_message(channel_id)
            
        # 更新最后使用时间
        self.channels[channel_id]["last_used"] = time.time()
        
        # 获取优先级最高的消息 | Get highest priority message
        messages = self.channels[channel_id]["messages"]
        if not messages:
            return None
            
        # 按优先级排序，然后按时间排序
        messages.sort(key=lambda x: (x["_meta"]["priority"], x["_meta"]["timestamp"]))
        
        # 记录接收时间用于性能统计
        start_time = time.time()
        message = messages.pop(0)
        
        # 更新性能指标
        processing_time = time.time() - start_time
        self._update_channel_performance(channel_id, processing_time)
        
        return message
        
    def _persist_message(self, channel_id: str, message: Dict):
        """持久化消息 | Persist message"""
        filename = os.path.join(self.storage_path, f"{channel_id}.log")
        try:
            with open(filename, 'a', encoding='utf-8') as f:
                f.write(json.dumps(message) + '\n')
        except Exception as e:
            self.logger.error(f"持久化消息失败: {str(e)} | Failed to persist message: {str(e)}")
            
    def _recover_message(self, channel_id: str) -> Optional[Dict]:
        """从持久化存储恢复消息 | Recover message from persistent storage"""
        filename = os.path.join(self.storage_path, f"{channel_id}.log")
        if not os.path.exists(filename):
            return None
            
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines:
                    return None
                    
                # 获取第一条消息 | Get first message
                message = json.loads(lines[0])
                
                # 移除已处理的消息 | Remove processed message
                with open(filename, 'w', encoding='utf-8') as fw:
                    fw.writelines(lines[1:])
                    
                return message
        except Exception as e:
            self.logger.error(f"恢复消息失败: {str(e)} | Failed to recover message: {str(e)}")
            return None
    
    def get_active_channels(self, pattern: str = "*") -> List[Dict]:
        """获取匹配的活动通道信息 | Get active channels matching pattern with info
        参数:
            pattern: 匹配模式 (支持*通配符) | Match pattern (supports * wildcard)
        返回:
            通道信息字典列表 | List of channel info dictionaries
        """
        now = time.time()
        return [
            {
                "id": cid,
                "message_count": len(self.channels[cid]["messages"]),
                "throughput": self.channels[cid]["throughput"],
                "priority": self.channels[cid]["priority"],
                "is_knowledge": cid in self.knowledge_channels,
                "last_used": self.channels[cid]["last_used"],
                "age": now - self.channels[cid]["created_at"],
                "avg_processing_time": self.channels[cid]["avg_processing_time"]
            }
            for cid in self.channels if self._match_pattern(cid, pattern)
        ]
        
    def get_task_channels(self, task_id: str) -> List[Dict]:
        """获取特定任务相关的通道 | Get channels related to specific task
        参数:
            task_id: 任务ID | Task ID
        返回:
            通道信息字典列表 | List of channel info dictionaries
        """
        return self.get_active_channels(f"{task_id}-*")
        
    def get_knowledge_channels(self) -> List[str]:
        """获取所有知识库通道 | Get all knowledge channels
        返回:
            知识库通道ID列表 | List of knowledge channel IDs
        """
        return list(self.knowledge_channels)
    
    def _match_pattern(self, channel_id: str, pattern: str) -> bool:
        """匹配通道ID模式 | Match channel ID pattern
        参数:
            channel_id: 通道ID | Channel ID
            pattern: 匹配模式 | Match pattern
        返回:
            是否匹配 | Whether matched
        """
        if pattern == "*":
            return True
            
        if '*' not in pattern:
            return channel_id == pattern
            
        regex = pattern.replace("*", ".*")
        return re.match(regex, channel_id) is not None
    
    def create_direct_channel(self, model1: str, model2: str):
        """创建两个模型之间的直接通信通道 | Create direct communication channel between two models
        参数:
            model1: 模型1的ID | ID of model1
            model2: 模型2的ID | ID of model2
        返回:
            通道ID | Channel ID
        """
        # 生成唯一的通道ID，按字母顺序排序以确保唯一性
        sorted_models = sorted([model1, model2])
        channel_id = f"direct-{sorted_models[0]}-{sorted_models[1]}"
        
        if channel_id in self.direct_channels:
            self.logger.info(f"直接通道已存在: {channel_id} | Direct channel already exists: {channel_id}")
            return channel_id
            
        # 创建一个新的通道，这里我们使用一个简单的队列实现
        self.direct_channels[channel_id] = {
            "queue": [],
            "lock": threading.Lock(),
            "created_at": time.time(),
            "last_used": time.time(),
            "throughput": 0
        }
        self.logger.info(f"创建直接通道: {channel_id} | Direct channel created: {channel_id}")
        return channel_id
        
    def send_direct_message(self, channel_id: str, message: Dict):
        """通过直接通道发送消息 | Send message via direct channel
        参数:
            channel_id: 直接通道ID | Direct channel ID
            message: 消息内容 | Message content
        返回:
            是否成功 | Whether successful
        """
        if channel_id not in self.direct_channels:
            self.logger.error(f"直接通道不存在: {channel_id} | Direct channel not found: {channel_id}")
            return False
            
        with self.direct_channels[channel_id]["lock"]:
            # 添加消息ID和时间戳
            message.setdefault("_meta", {})
            message["_meta"].update({
                "timestamp": time.time(),
                "message_id": f"direct_msg_{int(time.time()*1000)}_{os.urandom(4).hex()}"
            })
            self.direct_channels[channel_id]["queue"].append(message)
            self.direct_channels[channel_id]["throughput"] += 1
            self.direct_channels[channel_id]["last_used"] = time.time()
            
        return True
        
    def receive_direct_message(self, channel_id: str) -> Optional[Dict]:
        """从直接通道接收消息 | Receive message from direct channel
        参数:
            channel_id: 直接通道ID | Direct channel ID
        返回:
            消息内容或None | Message content or None
        """
        if channel_id not in self.direct_channels:
            self.logger.error(f"直接通道不存在: {channel_id} | Direct channel not found: {channel_id}")
            return None
            
        with self.direct_channels[channel_id]["lock"]:
            if not self.direct_channels[channel_id]["queue"]:
                return None
            
            message = self.direct_channels[channel_id]["queue"].pop(0)
            self.direct_channels[channel_id]["last_used"] = time.time()
            return message
    
    def enhance_collaboration(self, source_model, target_model, task_id):
        """增强模型间协作 | Enhance inter-model collaboration
        参数:
            source_model: 源模型ID | Source model ID
            target_model: 目标模型ID | Target model ID
            task_id: 任务ID | Task ID
        返回:
            协作通道ID | Collaboration channel ID
        """
        # 创建专用协作通道 | Create dedicated collaboration channel
        channel_id = f"{task_id}-{source_model}-{target_model}-collab"
        
        # 检查通道是否已存在
        if channel_id not in self.channels:
            # 为协作任务创建高优先级通道
            self.create_channel(channel_id, capacity=20, priority=1)
            self.logger.info(f"创建协作通道: {channel_id} | Collaboration channel created: {channel_id}")
        
        # 记录协作关系
        collaboration_key = f"{source_model}:{target_model}"
        reverse_key = f"{target_model}:{source_model}"
        
        # 初始化协作统计
        if "collaboration_stats" not in self.performance_metrics:
            self.performance_metrics["collaboration_stats"] = {}
        
        if collaboration_key not in self.performance_metrics["collaboration_stats"] and \
           reverse_key not in self.performance_metrics["collaboration_stats"]:
            self.performance_metrics["collaboration_stats"][collaboration_key] = {
                "tasks": 0,
                "success_count": 0,
                "failure_count": 0,
                "total_response_time": 0,
                "last_collaboration": time.time()
            }
        
        # 更新协作任务计数
        if collaboration_key in self.performance_metrics["collaboration_stats"]:
            self.performance_metrics["collaboration_stats"][collaboration_key]["tasks"] += 1
            self.performance_metrics["collaboration_stats"][collaboration_key]["last_collaboration"] = time.time()
        elif reverse_key in self.performance_metrics["collaboration_stats"]:
            self.performance_metrics["collaboration_stats"][reverse_key]["tasks"] += 1
            self.performance_metrics["collaboration_stats"][reverse_key]["last_collaboration"] = time.time()
        
        return channel_id
    
    def register_component(self, component_id: str, component: Any, metadata: Dict = None):
        """注册组件到数据总线 | Register component to data bus
        增强版: 添加组件元数据存储和状态管理
        
        参数:
            component_id: 组件ID | Component ID
            component: 组件实例 | Component instance
            metadata: 组件元数据 (可选) | Component metadata (optional)
        """
        # 存储组件实例
        self.components[component_id] = component
        
        # 初始化组件状态
        self.component_status[component_id] = {
            "status": "active",
            "last_heartbeat": time.time(),
            "error_count": 0,
            "restart_count": 0,
            "resource_usage": {}
        }
        
        # 存储组件元数据
        self.component_metadata[component_id] = metadata or {
            "type": "generic",
            "version": "1.0.0",
            "description": "",
            "dependencies": [],
            "capabilities": []
        }
        
        self.logger.info(f"注册组件: {component_id} | Registered component: {component_id}")
        self.logger.debug(f"组件元数据: {json.dumps(self.component_metadata[component_id], ensure_ascii=False)}")
    
    def publish(self, channel_id: str, message: Any):
        """发布消息到指定通道"""
        try:
            # 检查通道是否存在
            if channel_id not in self.channels:
                self.create_channel(channel_id)
                self.logger.warning(f"通道 {channel_id} 不存在，已自动创建")
            
            # 为消息添加元数据
            message_with_metadata = {
                'message': message,
                'timestamp': datetime.now().isoformat(),
                'channel': channel_id,
                'sender': f'data_bus_{id(self)}',
                'message_id': str(uuid.uuid4())
            }
            
            # 发布消息到通道
            self.channels[channel_id].append(message_with_metadata)
            
            # 记录发布日志
            self.logger.info(f"消息已发布到通道 {channel_id}: {str(message)[:100]}...")
            
            # 通知订阅者
            self.notify_subscribers(channel_id, message_with_metadata)
            
            return True
        except Exception as e:
            self.logger.error(f"发布消息失败: {e}")
            return False
    
    # 添加publish_message方法作为publish方法的别名，以兼容app.py中的调用
    def publish_message(self, channel_id: str, message: Any):
        """发布消息到指定通道（兼容app.py的方法名）"""
        return self.publish(channel_id, message)

    def subscribe(self, channel_id: str, callback: Callable):
        """订阅通道消息"""
        try:
            if channel_id not in self.subscriptions:
                self.subscriptions[channel_id] = []
            self.subscriptions[channel_id].append(callback)
            self.logger.info(f"订阅通道 {channel_id} | Subscribed to channel {channel_id}")
        except Exception as e:
            self.logger.error(f"订阅通道失败: {str(e)} | Failed to subscribe to channel {channel_id}: {str(e)}")

    def notify_subscribers(self, channel_id: str, message: Dict):
        """通知订阅者"""
        if channel_id in self.subscriptions:
            for callback in self.subscriptions[channel_id]:
                try:
                    callback(message)
                except Exception as e:
                    self.logger.error(f"回调执行失败: {str(e)} | Callback execution failed: {str(e)}")
                        
    # ===== 新增编程模型文件操作接口 =====
    # ===== Added Programming Model File Operations =====
    def programming_file_read(self, file_path: str) -> Dict:
        """编程模型文件读取接口 | Programming model file read interface
        参数:
            file_path: 文件路径 | File path
        返回:
            {'status': 'success/error', 'content': 文件内容}
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return {'status': 'success', 'content': content}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def programming_file_write(self, file_path: str, content: str) -> Dict:
        """编程模型文件写入接口 | Programming model file write interface
        参数:
            file_path: 文件路径 | File path
            content: 要写入的内容 | Content to write
        返回:
            {'status': 'success/error'}
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return {'status': 'success'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def programming_file_append(self, file_path: str, content: str) -> Dict:
        """编程模型文件追加接口 | Programming model file append interface
        参数:
            file_path: 文件路径 | File path
            content: 要追加的内容 | Content to append
        返回:
            {'status': 'success/error'}
        """
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)
            return {'status': 'success'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def programming_file_delete(self, file_path: str) -> Dict:
        """编程模型文件删除接口 | Programming model file delete interface
        参数:
            file_path: 文件路径 | File path
        返回:
            {'status': 'success/error'}
        """
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                return {'status': 'success'}
            return {'status': 'error', 'message': '文件不存在'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def programming_list_files(self, dir_path: str) -> Dict:
        """编程模型文件列表接口 | Programming model list files interface
        参数:
            dir_path: 目录路径 | Directory path
        返回:
            {'status': 'success/error', 'files': 文件列表}
        """
        try:
            files = os.listdir(dir_path)
            # 增强：添加文件类型和大小信息
            file_details = []
            for file in files:
                full_path = os.path.join(dir_path, file)
                file_info = {
                    'name': file,
                    'is_dir': os.path.isdir(full_path),
                    'size': os.path.getsize(full_path) if os.path.isfile(full_path) else 0,
                    'modified_time': os.path.getmtime(full_path)
                }
                file_details.append(file_info)
            
            return {'status': 'success', 'files': file_details}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
            
    def programming_execute_command(self, command: str, cwd: str = None, env: Dict[str, str] = None, timeout: int = 30, sandbox: bool = False) -> Dict:
        """编程模型命令执行接口 | Programming model command execution interface
        参数:
            command: 要执行的命令 | Command to execute
            cwd: 工作目录 (可选) | Working directory (optional)
            env: 环境变量 (可选) | Environment variables (optional)
            timeout: 超时时间(秒) (默认30秒) | Timeout in seconds (default 30)
            sandbox: 是否在沙箱环境中执行 (默认False) | Whether to execute in sandbox environment (default False)
        返回:
            {'status': 'success/error', 'output': 命令输出, 'analysis': 智能分析结果}
        """
        # 安全验证：只允许字母、数字、空格和特定安全字符
        if not re.match(r'^[\w\s\-_\.\/:;=@]+$', command):
            self.logger.warning("命令包含非法字符")
            return {'status': 'error', 'message': '命令包含非法字符'}
            
        # 防止过长命令导致的DoS攻击
        if len(command) > 1000:
            self.logger.warning("命令过长")
            return {'status': 'error', 'message': '命令过长'}
            
        # 验证工作目录是否存在
        if cwd and not os.path.isdir(cwd):
            self.logger.error(f"工作目录不存在: {cwd}")
            return {'status': 'error', 'message': f'工作目录不存在: {cwd}'}
            
        # 沙箱环境准备
        if sandbox:
            # 创建临时沙箱目录
            sandbox_dir = os.path.join(self.storage_path, "sandbox", str(int(time.time())))
            os.makedirs(sandbox_dir, exist_ok=True)
            cwd = sandbox_dir  # 覆盖工作目录为沙箱目录
            self.logger.info(f"在沙箱环境中执行命令: {sandbox_dir}")
            
        try:
            # 使用shlex安全拆分命令，正确处理带引号的参数
            command_list = shlex.split(command)
            
            # 安全日志：只记录命令名称
            cmd_name = command_list[0] if command_list else "unknown"
            self.logger.info(f"执行命令: {cmd_name} (参数数量: {len(command_list)-1}, 超时: {timeout}s, 沙箱: {sandbox})")
            
            # 记录开始时间
            start_time = time.time()
            
            # 设置执行环境
            exec_env = os.environ.copy()
            if env:
                exec_env.update(env)
            
            # 设置超时防止阻塞
            result = subprocess.run(
                command_list, 
                shell=False,  # 更安全的执行方式
                capture_output=True, 
                text=True,
                timeout=timeout,
                cwd=cwd,      # 指定工作目录
                env=exec_env  # 指定环境变量
            )
            
            # 计算执行时间
            exec_time = time.time() - start_time
            
            # 构建响应
            response = {
                'status': 'success',
                'output': result.stdout,
                'error': result.stderr,
                'returncode': result.returncode,
                'execution_time': round(exec_time, 2),
                'analysis': self._analyze_command_result(command, result)  # 添加智能分析
            }
            
            # 记录执行结果
            if result.returncode != 0:
                self.logger.warning(f"命令执行失败 (代码: {result.returncode}, 时间: {exec_time:.2f}s)")
            else:
                self.logger.info(f"命令执行成功 (时间: {exec_time:.2f}s)")
                
            # 记录命令历史
            self._log_command_history(command, response)
                
            return response
        except subprocess.TimeoutExpired as e:
            error_response = {
                'status': 'error', 
                'message': f'命令执行超时（{timeout}秒），请简化命令或增加超时时间',
                'timeout': timeout,
                'analysis': self._analyze_timeout_error(command, timeout)
            }
            self.logger.error(f"命令执行超时: {e.cmd} (超时: {e.timeout}s)")
            return error_response
        except FileNotFoundError as e:
            error_response = {
                'status': 'error', 
                'message': f'命令未找到: {e.filename}',
                'analysis': self._analyze_file_not_found(command, e.filename)
            }
            self.logger.error(f"命令未找到: {e.filename}")
            return error_response
        except Exception as e:
            error_msg = f"命令执行失败: {type(e).__name__}"
            error_response = {
                'status': 'error', 
                'message': error_msg,
                'analysis': self._analyze_general_error(command, e)
            }
            self.logger.error(error_msg, exc_info=True)
            return error_response
        finally:
            # 清理沙箱环境
            if sandbox:
                try:
                    import shutil
                    shutil.rmtree(cwd)
                    self.logger.info(f"清理沙箱环境: {cwd}")
                except Exception as e:
                    self.logger.error(f"沙箱清理失败: {str(e)}")
                    
    def _analyze_command_result(self, command: str, result: subprocess.CompletedProcess) -> Dict:
        """智能分析命令执行结果"""
        analysis = {
            'success': result.returncode == 0,
            'suggestions': [],
            'potential_issues': []
        }
        
        # 分析错误输出
        if result.stderr:
            # 常见错误模式识别
            if "permission denied" in result.stderr.lower():
                analysis['suggestions'].append("尝试使用管理员权限执行命令")
                analysis['potential_issues'].append("权限不足")
            if "no such file or directory" in result.stderr.lower():
                analysis['suggestions'].append("检查文件路径是否正确")
                analysis['potential_issues'].append("文件或目录不存在")
            if "command not found" in result.stderr.lower():
                analysis['suggestions'].append("检查命令是否安装或路径是否配置正确")
                analysis['potential_issues'].append("命令未找到")
                
        # 分析输出内容
        if result.stdout:
            # 检测潜在问题
            if "warning" in result.stdout.lower():
                analysis['suggestions'].append("输出包含警告信息，建议检查")
                analysis['potential_issues'].append("存在警告")
            if "error" in result.stdout.lower():
                analysis['suggestions'].append("输出包含错误信息，需要修复")
                analysis['potential_issues'].append("存在错误")
                
        # 分析执行时间
        if hasattr(result, 'execution_time') and result.execution_time > 10:
            analysis['suggestions'].append("命令执行时间较长，考虑优化命令或拆分任务")
            
        return analysis
        
    def _analyze_timeout_error(self, command: str, timeout: int) -> Dict:
        """分析超时错误"""
        return {
            'error_type': 'timeout',
            'suggestions': [
                f"增加超时时间（当前: {timeout}秒）",
                "优化命令以减少执行时间",
                "将复杂命令拆分为多个简单命令"
            ]
        }
        
    def _analyze_file_not_found(self, command: str, filename: str) -> Dict:
        """分析文件未找到错误"""
        return {
            'error_type': 'file_not_found',
            'suggestions': [
                f"检查文件路径: {filename}",
                "确认文件是否存在",
                "检查文件权限"
            ]
        }
        
    def _analyze_general_error(self, command: str, exception: Exception) -> Dict:
        """分析通用错误"""
        return {
            'error_type': type(exception).__name__,
            'suggestions': [
                "检查命令语法",
                "查看详细错误日志",
                "尝试在更简单的环境中测试命令"
            ]
        }
        
    def _log_command_history(self, command: str, response: Dict):
        """记录命令执行历史"""
        history_file = os.path.join(self.storage_path, "command_history.log")
        try:
            entry = {
                'timestamp': time.time(),
                'command': command,
                'status': response['status'],
                'execution_time': response.get('execution_time', 0),
                'returncode': response.get('returncode', -1)
            }
            with open(history_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')
        except Exception as e:
            self.logger.error(f"记录命令历史失败: {str(e)}")
    
        # ===== 新增性能监控和组件管理接口 =====
        
        def _start_performance_monitoring(self):
            """启动性能监控任务"""
            self.performance_metrics = {
                "total_messages": 0,
                "success_rate": 0.0,
                "avg_response_time": 0.0,
                "error_count": 0,
                "collaboration_stats": {},
                "channel_usage": {}
            }
            
            # 启动定期清理线程
            def cleanup_task():
                while True:
                    try:
                        self._cleanup_old_data()
                        time.sleep(3600)  # 每小时清理一次
                    except Exception as e:
                        self.logger.error(f"清理任务失败: {str(e)}")
            
            cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
            cleanup_thread.start()
        
        def _update_channel_performance(self, channel_id: str, processing_time: float):
            """更新通道性能指标"""
            if channel_id not in self.channels:
                return
            
            channel = self.channels[channel_id]
            channel["message_processing_time"].append(processing_time)
            
            # 限制历史数据点数量，防止内存占用过大
            if len(channel["message_processing_time"]) > 100:
                channel["message_processing_time"] = channel["message_processing_time"][-100:]
            
            # 计算平均处理时间
            channel["avg_processing_time"] = sum(channel["message_processing_time"]) / len(channel["message_processing_time"])
            
            # 更新全局性能指标
            self.performance_metrics["total_messages"] += 1
            
        def _cleanup_old_data(self):
            """清理过期数据和不活跃通道"""
            now = time.time()
            self.last_cleanup_time = now
            
            # 清理30天以上的命令历史
            try:
                history_file = os.path.join(self.storage_path, "command_history.log")
                if os.path.exists(history_file):
                    cutoff_time = now - (30 * 24 * 3600)  # 30天前
                    
                    with open(history_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                    
                    # 保留30天内的记录
                    new_lines = []
                    for line in lines:
                        try:
                            entry = json.loads(line)
                            if entry.get('timestamp', 0) >= cutoff_time:
                                new_lines.append(line)
                        except:
                            pass
                    
                    if len(new_lines) < len(lines):
                        with open(history_file, 'w', encoding='utf-8') as f:
                            f.writelines(new_lines)
                        self.logger.info(f"清理命令历史: 保留 {len(new_lines)} 条记录，移除 {len(lines)-len(new_lines)} 条旧记录")
            except Exception as e:
                self.logger.error(f"清理命令历史失败: {str(e)}")
            
            # 清理6小时未使用的不活跃通道
            inactive_threshold = now - (6 * 3600)  # 6小时前
            inactive_channels = []
            
            for channel_id, channel in self.channels.items():
                # 跳过知识库通道和高优先级通道
                if channel_id in self.knowledge_channels or channel["priority"] <= 2:
                    continue
                
                # 检查是否不活跃且没有消息
                if channel["last_used"] < inactive_threshold and len(channel["messages"]) == 0:
                    inactive_channels.append(channel_id)
            
            for channel_id in inactive_channels:
                del self.channels[channel_id]
                if channel_id in self.subscriptions:
                    del self.subscriptions[channel_id]
                self.logger.info(f"清理不活跃通道: {channel_id}")
        
        def update_component_heartbeat(self, component_id: str):
            """更新组件心跳
            参数:
                component_id: 组件ID
            """
            if component_id in self.component_status:
                self.component_status[component_id]["last_heartbeat"] = time.time()
        
        def update_component_status(self, component_id: str, status: str, resource_usage: Dict = None):
            """更新组件状态
            参数:
                component_id: 组件ID
                status: 组件状态 (active, error, warning, offline)
                resource_usage: 资源使用情况 (可选)
            """
            if component_id in self.component_status:
                self.component_status[component_id]["status"] = status
                if resource_usage:
                    self.component_status[component_id]["resource_usage"] = resource_usage
                
                # 如果状态是error，增加错误计数
                if status == "error":
                    self.component_status[component_id]["error_count"] += 1
        
        def get_component_status(self, component_id: str = None) -> Dict:
            """获取组件状态信息
            参数:
                component_id: 组件ID，如果为None则返回所有组件状态
            返回:
                组件状态信息字典
            """
            if component_id:
                if component_id in self.component_status:
                    return {
                        component_id: {
                            **self.component_status[component_id],
                            "metadata": self.component_metadata.get(component_id, {})
                        }
                    }
                return {}
            
            # 返回所有组件状态
            result = {}
            for cid in self.component_status:
                result[cid] = {
                    **self.component_status[cid],
                    "metadata": self.component_metadata.get(cid, {})
                }
            
            return result
        
        def get_performance_metrics(self) -> Dict:
            """获取系统性能指标
            返回:
                性能指标字典
            """
            return {
                **self.performance_metrics,
                "channel_count": len(self.channels),
                "direct_channel_count": len(self.direct_channels),
                "registered_components": len(self.components),
                "last_cleanup": self.last_cleanup_time
            }
        
        def reset_channel(self, channel_id: str):
            """重置指定通道
            参数:
                channel_id: 通道ID
            """
            if channel_id in self.channels:
                # 保留通道配置，但清空消息
                original_config = self.channels[channel_id].copy()
                self.channels[channel_id] = {
                    "messages": [],
                    "capacity": original_config["capacity"],
                    "throughput": 0,
                    "persistent": original_config["persistent"],
                    "priority": original_config["priority"],
                    "created_at": original_config["created_at"],
                    "last_used": time.time(),
                    "message_processing_time": [],
                    "avg_processing_time": 0.0
                }
                self.logger.info(f"重置通道: {channel_id}")
                
                # 删除持久化消息文件
                try:
                    persist_file = os.path.join(self.storage_path, f"{channel_id}.log")
                    if os.path.exists(persist_file):
                        os.remove(persist_file)
                except Exception as e:
                    self.logger.error(f"删除持久化文件失败: {str(e)}")
        
        def remove_component(self, component_id: str):
            """移除已注册的组件
            参数:
                component_id: 组件ID
            """
            if component_id in self.components:
                del self.components[component_id]
                if component_id in self.component_status:
                    del self.component_status[component_id]
                if component_id in self.component_metadata:
                    del self.component_metadata[component_id]
                self.logger.info(f"移除组件: {component_id}")
        
        def get_component_dependencies(self, component_id: str) -> List[str]:
            """获取组件依赖信息
            参数:
                component_id: 组件ID
            返回:
                依赖组件ID列表
            """
            if component_id in self.component_metadata:
                return self.component_metadata[component_id].get("dependencies", [])
            return []
        
        def get_component_capabilities(self, component_id: str) -> List[str]:
            """获取组件功能列表
            参数:
                component_id: 组件ID
            返回:
                组件功能列表
            """
            if component_id in self.component_metadata:
                return self.component_metadata[component_id].get("capabilities", [])
            return []


def get_data_bus(storage_path: str = "data_bus_storage") -> DataBus:
    """获取DataBus的单例实例
    参数:
        storage_path: 存储路径，仅在首次创建实例时有效
    返回:
        DataBus实例
    """
    global _data_bus_instance
    
    # 使用双重检查锁定模式确保线程安全
    if _data_bus_instance is None:
        with _data_bus_lock:
            if _data_bus_instance is None:
                _data_bus_instance = DataBus(storage_path)
                logging.getLogger("DataBus").info("DataBus singleton instance created")
    
    return _data_bus_instance
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
