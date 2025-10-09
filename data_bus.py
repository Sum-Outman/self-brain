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
import threading
import time
from collections import defaultdict, deque
from typing import Dict, Any, List, Optional, Callable, Union
from datetime import datetime
import json
import os

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataBus")

class DataBus:
    """数据总线，用于在不同模块之间传递数据和消息"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式实现"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DataBus, cls).__new__(cls)
                cls._instance._initialize()
                logger.info("DataBus singleton instance created")
            return cls._instance
    
    def _initialize(self):
        """初始化数据总线"""
        # 数据存储
        self.data_store = {}
        self.data_store_lock = threading.RLock()
        
        # 消息队列
        self.message_queues = defaultdict(deque)
        self.message_queues_lock = threading.RLock()
        
        # 订阅者管理
        self.subscribers = defaultdict(list)
        self.subscribers_lock = threading.RLock()
        
        # 事件历史
        self.event_history = deque(maxlen=1000)  # 保留最近1000个事件
        self.history_lock = threading.RLock()
        
        # 数据有效期管理
        self.data_ttl = defaultdict(int)
        self.data_expiry = defaultdict(float)
        self.data_expiry_lock = threading.RLock()
        
        # 启动清理线程
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_task, daemon=True)
        self.cleanup_thread.start()
        
        logger.info("DataBus initialized with thread-safe data structures")
    
    def _cleanup_task(self):
        """定期清理过期数据的后台任务"""
        while self.running:
            try:
                current_time = time.time()
                with self.data_store_lock, self.data_expiry_lock:
                    # 查找并删除过期数据
                    expired_keys = [key for key, expiry in self.data_expiry.items() if expiry > 0 and current_time > expiry]
                    for key in expired_keys:
                        if key in self.data_store:
                            del self.data_store[key]
                        if key in self.data_expiry:
                            del self.data_expiry[key]
                        logger.debug(f"Expired data removed: {key}")
                
                # 休眠一段时间后再次检查
                time.sleep(60)  # 每分钟检查一次
            except Exception as e:
                logger.error(f"Error in cleanup task: {str(e)}")
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置数据
        
        Args:
            key: 数据键名
            value: 数据值
            ttl: 数据有效期（秒），None表示永不过期
        """
        try:
            with self.data_store_lock:
                self.data_store[key] = value
                
                # 设置过期时间
                if ttl is not None and ttl > 0:
                    self.data_expiry[key] = time.time() + ttl
                elif key in self.data_expiry:
                    del self.data_expiry[key]  # 移除已存在的过期时间
            
            # 记录事件
            self._record_event('data_set', {'key': key, 'has_value': value is not None})
            
            # 通知订阅者
            self._notify_subscribers(key, value)
            
            logger.debug(f"Data set: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to set data '{key}': {str(e)}")
            return False
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取数据
        
        Args:
            key: 数据键名
            default: 默认值
        
        Returns:
            数据值或默认值
        """
        try:
            with self.data_store_lock:
                # 检查是否过期
                if key in self.data_expiry and time.time() > self.data_expiry[key]:
                    # 立即删除过期数据
                    del self.data_store[key]
                    del self.data_expiry[key]
                    logger.debug(f"Expired data accessed and removed: {key}")
                    return default
                
                # 返回数据或默认值
                value = self.data_store.get(key, default)
                
                # 记录事件
                if key in self.data_store:
                    self._record_event('data_get', {'key': key})
                
                return value
        except Exception as e:
            logger.error(f"Failed to get data '{key}': {str(e)}")
            return default
    
    def delete(self, key: str) -> bool:
        """删除数据
        
        Args:
            key: 数据键名
        
        Returns:
            是否删除成功
        """
        try:
            with self.data_store_lock:
                if key in self.data_store:
                    del self.data_store[key]
                    
                    # 同时删除过期时间
                    if key in self.data_expiry:
                        del self.data_expiry[key]
                    
                    # 记录事件
                    self._record_event('data_delete', {'key': key})
                    
                    logger.debug(f"Data deleted: {key}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to delete data '{key}': {str(e)}")
            return False
    
    def publish(self, channel: str, message: Any):
        """发布消息到指定频道
        
        Args:
            channel: 频道名称
            message: 消息内容
        """
        try:
            # 记录事件
            self._record_event('message_publish', {'channel': channel})
            
            # 将消息添加到队列
            with self.message_queues_lock:
                self.message_queues[channel].append({
                    'message': message,
                    'timestamp': time.time()
                })
            
            # 通知订阅者
            self._notify_subscribers(channel, message)
            
            logger.debug(f"Message published to channel: {channel}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish message to channel '{channel}': {str(e)}")
            return False
    
    def subscribe(self, channel: str, callback: Callable):
        """订阅指定频道的消息
        
        Args:
            channel: 频道名称
            callback: 回调函数，接收消息内容作为参数
        """
        try:
            with self.subscribers_lock:
                # 检查回调是否已经订阅
                if callback not in self.subscribers[channel]:
                    self.subscribers[channel].append(callback)
                    logger.debug(f"Subscriber added to channel: {channel}")
                else:
                    logger.warning(f"Callback already subscribed to channel: {channel}")
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to channel '{channel}': {str(e)}")
            return False
    
    def unsubscribe(self, channel: str, callback: Callable):
        """取消订阅指定频道的消息
        
        Args:
            channel: 频道名称
            callback: 回调函数
        """
        try:
            with self.subscribers_lock:
                if channel in self.subscribers and callback in self.subscribers[channel]:
                    self.subscribers[channel].remove(callback)
                    # 如果没有订阅者了，清理频道
                    if not self.subscribers[channel]:
                        del self.subscribers[channel]
                    logger.debug(f"Subscriber removed from channel: {channel}")
            return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe from channel '{channel}': {str(e)}")
            return False
    
    def get_next_message(self, channel: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        """获取指定频道的下一条消息
        
        Args:
            channel: 频道名称
            timeout: 超时时间（秒），None表示无限等待
        
        Returns:
            消息字典或None
        """
        start_time = time.time()
        
        while True:
            with self.message_queues_lock:
                if channel in self.message_queues and self.message_queues[channel]:
                    # 获取并移除队列中的第一条消息
                    message_info = self.message_queues[channel].popleft()
                    
                    # 记录事件
                    self._record_event('message_receive', {'channel': channel})
                    
                    logger.debug(f"Message received from channel: {channel}")
                    return message_info
            
            # 检查是否超时
            if timeout is not None and time.time() - start_time > timeout:
                logger.debug(f"Timeout waiting for message from channel: {channel}")
                return None
            
            # 短暂休眠以避免CPU占用过高
            time.sleep(0.01)
    
    def _notify_subscribers(self, channel: str, data: Any):
        """通知指定频道的所有订阅者
        
        Args:
            channel: 频道名称
            data: 数据内容
        """
        try:
            # 获取订阅者列表的副本，避免在遍历过程中修改
            with self.subscribers_lock:
                if channel not in self.subscribers:
                    return
                subscribers_copy = self.subscribers[channel].copy()
            
            # 通知每个订阅者
            for callback in subscribers_copy:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in subscriber callback for channel '{channel}': {str(e)}")
                    # 可以选择移除出错的订阅者
                    # self.unsubscribe(channel, callback)
        except Exception as e:
            logger.error(f"Failed to notify subscribers for channel '{channel}': {str(e)}")
    
    def _record_event(self, event_type: str, event_data: Dict[str, Any]):
        """记录事件到历史
        
        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        try:
            with self.history_lock:
                self.event_history.append({
                    'type': event_type,
                    'data': event_data,
                    'timestamp': datetime.now().isoformat()
                })
        except Exception as e:
            # 记录历史失败不应该影响主流程
            logger.debug(f"Failed to record event: {str(e)}")
    
    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取事件历史
        
        Args:
            limit: 返回的事件数量限制
        
        Returns:
            事件历史列表
        """
        try:
            with self.history_lock:
                return list(self.event_history)[-limit:]
        except Exception as e:
            logger.error(f"Failed to get event history: {str(e)}")
            return []
    
    def clear_history(self):
        """清空事件历史"""
        try:
            with self.history_lock:
                self.event_history.clear()
            logger.info("Event history cleared")
        except Exception as e:
            logger.error(f"Failed to clear event history: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取数据总线状态
        
        Returns:
            状态信息字典
        """
        try:
            with self.data_store_lock, self.message_queues_lock, self.subscribers_lock:
                return {
                    'data_count': len(self.data_store),
                    'channels': {
                        channel: {
                            'message_count': len(queue),
                            'subscriber_count': len(self.subscribers.get(channel, []))
                        }
                        for channel, queue in self.message_queues.items()
                    },
                    'total_subscribers': sum(len(subscribers) for subscribers in self.subscribers.values()),
                    'history_length': len(self.event_history),
                    'running': self.running
                }
        except Exception as e:
            logger.error(f"Failed to get DataBus status: {str(e)}")
            return {
                'error': str(e),
                'running': self.running
            }
    
    def clear(self, channel: Optional[str] = None):
        """清空数据或指定频道的消息
        
        Args:
            channel: 频道名称，如果为None则清空所有数据和消息
        """
        try:
            if channel is None:
                # 清空所有数据
                with self.data_store_lock:
                    self.data_store.clear()
                    self.data_expiry.clear()
                
                # 清空所有消息队列
                with self.message_queues_lock:
                    self.message_queues.clear()
                
                logger.info("DataBus cleared completely")
            else:
                # 仅清空指定频道的消息
                with self.message_queues_lock:
                    if channel in self.message_queues:
                        self.message_queues[channel].clear()
                
                logger.info(f"Channel '{channel}' cleared")
            
            # 记录事件
            self._record_event('clear', {'channel': channel})
            
            return True
        except Exception as e:
            logger.error(f"Failed to clear DataBus: {str(e)}")
            return False
    
    def shutdown(self):
        """关闭数据总线"""
        try:
            self.running = False
            
            # 等待清理线程结束
            if hasattr(self, 'cleanup_thread') and self.cleanup_thread.is_alive():
                self.cleanup_thread.join(timeout=5.0)
            
            # 清空所有数据
            self.clear()
            
            logger.info("DataBus shut down gracefully")
        except Exception as e:
            logger.error(f"Failed to shut down DataBus properly: {str(e)}")
    
    # 上下文管理器支持
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

# 全局函数，用于获取数据总线实例
def get_data_bus() -> DataBus:
    """获取数据总线单例实例
    
    Returns:
        DataBus实例
    """
    return DataBus()

# 测试代码
if __name__ == '__main__':
    # 创建数据总线实例
    data_bus = get_data_bus()
    
    # 测试数据设置和获取
    data_bus.set('test_key', 'test_value')
    value = data_bus.get('test_key')
    print(f"Get 'test_key': {value}")
    
    # 测试TTL功能
    data_bus.set('temp_key', 'will_expire', ttl=2)  # 2秒后过期
    print(f"Before expiry: {data_bus.get('temp_key')}")
    time.sleep(3)
    print(f"After expiry: {data_bus.get('temp_key', 'expired')}")
    
    # 测试发布订阅功能
    received_messages = []
    
    def subscriber_callback(message):
        received_messages.append(message)
        print(f"Received message: {message}")
    
    # 订阅频道
    data_bus.subscribe('test_channel', subscriber_callback)
    
    # 发布消息
    data_bus.publish('test_channel', 'Hello, DataBus!')
    
    # 直接从队列获取消息
    message = data_bus.get_next_message('test_channel', timeout=1)
    if message:
        print(f"Directly received from queue: {message['message']}")
    
    # 获取状态
    status = data_bus.get_status()
    print(f"DataBus status: {json.dumps(status, indent=2)}")
    
    # 清理
    data_bus.unsubscribe('test_channel', subscriber_callback)
    data_bus.clear()
    
    print("DataBus test completed")
