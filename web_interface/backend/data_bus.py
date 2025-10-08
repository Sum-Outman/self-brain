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

"""
DataBus implementation for model communication
Provides publish-subscribe pattern for inter-model communication
"""

import logging
import threading
import time
from typing import Dict, List, Callable, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DataBus")

class DataBus:
    """Central communication hub for model interactions"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern implementation"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DataBus, cls).__new__(cls)
                cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize DataBus with empty subscribers and topics"""
        # Dictionary to hold subscribers for each topic
        self._subscribers: Dict[str, List[Callable]] = {}
        # Lock for thread safety
        self._subscribers_lock = threading.RLock()
        # Store last message for each topic
        self._last_messages: Dict[str, Dict] = {}
        
        logger.info("DataBus initialized successfully")
    
    def publish(self, topic: str, data: Dict) -> bool:
        """Publish data to a specific topic
        
        Args:
            topic: The topic to publish to
            data: The data to publish
            
        Returns:
            bool: True if data was published successfully
        """
        try:
            # Add timestamp to data
            data_with_timestamp = {
                **data,
                'timestamp': time.time()
            }
            
            # Store last message
            with self._subscribers_lock:
                self._last_messages[topic] = data_with_timestamp
                
                # Get subscribers for this topic
                topic_subscribers = self._subscribers.get(topic, [])
            
            # Notify all subscribers in separate threads
            for callback in topic_subscribers:
                try:
                    # Run callback in a separate thread to avoid blocking
                    threading.Thread(
                        target=self._safe_callback,
                        args=(callback, data_with_timestamp),
                        daemon=True
                    ).start()
                except Exception as e:
                    logger.error(f"Error creating thread for subscriber callback: {str(e)}")
            
            # Log successful publication
            logger.debug(f"Published data to topic '{topic}'")
            return True
        except Exception as e:
            logger.error(f"Failed to publish to topic '{topic}': {str(e)}")
            return False
    
    def subscribe(self, topic: str, callback: Callable) -> bool:
        """Subscribe to a specific topic
        
        Args:
            topic: The topic to subscribe to
            callback: The function to call when data is published
            
        Returns:
            bool: True if subscription was successful
        """
        try:
            with self._subscribers_lock:
                # Initialize topic if it doesn't exist
                if topic not in self._subscribers:
                    self._subscribers[topic] = []
                
                # Add callback if not already present
                if callback not in self._subscribers[topic]:
                    self._subscribers[topic].append(callback)
                    logger.info(f"Subscribed to topic '{topic}'")
                else:
                    logger.warning(f"Already subscribed to topic '{topic}'")
            
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to topic '{topic}': {str(e)}")
            return False
    
    def unsubscribe(self, topic: str, callback: Callable) -> bool:
        """Unsubscribe from a specific topic
        
        Args:
            topic: The topic to unsubscribe from
            callback: The function to remove from subscribers
            
        Returns:
            bool: True if unsubscription was successful
        """
        try:
            with self._subscribers_lock:
                if topic in self._subscribers and callback in self._subscribers[topic]:
                    self._subscribers[topic].remove(callback)
                    logger.info(f"Unsubscribed from topic '{topic}'")
                    
                    # Clean up empty topics
                    if not self._subscribers[topic]:
                        del self._subscribers[topic]
                        logger.debug(f"Removed empty topic '{topic}'")
                else:
                    logger.warning(f"Not subscribed to topic '{topic}'")
            
            return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe from topic '{topic}': {str(e)}")
            return False
    
    def get_last_message(self, topic: str) -> Optional[Dict]:
        """Get the last message published to a topic
        
        Args:
            topic: The topic to get the last message from
            
        Returns:
            Optional[Dict]: The last message, or None if no message exists
        """
        try:
            with self._subscribers_lock:
                return self._last_messages.get(topic)
        except Exception as e:
            logger.error(f"Failed to get last message for topic '{topic}': {str(e)}")
            return None
    
    def get_topics(self) -> List[str]:
        """Get all available topics
        
        Returns:
            List[str]: List of available topics
        """
        try:
            with self._subscribers_lock:
                return list(self._subscribers.keys())
        except Exception as e:
            logger.error(f"Failed to get topics: {str(e)}")
            return []
    
    def _safe_callback(self, callback: Callable, data: Dict):
        """Safely execute a callback function with error handling
        
        Args:
            callback: The callback function to execute
            data: The data to pass to the callback
        """
        try:
            callback(data)
        except Exception as e:
            logger.error(f"Error in subscriber callback: {str(e)}")

# Global function to get DataBus instance
def get_data_bus() -> DataBus:
    """Get the singleton instance of DataBus
    
    Returns:
        DataBus: The singleton instance
    """
    return DataBus()