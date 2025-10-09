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