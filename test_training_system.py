#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Self Brain AGI System - Training Management System Test Script"""

import time
import random
import json
import os
from datetime import datetime

class TrainingManager:
    """Simple Training Manager Simulation"""
    
    def __init__(self):
        self.active_tasks = []
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.queued_tasks = 0
        self.task_id_counter = 1
        self.models = [
            "B_language", "C_audio", "D_image", 
            "E_video", "F_spatial", "G_sensor",
            "H_computer_control", "I_knowledge", 
            "J_motion", "K_programming"
        ]
        
    def get_system_status(self):
        """Get current system status"""
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "queued_tasks": self.queued_tasks,
            "resource_limits": {
                "cpu_usage": random.randint(50, 90),
                "memory_usage": random.randint(50, 85)
            }
        }
    
    def get_resource_usage(self):
        """Simulate resource usage"""
        return {
            "cpu_usage": random.randint(30, 80),
            "memory_usage": random.randint(40, 75)
        }
    
    def create_task(self, model_ids, training_type="single", epochs=10, 
                   batch_size=32, learning_rate=0.001, priority=5):
        """Create a new training task"""
        task_id = f"TASK-{self.task_id_counter}"
        self.task_id_counter += 1
        
        task = {
            "id": task_id,
            "model_ids": model_ids,
            "status": "preparing",
            "progress": 0,
            "current_epoch": 0,
            "total_epochs": epochs,
            "training_type": training_type,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "priority": priority,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.active_tasks.append(task)
        print(f"Created new task: {task_id} for models: {', '.join(model_ids)}")
        return task_id
    
    def update_tasks(self):
        """Update the status of all active tasks"""
        completed_tasks_this_round = []
        
        for task in self.active_tasks:
            if task["status"] == "preparing":
                # Transition from preparing to training
                if random.random() < 0.3:
                    task["status"] = "training"
            elif task["status"] == "training":
                # Update progress
                progress_increment = random.randint(1, 5)
                task["progress"] = min(100, task["progress"] + progress_increment)
                
                # Update epoch
                epoch_progress = task["progress"] / 100.0
                task["current_epoch"] = int(epoch_progress * task["total_epochs"])
                
                # Check for completion or failure
                if task["progress"] >= 100:
                    task["status"] = "completed"
                    completed_tasks_this_round.append(task)
                    self.completed_tasks += 1
                elif random.random() < 0.05:
                    task["status"] = "error"
                    completed_tasks_this_round.append(task)
                    self.failed_tasks += 1
        
        # Remove completed tasks from active list
        self.active_tasks = [t for t in self.active_tasks if t not in completed_tasks_this_round]
        
    def get_active_tasks(self):
        """Get all active tasks"""
        return self.active_tasks
    
    def stop_task(self, task_id):
        """Stop a specific task"""
        for task in self.active_tasks:
            if task["id"] == task_id:
                task["status"] = "stopped"
                self.active_tasks.remove(task)
                print(f"Stopped task: {task_id}")
                return True
        print(f"Task not found: {task_id}")
        return False
    
    def get_available_models(self):
        """Get list of available models"""
        return self.models
    
    def get_task_history(self, limit=10):
        """Get recent task history (simulated)"""
        history = []
        for i in range(1, min(limit + 1, self.task_id_counter)):
            task_id = f"TASK-{i}"
            status = random.choice(["completed", "failed", "stopped"])
            
            # Randomly select 1-3 models for this history item
            num_models = random.randint(1, 3)
            models = random.sample(self.models, num_models)
            
            # Generate a timestamp within the last 24 hours
            timestamp = (datetime.now() - 
                        timedelta(seconds=random.randint(30, 86400))).strftime("%Y-%m-%d %H:%M:%S")
            
            history.append({
                "task_id": task_id,
                "status": status,
                "models": models,
                "timestamp": timestamp
            })
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        return history

# Simple test simulation
if __name__ == "__main__":
    from datetime import timedelta
    
    print("=== Self Brain AGI Training Management System Test ===")
    
    # Initialize manager
    manager = TrainingManager()
    
    # Create some initial tasks
    print("\nCreating initial tasks...")
    manager.create_task(["B_language"], epochs=20)
    manager.create_task(["D_image", "E_video"], training_type="joint", epochs=15)
    manager.create_task(["C_audio"], epochs=10)
    
    # Simulate training process
    print("\nSimulating training process...")
    for _ in range(10):
        # Update task statuses
        manager.update_tasks()
        
        # Display current status
        status = manager.get_system_status()
        resources = manager.get_resource_usage()
        
        print(f"\n--- Status Update ---")
        print(f"Active Tasks: {status['active_tasks']}")
        print(f"Completed Tasks: {status['completed_tasks']}")
        print(f"Failed Tasks: {status['failed_tasks']}")
        print(f"CPU Usage: {resources['cpu_usage']}%")
        print(f"Memory Usage: {resources['memory_usage']}%")
        
        # Print active task details
        if manager.active_tasks:
            print(f"\nActive Task Details:")
            for task in manager.active_tasks:
                print(f"  {task['id']}: {task['progress']}% - {task['status']}")
                print(f"    Models: {', '.join(task['model_ids'])}")
                print(f"    Epoch: {task['current_epoch']}/{task['total_epochs']}")
        
        # Sleep to simulate time passing
        time.sleep(1)
    
    print("\n=== Simulation Complete ===")
    print(f"Final Status:")
    final_status = manager.get_system_status()
    print(f"  Total Tasks Created: {manager.task_id_counter - 1}")
    print(f"  Completed Tasks: {final_status['completed_tasks']}")
    print(f"  Failed Tasks: {final_status['failed_tasks']}")
    print(f"  Remaining Active Tasks: {final_status['active_tasks']}")