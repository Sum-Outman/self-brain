#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
A Management Model - Chat Interface Test Script
This script tests the interaction with the chat interface and API endpoints.
"""

import requests
import json
import time
import os

class ChatInterfaceTester:
    """Class to test the chat interface functionality"""
    
    def __init__(self, base_url="http://localhost:5001"):
        """Initialize the tester with base URL"""
        self.base_url = base_url
        self.chat_history = []
        self.current_conversation_id = None
        print(f"Initialized Chat Interface Tester for {self.base_url}")
    
    def health_check(self):
        """Check if the API is running"""
        try:
            response = requests.get(f"{self.base_url}/api/health")
            if response.status_code == 200:
                print("✅ API Health Check: OK")
                return True
            else:
                print(f"❌ API Health Check: Failed with status code {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API Health Check: Connection Error - {str(e)}")
            return False
    
    def get_system_stats(self):
        """Get system statistics"""
        try:
            response = requests.get(f"{self.base_url}/api/stats")
            if response.status_code == 200:
                stats = response.json()
                print("✅ System Stats Retrieved")
                print(f"   - Active Models: {stats.get('active_models', 0)}")
                print(f"   - Total Requests: {stats.get('total_requests', 0)}")
                print(f"   - Response Time: {stats.get('avg_response_time', 0)}ms")
                return stats
            else:
                print(f"❌ Failed to get system stats: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error retrieving system stats: {str(e)}")
            return None
    
    def send_message(self, message, files=None):
        """Send a message to the chat API"""
        try:
            url = f"{self.base_url}/api/chat"
            data = {
                "message": message,
                "conversation_id": self.current_conversation_id
            }
            
            # Prepare files for upload if any
            files_data = None
            if files:
                files_data = []
                for file_path in files:
                    if os.path.exists(file_path):
                        files_data.append(('files', (os.path.basename(file_path), open(file_path, 'rb'))))
                    else:
                        print(f"❌ File not found: {file_path}")
                        continue
            
            # Send the request
            print(f"\n🔄 Sending message: {message}")
            start_time = time.time()
            if files_data:
                response = requests.post(url, data=data, files=files_data)
            else:
                response = requests.post(url, json=data)
            
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                result = response.json()
                self.current_conversation_id = result.get('conversation_id', self.current_conversation_id)
                
                print(f"✅ Message Sent Successfully (Response Time: {response_time:.2f}ms)")
                print(f"   - Conversation ID: {self.current_conversation_id}")
                print(f"   - Response: {result.get('response', 'No response text')}")
                
                # Add to chat history
                self.chat_history.append({
                    "role": "user",
                    "content": message
                })
                self.chat_history.append({
                    "role": "assistant",
                    "content": result.get('response', '')
                })
                
                return result
            else:
                print(f"❌ Failed to send message: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"❌ Error sending message: {str(e)}")
            return None
    
    def test_file_upload(self, file_path):
        """Test file upload functionality"""
        if not os.path.exists(file_path):
            print(f"❌ Test file not found: {file_path}")
            return False
        
        return self.send_message("Please analyze this file.", [file_path]) is not None
    
    def run_basic_test(self):
        """Run a basic test sequence"""
        print("\n=== Running Basic Test Sequence ===")
        
        # Check health
        if not self.health_check():
            print("❌ Test aborted due to API health check failure")
            return False
        
        # Get system stats
        self.get_system_stats()
        
        # Test basic conversation
        self.send_message("Hello, I'm testing the A Management Model chat interface.")
        time.sleep(1)  # Wait for response
        
        self.send_message("Can you tell me about the system architecture?")
        time.sleep(1)  # Wait for response
        
        self.send_message("What models are currently active?")
        time.sleep(1)  # Wait for response
        
        print("\n=== Basic Test Sequence Completed ===")
        return True
    
    def save_chat_history(self, filename="chat_history.json"):
        """Save chat history to a JSON file"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    "conversation_id": self.current_conversation_id,
                    "history": self.chat_history,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }, f, indent=2, ensure_ascii=False)
            print(f"✅ Chat history saved to {filename}")
            return True
        except Exception as e:
            print(f"❌ Error saving chat history: {str(e)}")
            return False

if __name__ == "__main__":
    # Create tester instance
    tester = ChatInterfaceTester()
    
    try:
        # Run basic test
        tester.run_basic_test()
        
        # Save chat history
        tester.save_chat_history()
        
        print("\n📝 Test complete. You can now try:")
        print("   1. Visit http://localhost:5001 to use the enhanced chat interface")
        print("   2. Check the chat_history.json file for conversation logs")
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
        # Save chat history if any
        if tester.chat_history:
            tester.save_chat_history("chat_history_interrupted.json")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")