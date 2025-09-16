import requests
import json
import time

# Base URL
BASE_URL = 'http://localhost:5000/api/chat'

# Function to print with colors
class Colors:
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RED = '\033[91m'
    ENDC = '\033[0m'

def print_color(text, color):
    print(f"{color}{text}{Colors.ENDC}")

# Test full conversation flow
def test_full_conversation():
    try:
        print_color("===== Testing Full Conversation Flow =====", Colors.GREEN)
        
        # 1. Create a new conversation
        print_color("\n1. Creating a new conversation...", Colors.BLUE)
        new_conv_response = requests.post(
            f'{BASE_URL}/new_conversation',
            json={
                'title': 'Test Conversation',
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ')
            },
            headers={'Content-Type': 'application/json'}
        )
        
        if new_conv_response.status_code == 200:
            new_conv_data = new_conv_response.json()
            if new_conv_data['status'] == 'success':
                conversation_id = new_conv_data['conversation_id']
                print_color(f"Success: New conversation created with ID: {conversation_id}", Colors.GREEN)
            else:
                print_color(f"Failed to create new conversation: {new_conv_data}", Colors.RED)
                return
        else:
            print_color(f"API call failed with status code: {new_conv_response.status_code}", Colors.RED)
            print(f"Response: {new_conv_response.text}")
            return
        
        # 2. Send first message
        print_color(f"\n2. Sending first message to conversation {conversation_id}...", Colors.BLUE)
        message1 = "Tell me about the available models in the system"
        response1 = requests.post(
            f'{BASE_URL}/send',
            json={
                'message': message1,
                'conversation_id': conversation_id,
                'knowledge_base': 'all',
                'attachments': []
            },
            headers={'Content-Type': 'application/json'}
        )
        
        if response1.status_code == 200:
            data1 = response1.json()
            if data1['status'] == 'success':
                print_color(f"Success: First message sent", Colors.GREEN)
                print_color(f"AI Response:", Colors.BLUE)
                print(f"{data1['response'][:200]}...")  # Print first 200 chars
            else:
                print_color(f"Failed to send message: {data1}", Colors.RED)
        else:
            print_color(f"API call failed with status code: {response1.status_code}", Colors.RED)
            print(f"Response: {response1.text}")
        
        # 3. Send second message
        print_color(f"\n3. Sending second message to conversation {conversation_id}...", Colors.BLUE)
        message2 = "What's the system status?"
        response2 = requests.post(
            f'{BASE_URL}/send',
            json={
                'message': message2,
                'conversation_id': conversation_id,
                'knowledge_base': 'all',
                'attachments': []
            },
            headers={'Content-Type': 'application/json'}
        )
        
        if response2.status_code == 200:
            data2 = response2.json()
            if data2['status'] == 'success':
                print_color(f"Success: Second message sent", Colors.GREEN)
                print_color(f"AI Response:", Colors.BLUE)
                print(f"{data2['response'][:200]}...")  # Print first 200 chars
            else:
                print_color(f"Failed to send message: {data2}", Colors.RED)
        else:
            print_color(f"API call failed with status code: {response2.status_code}", Colors.RED)
            print(f"Response: {response2.text}")
        
        # 4. Get conversation history
        print_color(f"\n4. Getting conversation history for {conversation_id}...", Colors.BLUE)
        history_response = requests.get(f'{BASE_URL}/messages/{conversation_id}')
        
        if history_response.status_code == 200:
            history_data = history_response.json()
            if history_data['status'] == 'success':
                print_color(f"Success: Retrieved {len(history_data['messages'])} messages", Colors.GREEN)
                # Print message count and types
                message_types = {}
                for msg in history_data['messages']:
                    if msg['role'] in message_types:
                        message_types[msg['role']] += 1
                    else:
                        message_types[msg['role']] = 1
                print(f"Message types: {message_types}")
            else:
                print_color(f"Failed to get history: {history_data}", Colors.RED)
        else:
            print_color(f"API call failed with status code: {history_response.status_code}", Colors.RED)
            print(f"Response: {history_response.text}")
        
        # 5. List all conversations
        print_color("\n5. Listing all conversations...", Colors.BLUE)
        conv_list_response = requests.get(f'{BASE_URL}/conversations')
        
        if conv_list_response.status_code == 200:
            conv_list_data = conv_list_response.json()
            if conv_list_data['status'] == 'success':
                print_color(f"Success: Found {len(conv_list_data['conversations'])} conversations", Colors.GREEN)
                # Print conversation titles
                for i, conv in enumerate(conv_list_data['conversations'][:3]):  # Print first 3
                    print(f"{i+1}. {conv['title']}")
                if len(conv_list_data['conversations']) > 3:
                    print(f"... and {len(conv_list_data['conversations']) - 3} more")
            else:
                print_color(f"Failed to list conversations: {conv_list_data}", Colors.RED)
        else:
            print_color(f"API call failed with status code: {conv_list_response.status_code}", Colors.RED)
            print(f"Response: {conv_list_response.text}")
            
        print_color("\n===== Testing Complete =====", Colors.GREEN)
        
    except requests.exceptions.ConnectionError:
        print_color("Connection error: Could not connect to the API", Colors.RED)
    except Exception as e:
        print_color(f"An error occurred: {str(e)}", Colors.RED)

if __name__ == "__main__":
    test_full_conversation()