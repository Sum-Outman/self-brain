import requests
import json

# Test data
test_message = "Hello, who are you?"

# Test the API endpoint
def test_chat_api():
    try:
        # First, check if the web interface is running
        response = requests.get('http://localhost:5000', timeout=5)
        if response.status_code != 200:
            print("Web interface is not running on http://localhost:5000")
            return
        
        print("Web interface is running. Testing chat API...")
        
        # Test the chat API endpoint
        chat_response = requests.post(
            'http://localhost:5000/api/chat/send',
            json={
                'message': test_message,
                'conversation_id': 'test_conversation',
                'knowledge_base': 'all',
                'attachments': []
            },
            headers={'Content-Type': 'application/json'},
            timeout=30
        )
        
        if chat_response.status_code == 200:
            result = chat_response.json()
            print(f"API call succeeded with status: {result['status']}")
            print(f"AI Response: {result['response']}")
            print(f"Conversation ID: {result['conversation_id']}")
            print(f"Timestamp: {result['timestamp']}")
        else:
            print(f"API call failed with status code: {chat_response.status_code}")
            print(f"Response content: {chat_response.text}")
            
    except requests.exceptions.ConnectionError:
        print("Connection error: Could not connect to http://localhost:5000")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    test_chat_api()