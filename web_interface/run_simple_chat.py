from flask import Flask, render_template, request, jsonify
import requests
import json
from datetime import datetime
import os

# Create Flask app
app = Flask(__name__)

# Set up logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Render the simple chat interface"""
    return render_template('simple_chat_interface.html')

@app.route('/api/chat/send', methods=['POST'])
def send_message():
    """Handle message sending and forward to the main system's API"""
    try:
        # Get data from request
        data = request.get_json()
        message = data.get('message')
        model_id = data.get('model_id', 'a_manager')
        
        logger.info(f"Received message: {message}, model: {model_id}")
        
        # Forward the message to the main system's API
        # This is assuming the main system is running on localhost:5015
        try:
            response = requests.post(
                "http://localhost:5015/api/chat",
                json={
                    "message": message,
                    "conversation_id": "simple_chat",
                    "knowledge_base": "all",
                    "attachments": []
                },
                headers={
                    'Content-Type': 'application/json; charset=utf-8',
                    'Accept': 'application/json; charset=utf-8'
                },
                timeout=30
            )
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    
                    # Extract response based on the structure we've seen
                    if 'conversation_data' in result and isinstance(result['conversation_data'], dict):
                        conv_data = result['conversation_data']
                        if 'response' in conv_data:
                            return jsonify({
                                'status': 'success',
                                'response': str(conv_data['response']),
                                'timestamp': datetime.now().isoformat(),
                                'model_used': model_id
                            })
                    
                    # Case 2: Direct response field
                    if 'response' in result:
                        return jsonify({
                            'status': 'success',
                            'response': str(result['response']),
                            'timestamp': datetime.now().isoformat(),
                            'model_used': model_id
                        })
                    
                    # Case 3: Return any available content
                    if 'content' in result:
                        return jsonify({
                            'status': 'success',
                            'response': str(result['content']),
                            'timestamp': datetime.now().isoformat(),
                            'model_used': model_id
                        })
                    
                    # Default fallback
                    return jsonify({
                        'status': 'success',
                        'response': str(result),
                        'timestamp': datetime.now().isoformat(),
                        'model_used': model_id
                    })
                    
                except (ValueError, KeyError) as e:
                    logger.error(f"Error parsing response: {str(e)}")
                    return jsonify({
                        'status': 'success',
                        'response': response.text,
                        'timestamp': datetime.now().isoformat(),
                        'model_used': model_id
                    })
            else:
                logger.warning(f"Main system API call failed: {response.status_code} - {response.text}")
                return jsonify({
                    'status': 'success',
                    'response': f"I'm sorry, but I'm having trouble connecting to the main AI system. Please check if the system is running and try again.",
                    'timestamp': datetime.now().isoformat(),
                    'model_used': model_id
                })
                
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to the main AI system")
            return jsonify({
                'status': 'success',
                'response': "I'm sorry, but the main AI system is not reachable at the moment. Please make sure it's running on localhost:5015 and try again.",
                'timestamp': datetime.now().isoformat(),
                'model_used': model_id
            })
        except requests.exceptions.Timeout:
            logger.error("Main system request timed out")
            return jsonify({
                'status': 'success',
                'response': "I'm sorry, but the main AI system is taking too long to respond. Please try again later.",
                'timestamp': datetime.now().isoformat(),
                'model_used': model_id
            })
        except Exception as e:
            logger.error(f"Failed to call main system: {str(e)}")
            return jsonify({
                'status': 'success',
                'response': f"I'm sorry, but I encountered an error while processing your request: {str(e)}",
                'timestamp': datetime.now().isoformat(),
                'model_used': model_id
            })
            
    except Exception as e:
        logger.error(f"Error in send_message: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/health')
def health_check():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

if __name__ == '__main__':
    print("=== Starting Simple AI Chat Interface ===")
    print("Features:")
    print("1. ✅ Basic chat functionality")
    print("2. ✅ Clean, minimalist interface")
    print("3. ✅ Real-time message sending")
    print("4. ✅ Connection to the main AI system")
    print()
    print("Access http://localhost:8080 to use the main web interface")
    print("Note: Make sure the main AI system is running on localhost:5000")
    
    # Run the app
    app.run(host='0.0.0.0', port=8080, debug=True)