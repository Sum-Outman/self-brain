#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
A Manager Model API Server
Handles API requests for A manager model, including model coordination, task assignment, and emotion analysis
"""

import os
import sys
import json
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
from datetime import datetime

# Add project root directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AManagerAPI")

# Create Flask application
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Simulate A manager model status
model_status = {
    "status": "ready",
    "last_heartbeat": datetime.now().isoformat(),
    "connected_models": [],
    "current_tasks": [],
    "emotional_state": {
        "valence": 0.7,  # Valence (0-1)
        "arousal": 0.5,   # Arousal (0-1)
        "dominance": 0.8  # Dominance (0-1)
    }
}

# Simulate registered models
registered_models = {
    "A_manager": {
        "model_id": "A_manager",
        "name": "Management Model",
        "model_type": "management",
        "description": "Main interactive AI model that coordinates all sub-models",
        "status": "active",
        "is_local": True,
        "config": {
            "model_source": "local",
            "api_endpoint": "http://localhost:5001",
            "version": "1.0.0"
        }
    },
    "B_language": {
        "model_id": "B_language",
        "name": "Large Language Model",
        "model_type": "language",
        "description": "Capable of multi-language interaction and emotional reasoning",
        "status": "active",
        "is_local": True,
        "config": {
            "model_source": "local",
            "api_endpoint": "http://localhost:5002",
            "version": "1.0.0"
        }
    },
    "C_audio": {
        "model_id": "C_audio",
        "name": "Audio Processing Model",
        "model_type": "audio",
        "description": "Speech recognition, tone recognition, audio synthesis",
        "status": "active",
        "is_local": True,
        "config": {
            "model_source": "local",
            "api_endpoint": "http://localhost:5003",
            "version": "1.0.0"
        }
    },
    "D_image": {
        "model_id": "D_image",
        "name": "Image Visual Processing Model",
        "model_type": "image",
        "description": "Image content recognition, modification, and generation",
        "status": "active",
        "is_local": True,
        "config": {
            "model_source": "local",
            "api_endpoint": "http://localhost:5004",
            "version": "1.0.0"
        }
    },
    "E_video": {
        "model_id": "E_video",
        "name": "Video Stream Visual Processing Model",
        "model_type": "video",
        "description": "Video content recognition, editing, and generation",
        "status": "active",
        "is_local": True,
        "config": {
            "model_source": "local",
            "api_endpoint": "http://localhost:5005",
            "version": "1.0.0"
        }
    },
    "F_spatial": {
        "model_id": "F_spatial",
        "name": "Binocular Spatial Positioning Perception Model",
        "model_type": "spatial",
        "description": "Spatial recognition, visual spatial modeling, spatial positioning",
        "status": "active",
        "is_local": True,
        "config": {
            "model_source": "local",
            "api_endpoint": "http://localhost:5006",
            "version": "1.0.0"
        }
    },
    "G_sensor": {
        "model_id": "G_sensor",
        "name": "Sensor Perception Model",
        "model_type": "sensor",
        "description": "Multi-sensor data processing and analysis",
        "status": "active",
        "is_local": True,
        "config": {
            "model_source": "local",
            "api_endpoint": "http://localhost:5007",
            "version": "1.0.0"
        }
    },
    "H_computer": {
        "model_id": "H_computer",
        "name": "Computer Control Model",
        "model_type": "computer_control",
        "description": "Control computer operations through commands",
        "status": "active",
        "is_local": True,
        "config": {
            "model_source": "local",
            "api_endpoint": "http://localhost:5008",
            "version": "1.0.0"
        }
    },
    "I_knowledge": {
        "model_id": "I_knowledge",
        "name": "Knowledge Base Expert Model",
        "model_type": "knowledge",
        "description": "Comprehensive knowledge system to assist other models in task completion",
        "status": "active",
        "is_local": True,
        "config": {
            "model_source": "local",
            "api_endpoint": "http://localhost:5009",
            "version": "1.0.0"
        }
    },
    "J_motion": {
        "model_id": "J_motion",
        "name": "Motion and Actuator Control Model",
        "model_type": "motion",
        "description": "Perform complex control based on perception data",
        "status": "active",
        "is_local": True,
        "config": {
            "model_source": "local",
            "api_endpoint": "http://localhost:5010",
            "version": "1.0.0"
        }
    },
    "K_programming": {
        "model_id": "K_programming",
        "name": "Programming Model",
        "model_type": "programming",
        "description": "Programming assistance, autonomous programming to improve environment",
        "status": "active",
        "is_local": True,
        "config": {
            "model_source": "local",
            "api_endpoint": "http://localhost:5011",
            "version": "1.0.0"
        }
    }
}

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get A manager model status"""
    try:
        # Update heartbeat time
        model_status['last_heartbeat'] = datetime.now().isoformat()
        
        return jsonify({
            'status': 'success',
            'data': model_status
        })
    except Exception as e:
        logger.error(f"Failed to get status: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get all models information"""
    try:
        return jsonify({
            'status': 'success',
            'models': list(registered_models.values())
        })
    except Exception as e:
        logger.error(f"Failed to get model list: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/models/<model_id>', methods=['GET'])
def get_model(model_id):
    """Get specific model information"""
    try:
        if model_id in registered_models:
            return jsonify({
                'status': 'success',
                'model': registered_models[model_id]
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Model {model_id} not found'
            }), 404
    except Exception as e:
        logger.error(f"Failed to get model information: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/task/assign', methods=['POST'])
def assign_task():
    """Assign task to model"""
    try:
        data = request.get_json()
        task_description = data.get('task')
        target_model = data.get('model_id')
        priority = data.get('priority', 'normal')
        
        if not task_description or not target_model:
            return jsonify({
                'status': 'error',
                'message': 'Task description and model_id are required'
            }), 400
        
        if target_model not in registered_models:
            return jsonify({
                'status': 'error',
                'message': f'Model {target_model} not found'
            }), 404
        
        # Simulate task assignment
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        task = {
            'task_id': task_id,
            'description': task_description,
            'model_id': target_model,
            'priority': priority,
            'status': 'assigned',
            'created_at': datetime.now().isoformat(),
            'assigned_at': datetime.now().isoformat()
        }
        
        # Add to current tasks list
        model_status['current_tasks'].append(task)
        
        logger.info(f"Task assigned successfully: {task_id} -> {target_model}")
        
        return jsonify({
            'status': 'success',
            'task_id': task_id,
            'message': f'Task assigned to {target_model} successfully'
        })
    except Exception as e:
        logger.error(f"Task assignment failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/task/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    """Get task status"""
    try:
        # Find the task
        task = next((t for t in model_status['current_tasks'] if t['task_id'] == task_id), None)
        
        if task:
            return jsonify({
                'status': 'success',
                'task': task
            })
        else:
            return jsonify({
                'status': 'error',
                'message': f'Task {task_id} not found'
            }), 404
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/emotion/analyze', methods=['POST'])
def analyze_emotion():
    """Analyze emotion"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({
                'status': 'error',
                'message': 'Text is required for emotion analysis'
            }), 400
        
        # Simulate emotion analysis
        # Using simple keyword matching to simulate emotion analysis
        positive_words = ['good', 'happy', 'joy', 'like', 'love', 'great', 'excellent', 'perfect', 'wonderful']
        negative_words = ['bad', 'sad', 'sorrow', 'hate', 'dislike', 'poor', 'terrible', 'failure', 'pain']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            emotion = "positive"
            intensity = min(0.9, 0.5 + (positive_count * 0.1))
        elif negative_count > positive_count:
            emotion = "negative"
            intensity = min(0.9, 0.5 + (negative_count * 0.1))
        else:
            emotion = "neutral"
            intensity = 0.5
        
        return jsonify({
            'status': 'success',
            'emotion': emotion,
            'intensity': intensity,
            'analysis': {
                'positive_keywords': positive_count,
                'negative_keywords': negative_count,
                'text_length': len(text)
            }
        })
    except Exception as e:
        logger.error(f"Emotion analysis failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/emotion/set', methods=['POST'])
def set_emotion():
    """Set emotional state"""
    try:
        data = request.get_json()
        valence = data.get('valence')
        arousal = data.get('arousal')
        dominance = data.get('dominance')
        
        if valence is not None:
            model_status['emotional_state']['valence'] = max(0, min(1, valence))
        if arousal is not None:
            model_status['emotional_state']['arousal'] = max(0, min(1, arousal))
        if dominance is not None:
            model_status['emotional_state']['dominance'] = max(0, min(1, dominance))
        
        return jsonify({
            'status': 'success',
            'emotional_state': model_status['emotional_state'],
            'message': 'Emotional state updated successfully'
        })
    except Exception as e:
        logger.error(f"Failed to set emotional state: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/coordinate', methods=['POST'])
def coordinate_models():
    """Coordinate multiple models"""
    try:
        data = request.get_json()
        task_description = data.get('task')
        involved_models = data.get('models', [])
        coordination_strategy = data.get('strategy', 'sequential')
        
        if not task_description or not involved_models:
            return jsonify({
                'status': 'error',
                'message': 'Task description and models are required'
            }), 400
        
        # Verify all models exist
        for model_id in involved_models:
            if model_id not in registered_models:
                return jsonify({
                    'status': 'error',
                    'message': f'Model {model_id} not found'
                }), 404
        
        # Simulate coordination process
        coordination_id = f"coord_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        coordination_plan = {
            'coordination_id': coordination_id,
            'task': task_description,
            'involved_models': involved_models,
            'strategy': coordination_strategy,
            'status': 'planning',
            'created_at': datetime.now().isoformat(),
            'steps': []
        }
        
        # Generate coordination steps based on strategy
        if coordination_strategy == 'sequential':
            for i, model_id in enumerate(involved_models):
                coordination_plan['steps'].append({
                    'step': i + 1,
                    'model_id': model_id,
                    'action': f'Execute part {i + 1} of task',
                    'status': 'pending'
                })
        elif coordination_strategy == 'parallel':
            for model_id in involved_models:
                coordination_plan['steps'].append({
                    'step': 1,
                    'model_id': model_id,
                    'action': 'Execute parallel task component',
                    'status': 'pending'
                })
        else:
            coordination_plan['steps'].append({
                'step': 1,
                'model_id': involved_models[0],
                'action': 'Execute main task',
                'status': 'pending'
            })
        
        logger.info(f"Coordination plan created successfully: {coordination_id}")
        
        return jsonify({
            'status': 'success',
            'coordination_id': coordination_id,
            'plan': coordination_plan,
            'message': 'Coordination plan created successfully'
        })
    except Exception as e:
        logger.error(f"Failed to coordinate models: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/knowledge/query', methods=['POST'])
def query_knowledge():
    """Query knowledge base"""
    try:
        data = request.get_json()
        query = data.get('query')
        knowledge_domain = data.get('domain', 'general')
        
        if not query:
            return jsonify({
                'status': 'error',
                'message': 'Query is required'
            }), 400
        
        # Simulate knowledge query response
        responses = {
            'machine learning': "Machine learning is a branch of artificial intelligence that enables computer systems to learn from data and improve without explicit programming.",
            'deep learning': "Deep learning is a subfield of machine learning that uses multi-layer neural networks to process complex data patterns.",
            'natural language processing': "Natural language processing is a field of artificial intelligence focused on interaction between computers and human language.",
            'computer vision': "Computer vision enables computers to derive information from digital images, videos, and other visual inputs and take action.",
            'reinforcement learning': "Reinforcement learning is a machine learning method where agents learn optimal behavior strategies through interaction with the environment."
        }
        
        # Simple keyword matching
        query_lower = query.lower()
        response = "Based on my knowledge base, this is a good question. Let me provide you with relevant information."
        
        for keyword, answer in responses.items():
            if keyword in query_lower:
                response = answer
                break
        
        return jsonify({
            'status': 'success',
            'query': query,
            'response': response,
            'domain': knowledge_domain,
            'confidence': 0.85,
            'sources': ['internal_knowledge_base']
        })
    except Exception as e:
        logger.error(f"Knowledge query failed: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0',
        'service': 'A Manager Model API',
        'uptime': '0 days 0 hours 0 minutes'
    })

if __name__ == '__main__':
    # Load system configuration to get the correct port
    import yaml
    import os
    from pathlib import Path
    
    # Get base directory
    BASE_DIR = Path(__file__).parent.parent.parent.absolute()
    
    # Load config
    config_path = BASE_DIR / "config" / "system_config.yaml"
    config = {}
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
    
    # Get port from config or use default
    port = config.get('ports', {}).get('manager_api', 5015)
    
    logger.info(f"Starting A Manager Model API Server")
    logger.info(f"API service running at http://localhost:{port}")
    
    # Run Flask application
    app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
