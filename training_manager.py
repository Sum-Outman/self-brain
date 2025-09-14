# Copyright 2025 The AI Management System Authors
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

# 训练管理器
# Training Manager

from flask import Flask, request, jsonify
import threading
import time
import uuid

app = Flask(__name__)

class TrainingManager:
    def __init__(self):
        self.active_trainings = {}
        self.completed_trainings = {}
    
    def start_training(self, config):
        training_id = str(uuid.uuid4())
        self.active_trainings[training_id] = {
            'config': config,
            'status': 'running',
            'progress': 0,
            'start_time': time.time()
        }
        
        def training_thread():
            for i in range(100):
                if training_id in self.active_trainings:
                    self.active_trainings[training_id]['progress'] = i + 1
                    time.sleep(0.1)
                else:
                    break
            
            if training_id in self.active_trainings:
                training = self.active_trainings.pop(training_id)
                training['status'] = 'completed'
                training['end_time'] = time.time()
                self.completed_trainings[training_id] = training
        
        thread = threading.Thread(target=training_thread)
        thread.start()
        
        return training_id
    
    def get_status(self, training_id):
        if training_id in self.active_trainings:
            return self.active_trainings[training_id]
        elif training_id in self.completed_trainings:
            return self.completed_trainings[training_id]
        else:
            return None
    
    def list_trainings(self):
        return {
            'active': list(self.active_trainings.keys()),
            'completed': list(self.completed_trainings.keys())
        }

manager = TrainingManager()

@app.route('/start', methods=['POST'])
def start_training():
    data = request.json
    config = data.get('config', {})
    
    training_id = manager.start_training(config)
    return jsonify({
        'status': 'success',
        'training_id': training_id,
        'message': '训练已启动'
    })

@app.route('/status/<training_id>', methods=['GET'])
def get_training_status(training_id):
    status = manager.get_status(training_id)
    if status:
        return jsonify(status)
    else:
        return jsonify({'error': '训练ID不存在'}), 404

@app.route('/list', methods=['GET'])
def list_trainings():
    return jsonify(manager.list_trainings())

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "service": "Training Manager"})

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5012))
    app.run(host='0.0.0.0', port=port, debug=True)