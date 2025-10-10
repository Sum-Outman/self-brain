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

from flask import Flask, request, jsonify
from device_control import MotionController

app = Flask(__name__)

# Health check endpoints
@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        "status": "active",
        "model": "J_motion",
        "version": "1.0.0",
        "capabilities": ["motion_control", "device_communication", "multi_protocol_support"]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check"""
    return jsonify({"status": "healthy", "model": "J_motion"})

motion_controller = MotionController()

@app.route('/control', methods=['POST'])
def handle_control():
    """Motion control API endpoint"""
    data = request.json
    protocol = data.get('protocol')
    command = data.get('command')
    
    if not protocol or not command:
        return jsonify({"error": "Missing protocol or command"}), 400
    
    # Execute device control
    result = motion_controller.control_device(protocol, command)
    return jsonify(result)

import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5009))
    app.run(host='0.0.0.0', port=port, debug=True)
