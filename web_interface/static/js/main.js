// Self Brain Main JavaScript File
// Version 1.0
// Author: Silence Crow Team
// Email: silencecrowtom@qq.com

// Main application object
const selfBrain = {
    // System configuration
    config: {
        apiBaseUrl: '/api/v1',
        updateInterval: 5000, // 5 seconds
        systemLanguage: 'en',
        enableGPU: true,
        enableMultithreading: false
    },

    // System state
    state: {
        isConnected: false,
        activeModels: [],
        hardwareStatus: {
            cameras: [],
            serialPorts: [],
            sensors: []
        },
        resources: {
            cpu: 0,
            memory: 0,
            disk: 0,
            temperature: 0
        },
        isTraining: false,
        trainingProgress: 0
    },

    // Initialize application
    init: function() {
        console.log('Self Brain AGI system initializing...');
        
        // Load configuration from localStorage if available
        this.loadConfig();
        
        // Initialize API communication
        this.api.init();
        
        // Initialize hardware management
        this.hardware.init();
        
        // Initialize model management
        this.models.init();
        
        // Start system monitoring
        this.startMonitoring();
        
        console.log('Self Brain AGI system initialized successfully.');
    },

    // Load configuration from localStorage
    loadConfig: function() {
        const savedConfig = localStorage.getItem('selfBrainConfig');
        if (savedConfig) {
            try {
                const config = JSON.parse(savedConfig);
                this.config = { ...this.config, ...config };
            } catch (e) {
                console.error('Failed to parse saved configuration:', e);
            }
        }
    },

    // Save configuration to localStorage
    saveConfig: function() {
        localStorage.setItem('selfBrainConfig', JSON.stringify(this.config));
    },

    // Start system monitoring
    startMonitoring: function() {
        this.updateSystemStatus();
        setInterval(() => {
            this.updateSystemStatus();
        }, this.config.updateInterval);
    },

    // Update system status
    updateSystemStatus: function() {
        // In a real implementation, this would fetch data from the server
        this.api.getSystemStatus()
            .then(data => {
                if (data) {
                    this.state = { ...this.state, ...data };
                    this.updateUI();
                }
            })
            .catch(error => {
                console.error('Failed to update system status:', error);
                // Fallback to simulated data
                this.simulateSystemStatus();
            });
    },

    // Simulate system status (for offline testing)
    simulateSystemStatus: function() {
        // Simulate resource usage
        this.state.resources = {
            cpu: Math.random() * 50 + 10, // 10-60%
            memory: Math.random() * 30 + 20, // 20-50%
            disk: Math.random() * 10 + 5, // 5-15%
            temperature: Math.random() * 15 + 35 // 35-50°C
        };
        
        this.updateUI();
    },

    // Update UI elements
    updateUI: function() {
        // Update resource usage displays
        const resourceElements = document.querySelectorAll('.resource-usage');
        resourceElements.forEach(el => {
            const type = el.dataset.type;
            if (type && this.state.resources[type] !== undefined) {
                el.textContent = this.state.resources[type].toFixed(1) + '%';
                
                // Update progress bars if available
                const progressBar = el.closest('.resource-item').querySelector('.progress-bar');
                if (progressBar) {
                    progressBar.style.width = this.state.resources[type] + '%';
                    
                    // Change color based on usage level
                    if (this.state.resources[type] > 70) {
                        progressBar.className = 'progress-bar critical';
                    } else if (this.state.resources[type] > 50) {
                        progressBar.className = 'progress-bar warning';
                    } else {
                        progressBar.className = 'progress-bar';
                    }
                }
            }
        });
        
        // Update model status displays
        const modelStatusElements = document.querySelectorAll('.model-status');
        modelStatusElements.forEach(el => {
            const modelId = el.closest('.model-item').querySelector('.model-id').textContent;
            const model = this.state.activeModels.find(m => m.id === modelId);
            if (model) {
                el.textContent = model.status;
                el.className = 'model-status ' + model.status.toLowerCase();
            }
        });
        
        // Update training progress
        const trainingProgress = document.getElementById('training-progress');
        if (trainingProgress) {
            trainingProgress.textContent = this.state.trainingProgress.toFixed(1) + '%';
            trainingProgress.style.width = this.state.trainingProgress + '%';
        }
        
        // Update training status
        const trainingStatus = document.getElementById('training-status');
        if (trainingStatus) {
            trainingStatus.textContent = this.state.isTraining ? 'Training in progress' : 'Not training';
        }
    },

    // API communication module
    api: {
        // Initialize API module
        init: function() {
            console.log('API module initialized.');
        },
        
        // Make HTTP request
        request: function(endpoint, method = 'GET', data = null) {
            const url = selfBrain.config.apiBaseUrl + endpoint;
            
            const options = {
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                }
            };
            
            if (data) {
                options.body = JSON.stringify(data);
            }
            
            return fetch(url, options)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`API request failed: ${response.status}`);
                    }
                    return response.json();
                })
                .catch(error => {
                    console.error('API error:', error);
                    // Return mock data for offline testing
                    return selfBrain.api.getMockData(endpoint, method);
                });
        },
        
        // Get mock data for offline testing
        getMockData: function(endpoint, method) {
            // Mock data for various API endpoints
            const mockData = {
                '/system/status': {
                    isConnected: true,
                    activeModels: [
                        { id: 'A', name: 'Manager Model', status: 'Running' },
                        { id: 'B', name: 'Language Model', status: 'Running' },
                        { id: 'C', name: 'Audio Model', status: 'Running' },
                        { id: 'D', name: 'Image Model', status: 'Running' },
                        { id: 'E', name: 'Video Model', status: 'Running' },
                        { id: 'F', name: 'Spatial Model', status: 'Running' },
                        { id: 'G', name: 'Sensor Model', status: 'Running' },
                        { id: 'H', name: 'Computer Control', status: 'Running' },
                        { id: 'I', name: 'Motion Control', status: 'Running' },
                        { id: 'J', name: 'Knowledge Base', status: 'Running' },
                        { id: 'K', name: 'Programming Model', status: 'Running' }
                    ],
                    resources: {
                        cpu: 35.2,
                        memory: 42.8,
                        disk: 8.7,
                        temperature: 42.1
                    },
                    isTraining: false,
                    trainingProgress: 0
                },
                '/hardware/cameras': [
                    { id: '0', name: 'Camera 0 (Default Webcam)', resolution: '1280x720', fps: 30 },
                    { id: '1', name: 'Camera 1 (External USB Camera)', resolution: '1920x1080', fps: 60 }
                ],
                '/hardware/serial_ports': [
                    { port: 'COM1', name: 'COM1 - Communications Port' },
                    { port: 'COM3', name: 'COM3 - Arduino Uno' },
                    { port: 'COM5', name: 'COM5 - USB Serial Port' }
                ],
                '/hardware/sensors': [
                    { id: 'temp1', name: 'Temperature Sensor (TMP36)', type: 'temperature', value: '25.5 °C', timestamp: new Date().toLocaleString() },
                    { id: 'hum1', name: 'Humidity Sensor (DHT22)', type: 'humidity', value: '48.2 %', timestamp: new Date().toLocaleString() },
                    { id: 'motion1', name: 'Motion Sensor (PIR)', type: 'motion', value: 'No motion', timestamp: new Date().toLocaleString() },
                    { id: 'light1', name: 'Light Sensor (LDR)', type: 'light', value: '320 lux', timestamp: new Date().toLocaleString() }
                ]
            };
            
            // Return appropriate mock data based on endpoint
            for (const [key, value] of Object.entries(mockData)) {
                if (endpoint.includes(key)) {
                    return value;
                }
            }
            
            return null;
        },
        
        // Get system status
        getSystemStatus: function() {
            return this.request('/system/status');
        },
        
        // Get available cameras
        getCameras: function() {
            return this.request('/hardware/cameras');
        },
        
        // Get available serial ports
        getSerialPorts: function() {
            return this.request('/hardware/serial_ports');
        },
        
        // Connect to serial port
        connectSerialPort: function(port, baudrate, timeout, deviceId) {
            return this.request('/hardware/serial/connect', 'POST', {
                port: port,
                baudrate: baudrate,
                timeout: timeout,
                deviceId: deviceId
            });
        },
        
        // Disconnect from serial port
        disconnectSerialPort: function(port) {
            return this.request('/hardware/serial/disconnect', 'POST', {
                port: port
            });
        },
        
        // Send command to serial port
        sendSerialCommand: function(port, command) {
            return this.request('/hardware/serial/command', 'POST', {
                port: port,
                command: command
            });
        },
        
        // Get connected sensors
        getSensors: function() {
            return this.request('/hardware/sensors');
        },
        
        // Get sensor data
        getSensorData: function(sensorId) {
            return this.request(`/hardware/sensors/${sensorId}`);
        },
        
        // Update sensor configuration
        updateSensorConfig: function(config) {
            return this.request('/hardware/sensors/config', 'POST', config);
        },
        
        // Send message to AI
        sendMessage: function(message, messageType = 'text') {
            return this.request('/ai/message', 'POST', {
                message: message,
                type: messageType
            });
        },
        
        // Start training
        startTraining: function(modelId, parameters = {}) {
            return this.request('/ai/train/start', 'POST', {
                modelId: modelId,
                parameters: parameters
            });
        },
        
        // Stop training
        stopTraining: function() {
            return this.request('/ai/train/stop', 'POST');
        }
    },

    // Hardware management module
    hardware: {
        // Active hardware connections
        activeConnections: {
            cameras: [],
            serialPorts: [],
            sensors: []
        },
        
        // Initialize hardware module
        init: function() {
            console.log('Hardware management module initialized.');
            
            // Initialize camera management
            this.camera.init();
            
            // Initialize serial port management
            this.serial.init();
            
            // Initialize sensor management
            this.sensor.init();
        },
        
        // Camera management
        camera: {
            // Initialize camera management
            init: function() {
                console.log('Camera management initialized.');
            },
            
            // Get available cameras
            getCameras: function() {
                return selfBrain.api.getCameras();
            },
            
            // Start camera stream
            startStream: function(cameraId, options = {}) {
                const defaultOptions = { resolution: '1280x720', fps: 30, brightness: 50 };
                const config = { ...defaultOptions, ...options };
                
                // In a real implementation, this would start the camera stream
                return new Promise((resolve) => {
                    setTimeout(() => {
                        // Add to active connections
                        selfBrain.hardware.activeConnections.cameras.push({
                            id: cameraId,
                            ...config
                        });
                        
                        resolve({
                            success: true,
                            streamUrl: `https://via.placeholder.com/${config.resolution.split('x')[0]}x${config.resolution.split('x')[1]}?text=Camera+${cameraId}`
                        });
                    }, 1000);
                });
            },
            
            // Stop camera stream
            stopStream: function(cameraId) {
                // In a real implementation, this would stop the camera stream
                return new Promise((resolve) => {
                    setTimeout(() => {
                        // Remove from active connections
                        selfBrain.hardware.activeConnections.cameras = 
                            selfBrain.hardware.activeConnections.cameras.filter(cam => cam.id !== cameraId);
                        
                        resolve({ success: true });
                    }, 500);
                });
            },
            
            // Start stereo vision
            startStereoVision: function(leftCameraId, rightCameraId, options = {}) {
                return Promise.all([
                    this.startStream(leftCameraId, options),
                    this.startStream(rightCameraId, options)
                ]).then(results => {
                    return {
                        success: results.every(r => r.success),
                        leftStreamUrl: results[0].streamUrl,
                        rightStreamUrl: results[1].streamUrl
                    };
                });
            }
        },
        
        // Serial port management
        serial: {
            // Initialize serial port management
            init: function() {
                console.log('Serial port management initialized.');
            },
            
            // Get available serial ports
            getPorts: function() {
                return selfBrain.api.getSerialPorts();
            },
            
            // Connect to serial port
            connect: function(port, baudrate, timeout, deviceId) {
                return selfBrain.api.connectSerialPort(port, baudrate, timeout, deviceId)
                    .then(result => {
                        if (result.success) {
                            // Add to active connections
                            selfBrain.hardware.activeConnections.serialPorts.push({
                                port: port,
                                baudrate: baudrate,
                                timeout: timeout,
                                deviceId: deviceId,
                                connectedAt: new Date()
                            });
                        }
                        return result;
                    });
            },
            
            // Disconnect from serial port
            disconnect: function(port) {
                return selfBrain.api.disconnectSerialPort(port)
                    .then(result => {
                        if (result.success) {
                            // Remove from active connections
                            selfBrain.hardware.activeConnections.serialPorts = 
                                selfBrain.hardware.activeConnections.serialPorts.filter(sp => sp.port !== port);
                        }
                        return result;
                    });
            },
            
            // Send command to serial port
            sendCommand: function(port, command) {
                return selfBrain.api.sendSerialCommand(port, command);
            }
        },
        
        // Sensor management
        sensor: {
            // Initialize sensor management
            init: function() {
                console.log('Sensor management initialized.');
            },
            
            // Get connected sensors
            getSensors: function() {
                return selfBrain.api.getSensors();
            },
            
            // Get sensor data
            getSensorData: function(sensorId) {
                return selfBrain.api.getSensorData(sensorId);
            },
            
            // Update sensor configuration
            updateConfig: function(config) {
                return selfBrain.api.updateSensorConfig(config);
            }
        }
    },

    // Model management module
    models: {
        // Initialize model management
        init: function() {
            console.log('Model management module initialized.');
        },
        
        // Toggle model activation
        toggleModel: function(modelId, enable) {
            // In a real implementation, this would send a request to enable/disable the model
            console.log(`${enable ? 'Enabling' : 'Disabling'} model ${modelId}`);
            
            return new Promise((resolve) => {
                setTimeout(() => {
                    resolve({ success: true });
                }, 500);
            });
        },
        
        // Start training
        startTraining: function(modelId, parameters = {}) {
            selfBrain.state.isTraining = true;
            selfBrain.state.trainingProgress = 0;
            
            return selfBrain.api.startTraining(modelId, parameters)
                .then(result => {
                    if (result.success) {
                        // Start progress simulation
                        this.simulateTrainingProgress();
                    }
                    return result;
                });
        },
        
        // Stop training
        stopTraining: function() {
            selfBrain.state.isTraining = false;
            
            return selfBrain.api.stopTraining();
        },
        
        // Simulate training progress (for demonstration)
        simulateTrainingProgress: function() {
            if (!selfBrain.state.isTraining) return;
            
            // Increment progress
            selfBrain.state.trainingProgress += Math.random() * 5;
            
            if (selfBrain.state.trainingProgress >= 100) {
                selfBrain.state.trainingProgress = 100;
                selfBrain.state.isTraining = false;
            } else {
                // Continue simulation
                setTimeout(() => this.simulateTrainingProgress(), 1000);
            }
            
            // Update UI
            selfBrain.updateUI();
        }
    },

    // Dialog management module
    dialog: {
        // Send message to AI
        sendMessage: function(message, messageType = 'text') {
            // Add user message to UI
            this.addMessageToUI('user', message, messageType);
            
            // Send to API
            return selfBrain.api.sendMessage(message, messageType)
                .then(response => {
                    // Add AI response to UI
                    if (response && response.message) {
                        this.addMessageToUI('ai', response.message, response.type || 'text');
                    }
                    return response;
                })
                .catch(error => {
                    console.error('Failed to send message:', error);
                    // Add error message to UI
                    this.addMessageToUI('ai', 'Sorry, I couldn\'t process your request at the moment.', 'text');
                });
        },
        
        // Add message to UI
        addMessageToUI: function(sender, content, type = 'text') {
            const chatContainer = document.getElementById('conversation-container');
            if (!chatContainer) return;
            
            const messageElement = document.createElement('div');
            messageElement.className = `message-bubble ${sender}-message`;
            
            // Create message content based on type
            if (type === 'text') {
                // Convert newlines to <br> tags
                const formattedContent = content.replace(/\n/g, '<br>');
                messageElement.innerHTML = `
                    <p>${formattedContent}</p>
                `;
            } else if (type === 'image') {
                messageElement.innerHTML = `
                    <img src="${content}" alt="Image" class="chat-image img-fluid rounded">
                `;
            } else if (type === 'audio') {
                messageElement.innerHTML = `
                    <audio controls src="${content}"></audio>
                `;
            }
            
            // Add to chat container
            chatContainer.appendChild(messageElement);
            
            // Scroll to bottom
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    }
};

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    selfBrain.init();
});

// Export the selfBrain object for global access
window.selfBrain = selfBrain;

// Global command functions
export function quickCommand(command) {
    if (window.selfBrain && window.selfBrain.dialog) {
        selfBrain.dialog.sendMessage(command, 'text');
    }
}

export function startTraining() {
    if (window.selfBrain && window.selfBrain.models) {
        // Start training for model A (Management Model) by default
        selfBrain.models.startTraining('A');
    }
}

export function stopTraining() {
    if (window.selfBrain && window.selfBrain.models) {
        selfBrain.models.stopTraining();
    }
}

export function importKnowledge() {
    alert('Knowledge import functionality coming soon!');
}

export function uploadTrainingData() {
    alert('Training data upload functionality coming soon!');
}

export function exportData() {
    alert('Data export functionality coming soon!');
}

export function systemHealth() {
    alert('System health check functionality coming soon!');
}

export function clearConversation() {
    const conversationContainer = document.getElementById('conversation-container');
    if (conversationContainer) {
        conversationContainer.innerHTML = `
            <div class="text-center text-gray-500 py-5">
                <i class="bi bi-robot fs-1"></i>
                <p class="mt-3 text-dark">Conversation Cleared</p>
                <small class="text-gray-500">Start a new conversation</small>
            </div>
        `;
    }
}

export function updateEmotion() {
    const emotionSelect = document.getElementById('emotion-select');
    if (emotionSelect) {
        const selectedMode = emotionSelect.value;
        console.log(`System mode changed to: ${selectedMode}`);
        
        // In a real implementation, this would update the system's behavior
        alert(`System mode changed to: ${selectedMode}`);
    }
}

export function executeCommand() {
    const messageInput = document.getElementById('message-input');
    if (messageInput) {
        const command = messageInput.value.trim();
        if (command) {
            console.log(`Executing command: ${command}`);
            alert(`Command execution simulation: "${command}"\n\nIn a real implementation, this would execute system commands.`);
        } else {
            alert('Please enter a command to execute.');
        }
    }
}

// Camera control functions
export function listAllCameras() {
    if (window.selfBrain && window.selfBrain.hardware && window.selfBrain.hardware.camera) {
        selfBrain.hardware.camera.getCameras()
            .then(cameras => {
                const camera1Selector = document.getElementById('camera1Selector');
                const camera2Selector = document.getElementById('camera2Selector');
                
                if (camera1Selector && camera2Selector && cameras && cameras.length > 0) {
                    // Clear existing options
                    camera1Selector.innerHTML = '<option value="default">Select Camera</option>';
                    camera2Selector.innerHTML = '<option value="default">Select Camera</option>';
                    
                    // Add camera options
                    cameras.forEach(camera => {
                        const option1 = document.createElement('option');
                        option1.value = camera.id;
                        option1.textContent = `${camera.name} (${camera.resolution}, ${camera.fps}fps)`;
                        camera1Selector.appendChild(option1);
                        
                        const option2 = document.createElement('option');
                        option2.value = camera.id;
                        option2.textContent = `${camera.name} (${camera.resolution}, ${camera.fps}fps)`;
                        camera2Selector.appendChild(option2);
                    });
                }
            });
    }
}

export function startStereoVision() {
    if (window.selfBrain && window.selfBrain.hardware && window.selfBrain.hardware.camera) {
        const camera1Selector = document.getElementById('camera1Selector');
        const camera2Selector = document.getElementById('camera2Selector');
        const camera1Preview = document.getElementById('camera1Preview');
        const camera2Preview = document.getElementById('camera2Preview');
        const camera1Status = document.getElementById('camera1Status');
        const camera2Status = document.getElementById('camera2Status');
        
        if (camera1Selector && camera2Selector && camera1Selector.value !== 'default' && camera2Selector.value !== 'default') {
            const leftCameraId = camera1Selector.value;
            const rightCameraId = camera2Selector.value;
            
            selfBrain.hardware.camera.startStereoVision(leftCameraId, rightCameraId)
                .then(result => {
                    if (result.success && camera1Preview && camera2Preview) {
                        // Update camera status
                        if (camera1Status) camera1Status.textContent = 'Active';
                        if (camera2Status) camera2Status.textContent = 'Active';
                        
                        // Create video elements
                        camera1Preview.innerHTML = `<video id="camera1Video" autoplay muted playsinline></video>`;
                        camera2Preview.innerHTML = `<video id="camera2Video" autoplay muted playsinline></video>`;
                        
                        // Set video sources with placeholder images for now
                        const camera1Video = document.getElementById('camera1Video');
                        const camera2Video = document.getElementById('camera2Video');
                        
                        if (camera1Video && camera2Video) {
                            // In a real implementation, we would use the actual stream
                            // For now, we'll use placeholder images
                            camera1Video.style.display = 'none';
                            camera2Video.style.display = 'none';
                            
                            camera1Preview.innerHTML = `
                                <span id="camera1Status" class="absolute top-1 right-1 text-white text-xs px-1 rounded bg-green-700">Active</span>
                                <img src="${result.leftStreamUrl}" alt="Left Camera" class="w-full h-full object-cover">
                            `;
                            camera2Preview.innerHTML = `
                                <span id="camera2Status" class="absolute top-1 right-1 text-white text-xs px-1 rounded bg-green-700">Active</span>
                                <img src="${result.rightStreamUrl}" alt="Right Camera" class="w-full h-full object-cover">
                            `;
                        }
                    }
                });
        } else {
            alert('Please select both cameras for stereo vision.');
        }
    }
}

export function stopAllCameras() {
    if (window.selfBrain && window.selfBrain.hardware && window.selfBrain.hardware.camera) {
        // Get all active cameras and stop them
        const activeCameras = selfBrain.hardware.activeConnections.cameras;
        activeCameras.forEach(camera => {
            selfBrain.hardware.camera.stopStream(camera.id);
        });
        
        // Reset camera previews
        const camera1Preview = document.getElementById('camera1Preview');
        const camera2Preview = document.getElementById('camera2Preview');
        
        if (camera1Preview && camera2Preview) {
            camera1Preview.innerHTML = `
                <span id="camera1Status" class="absolute top-1 right-1 text-white text-xs px-1 rounded bg-gray-700">Inactive</span>
                <div class="text-center">
                    <i class="bi bi-camera text-gray-400 fs-2"></i>
                    <p class="text-xs text-gray-500 mt-1">Left Eye</p>
                </div>
            `;
            camera2Preview.innerHTML = `
                <span id="camera2Status" class="absolute top-1 right-1 text-white text-xs px-1 rounded bg-gray-700">Inactive</span>
                <div class="text-center">
                    <i class="bi bi-camera text-gray-400 fs-2"></i>
                    <p class="text-xs text-gray-500 mt-1">Right Eye</p>
                </div>
            `;
        }
    }
}

export function takeAllSnapshots() {
    alert('Camera snapshot functionality coming soon!');
}

// Export global functions to window
window.quickCommand = quickCommand;
window.startTraining = startTraining;
window.stopTraining = stopTraining;
window.importKnowledge = importKnowledge;
window.uploadTrainingData = uploadTrainingData;
window.exportData = exportData;
window.systemHealth = systemHealth;
window.clearConversation = clearConversation;
window.updateEmotion = updateEmotion;
window.executeCommand = executeCommand;
window.listAllCameras = listAllCameras;
window.startStereoVision = startStereoVision;
window.stopAllCameras = stopAllCameras;
window.takeAllSnapshots = takeAllSnapshots;