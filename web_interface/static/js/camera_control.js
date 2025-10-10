// Camera Control Module for Self Brain AGI System
// Implements multi-camera management and stereo vision capabilities

class CameraControl {
    constructor() {
        // API endpoints
        this.apiEndpoints = {
            listCameras: '/api/cameras',
            startCamera: '/api/cameras/',
            stopCamera: '/api/cameras/',
            stopAllCameras: '/api/cameras/active',
            getFrame: '/api/cameras/',
            takeSnapshot: '/api/cameras/',
            getSettings: '/api/cameras/',
            updateSettings: '/api/cameras/',
            createStereoPair: '/api/stereo/pairs',
            enableStereoVision: '/api/stereo/pairs/',
            disableStereoVision: '/api/stereo/pairs/',
            getDepthData: '/api/stereo/pairs/'
        };
        
        // Active camera streams
        this.activeCameras = {};
        // Active depth streams
        this.activeDepthStreams = {};
        // Frame rate counters
        this.frameCounters = {};
        this.frameRates = {};
        // Stereo pairs
        this.stereoPairs = {};
        // Depth map interval
        this.depthMapInterval = null;
        
        // Camera constraints
        this.cameraConstraints = {
            '640x480': { width: 640, height: 480 },
            '1280x720': { width: 1280, height: 720 },
            '1920x1080': { width: 1920, height: 1080 }
        };
        
        // Initialize UI elements
        this.initializeUI();
        // Initialize event listeners
        this.initializeEventListeners();
        // Initialize WebSocket connections
        this.initializeWebSockets();
    }

    // Static method to get an instance (for backward compatibility)
    static getInstance() {
        if (!window.cameraControl) {
            window.cameraControl = new CameraControl();
        }
        return window.cameraControl;
    }
    
    // Initialize UI elements
    initializeUI() {
        // Get references to UI elements
        this.ui = {
            // Camera selectors
            leftCameraSelector: document.getElementById('camera1Selector'),
            rightCameraSelector: document.getElementById('camera2Selector'),
            topCameraSelector: document.getElementById('camera3Selector'),
            resolutionSelector: document.getElementById('cameraResolution'),
            
            // Status indicators
            leftCameraStatus: document.getElementById('camera1Status'),
            rightCameraStatus: document.getElementById('camera2Status'),
            topCameraStatus: document.getElementById('camera3Status'),
            depthMapStatus: document.getElementById('depthMapStatus'),
            
            // Video elements
            leftCameraVideo: document.getElementById('camera1Video'),
            rightCameraVideo: document.getElementById('camera2Video'),
            topCameraVideo: document.getElementById('camera3Video'),
            
            // Depth map canvas
            depthMapCanvas: document.getElementById('depthMapCanvas'),
            
            // Buttons
            refreshCamerasBtn: document.getElementById('refreshCameras'),
            startStereoVisionBtn: document.getElementById('startStereoVision'),
            takeSnapshotsBtn: document.getElementById('takeSnapshots'),
            stopAllCamerasBtn: document.getElementById('stopAllCamerasBtn'),
            startLeftCameraBtn: document.getElementById('startCamera1'),
            stopLeftCameraBtn: document.getElementById('stopCamera1'),
            startRightCameraBtn: document.getElementById('startCamera2'),
            stopRightCameraBtn: document.getElementById('stopCamera2'),
            startTopCameraBtn: document.getElementById('startCamera3'),
            stopTopCameraBtn: document.getElementById('stopCamera3'),
            
            // Depth map color scheme
            depthMapColorScheme: document.getElementById('depthMapColorScheme')
        };
    }
    
    // Initialize event listeners
    initializeEventListeners() {
        // Refresh cameras button
        if (this.ui.refreshCamerasBtn) {
            this.ui.refreshCamerasBtn.addEventListener('click', () => {
                this.listCameras();
                this.showNotification('Refreshing camera list...', 'info');
            });
        }
        
        // Camera control buttons
        if (this.ui.startLeftCameraBtn) this.ui.startLeftCameraBtn.addEventListener('click', () => this.startCamera('left'));
        if (this.ui.stopLeftCameraBtn) this.ui.stopLeftCameraBtn.addEventListener('click', () => this.stopCamera('left'));
        if (this.ui.startRightCameraBtn) this.ui.startRightCameraBtn.addEventListener('click', () => this.startCamera('right'));
        if (this.ui.stopRightCameraBtn) this.ui.stopRightCameraBtn.addEventListener('click', () => this.stopCamera('right'));
        if (this.ui.startTopCameraBtn) this.ui.startTopCameraBtn.addEventListener('click', () => this.startCamera('top'));
        if (this.ui.stopTopCameraBtn) this.ui.stopTopCameraBtn.addEventListener('click', () => this.stopCamera('top'));
        
        // Start stereo vision button
        if (this.ui.startStereoVisionBtn) {
            this.ui.startStereoVisionBtn.addEventListener('click', () => {
                this.enableStereoVision();
            });
        }
        
        // Take snapshots button
        if (this.ui.takeSnapshotsBtn) {
            this.ui.takeSnapshotsBtn.addEventListener('click', () => {
                this.takeAllCameraSnapshots();
            });
        }
        
        // Stop all cameras button
        if (this.ui.stopAllCamerasBtn) {
            this.ui.stopAllCamerasBtn.addEventListener('click', () => {
                this.stopAllCameras();
                this.showNotification('All cameras stopped', 'info');
            });
        }
        
        // Resolution change handler
        if (this.ui.resolutionSelector) {
            this.ui.resolutionSelector.addEventListener('change', () => {
                // Update resolution for all active cameras
                Object.keys(this.activeCameras).forEach(cameraId => {
                    if (this.activeCameras[cameraId].stream) {
                        this.restartCameraWithNewResolution(cameraId);
                    }
                });
            });
        }
        
        // Depth map color scheme change
        if (this.ui.depthMapColorScheme) {
            this.ui.depthMapColorScheme.addEventListener('change', () => {
                // Update color scheme for existing depth maps
                this.updateDepthMapColorScheme();
            });
        }
    }
    
    // Restart camera with new resolution
    restartCameraWithNewResolution(cameraId) {
        if (this.activeCameras[cameraId] && this.activeCameras[cameraId].stream) {
            // Stop the current stream
            this.activeCameras[cameraId].stream.getTracks().forEach(track => track.stop());
            
            // Start new stream with updated resolution
            setTimeout(() => {
                this.startCameraByID(cameraId);
            }, 100);
        }
    }
    
    // Initialize WebSocket connections
    initializeWebSockets() {
        // Check if WebSocket is supported
        if ('WebSocket' in window) {
            try {
                // Create WebSocket connection
                this.socket = new WebSocket(`ws://${window.location.host}/ws/camera`);
                
                // Connection opened
                this.socket.addEventListener('open', (event) => {
                    console.log('Camera WebSocket connection established');
                });
                
                // Listen for messages
                this.socket.addEventListener('message', (event) => {
                    this.handleWebSocketMessage(event.data);
                });
                
                // Handle errors
                this.socket.addEventListener('error', (error) => {
                    console.error('Camera WebSocket error:', error);
                });
                
                // Handle close
                this.socket.addEventListener('close', (event) => {
                    console.log('Camera WebSocket connection closed');
                    // Try to reconnect after a delay
                    setTimeout(() => {
                        this.initializeWebSockets();
                    }, 5000);
                });
            } catch (error) {
                console.error('Failed to initialize WebSocket:', error);
                // Fallback to HTTP polling
                console.log('Falling back to browser camera access for local development');
            }
        } else {
            console.log('WebSocket is not supported. Using browser camera access for local development');
        }
    }
    
    // Handle WebSocket messages
    handleWebSocketMessage(data) {
        try {
            const message = JSON.parse(data);
            
            switch (message.type) {
                case 'camera_frame':
                    this.updateCameraFrame(message.camera_id, message.frame_data);
                    break;
                case 'depth_map':
                    this.updateDepthMap(message.pair_id, message.depth_data);
                    break;
                case 'camera_status':
                    this.updateCameraStatus(message.camera_id, message.status);
                    break;
                case 'stereo_status':
                    this.updateStereoStatus(message.pair_id, message.status);
                    break;
                default:
                    console.log('Unknown WebSocket message type:', message.type);
            }
        } catch (error) {
            console.error('Error handling WebSocket message:', error);
        }
    }
    
    // List available cameras
    listCameras() {
        // Use browser's media devices API to list cameras for local development
        navigator.mediaDevices.enumerateDevices()
            .then(devices => {
                const videoDevices = devices.filter(device => device.kind === 'videoinput');
                
                // Clear existing options
                if (this.ui.leftCameraSelector) this.ui.leftCameraSelector.innerHTML = '<option value="default">Select Camera</option>';
                if (this.ui.rightCameraSelector) this.ui.rightCameraSelector.innerHTML = '<option value="default">Select Camera</option>';
                if (this.ui.topCameraSelector) this.ui.topCameraSelector.innerHTML = '<option value="default">Select Camera (Optional)</option>';
                
                // Add new options
                videoDevices.forEach(device => {
                    const option = document.createElement('option');
                    option.value = device.deviceId;
                    option.textContent = device.label || `Camera ${videoDevices.indexOf(device) + 1}`;
                    
                    if (this.ui.leftCameraSelector) this.ui.leftCameraSelector.appendChild(option.cloneNode(true));
                    if (this.ui.rightCameraSelector) this.ui.rightCameraSelector.appendChild(option.cloneNode(true));
                    if (this.ui.topCameraSelector) this.ui.topCameraSelector.appendChild(option.cloneNode(true));
                });
                
                this.showNotification(`Found ${videoDevices.length} cameras`, 'success');
            })
            .catch(error => {
                console.error('Error listing cameras:', error);
                this.showNotification('Failed to list cameras: ' + error.message, 'error');
            });
        
        // Also call the API in case there are additional cameras available through backend
        return fetch(this.apiEndpoints.listCameras)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success' && data.cameras && data.cameras.length > 0) {
                    this.updateCameraSelectors(data.cameras);
                }
                return data;
            })
            .catch(error => console.error('Error listing cameras via API:', error));
    }
    
    // Update camera selectors with available cameras from API
    updateCameraSelectors(cameras) {
        const selectors = [this.ui.leftCameraSelector, this.ui.rightCameraSelector, this.ui.topCameraSelector];
        
        selectors.forEach(selector => {
            if (selector) {
                // Add API cameras to existing options
                cameras.forEach(camera => {
                    const option = document.createElement('option');
                    option.value = `api_${camera.id}`;
                    option.textContent = `API: ${camera.name} (${camera.width}x${camera.height} @ ${camera.fps} FPS)`;
                    selector.appendChild(option);
                });
            }
        });
    }
    
    // Start camera by position (left, right, top)
    startCamera(position) {
        // Get the appropriate UI elements based on camera position
        let selector, statusIndicator, videoElement, cameraName;
        
        switch (position) {
            case 'left':
                selector = this.ui.leftCameraSelector;
                statusIndicator = this.ui.leftCameraStatus;
                videoElement = this.ui.leftCameraVideo;
                cameraName = 'Left Camera';
                break;
            case 'right':
                selector = this.ui.rightCameraSelector;
                statusIndicator = this.ui.rightCameraStatus;
                videoElement = this.ui.rightCameraVideo;
                cameraName = 'Right Camera';
                break;
            case 'top':
                selector = this.ui.topCameraSelector;
                statusIndicator = this.ui.topCameraStatus;
                videoElement = this.ui.topCameraVideo;
                cameraName = 'Top Camera';
                break;
            default:
                throw new Error(`Invalid camera position: ${position}`);
        }
        
        // Check if selector and other elements exist
        if (!selector || !statusIndicator || !videoElement) {
            console.error(`UI elements for ${position} camera not found`);
            return;
        }
        
        const cameraId = selector.value;
        
        // Check if a camera is selected
        if (cameraId === 'default') {
            this.showNotification('Please select a camera first', 'info');
            return;
        }
        
        // Check if camera is already active
        if (this.activeCameras[cameraId]) {
            this.showNotification(`${cameraName} is already active`, 'info');
            return;
        }
        
        // Show connecting status
        this.updateCameraStatus(statusIndicator, 'connecting');
        
        // Prepare camera settings
        const settings = {
            resolution: this.ui.resolutionSelector ? this.ui.resolutionSelector.value : '640x480',
            framerate: 30
        };
        
        // Check if it's an API camera or local camera
        if (cameraId.startsWith('api_')) {
            // Handle API camera
            const apiCameraId = cameraId.replace('api_', '');
            this.startAPICamera(apiCameraId, position, statusIndicator, videoElement, cameraName, settings);
        } else {
            // Handle local camera using browser's media devices API
            this.startLocalCamera(cameraId, position, statusIndicator, videoElement, cameraName, settings);
        }
    }
    
    // Start local camera using browser's media devices API
    startLocalCamera(cameraId, position, statusIndicator, videoElement, cameraName, settings) {
        // Get selected resolution
        let resolution = '1280x720'; // Default
        if (this.ui.resolutionSelector) {
            resolution = this.ui.resolutionSelector.value;
        }
        
        const constraints = {
            video: {
                deviceId: cameraId,
                ...this.cameraConstraints[resolution]
            }
        };
        
        navigator.mediaDevices.getUserMedia(constraints)
            .then(stream => {
                // Store the stream
                this.activeCameras[cameraId] = {
                    stream: stream,
                    position: position,
                    statusIndicator: statusIndicator,
                    videoElement: videoElement
                };
                
                // Assign stream to the video element
                videoElement.srcObject = stream;
                this.updateCameraStatus(statusIndicator, 'active');
                
                // Start frame rate counter
                this.startFrameRateCounter(cameraId);
                
                this.showNotification(`${cameraName} started successfully`, 'success');
            })
            .catch(error => {
                console.error(`Error starting camera ${cameraId}:`, error);
                this.showNotification('Failed to start camera: ' + error.message, 'error');
                this.updateCameraStatus(statusIndicator, 'error');
            });
    }
    
    // Start camera by ID (for internal use)
    startCameraByID(cameraId) {
        // Find the camera position based on ID
        const cameraInfo = this.activeCameras[cameraId];
        if (cameraInfo) {
            this.startCamera(cameraInfo.position);
        }
    }
    
    // Start API camera
    startAPICamera(apiCameraId, position, statusIndicator, videoElement, cameraName, settings) {
        // Start camera via API
        fetch(`${this.apiEndpoints.startCamera}${apiCameraId}/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Update status indicator
                this.updateCameraStatus(statusIndicator, 'active');
                this.showNotification(`${cameraName} started successfully`, 'success');
                
                // Store active camera information
                this.activeCameras[`api_${apiCameraId}`] = {
                    position: position,
                    statusIndicator: statusIndicator,
                    videoElement: videoElement,
                    settings: settings
                };
                
                // Start frame rate counter
                this.startFrameRateCounter(`api_${apiCameraId}`);
                
                // Start receiving video stream
                if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                    // Send start stream command via WebSocket
                    this.socket.send(JSON.stringify({
                        type: 'start_stream',
                        camera_id: apiCameraId
                    }));
                } else {
                    // Fallback to HTTP polling
                    this.startHttpPolling(apiCameraId, videoElement);
                }
            } else {
                this.updateCameraStatus(statusIndicator, 'error');
                this.showNotification(`Failed to start ${cameraName}: ${data.message}`, 'error');
            }
        })
        .catch(error => {
            console.error(`Error starting ${cameraName}:`, error);
            this.updateCameraStatus(statusIndicator, 'error');
            this.showNotification(`Error starting ${cameraName}: ${error.message}`, 'error');
        });
    }
    
    // Update camera status indicator
    updateCameraStatus(indicator, status) {
        if (!indicator) return;
        
        // Remove all status classes
        indicator.className = 'camera-status-indicator';
        
        // Add the new status class
        indicator.classList.add(status);
        
        // Update tooltip text
        const statusTexts = {
            inactive: 'Camera Inactive',
            active: 'Camera Active',
            connecting: 'Connecting...',
            stopping: 'Stopping...',
            error: 'Error'
        };
        
        indicator.title = statusTexts[status] || status;
    }
    
    // Stop a camera stream by position
    stopCamera(position) {
        // Get the appropriate UI elements based on camera position
        let selector, statusIndicator, videoElement, cameraName;
        
        switch (position) {
            case 'left':
                selector = this.ui.leftCameraSelector;
                statusIndicator = this.ui.leftCameraStatus;
                videoElement = this.ui.leftCameraVideo;
                cameraName = 'Left Camera';
                break;
            case 'right':
                selector = this.ui.rightCameraSelector;
                statusIndicator = this.ui.rightCameraStatus;
                videoElement = this.ui.rightCameraVideo;
                cameraName = 'Right Camera';
                break;
            case 'top':
                selector = this.ui.topCameraSelector;
                statusIndicator = this.ui.topCameraStatus;
                videoElement = this.ui.topCameraVideo;
                cameraName = 'Top Camera';
                break;
            default:
                throw new Error(`Invalid camera position: ${position}`);
        }
        
        // Check if selector and other elements exist
        if (!selector || !statusIndicator || !videoElement) {
            console.error(`UI elements for ${position} camera not found`);
            return;
        }
        
        const cameraId = selector.value;
        
        // Check if a camera is selected
        if (cameraId === 'default') {
            this.showNotification('No camera selected', 'info');
            return;
        }
        
        // Check if camera is active
        if (!this.activeCameras[cameraId]) {
            this.showNotification(`${cameraName} is not active`, 'info');
            return;
        }
        
        // Show stopping status
        this.updateCameraStatus(statusIndicator, 'stopping');
        
        // Check if it's an API camera or local camera
        if (cameraId.startsWith('api_')) {
            // Handle API camera
            const apiCameraId = cameraId.replace('api_', '');
            this.stopAPICamera(apiCameraId, cameraId, statusIndicator, videoElement);
        } else {
            // Handle local camera
            this.stopLocalCamera(cameraId, statusIndicator, videoElement);
        }
    }
    
    // Stop local camera
    stopLocalCamera(cameraId, statusIndicator, videoElement) {
        // Stop the local stream
        if (this.activeCameras[cameraId] && this.activeCameras[cameraId].stream) {
            this.activeCameras[cameraId].stream.getTracks().forEach(track => track.stop());
        }
        
        // Update UI
        videoElement.srcObject = null;
        this.updateCameraStatus(statusIndicator, 'inactive');
        
        // Stop frame rate counter
        this.stopFrameRateCounter(cameraId);
        
        // Remove from active cameras
        delete this.activeCameras[cameraId];
    }
    
    // Stop API camera
    stopAPICamera(apiCameraId, cameraId, statusIndicator, videoElement) {
        // Stop camera via API
        fetch(`${this.apiEndpoints.stopCamera}${apiCameraId}/stop`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Update status indicator
                this.updateCameraStatus(statusIndicator, 'inactive');
                
                // Stop frame rate counter
                this.stopFrameRateCounter(cameraId);
                
                // Clear video element
                videoElement.src = '';
                
                // Remove from active cameras
                delete this.activeCameras[cameraId];
                
                // If using WebSocket, send stop stream command
                if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                    this.socket.send(JSON.stringify({
                        type: 'stop_stream',
                        camera_id: apiCameraId
                    }));
                }
            } else {
                this.updateCameraStatus(statusIndicator, 'error');
                this.showNotification(`Failed to stop camera: ${data.message}`, 'error');
            }
        })
        .catch(error => {
            console.error('Error stopping camera via API:', error);
            this.updateCameraStatus(statusIndicator, 'error');
        });
    }
    
    // Stop all camera streams
    stopAllCameras() {
        // Show stopping status for all cameras
        if (this.ui.leftCameraStatus) this.updateCameraStatus(this.ui.leftCameraStatus, 'stopping');
        if (this.ui.rightCameraStatus) this.updateCameraStatus(this.ui.rightCameraStatus, 'stopping');
        if (this.ui.topCameraStatus) this.updateCameraStatus(this.ui.topCameraStatus, 'stopping');
        if (this.ui.depthMapStatus) this.updateCameraStatus(this.ui.depthMapStatus, 'stopping');
        
        // Stop all local streams
        Object.keys(this.activeCameras).forEach(cameraId => {
            const camera = this.activeCameras[cameraId];
            if (camera.stream) {
                // Local camera
                camera.stream.getTracks().forEach(track => track.stop());
                if (camera.videoElement) {
                    camera.videoElement.srcObject = null;
                }
            } else {
                // API camera
                if (camera.videoElement) {
                    camera.videoElement.src = '';
                }
                // API stop handled separately
            }
        });
        
        // Clear depth map interval
        if (this.depthMapInterval) {
            clearInterval(this.depthMapInterval);
            this.depthMapInterval = null;
        }
        
        // Clear depth map canvas
        if (this.ui.depthMapCanvas) {
            const ctx = this.ui.depthMapCanvas.getContext('2d');
            ctx.clearRect(0, 0, this.ui.depthMapCanvas.width, this.ui.depthMapCanvas.height);
        }
        
        this.updateCameraStatus(this.ui.depthMapStatus, 'inactive');
        
        // Stop all frame rate counters
        Object.keys(this.frameCounters).forEach(cameraId => {
            if (cameraId.endsWith('_interval')) {
                clearInterval(this.frameCounters[cameraId]);
            }
        });
        
        // Clear active cameras and depth streams
        this.activeCameras = {};
        this.activeDepthStreams = {};
        this.stereoPairs = {};
        this.frameCounters = {};
        this.frameRates = {};
        
        // If using WebSocket, send stop all streams command
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.send(JSON.stringify({
                type: 'stop_all_streams'
            }));
        }
        
        // Also call the API
        fetch(this.apiEndpoints.stopAllCameras, {
            method: 'DELETE'
        })
        .catch(error => console.error('Error stopping all cameras via API:', error));
    }
    
    // Start frame rate counter
    startFrameRateCounter(cameraId) {
        // Initialize counters
        this.frameCounters[cameraId] = 0;
        this.frameRates[cameraId] = 0;
        
        // Set up interval to calculate frame rate
        this.frameCounters[cameraId + '_interval'] = setInterval(() => {
            // Calculate frames per second
            this.frameRates[cameraId] = this.frameCounters[cameraId];
            
            // Reset counter
            this.frameCounters[cameraId] = 0;
        }, 1000); // Update every second
    }
    
    // Stop frame rate counter
    stopFrameRateCounter(cameraId) {
        // Clear interval if it exists
        if (this.frameCounters[cameraId + '_interval']) {
            clearInterval(this.frameCounters[cameraId + '_interval']);
            delete this.frameCounters[cameraId + '_interval'];
        }
        
        // Reset counters
        delete this.frameCounters[cameraId];
        delete this.frameRates[cameraId];
    }
    
    // Show notification
    showNotification(message, type = 'info') {
        // Simple toast notification implementation
        const toast = document.createElement('div');
        toast.className = `position-fixed bottom-3 right-3 p-3 rounded shadow-lg z-50 transition-all duration-300 bg-${type === 'error' ? 'danger' : type === 'success' ? 'success' : 'info'} text-white`;
        toast.textContent = message;
        document.body.appendChild(toast);
        
        // Fade in
        setTimeout(() => toast.classList.add('opacity-100'), 10);
        
        // Remove after 3 seconds
        setTimeout(() => {
            toast.classList.add('opacity-0');
            setTimeout(() => document.body.removeChild(toast), 300);
        }, 3000);
    }
    
    // Get camera settings
    getCameraSettings(cameraId) {
        // Implementation to get camera settings
        return fetch(`${this.apiEndpoints.getSettings}${cameraId}/settings`)
            .then(response => response.json())
            .catch(error => console.error(`Error getting settings for camera:`, error));
    }
    
    // Update camera settings
    updateCameraSettings(cameraId, settings) {
        // Implementation to update camera settings
        return fetch(`${this.apiEndpoints.updateSettings}${cameraId}/settings`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        })
        .then(response => response.json())
        .catch(error => console.error(`Error updating settings for camera:`, error));
    }
    
    // Enable stereo vision
    enableStereoVision() {
        // Get selected cameras
        const leftCameraId = this.ui.leftCameraSelector ? this.ui.leftCameraSelector.value : 'default';
        const rightCameraId = this.ui.rightCameraSelector ? this.ui.rightCameraSelector.value : 'default';
        
        // Check if both cameras are selected
        if (leftCameraId === 'default' || rightCameraId === 'default') {
            this.showNotification('Please select both left and right cameras for stereo vision', 'error');
            return;
        }
        
        // First start both cameras if not already started
        if (!this.activeCameras[leftCameraId]) {
            // Find position of left camera
            if (this.ui.leftCameraSelector.value === leftCameraId) {
                this.startCamera('left');
            }
        }
        
        if (!this.activeCameras[rightCameraId]) {
            // Find position of right camera
            if (this.ui.rightCameraSelector.value === rightCameraId) {
                this.startCamera('right');
            }
        }
        
        // Create stereo pair data
        const pairData = {
            left_camera_id: leftCameraId.startsWith('api_') ? leftCameraId.replace('api_', '') : leftCameraId,
            right_camera_id: rightCameraId.startsWith('api_') ? rightCameraId.replace('api_', '') : rightCameraId
        };
        
        // Update UI status
        this.updateCameraStatus(this.ui.depthMapStatus, 'connecting');
        
        const pairId = 'main_stereo_pair';
        
        // Create stereo pair via API
        fetch(this.apiEndpoints.createStereoPair, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(pairData)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Enable stereo vision
                return fetch(`${this.apiEndpoints.enableStereoVision}${pairId}/enable`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
            } else {
                throw new Error(data.message || 'Failed to create stereo pair');
            }
        })
        .then(() => {
            // Update UI status
            this.updateCameraStatus(this.ui.depthMapStatus, 'active');
            
            // Start generating depth map (fallback if API doesn't provide it)
            setTimeout(() => {
                if (!this.activeDepthStreams[pairId]) {
                    this.startSimulatedDepthMap();
                }
            }, 2000);
            
            // Store stereo pair information
            this.stereoPairs[pairId] = pairData;
            
            this.showNotification('Stereo vision enabled', 'success');
            
            // If using WebSocket, send start depth stream command
            if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                this.socket.send(JSON.stringify({
                    type: 'start_depth_stream',
                    pair_id: pairId
                }));
            } else {
                // Fallback to HTTP polling for depth data
                this.startDepthHttpPolling(pairId);
            }
        })
        .catch(error => {
            console.error('Error creating stereo pair:', error);
            this.updateCameraStatus(this.ui.depthMapStatus, 'error');
            this.showNotification('Failed to enable stereo vision: ' + error.message, 'error');
            
            // If API fails, start local simulated depth map
            this.startSimulatedDepthMap();
            this.updateCameraStatus(this.ui.depthMapStatus, 'active');
        });
    }
    
    // Start generating simulated depth map
    startSimulatedDepthMap() {
        if (this.ui.depthMapCanvas && !this.depthMapInterval) {
            const canvas = this.ui.depthMapCanvas;
            const ctx = canvas.getContext('2d');
            
            // Set canvas size
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
            
            // Create interval to update depth map
            this.depthMapInterval = setInterval(() => {
                this.generateSimulatedDepthMap(ctx, canvas.width, canvas.height);
            }, 100); // Update every 100ms
        }
    }
    
    // Generate simulated depth map
    generateSimulatedDepthMap(ctx, width, height) {
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        // Generate random depth values for demonstration
        const imageData = ctx.createImageData(width, height);
        const data = imageData.data;
        
        for (let i = 0; i < data.length; i += 4) {
            // Create a wave pattern for depth visualization
            const x = (i / 4) % width;
            const y = Math.floor((i / 4) / width);
            
            // Generate a simple wave pattern based on time
            const time = Date.now() / 5000;
            const wave = Math.sin(x * 0.02 + time) * Math.cos(y * 0.02 + time);
            
            // Map wave to color (blue = far, red = near)
            const intensity = Math.floor((wave + 1) * 128);
            data[i] = intensity; // R
            data[i + 1] = 0;     // G
            data[i + 2] = 255 - intensity; // B
            data[i + 3] = 255;   // A
        }
        
        // Draw depth map
        ctx.putImageData(imageData, 0, 0);
        
        // Add some labels for better understanding
        ctx.fillStyle = 'white';
        ctx.font = '12px Arial';
        ctx.fillText('Simulated Depth Map (Red = Near, Blue = Far)', 10, 20);
    }
    
    // Disable stereo vision
    disableStereoVision(pairId = 'main_stereo_pair') {
        // Clear depth map interval
        if (this.depthMapInterval) {
            clearInterval(this.depthMapInterval);
            this.depthMapInterval = null;
        }
        
        // Clear depth map canvas
        if (this.ui.depthMapCanvas) {
            const ctx = this.ui.depthMapCanvas.getContext('2d');
            ctx.clearRect(0, 0, this.ui.depthMapCanvas.width, this.ui.depthMapCanvas.height);
        }
        
        this.updateCameraStatus(this.ui.depthMapStatus, 'inactive');
        
        this.showNotification('Stereo vision disabled', 'info');
        
        // Also call the API
        return fetch(`${this.apiEndpoints.disableStereoVision}${pairId}/disable`, {
            method: 'POST'
        })
        .then(response => response.json())
        .catch(error => console.error(`Error disabling stereo vision via API:`, error));
    }
    
    // Stop stereo vision
    stopStereoVision(pairId = 'main_stereo_pair') {
        return this.disableStereoVision(pairId);
    }
    
    // Take snapshots from all active cameras
    takeAllCameraSnapshots() {
        let snapshotCount = 0;
        
        // Check left camera
        if (this.ui.leftCameraVideo && this.ui.leftCameraVideo.srcObject) {
            this.takeSnapshotFromVideo(this.ui.leftCameraVideo, 'left_camera');
            snapshotCount++;
        }
        
        // Check right camera
        if (this.ui.rightCameraVideo && this.ui.rightCameraVideo.srcObject) {
            this.takeSnapshotFromVideo(this.ui.rightCameraVideo, 'right_camera');
            snapshotCount++;
        }
        
        // Check top camera
        if (this.ui.topCameraVideo && this.ui.topCameraVideo.srcObject) {
            this.takeSnapshotFromVideo(this.ui.topCameraVideo, 'top_camera');
            snapshotCount++;
        }
        
        // Check if any snapshots were taken
        if (snapshotCount > 0) {
            this.showNotification(`Successfully captured ${snapshotCount} snapshots`, 'success');
        } else {
            this.showNotification('No active cameras to capture snapshots from', 'warning');
        }
        
        // Also take snapshots from API cameras
        Object.keys(this.activeCameras).forEach(cameraId => {
            if (cameraId.startsWith('api_')) {
                const apiCameraId = cameraId.replace('api_', '');
                this.takeSnapshot(apiCameraId);
            }
        });
    }
    
    // Take snapshot from a video element
    takeSnapshotFromVideo(videoElement, cameraName) {
        // Create canvas for snapshot
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        
        // Draw current frame to canvas
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        
        // Convert to data URL
        const dataURL = canvas.toDataURL('image/png');
        
        // Create download link
        const link = document.createElement('a');
        link.href = dataURL;
        link.download = `${cameraName}_snapshot_${new Date().toISOString().replace(/[:.]/g, '-')}.png`;
        
        // Trigger download
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        // Also send snapshot to backend
        this.sendSnapshotToBackend(dataURL, cameraName);
    }
    
    // Send snapshot to backend
    sendSnapshotToBackend(dataURL, cameraName) {
        // Extract base64 data without prefix
        const base64Data = dataURL.replace(/^data:image\/png;base64,/, '');
        
        fetch('/api/cameras/snapshot', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                cameraName: cameraName,
                imageData: base64Data
            })
        })
        .catch(error => console.error('Error sending snapshot to backend:', error));
    }
    
    // Take a snapshot from a camera via API
    takeSnapshot(cameraId) {
        return fetch(`${this.apiEndpoints.takeSnapshot}${cameraId}/snapshot`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    this.showNotification('Snapshot taken successfully', 'success');
                    return data;
                } else {
                    throw new Error(data.message || 'Failed to take snapshot');
                }
            })
            .catch(error => {
                console.error(`Error taking snapshot for camera ${cameraId}:`, error);
                throw error;
            });
    }
    
    // Get all snapshots for a camera
    getCameraSnapshots(cameraId) {
        // Implementation to get all snapshots for a camera
        return fetch(`${this.apiEndpoints.takeSnapshot}${cameraId}/snapshots`)
            .then(response => response.json())
            .catch(error => {
                console.error(`Error getting snapshots for camera:`, error);
                return { status: 'error', message: error.message };
            });
    }
    
    // Get active camera inputs
    getActiveCameraInputs() {
        return fetch(this.apiEndpoints.stopAllCameras)
            .then(response => response.json())
            .catch(error => {
                console.error('Error getting active camera inputs:', error);
                return { status: 'error', message: error.message };
            });
    }
    
    // Start multiple cameras at once
    startMultipleCameras(cameraIds) {
        return fetch('/api/cameras/multiple/start', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ cameraIds: cameraIds })
        })
        .then(response => response.json())
        .catch(error => {
            console.error('Error starting multiple cameras:', error);
            return { status: 'error', message: error.message };
        });
    }
    
    // Get depth data for a stereo pair
    getDepthData(pairId) {
        return fetch(`${this.apiEndpoints.getDepthData}${pairId}`)
            .then(response => response.json())
            .catch(error => {
                console.error(`Error getting depth data for pair:`, error);
                return { status: 'error', message: error.message };
            });
    }
    
    // Update depth map display
    updateDepthMap(pairId, depthData) {
        if (!this.ui.depthMapCanvas) return;
        
        const ctx = this.ui.depthMapCanvas.getContext('2d');
        const canvas = this.ui.depthMapCanvas;
        
        // Set canvas size
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;
        
        try {
            // If depthData is provided, use it
            if (depthData && depthData.imageData) {
                // Create image from depth data
                const imageData = ctx.createImageData(canvas.width, canvas.height);
                const data = imageData.data;
                
                // Process depth data into image data
                for (let i = 0; i < data.length; i += 4) {
                    const depthIndex = Math.floor(i / 4);
                    const depthValue = depthData.imageData[depthIndex] || 0;
                    
                    // Map depth value to color
                    data[i] = depthValue;     // R
                    data[i + 1] = 0;         // G
                    data[i + 2] = 255 - depthValue; // B
                    data[i + 3] = 255;       // A
                }
                
                ctx.putImageData(imageData, 0, 0);
            } else if (depthData && typeof depthData === 'string') {
                try {
                    // Handle base64 depth data from WebSocket
                    const binaryData = new Uint8Array(atob(depthData).split('').map(char => char.charCodeAt(0)));
                    const blob = new Blob([binaryData], { type: 'image/png' });
                    const imageUrl = URL.createObjectURL(blob);
                    
                    const img = new Image();
                    img.onload = () => {
                        // Update canvas dimensions
                        canvas.width = img.width;
                        canvas.height = img.height;
                        
                        // Draw the depth map
                        ctx.drawImage(img, 0, 0);
                        
                        // Apply color scheme if needed
                        this.applyDepthMapColorScheme();
                        
                        // Revoke the URL after use
                        URL.revokeObjectURL(imageUrl);
                    };
                    img.onerror = () => {
                        console.error('Failed to load depth map image');
                        URL.revokeObjectURL(imageUrl);
                    };
                    img.src = imageUrl;
                } catch (error) {
                    console.error('Error parsing depth data:', error);
                    // Fallback to simulated depth map
                    this.generateSimulatedDepthMap(ctx, canvas.width, canvas.height);
                }
            } else {
                // Fallback to simulated depth map
                this.generateSimulatedDepthMap(ctx, canvas.width, canvas.height);
            }
        } catch (error) {
            console.error('Error updating depth map:', error);
            // Fallback to simulated depth map
            this.generateSimulatedDepthMap(ctx, canvas.width, canvas.height);
        }
    }
    
    // Start HTTP polling for camera frames
    startHttpPolling(cameraId, videoElement) {
        // Store interval ID on video element for later cleanup
        videoElement.frameInterval = setInterval(() => {
            fetch(`${this.apiEndpoints.getFrame}${cameraId}/frame`)
                .then(response => {
                    if (response.ok) {
                        return response.blob();
                    }
                    throw new Error('Failed to get frame');
                })
                .then(blob => {
                    // Update frame counter
                    if (this.frameCounters[`api_${cameraId}`]) {
                        this.frameCounters[`api_${cameraId}`]++;
                    }
                    
                    // Create a URL for the blob
                    const imageUrl = URL.createObjectURL(blob);
                    
                    // Update video element source
                    videoElement.src = imageUrl;
                    
                    // Revoke the URL after a short delay to prevent memory leaks
                    setTimeout(() => {
                        URL.revokeObjectURL(imageUrl);
                    }, 100);
                })
                .catch(error => {
                    console.error(`Error fetching frame for camera ${cameraId}:`, error);
                });
        }, 100); // Poll every 100ms (10 FPS)
    }
    
    // Update camera frame from WebSocket
    updateCameraFrame(cameraId, frameData) {
        const camera = this.activeCameras[`api_${cameraId}`];
        if (!camera) return;
        
        // Update frame counter
        if (this.frameCounters[`api_${cameraId}`]) {
            this.frameCounters[`api_${cameraId}`]++;
        }
        
        // Convert base64 frame data to blob
        const binaryString = atob(frameData);
        const length = binaryString.length;
        const bytes = new Uint8Array(length);
        for (let i = 0; i < length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        const blob = new Blob([bytes], { type: 'image/jpeg' });
        
        // Create a URL for the blob
        const imageUrl = URL.createObjectURL(blob);
        
        // Update video element source
        camera.videoElement.src = imageUrl;
        
        // Revoke the URL after a short delay to prevent memory leaks
        setTimeout(() => {
            URL.revokeObjectURL(imageUrl);
        }, 100);
    }
    
    // Start HTTP polling for depth data
    startDepthHttpPolling(pairId) {
        if (!this.ui.depthMapCanvas) return;
        
        // Store interval ID on canvas element for later cleanup
        this.ui.depthMapCanvas.depthInterval = setInterval(() => {
            fetch(`${this.apiEndpoints.getDepthData}${pairId}/depth`)
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success' && data.depth_data) {
                        this.updateDepthMap(pairId, data.depth_data);
                    }
                })
                .catch(error => {
                    console.error(`Error fetching depth data for pair ${pairId}:`, error);
                });
        }, 200); // Poll every 200ms (5 FPS)
    }
    
    // Apply color scheme to depth map
    applyDepthMapColorScheme() {
        if (!this.ui.depthMapCanvas || !this.ui.depthMapColorScheme) return;
        
        const ctx = this.ui.depthMapCanvas.getContext('2d');
        const imageData = ctx.getImageData(0, 0, this.ui.depthMapCanvas.width, this.ui.depthMapCanvas.height);
        const data = imageData.data;
        
        const colorScheme = this.ui.depthMapColorScheme.value;
        
        // Apply color scheme
        for (let i = 0; i < data.length; i += 4) {
            const gray = data[i]; // Assuming grayscale image
            let r, g, b;
            
            switch (colorScheme) {
                case 'hot':
                    // Hot color map
                    [r, g, b] = this.grayToHot(gray);
                    break;
                case 'rainbow':
                    // Rainbow color map
                    [r, g, b] = this.grayToRainbow(gray);
                    break;
                default:
                    // Grayscale (no change)
                    r = g = b = gray;
            }
            
            data[i] = r;
            data[i + 1] = g;
            data[i + 2] = b;
            // Alpha remains unchanged
        }
        
        // Put the modified image data back
        ctx.putImageData(imageData, 0, 0);
    }
    
    // Update color scheme for all depth maps
    updateDepthMapColorScheme() {
        this.applyDepthMapColorScheme();
    }
    
    // Helper function to convert grayscale to hot color map
    grayToHot(gray) {
        const normalized = gray / 255;
        let r, g, b;
        
        if (normalized < 0.25) {
            r = 0;
            g = 0;
            b = 4 * normalized;
        } else if (normalized < 0.5) {
            r = 0;
            g = 4 * (normalized - 0.25);
            b = 1;
        } else if (normalized < 0.75) {
            r = 4 * (normalized - 0.5);
            g = 1;
            b = 1 - 4 * (normalized - 0.5);
        } else {
            r = 1;
            g = 1 - 4 * (normalized - 0.75);
            b = 0;
        }
        
        return [
            Math.round(r * 255),
            Math.round(g * 255),
            Math.round(b * 255)
        ];
    }
    
    // Helper function to convert grayscale to rainbow color map
    grayToRainbow(gray) {
        const normalized = gray / 255;
        let r, g, b;
        
        if (normalized < 0.2) {
            r = 0;
            g = 0;
            b = 0.5 + 2.5 * normalized;
        } else if (normalized < 0.4) {
            r = 0;
            g = 2.5 * (normalized - 0.2);
            b = 1;
        } else if (normalized < 0.6) {
            r = 2.5 * (normalized - 0.4);
            g = 1;
            b = 1 - 2.5 * (normalized - 0.4);
        } else if (normalized < 0.8) {
            r = 1;
            g = 1 - 2.5 * (normalized - 0.6);
            b = 0;
        } else {
            r = 1;
            g = 0;
            b = 0;
        }
        
        return [
            Math.round(r * 255),
            Math.round(g * 255),
            Math.round(b * 255)
        ];
    }
    
    // Update stereo status
    updateStereoStatus(pairId, status) {
        console.log(`Stereo pair ${pairId} status: ${status}`);
        if (status === 'active' && pairId === 'main_stereo_pair') {
            this.updateCameraStatus(this.ui.depthMapStatus, 'active');
        } else if (status === 'inactive' && pairId === 'main_stereo_pair') {
            this.updateCameraStatus(this.ui.depthMapStatus, 'inactive');
        }
    }
    
    // Create depth map color scheme selector
    createDepthMapColorSchemeSelector() {
        // Check if selector already exists
        if (this.ui.depthMapColorScheme) return;
        
        // Find the depth map container
        const depthMapContainer = this.ui.depthMapCanvas ? this.ui.depthMapCanvas.parentElement : null;
        if (!depthMapContainer) return;
        
        // Create selector element
        const selector = document.createElement('select');
        selector.id = 'depthMapColorScheme';
        selector.className = 'form-select mt-2';
        
        // Add options
        const options = [
            { value: 'grayscale', text: 'Grayscale' },
            { value: 'hot', text: 'Hot' },
            { value: 'rainbow', text: 'Rainbow' }
        ];
        
        options.forEach(option => {
            const opt = document.createElement('option');
            opt.value = option.value;
            opt.textContent = option.text;
            selector.appendChild(opt);
        });
        
        // Add label
        const label = document.createElement('label');
        label.htmlFor = 'depthMapColorScheme';
        label.className = 'form-label mt-2';
        label.textContent = 'Depth Map Color Scheme:';
        
        // Add to container
        depthMapContainer.appendChild(label);
        depthMapContainer.appendChild(selector);
        
        // Update UI reference
        this.ui.depthMapColorScheme = selector;
        
        // Add event listener
        selector.addEventListener('change', () => {
            this.updateDepthMapColorScheme();
        });
    }
    
    // Initialize the camera control system
    initialize() {
        // List available cameras
        this.listCameras();
        
        // Create depth map color scheme selector if not exists
        this.createDepthMapColorSchemeSelector();
    }
}

// Create a single instance of CameraControl
const cameraControl = new CameraControl();

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    cameraControl.initialize();
});

// Expose public methods for global access
window.listAllCameras = function() {
    return cameraControl.listCameras();
};

window.startStereoVision = function(leftCameraId, rightCameraId) {
    // If no IDs provided, use selected ones from UI
    if (!leftCameraId && cameraControl.ui.leftCameraSelector) {
        leftCameraId = cameraControl.ui.leftCameraSelector.value;
    }
    
    if (!rightCameraId && cameraControl.ui.rightCameraSelector) {
        rightCameraId = cameraControl.ui.rightCameraSelector.value;
    }
    
    if (leftCameraId && rightCameraId && leftCameraId !== 'default' && rightCameraId !== 'default') {
        // Store the selected IDs and enable stereo vision
        if (cameraControl.ui.leftCameraSelector) cameraControl.ui.leftCameraSelector.value = leftCameraId;
        if (cameraControl.ui.rightCameraSelector) cameraControl.ui.rightCameraSelector.value = rightCameraId;
        return cameraControl.enableStereoVision();
    } else {
        cameraControl.showNotification('Please select both left and right cameras', 'error');
        return Promise.reject('Cameras not selected');
    }
};

window.startStereoVisionSystem = function() {
    return cameraControl.enableStereoVision();
};

window.stopStereoVision = function(pairId) {
    return cameraControl.stopStereoVision(pairId);
};

window.startVisionSystem = function() {
    // Start all three cameras for the full vision system
    setTimeout(() => cameraControl.startCamera('left'), 100);
    setTimeout(() => cameraControl.startCamera('right'), 300);
    setTimeout(() => cameraControl.startCamera('top'), 500);
    
    // Enable stereo vision after a delay to ensure cameras are started
    setTimeout(() => cameraControl.enableStereoVision(), 1000);
};

window.stopAllCameras = function() {
    return cameraControl.stopAllCameras();
};

window.takeAllCameraSnapshots = function() {
    return cameraControl.takeAllCameraSnapshots();
};

window.toggleDepthMap = function() {
    if (cameraControl.ui.depthMapCanvas) {
        const canvas = cameraControl.ui.depthMapCanvas;
        canvas.style.display = canvas.style.display === 'none' ? 'block' : 'none';
    }
};

window.openCameraSettings = function(cameraId) {
    console.log(`Opening settings for camera ${cameraId}`);
    cameraControl.showNotification('Camera settings functionality is being implemented', 'info');
};
