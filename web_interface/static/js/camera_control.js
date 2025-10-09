<<<<<<< HEAD
=======
// Camera Control Module
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
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
<<<<<<< HEAD
            getDepthData: '/api/stereo/depth/'
=======
            getDepthData: '/api/stereo/pairs/'
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
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
        
<<<<<<< HEAD
        // Camera constraints
        this.cameraConstraints = {
            '640x480': { width: 640, height: 480 },
            '1280x720': { width: 1280, height: 720 },
            '1920x1080': { width: 1920, height: 1080 }
        };
        
=======
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        // Initialize UI elements
        this.initializeUI();
        // Initialize event listeners
        this.initializeEventListeners();
<<<<<<< HEAD
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
=======
        // Initialize WebSocket connections (if available)
        this.initializeWebSockets();
    }
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
    
    // Initialize UI elements
    initializeUI() {
        // Get references to UI elements
        this.ui = {
            // Camera selectors
            leftCameraSelector: document.getElementById('camera1Selector'),
            rightCameraSelector: document.getElementById('camera2Selector'),
            topCameraSelector: document.getElementById('camera3Selector'),
<<<<<<< HEAD
            resolutionSelector: document.getElementById('cameraResolution'),
=======
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            
            // Status indicators
            leftCameraStatus: document.getElementById('camera1Status'),
            rightCameraStatus: document.getElementById('camera2Status'),
<<<<<<< HEAD
=======
            topCameraStatus: document.getElementById('camera3Status'),
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            depthMapStatus: document.getElementById('depthMapStatus'),
            
            // Video elements
            leftCameraVideo: document.getElementById('camera1Video'),
            rightCameraVideo: document.getElementById('camera2Video'),
<<<<<<< HEAD
            depthMapCanvas: document.getElementById('depthMapCanvas'),
            
            // Buttons
            refreshCamerasBtn: document.getElementById('refreshCameras'),
            startStereoVisionBtn: document.getElementById('startStereoVision'),
            takeSnapshotsBtn: document.getElementById('takeSnapshots'),
            stopAllCamerasBtn: document.getElementById('stopAllCamerasBtn')
=======
            topCameraVideo: document.getElementById('camera3Video'),
            
            // Depth map canvas
            depthMapCanvas: document.getElementById('depthMapCanvas'),
            
            // Buttons
            startLeftCamera: document.getElementById('startCamera1'),
            stopLeftCamera: document.getElementById('stopCamera1'),
            startRightCamera: document.getElementById('startCamera2'),
            stopRightCamera: document.getElementById('stopCamera2'),
            startTopCamera: document.getElementById('startCamera3'),
            stopTopCamera: document.getElementById('stopCamera3'),
            startStereoVision: document.getElementById('startStereoVision'),
            stopAllCameras: document.getElementById('stopAllCamerasBtn'),
            takeSnapshots: document.getElementById('takeSnapshots'),
            refreshCameras: document.getElementById('refreshCameras'),
            
            // Resolution selector
            resolutionSelector: document.getElementById('cameraResolution'),
            
            // Depth map color scheme
            depthMapColorScheme: document.getElementById('depthMapColorScheme')
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
        };
    }
    
    // Initialize event listeners
    initializeEventListeners() {
<<<<<<< HEAD
        // Refresh cameras button
        if (this.ui.refreshCamerasBtn) {
            this.ui.refreshCamerasBtn.addEventListener('click', () => {
                this.listCameras();
                this.showNotification('Refreshing camera list...', 'info');
            });
        }
        
        // Start stereo vision button
        if (this.ui.startStereoVisionBtn) {
            this.ui.startStereoVisionBtn.addEventListener('click', () => {
                const leftCameraId = this.ui.leftCameraSelector ? this.ui.leftCameraSelector.value : null;
                const rightCameraId = this.ui.rightCameraSelector ? this.ui.rightCameraSelector.value : null;
                
                if (leftCameraId && rightCameraId && leftCameraId !== 'default' && rightCameraId !== 'default') {
                    this.startStereoVision(leftCameraId, rightCameraId);
                } else {
                    this.showNotification('Please select both left and right cameras', 'error');
                }
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
=======
        // Camera control buttons
        if (this.ui.startLeftCamera) this.ui.startLeftCamera.addEventListener('click', () => this.startCamera('left'));
        if (this.ui.stopLeftCamera) this.ui.stopLeftCamera.addEventListener('click', () => this.stopCamera('left'));
        if (this.ui.startRightCamera) this.ui.startRightCamera.addEventListener('click', () => this.startCamera('right'));
        if (this.ui.stopRightCamera) this.ui.stopRightCamera.addEventListener('click', () => this.stopCamera('right'));
        if (this.ui.startTopCamera) this.ui.startTopCamera.addEventListener('click', () => this.startCamera('top'));
        if (this.ui.stopTopCamera) this.ui.stopTopCamera.addEventListener('click', () => this.stopCamera('top'));
        
        // Stereo vision control
        if (this.ui.startStereoVision) this.ui.startStereoVision.addEventListener('click', () => this.enableStereoVision());
        
        // Global controls
        if (this.ui.stopAllCameras) this.ui.stopAllCameras.addEventListener('click', () => this.stopAllCameras());
        if (this.ui.takeSnapshots) this.ui.takeSnapshots.addEventListener('click', () => this.takeAllSnapshots());
        if (this.ui.refreshCameras) this.ui.refreshCameras.addEventListener('click', () => this.listCameras());
        
        // Depth map color scheme change
        if (this.ui.depthMapColorScheme) {
            this.ui.depthMapColorScheme.addEventListener('change', () => {
                // Update color scheme for existing depth maps
                this.updateDepthMapColorScheme();
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            });
        }
    }
    
<<<<<<< HEAD
    // Restart camera with new resolution
    restartCameraWithNewResolution(cameraId) {
        if (this.activeCameras[cameraId] && this.activeCameras[cameraId].stream) {
            // Stop the current stream
            this.activeCameras[cameraId].stream.getTracks().forEach(track => track.stop());
            
            // Start new stream with updated resolution
            setTimeout(() => {
                this.startCamera(cameraId);
            }, 100);
        }
    }
    
    // Initialize WebSocket connections
    initializeWebSockets() {
        // WebSocket initialization would go here
        // This would be used for real-time depth data updates
        // For now, we'll use a simulated approach with canvas
    }

    // Static methods for compatibility with existing code
    static listCameras() {
        return this.getInstance().listCameras();
    }

    static startCamera(cameraId) {
        return this.getInstance().startCamera(cameraId);
    }

    static stopCamera(cameraId) {
        return this.getInstance().stopCamera(cameraId);
    }

    static stopAllCameras() {
        return this.getInstance().stopAllCameras();
    }

    static getCameraSettings(cameraId) {
        return this.getInstance().getCameraSettings(cameraId);
    }

    static updateCameraSettings(cameraId, settings) {
        return this.getInstance().updateCameraSettings(cameraId, settings);
    }

    static enableStereoVision(leftCameraId, rightCameraId) {
        return this.getInstance().enableStereoVision(leftCameraId, rightCameraId);
    }

    static disableStereoVision(pairId) {
        return this.getInstance().disableStereoVision(pairId);
    }

    static getDepthData(pairId) {
        return this.getInstance().getDepthData(pairId);
    }

    static takeSnapshot(cameraId) {
        return this.getInstance().takeSnapshot(cameraId);
    }

    static getCameraSnapshots(cameraId) {
        return this.getInstance().getCameraSnapshots(cameraId);
    }

    static updateDepthMap(pairId, depthData) {
        return this.getInstance().updateDepthMap(pairId, depthData);
    }

    static getActiveCameraInputs() {
        return this.getInstance().getActiveCameraInputs();
    }

    static startMultipleCameras(cameraIds) {
        return this.getInstance().startMultipleCameras(cameraIds);
=======
    // Initialize WebSocket connections for real-time streaming
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
                console.log('Falling back to HTTP polling for camera streams');
            }
        } else {
            console.log('WebSocket is not supported. Using HTTP polling for camera streams');
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
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
    }
    
    // List available cameras
    listCameras() {
<<<<<<< HEAD
        // Use browser's media devices API to list cameras
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
            .catch(error => console.error('Error listing cameras via API:', error));
    }

    // Take snapshot from a specific camera
    takeSnapshot(cameraId) {
        // Implementation to take a snapshot from a specific camera
        return fetch(`${this.apiEndpoints.takeSnapshot}${cameraId}/snapshot`, {
            method: 'POST'
        })
        .then(response => response.json())
        .catch(error => {
            console.error(`Error taking snapshot for camera:`, error);
            return { status: 'error', message: error.message };
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
        if (this.ui.depthMapCanvas) {
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


    
    // Start a camera stream
    startCamera(cameraId) {
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
                this.activeCameras[cameraId] = { stream: stream };
                
                // Assign stream to the appropriate video element
                if (this.ui.leftCameraSelector && this.ui.leftCameraSelector.value === cameraId && this.ui.leftCameraVideo) {
                    this.ui.leftCameraVideo.srcObject = stream;
                    this.updateCameraStatus('leftCameraStatus', 'active');
                } else if (this.ui.rightCameraSelector && this.ui.rightCameraSelector.value === cameraId && this.ui.rightCameraVideo) {
                    this.ui.rightCameraVideo.srcObject = stream;
                    this.updateCameraStatus('rightCameraStatus', 'active');
                }
                
                // Start frame rate counter
                this.startFrameRateCounter(cameraId);
                
                this.showNotification(`Camera started successfully`, 'success');
            })
            .catch(error => {
                console.error(`Error starting camera ${cameraId}:`, error);
                this.showNotification('Failed to start camera: ' + error.message, 'error');
                this.updateCameraStatus('leftCameraStatus', 'error');
                this.updateCameraStatus('rightCameraStatus', 'error');
            });
        
        // Also call the API for backend integration
        return fetch(`${this.apiEndpoints.startCamera}${cameraId}/start`, {
            method: 'POST'
        })
        .then(response => response.json())
        .catch(error => console.error(`Error starting camera via API:`, error));
    }
    
    // Update camera status indicator
    updateCameraStatus(statusElement, status) {
        if (this.ui[statusElement]) {
            this.ui[statusElement].className = 'camera-status-indicator';
            
            switch (status) {
                case 'active':
                    this.ui[statusElement].classList.add('status-active');
                    break;
                case 'inactive':
                    this.ui[statusElement].classList.add('status-inactive');
                    break;
                case 'error':
                    this.ui[statusElement].classList.add('status-error');
                    break;
                default:
                    this.ui[statusElement].classList.add('status-inactive');
            }
        }
    }
    
    // Stop a camera stream
    stopCamera(cameraId) {
        // Stop the local stream
        if (this.activeCameras[cameraId] && this.activeCameras[cameraId].stream) {
            this.activeCameras[cameraId].stream.getTracks().forEach(track => track.stop());
            delete this.activeCameras[cameraId];
        }
        
        // Update UI
        if (this.ui.leftCameraSelector && this.ui.leftCameraSelector.value === cameraId && this.ui.leftCameraVideo) {
            this.ui.leftCameraVideo.srcObject = null;
            this.updateCameraStatus('leftCameraStatus', 'inactive');
        } else if (this.ui.rightCameraSelector && this.ui.rightCameraSelector.value === cameraId && this.ui.rightCameraVideo) {
            this.ui.rightCameraVideo.srcObject = null;
            this.updateCameraStatus('rightCameraStatus', 'inactive');
        }
        
        // Stop frame rate counter
        this.stopFrameRateCounter(cameraId);
        
        // Also call the API
        return fetch(`${this.apiEndpoints.stopCamera}${cameraId}/stop`, {
            method: 'POST'
        })
        .then(response => response.json())
        .catch(error => console.error(`Error stopping camera:`, error));
    }
    
    // Stop all camera streams
    stopAllCameras() {
        // Stop all local streams
        Object.keys(this.activeCameras).forEach(cameraId => {
            this.stopCamera(cameraId);
        });
        
        // Clear depth map
        if (this.ui.depthMapCanvas) {
            const ctx = this.ui.depthMapCanvas.getContext('2d');
            ctx.clearRect(0, 0, this.ui.depthMapCanvas.width, this.ui.depthMapCanvas.height);
        }
        
        this.updateCameraStatus('depthMapStatus', 'inactive');
        
        // Also call the API
        return fetch(this.apiEndpoints.stopAllCameras, {
            method: 'DELETE'
        })
        .then(response => response.json())
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
=======
        return fetch(this.apiEndpoints.listCameras)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success' && data.cameras) {
                    this.updateCameraSelectors(data.cameras);
                    return data.cameras;
                } else {
                    throw new Error(data.message || 'Failed to list cameras');
                }
            })
            .catch(error => {
                console.error('Error listing cameras:', error);
                this.showNotification('Failed to load available cameras', 'error');
                throw error;
            });
    }
    
    // Update camera selectors with available cameras
    updateCameraSelectors(cameras) {
        const selectors = [this.ui.leftCameraSelector, this.ui.rightCameraSelector, this.ui.topCameraSelector];
        
        selectors.forEach(selector => {
            if (selector) {
                // Clear existing options
                selector.innerHTML = '<option value="default">Select Camera</option>';
                
                // Add new options
                cameras.forEach(camera => {
                    const option = document.createElement('option');
                    option.value = camera.id;
                    option.textContent = `${camera.name} (${camera.width}x${camera.height} @ ${camera.fps} FPS)`;
                    selector.appendChild(option);
                });
            }
        });
    }
    
    // Start a camera stream
    startCamera(cameraPosition) {
        // Get the appropriate UI elements based on camera position
        let selector, statusIndicator, videoElement, cameraName;
        
        switch (cameraPosition) {
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
                throw new Error(`Invalid camera position: ${cameraPosition}`);
        }
        
        // Check if selector and other elements exist
        if (!selector || !statusIndicator || !videoElement) {
            console.error(`UI elements for ${cameraPosition} camera not found`);
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
        this.updateStatusIndicator(statusIndicator, 'connecting');
        
        // Prepare camera settings
        const settings = {
            resolution: this.ui.resolutionSelector ? this.ui.resolutionSelector.value : '640x480',
            framerate: 30
        };
        
        // Start camera via API
        fetch(`${this.apiEndpoints.startCamera}${cameraId}/start`, {
            method: 'POST',
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        })
        .then(response => response.json())
<<<<<<< HEAD
        .catch(error => console.error(`Error updating settings for camera:`, error));
    }
    
    // Enable stereo vision
    enableStereoVision(leftCameraId, rightCameraId) {
        // First start both cameras if not already started
        if (!this.activeCameras[leftCameraId]) {
            this.startCamera(leftCameraId);
        }
        
        if (!this.activeCameras[rightCameraId]) {
            this.startCamera(rightCameraId);
        }
        
        // Create stereo pair data
        const pairData = {
            leftCameraId: leftCameraId,
            rightCameraId: rightCameraId
        };
        
        // Update UI status
        this.updateCameraStatus('depthMapStatus', 'active');
        
        // Start generating simulated depth map
        this.startSimulatedDepthMap();
        
        this.showNotification('Stereo vision enabled', 'success');
        
        // Also call the API for backend integration
        return fetch(this.apiEndpoints.createStereoPair, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(pairData)
        })
        .then(response => response.json())
        .catch(error => console.error('Error creating stereo pair via API:', error));
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
    disableStereoVision(pairId) {
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
        
        this.updateCameraStatus('depthMapStatus', 'inactive');
        
        this.showNotification('Stereo vision disabled', 'info');
        
        // Also call the API
        return fetch(`${this.apiEndpoints.disableStereoVision}${pairId}/disable`, {
            method: 'POST'
        })
        .then(response => response.json())
        .catch(error => console.error(`Error disabling stereo vision via API:`, error));
    }
    
    // Stop stereo vision
    stopStereoVision(pairId) {
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
        
        // Check if any snapshots were taken
        if (snapshotCount > 0) {
            this.showNotification(`Successfully captured ${snapshotCount} snapshots`, 'success');
        } else {
            this.showNotification('No active cameras to capture snapshots from', 'warning');
        }
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
=======
        .then(data => {
            if (data.status === 'success') {
                // Update status indicator
                this.updateStatusIndicator(statusIndicator, 'active');
                this.showNotification(`${cameraName} started successfully`, 'success');
                
                // Store active camera information
                this.activeCameras[cameraId] = {
                    position: cameraPosition,
                    statusIndicator: statusIndicator,
                    videoElement: videoElement,
                    settings: settings
                };
                
                // Start frame rate counter
                this.startFrameRateCounter(cameraId);
                
                // Start receiving video stream
                if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                    // Send start stream command via WebSocket
                    this.socket.send(JSON.stringify({
                        type: 'start_stream',
                        camera_id: cameraId
                    }));
                } else {
                    // Fallback to HTTP polling
                    this.startHttpPolling(cameraId, videoElement);
                }
            } else {
                this.updateStatusIndicator(statusIndicator, 'error');
                this.showNotification(`Failed to start ${cameraName}: ${data.message}`, 'error');
            }
        })
        .catch(error => {
            console.error(`Error starting ${cameraName}:`, error);
            this.updateStatusIndicator(statusIndicator, 'error');
            this.showNotification(`Error starting ${cameraName}: ${error.message}`, 'error');
        });
    }
    
    // Stop a camera stream
    stopCamera(cameraPosition) {
        // Get the appropriate UI elements based on camera position
        let selector, statusIndicator, videoElement, cameraName;
        
        switch (cameraPosition) {
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
                throw new Error(`Invalid camera position: ${cameraPosition}`);
        }
        
        // Check if selector and other elements exist
        if (!selector || !statusIndicator || !videoElement) {
            console.error(`UI elements for ${cameraPosition} camera not found`);
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
        this.updateStatusIndicator(statusIndicator, 'stopping');
        
        // Stop camera via API
        fetch(`${this.apiEndpoints.stopCamera}${cameraId}/stop`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Update status indicator
                this.updateStatusIndicator(statusIndicator, 'inactive');
                this.showNotification(`${cameraName} stopped successfully`, 'success');
                
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
                        camera_id: cameraId
                    }));
                }
            } else {
                this.updateStatusIndicator(statusIndicator, 'error');
                this.showNotification(`Failed to stop ${cameraName}: ${data.message}`, 'error');
            }
        })
        .catch(error => {
            console.error(`Error stopping ${cameraName}:`, error);
            this.updateStatusIndicator(statusIndicator, 'error');
            this.showNotification(`Error stopping ${cameraName}: ${error.message}`, 'error');
        });
    }
    
    // Stop all cameras
    stopAllCameras() {
        // Show stopping status for all cameras
        if (this.ui.leftCameraStatus) this.updateStatusIndicator(this.ui.leftCameraStatus, 'stopping');
        if (this.ui.rightCameraStatus) this.updateStatusIndicator(this.ui.rightCameraStatus, 'stopping');
        if (this.ui.topCameraStatus) this.updateStatusIndicator(this.ui.topCameraStatus, 'stopping');
        if (this.ui.depthMapStatus) this.updateStatusIndicator(this.ui.depthMapStatus, 'stopping');
        
        // First get list of active cameras
        fetch(this.apiEndpoints.stopAllCameras, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success' && data.active_camera_ids) {
                // Stop each active camera individually
                const stopPromises = data.active_camera_ids.map(cameraId => 
                    fetch(`${this.apiEndpoints.stopCamera}${cameraId}/stop`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        }
                    })
                );
                
                return Promise.all(stopPromises);
            } else {
                throw new Error(data.message || 'Failed to get active cameras');
            }
        })
        .then(responses => {
                // Reset status indicators
                if (this.ui.leftCameraStatus) this.updateStatusIndicator(this.ui.leftCameraStatus, 'inactive');
                if (this.ui.rightCameraStatus) this.updateStatusIndicator(this.ui.rightCameraStatus, 'inactive');
                if (this.ui.topCameraStatus) this.updateStatusIndicator(this.ui.topCameraStatus, 'inactive');
                if (this.ui.depthMapStatus) this.updateStatusIndicator(this.ui.depthMapStatus, 'inactive');
                
                // Clear video elements
                if (this.ui.leftCameraVideo) this.ui.leftCameraVideo.src = '';
                if (this.ui.rightCameraVideo) this.ui.rightCameraVideo.src = '';
                if (this.ui.topCameraVideo) this.ui.topCameraVideo.src = '';
                
                // Clear depth map canvas
                if (this.ui.depthMapCanvas) {
                    const ctx = this.ui.depthMapCanvas.getContext('2d');
                    ctx.clearRect(0, 0, this.ui.depthMapCanvas.width, this.ui.depthMapCanvas.height);
                }
                
                // Stop all frame rate counters
                Object.keys(this.frameCounters).forEach(cameraId => {
                    this.stopFrameRateCounter(cameraId);
                });
                
                // Clear active cameras and depth streams
                this.activeCameras = {};
                this.activeDepthStreams = {};
                this.stereoPairs = {};
                
                // If using WebSocket, send stop all streams command
                if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                    this.socket.send(JSON.stringify({
                        type: 'stop_all_streams'
                    }));
                }
                
                this.showNotification('All cameras stopped successfully', 'success');
            // Reset UI even if some camera stops failed
            this.showNotification('All cameras stopped successfully', 'success');
        })
        .catch(error => {
            console.error('Error stopping all cameras:', error);
            this.showNotification(`Error stopping all cameras: ${error.message}`, 'error');
        });
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
                    if (this.frameCounters[cameraId]) {
                        this.frameCounters[cameraId]++;
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
        const camera = this.activeCameras[cameraId];
        if (!camera) return;
        
        // Update frame counter
        if (this.frameCounters[cameraId]) {
            this.frameCounters[cameraId]++;
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
    
    // Take a snapshot from a camera
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
            this.showNotification(`Failed to take snapshot: ${error.message}`, 'error');
            throw error;
        });
    }
    
    // Take snapshots from all active cameras
    takeAllSnapshots() {
        const activeCameraIds = Object.keys(this.activeCameras);
        
        if (activeCameraIds.length === 0) {
            this.showNotification('No active cameras to take snapshots from', 'info');
            return;
        }
        
        // Take snapshot from each active camera
        activeCameraIds.forEach(cameraId => {
            this.takeSnapshot(cameraId);
        });
    }
    
    // Enable stereo vision
    enableStereoVision() {
        // Get selected cameras
        const leftCameraId = this.ui.leftCameraSelector ? this.ui.leftCameraSelector.value : 'default';
        const rightCameraId = this.ui.rightCameraSelector ? this.ui.rightCameraSelector.value : 'default';
        
        // Check if both cameras are selected
        if (leftCameraId === 'default' || rightCameraId === 'default') {
            this.showNotification('Please select both left and right cameras for stereo vision', 'info');
            return;
        }
        
        // Check if both cameras are active
        if (!this.activeCameras[leftCameraId] || !this.activeCameras[rightCameraId]) {
            this.showNotification('Both left and right cameras must be active for stereo vision', 'info');
            return;
        }
        
        // Show starting status
        if (this.ui.depthMapStatus) {
            this.updateStatusIndicator(this.ui.depthMapStatus, 'connecting');
        }
        
        const pairId = 'main_stereo_pair';
        
        // Create stereo pair
        fetch(`${this.apiEndpoints.createStereoPair}${pairId}`, {
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
<<<<<<< HEAD
                cameraName: cameraName,
                imageData: base64Data
            })
        })
        .catch(error => console.error('Error sending snapshot to backend:', error));
    }
    
    // Initialize the camera control system
    initialize() {
        // List available cameras
        this.listCameras();
    }
}

// Create a single instance of CameraControl
=======
                left_camera_id: leftCameraId,
                right_camera_id: rightCameraId
            })
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
                throw new Error(data.message);
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Update status indicator
                if (this.ui.depthMapStatus) {
                    this.updateStatusIndicator(this.ui.depthMapStatus, 'active');
                }
                
                this.showNotification('Stereo vision started successfully', 'success');
                
                // Store stereo pair information
                this.stereoPairs[pairId] = {
                    left_camera_id: leftCameraId,
                    right_camera_id: rightCameraId
                };
                
                // Start receiving depth data
                if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                    // Send start depth stream command via WebSocket
                    this.socket.send(JSON.stringify({
                        type: 'start_depth_stream',
                        pair_id: pairId
                    }));
                } else {
                    // Fallback to HTTP polling for depth data
                    this.startDepthHttpPolling(pairId);
                }
            } else {
                throw new Error(data.message);
            }
        })
        .catch(error => {
            console.error('Error enabling stereo vision:', error);
            if (this.ui.depthMapStatus) {
                this.updateStatusIndicator(this.ui.depthMapStatus, 'error');
            }
            this.showNotification(`Failed to start stereo vision: ${error.message}`, 'error');
        });
    }
    
    // Disable stereo vision
    disableStereoVision(pairId) {
        // Show stopping status
        if (this.ui.depthMapStatus) {
            this.updateStatusIndicator(this.ui.depthMapStatus, 'stopping');
        }
        
        return fetch(`${this.apiEndpoints.disableStereoVision}${pairId}/disable`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Clear depth map interval if it exists
                if (this.ui.depthMapCanvas && this.ui.depthMapCanvas.depthInterval) {
                    clearInterval(this.ui.depthMapCanvas.depthInterval);
                    delete this.ui.depthMapCanvas.depthInterval;
                }
                
                // Update status indicator
                if (this.ui.depthMapStatus) {
                    this.updateStatusIndicator(this.ui.depthMapStatus, 'inactive');
                }
                
                // Send WebSocket command to stop depth stream
                if (this.socket && this.socket.readyState === WebSocket.OPEN) {
                    this.socket.send(JSON.stringify({
                        command: 'stop_depth_stream',
                        pair_id: pairId
                    }));
                }
                
                return data;
            } else {
                throw new Error(data.message);
            }
        })
        .catch(error => {
            console.error('Error disabling stereo vision:', error);
            if (this.ui.depthMapStatus) {
                this.updateStatusIndicator(this.ui.depthMapStatus, 'error');
            }
            throw error;
        });
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
    
    // Update depth map from WebSocket or HTTP polling
    updateDepthMap(pairId, depthData) {
        if (!this.ui.depthMapCanvas) return;
        
        const ctx = this.ui.depthMapCanvas.getContext('2d');
        
        try {
            // Convert hex string back to binary
            const binaryData = new Uint8Array(depthData.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
            const blob = new Blob([binaryData], { type: 'image/png' });
            const imageUrl = URL.createObjectURL(blob);
            
            const img = new Image();
            img.onload = () => {
                // Update canvas dimensions
                this.ui.depthMapCanvas.width = img.width;
                this.ui.depthMapCanvas.height = img.height;
                
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
            console.error('Error updating depth map:', error);
        }
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
    
    // Update camera status indicator
    updateStatusIndicator(indicator, status) {
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
    
    // Start frame rate counter
    startFrameRateCounter(cameraId) {
        // Initialize counter
        this.frameCounters[cameraId] = 0;
        this.frameRates[cameraId] = 0;
        
        // Update frame rate every second
        this.frameCounters[`${cameraId}_interval`] = setInterval(() => {
            this.frameRates[cameraId] = this.frameCounters[cameraId];
            this.frameCounters[cameraId] = 0;
            
            // Update UI with frame rate (if needed)
            // This would require additional UI elements
        }, 1000);
    }
    
    // Stop frame rate counter
    stopFrameRateCounter(cameraId) {
        if (this.frameCounters[`${cameraId}_interval`]) {
            clearInterval(this.frameCounters[`${cameraId}_interval`]);
            delete this.frameCounters[`${cameraId}_interval`];
        }
        
        delete this.frameCounters[cameraId];
        delete this.frameRates[cameraId];
    }
    
    // Show notification
    showNotification(message, type = 'info') {
        // This function should be implemented to show user notifications
        // It could use a toast notification system or similar
        console.log(`[${type.toUpperCase()}] ${message}`);
        
        // If showNotification is defined globally (from index.html), use that
        if (window.showNotification) {
            window.showNotification(message, type);
        }
    }
    
    // Get camera settings
    getCameraSettings(cameraId) {
        return fetch(`${this.apiEndpoints.getSettings}${cameraId}/settings`)
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    return data.settings;
                } else {
                    throw new Error(data.message || 'Failed to get camera settings');
                }
            })
            .catch(error => {
                console.error(`Error getting settings for camera ${cameraId}:`, error);
                throw error;
            });
    }
    
    // Update camera settings
    updateCameraSettings(cameraId, settings) {
        return fetch(`${this.apiEndpoints.updateSettings}${cameraId}/settings`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                // Update local settings
                if (this.activeCameras[cameraId]) {
                    this.activeCameras[cameraId].settings = { ...this.activeCameras[cameraId].settings, ...settings };
                }
                
                return data;
            } else {
                throw new Error(data.message || 'Failed to update camera settings');
            }
        })
        .catch(error => {
            console.error(`Error updating settings for camera ${cameraId}:`, error);
            throw error;
        });
    }
    
    // Initialize the system
    initialize() {
        // List available cameras
        this.listCameras();
        
        // Create depth map color scheme selector if not exists
        this.createDepthMapColorSchemeSelector();
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
}

// Create a global instance of CameraControl
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
const cameraControl = new CameraControl();

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    cameraControl.initialize();
});

// Expose public methods for global access
window.listAllCameras = function() {
    return cameraControl.listCameras();
};

<<<<<<< HEAD
window.startStereoVision = function(leftCameraId, rightCameraId) {
    // If no IDs provided, use selected ones from UI
    if (!leftCameraId && cameraControl.ui.leftCameraSelector) {
        leftCameraId = cameraControl.ui.leftCameraSelector.value;
    }
    
    if (!rightCameraId && cameraControl.ui.rightCameraSelector) {
        rightCameraId = cameraControl.ui.rightCameraSelector.value;
    }
    
    if (leftCameraId && rightCameraId && leftCameraId !== 'default' && rightCameraId !== 'default') {
        return cameraControl.enableStereoVision(leftCameraId, rightCameraId);
    } else {
        cameraControl.showNotification('Please select both left and right cameras', 'error');
        return Promise.reject('Cameras not selected');
    }
};

window.startStereoVisionSystem = function() {
    return window.startStereoVision();
};

window.stopStereoVision = function(pairId) {
    return cameraControl.stopStereoVision(pairId);
=======
window.startStereoVisionSystem = function() {
    return cameraControl.enableStereoVision();
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
};

window.startVisionSystem = function() {
    // Start all three cameras for the full vision system
<<<<<<< HEAD
    setTimeout(() => {
        if (cameraControl.ui.leftCameraSelector && cameraControl.ui.leftCameraSelector.value !== 'default') {
            cameraControl.startCamera(cameraControl.ui.leftCameraSelector.value);
        }
    }, 100);
    
    setTimeout(() => {
        if (cameraControl.ui.rightCameraSelector && cameraControl.ui.rightCameraSelector.value !== 'default') {
            cameraControl.startCamera(cameraControl.ui.rightCameraSelector.value);
        }
    }, 300);
    
    setTimeout(() => {
        if (cameraControl.ui.topCameraSelector && cameraControl.ui.topCameraSelector.value !== 'default') {
            cameraControl.startCamera(cameraControl.ui.topCameraSelector.value);
        }
    }, 500);
    
    // Enable stereo vision after a delay to ensure cameras are started
    setTimeout(() => {
        window.startStereoVision();
    }, 1000);
=======
    setTimeout(() => cameraControl.startCamera('left'), 100);
    setTimeout(() => cameraControl.startCamera('right'), 300);
    setTimeout(() => cameraControl.startCamera('top'), 500);
    
    // Enable stereo vision after a delay to ensure cameras are started
    setTimeout(() => cameraControl.enableStereoVision(), 1000);
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
};

window.stopAllCameras = function() {
    return cameraControl.stopAllCameras();
};

window.takeAllCameraSnapshots = function() {
<<<<<<< HEAD
    return cameraControl.takeAllCameraSnapshots();
=======
    return cameraControl.takeAllSnapshots();
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
};

window.toggleDepthMap = function() {
    if (cameraControl.ui.depthMapCanvas) {
        const canvas = cameraControl.ui.depthMapCanvas;
        canvas.style.display = canvas.style.display === 'none' ? 'block' : 'none';
    }
};

window.openCameraSettings = function(cameraId) {
<<<<<<< HEAD
    console.log(`Opening settings for camera ${cameraId}`);
    cameraControl.showNotification('Camera settings functionality is being implemented', 'info');
};
=======
    // This would open a settings dialog for the specified camera
    // Implementation depends on the UI design
    console.log(`Opening settings for camera ${cameraId}`);
    cameraControl.showNotification('Camera settings functionality is being implemented', 'info');
};
>>>>>>> 55541e2569d492f61ad4c096b6721db4fe055a13
