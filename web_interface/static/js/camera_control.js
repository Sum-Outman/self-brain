// Camera Control Module
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
        
        // Initialize UI elements
        this.initializeUI();
        // Initialize event listeners
        this.initializeEventListeners();
        // Initialize WebSocket connections (if available)
        this.initializeWebSockets();
    }
    
    // Initialize UI elements
    initializeUI() {
        // Get references to UI elements
        this.ui = {
            // Camera selectors
            leftCameraSelector: document.getElementById('camera1Selector'),
            rightCameraSelector: document.getElementById('camera2Selector'),
            topCameraSelector: document.getElementById('camera3Selector'),
            
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
        };
    }
    
    // Initialize event listeners
    initializeEventListeners() {
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
            });
        }
    }
    
    // Initialize WebSocket connections for real-time streaming
    initializeWebSockets() {
        // Use the existing Socket.IO connection from the main application
        if (window.socket) {
            this.socket = window.socket;
            console.log('Camera Control connected to existing Socket.IO instance');
            
            // Set up event listeners
            this.setupSocketListeners();
        } else {
            console.log('Socket.IO is not available. Falling back to HTTP polling for camera streams');
            
            // Set up a check to see if socket becomes available later
            this.socketCheckInterval = setInterval(() => {
                if (window.socket) {
                    clearInterval(this.socketCheckInterval);
                    this.socket = window.socket;
                    console.log('Camera Control connected to Socket.IO instance (late connection)');
                    this.setupSocketListeners();
                }
            }, 2000);
        }
    }
    
    // Setup Socket.IO event listeners
    setupSocketListeners() {
        if (!this.socket) return;
        
        // Camera frame event
        this.socket.on('camera_frame', (data) => {
            this.updateCameraFrame(data.camera_id, data.frame_data);
        });
        
        // Depth map event
        this.socket.on('depth_map', (data) => {
            this.updateDepthMap(data.pair_id, data.depth_data);
        });
        
        // Camera status event
        this.socket.on('camera_status', (data) => {
            this.updateCameraStatus(data.camera_id, data.status);
        });
        
        // Stereo status event
        this.socket.on('stereo_status', (data) => {
            this.updateStereoStatus(data.pair_id, data.status);
        });
        
        // Error event
        this.socket.on('camera_error', (data) => {
            console.error('Camera error:', data.error);
            this.showNotification(`Camera error: ${data.error}`, 'error');
        });
    }
    
    // Update camera status
    updateCameraStatus(cameraId, status) {
        this.updateStatusIndicator(cameraId, status);
        
        // Additional status handling logic if needed
        if (status === 'active') {
            this.showNotification(`Camera ${cameraId} started successfully`, 'success');
        } else if (status === 'inactive') {
            this.showNotification(`Camera ${cameraId} stopped`, 'info');
        } else if (status === 'error') {
            this.showNotification(`Camera ${cameraId} encountered an error`, 'error');
        }
    }
    
    // Update stereo vision status
    updateStereoStatus(pairId, status) {
        // Update UI elements related to stereo vision status
        const stereoElement = document.getElementById(`stereo-pair-${pairId}`);
        if (stereoElement) {
            if (status === 'active') {
                stereoElement.classList.add('active');
                stereoElement.classList.remove('inactive', 'error');
            } else if (status === 'inactive') {
                stereoElement.classList.add('inactive');
                stereoElement.classList.remove('active', 'error');
            } else if (status === 'error') {
                stereoElement.classList.add('error');
                stereoElement.classList.remove('active', 'inactive');
            }
        }
        
        // Show notification based on status
        if (status === 'active') {
            this.showNotification(`Stereo pair ${pairId} activated`, 'success');
        } else if (status === 'error') {
            this.showNotification(`Stereo pair ${pairId} error`, 'error');
        }
    }
    
    // List available cameras
    listCameras() {
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
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        })
        .then(response => response.json())
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
                if (this.socket) {
                    // Send start stream command via Socket.IO
                    this.socket.emit('start_camera_stream', {
                        camera_id: cameraId
                    });
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
                
                // If using Socket.IO, send stop stream command
                if (this.socket) {
                    this.socket.emit('stop_camera_stream', {
                        camera_id: cameraId
                    });
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
                
                // If using Socket.IO, send stop all streams command
                if (this.socket) {
                    this.socket.emit('stop_all_camera_streams');
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
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
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
                if (this.socket) {
                    // Send start depth stream command via Socket.IO
                    this.socket.emit('start_depth_stream', {
                        pair_id: pairId
                    });
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
                if (this.socket) {
                    this.socket.emit('stop_depth_stream', {
                        pair_id: pairId
                    });
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
const cameraControl = new CameraControl();

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    cameraControl.initialize();
});

// Expose public methods for global access
window.listAllCameras = function() {
    return cameraControl.listCameras();
};

window.startStereoVisionSystem = function() {
    return cameraControl.enableStereoVision();
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
    return cameraControl.takeAllSnapshots();
};

window.toggleDepthMap = function() {
    if (cameraControl.ui.depthMapCanvas) {
        const canvas = cameraControl.ui.depthMapCanvas;
        canvas.style.display = canvas.style.display === 'none' ? 'block' : 'none';
    }
};

window.openCameraSettings = function(cameraId) {
    // This would open a settings dialog for the specified camera
    // Implementation depends on the UI design
    console.log(`Opening settings for camera ${cameraId}`);
    cameraControl.showNotification('Camera settings functionality is being implemented', 'info');
};
