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
            getDepthData: '/api/stereo/depth/'
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
            depthMapStatus: document.getElementById('depthMapStatus'),
            
            // Video elements
            leftCameraVideo: document.getElementById('camera1Video'),
            rightCameraVideo: document.getElementById('camera2Video'),
            depthMapCanvas: document.getElementById('depthMapCanvas'),
            
            // Buttons
            refreshCamerasBtn: document.getElementById('refreshCameras'),
            startStereoVisionBtn: document.getElementById('startStereoVision'),
            takeSnapshotsBtn: document.getElementById('takeSnapshots'),
            stopAllCamerasBtn: document.getElementById('stopAllCamerasBtn')
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
    }
    
    // List available cameras
    listCameras() {
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
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(settings)
        })
        .then(response => response.json())
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
    
    // Initialize the camera control system
    initialize() {
        // List available cameras
        this.listCameras();
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
};

window.startVisionSystem = function() {
    // Start all three cameras for the full vision system
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