// Camera Manager Module
// Self Brain AGI
// Enhanced Multi-Camera Support

class CameraManager {
    constructor() {
        // Active camera streams
        this.activeStreams = {};
        // Depth map processor
        this.depthMapProcessor = null;
        // Camera settings
        this.cameraSettings = {
            exposure: 50,
            brightness: 50,
            contrast: 50,
            saturation: 50,
            gain: 50,
            zoom: 1.0
        };
        // Vision processing options
        this.visionOptions = {
            depthCalculation: true,
            objectDetection: false,
            reconstruction3D: false,
            motionTracking: false
        };
        // Initialize the depth map processor
        this.initDepthMapProcessor();
    }

    // Initialize depth map processor
    initDepthMapProcessor() {
        this.depthMapProcessor = {
            isProcessing: false,
            canvas: null,
            ctx: null,
            lastFrameTime: 0,
            frameInterval: 100 // Process every 100ms
        };
    }

    // Get available cameras
    async getAvailableCameras() {
        try {
            // Get all media devices
            const devices = await navigator.mediaDevices.enumerateDevices();
            // Filter video input devices
            const cameras = devices.filter(device => device.kind === 'videoinput');
            
            // Add mock camera for testing if no real cameras found
            if (cameras.length === 0) {
                return this.getMockCameras();
            }
            
            // Format camera list for UI
            return cameras.map((camera, index) => ({
                id: camera.deviceId || index.toString(),
                name: camera.label || `Camera ${index + 1}`,
                resolution: 'Unknown',
                fps: 30
            }));
        } catch (error) {
            console.error('Error getting cameras:', error);
            // Return mock data if real data is unavailable
            return this.getMockCameras();
        }
    }

    // Get mock cameras for testing
    getMockCameras() {
        return [
            { id: '0', name: 'Webcam (Default)', resolution: '1280x720', fps: 30 },
            { id: '1', name: 'External Camera 1', resolution: '1920x1080', fps: 60 },
            { id: '2', name: 'External Camera 2', resolution: '800x600', fps: 15 }
        ];
    }

    // Start camera stream
    async startCameraStream(cameraId, videoElementId, options = {}) {
        try {
            // Stop existing stream if any
            await this.stopCameraStream(cameraId);
            
            // Default constraints
            const defaultConstraints = {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                frameRate: { ideal: 30 },
                deviceId: { exact: cameraId }
            };
            
            // Merge options with defaults
            const constraints = { ...defaultConstraints, ...options };
            
            // Get media stream
            const stream = await navigator.mediaDevices.getUserMedia({
                video: constraints,
                audio: false
            });
            
            // Update active streams
            this.activeStreams[cameraId] = {
                stream: stream,
                constraints: constraints,
                videoElementId: videoElementId
            };
            
            // Attach stream to video element
            const videoElement = document.getElementById(videoElementId);
            if (videoElement) {
                videoElement.srcObject = stream;
                videoElement.style.display = 'block';
                
                // Update camera status
                this.updateCameraStatus(cameraId, 'Active');
            }
            
            return { success: true, stream: stream };
        } catch (error) {
            console.error('Error starting camera stream:', error);
            // Handle permission or other errors
            this.handleCameraError(cameraId, error);
            
            // Return placeholder for testing
            return this.useCameraPlaceholder(cameraId, videoElementId);
        }
    }

    // Stop camera stream
    async stopCameraStream(cameraId) {
        try {
            const streamInfo = this.activeStreams[cameraId];
            if (streamInfo && streamInfo.stream) {
                // Stop all tracks
                streamInfo.stream.getTracks().forEach(track => track.stop());
                
                // Clear video element
                const videoElement = document.getElementById(streamInfo.videoElementId);
                if (videoElement) {
                    videoElement.srcObject = null;
                    videoElement.style.display = 'none';
                }
                
                // Update status
                this.updateCameraStatus(cameraId, 'Inactive');
                
                // Remove from active streams
                delete this.activeStreams[cameraId];
            }
        } catch (error) {
            console.error('Error stopping camera stream:', error);
        }
    }

    // Start stereo vision with two cameras
    async startStereoVision(leftCameraId, rightCameraId, topCameraId = null) {
        try {
            // Start both camera streams
            const [leftResult, rightResult] = await Promise.all([
                this.startCameraStream(leftCameraId, 'camera1Video'),
                this.startCameraStream(rightCameraId, 'camera2Video')
            ]);
            
            // Start top camera if available
            let topResult = null;
            if (topCameraId) {
                topResult = await this.startCameraStream(topCameraId, 'camera3Video');
            }
            
            // If depth calculation is enabled, start processing
            if (this.visionOptions.depthCalculation && leftResult.success && rightResult.success) {
                this.startDepthMapProcessing();
            }
            
            // Update camera info panel
            this.updateCameraInfoPanel(true);
            
            return {
                success: leftResult.success && rightResult.success,
                leftCameraId: leftCameraId,
                rightCameraId: rightCameraId,
                topCameraId: topCameraId
            };
        } catch (error) {
            console.error('Error starting stereo vision:', error);
            this.updateCameraInfoPanel(false);
            return { success: false, error: error.message };
        }
    }

    // Stop all camera streams
    async stopAllCameras() {
        try {
            // Stop depth map processing
            this.stopDepthMapProcessing();
            
            // Stop each active stream
            const cameraIds = Object.keys(this.activeStreams);
            await Promise.all(cameraIds.map(id => this.stopCameraStream(id)));
            
            // Update camera info panel
            this.updateCameraInfoPanel(false);
            
            return { success: true };
        } catch (error) {
            console.error('Error stopping all cameras:', error);
            return { success: false, error: error.message };
        }
    }

    // Start depth map processing
    startDepthMapProcessing() {
        const depthMapCanvas = document.getElementById('depthMapCanvas');
        if (!depthMapCanvas) return;
        
        // Initialize canvas context
        this.depthMapProcessor.canvas = depthMapCanvas;
        this.depthMapProcessor.ctx = depthMapCanvas.getContext('2d');
        this.depthMapProcessor.isProcessing = true;
        
        // Start processing loop
        this.processDepthMap();
        
        // Show depth map container
        const depthMapContainer = document.getElementById('depthMapContainer');
        if (depthMapContainer) {
            depthMapContainer.classList.remove('hidden');
        }
    }

    // Stop depth map processing
    stopDepthMapProcessing() {
        this.depthMapProcessor.isProcessing = false;
        
        // Hide depth map container
        const depthMapContainer = document.getElementById('depthMapContainer');
        if (depthMapContainer) {
            depthMapContainer.classList.add('hidden');
        }
    }

    // Process depth map (simplified implementation)
    processDepthMap() {
        if (!this.depthMapProcessor.isProcessing) return;
        
        const currentTime = Date.now();
        
        // Process at specified interval
        if (currentTime - this.depthMapProcessor.lastFrameTime >= this.depthMapProcessor.frameInterval) {
            this.depthMapProcessor.lastFrameTime = currentTime;
            
            // Get left and right video elements
            const leftVideo = document.getElementById('camera1Video');
            const rightVideo = document.getElementById('camera2Video');
            
            if (leftVideo && rightVideo && this.depthMapProcessor.ctx) {
                // Get canvas dimensions
                const canvas = this.depthMapProcessor.canvas;
                const ctx = this.depthMapProcessor.ctx;
                
                // Clear canvas
                ctx.fillStyle = 'black';
                ctx.fillRect(0, 0, canvas.width, canvas.height);
                
                // Check if videos are ready
                if (leftVideo.readyState >= 2 && rightVideo.readyState >= 2) {
                    // Draw placeholder depth map (simulated data)
                    this.drawSimulatedDepthMap(ctx, canvas.width, canvas.height);
                    
                    // Update FPS counter
                    this.updateCameraFps();
                }
            }
        }
        
        // Continue processing
        requestAnimationFrame(() => this.processDepthMap());
    }

    // Draw simulated depth map
    drawSimulatedDepthMap(ctx, width, height) {
        // Create gradient for depth visualization
        const depthGradient = ctx.createLinearGradient(0, 0, width, 0);
        depthGradient.addColorStop(0, '#0000ff');  // Blue - close
        depthGradient.addColorStop(0.5, '#00ffff'); // Cyan - medium
        depthGradient.addColorStop(1, '#ff0000');  // Red - far
        
        // Draw random depth patterns (in a real system, this would use actual stereo matching)
        const cellSize = 8;
        for (let y = 0; y < height; y += cellSize) {
            for (let x = 0; x < width; x += cellSize) {
                // Generate noise with some pattern to simulate depth
                const noise = Math.sin(x * 0.02) * Math.cos(y * 0.02) * 0.5 + 0.5;
                const depthValue = noise + (x / width) * 0.3;
                
                // Set opacity based on depth confidence
                const opacity = 0.7 + Math.random() * 0.3;
                ctx.fillStyle = depthGradient;
                ctx.globalAlpha = opacity;
                
                // Draw depth cell
                ctx.fillRect(x, y, cellSize, cellSize);
            }
        }
        
        // Reset alpha
        ctx.globalAlpha = 1.0;
    }

    // Toggle depth map display
    toggleDepthMap() {
        const depthMapContainer = document.getElementById('depthMapContainer');
        if (depthMapContainer) {
            depthMapContainer.classList.toggle('hidden');
        }
    }

    // Update camera status display
    updateCameraStatus(cameraId, status) {
        let statusElementId = '';
        let previewElementId = '';
        
        // Map camera ID to status element
        if (cameraId === 'camera1') {
            statusElementId = 'camera1Status';
            previewElementId = 'camera1Preview';
        } else if (cameraId === 'camera2') {
            statusElementId = 'camera2Status';
            previewElementId = 'camera2Preview';
        } else if (cameraId === 'camera3') {
            statusElementId = 'camera3Status';
            previewElementId = 'camera3Preview';
        }
        
        // Update status text
        const statusElement = document.getElementById(statusElementId);
        if (statusElement) {
            statusElement.textContent = status;
            // Update status color
            if (status === 'Active') {
                statusElement.className = 'absolute top-1 right-1 text-white text-xs px-1 rounded bg-green-700';
            } else if (status === 'Error') {
                statusElement.className = 'absolute top-1 right-1 text-white text-xs px-1 rounded bg-red-700';
            } else {
                statusElement.className = 'absolute top-1 right-1 text-white text-xs px-1 rounded bg-gray-700';
            }
        }
        
        // Update preview frame border
        const previewElement = document.getElementById(previewElementId);
        if (previewElement) {
            if (status === 'Active') {
                previewElement.classList.add('ring-2', 'ring-green-500/50');
            } else if (status === 'Error') {
                previewElement.classList.add('ring-2', 'ring-red-500/50');
            } else {
                previewElement.classList.remove('ring-2', 'ring-green-500/50', 'ring-red-500/50');
            }
        }
    }

    // Update camera info panel
    updateCameraInfoPanel(isConnected) {
        // Update connection status
        const connectionStatusElement = document.getElementById('cameraConnectionStatus');
        if (connectionStatusElement) {
            connectionStatusElement.textContent = isConnected ? 'Connected' : 'Not connected';
        }
        
        // Update resolution (mock data for now)
        const resolutionElement = document.getElementById('cameraResolution');
        if (resolutionElement) {
            resolutionElement.textContent = isConnected ? '1280x720' : '--x--';
        }
    }

    // Update camera FPS counter
    updateCameraFps() {
        const fpsElement = document.getElementById('cameraFps');
        if (fpsElement) {
            // In a real implementation, this would calculate actual FPS
            // For now, we'll use a simulated value
            fpsElement.textContent = Math.floor(Math.random() * 10) + 25;
        }
    }

    // Handle camera errors
    handleCameraError(cameraId, error) {
        console.error('Camera error for ID', cameraId, ':', error);
        
        // Update status
        this.updateCameraStatus(cameraId, 'Error');
        
        // Show error message
        let errorMsg = 'Camera error: ';
        if (error.name === 'NotAllowedError') {
            errorMsg += 'Camera permission denied. Please allow access in browser settings.';
        } else if (error.name === 'NotFoundError') {
            errorMsg += 'Camera not found.';
        } else {
            errorMsg += error.message || 'Unknown error';
        }
        
        alert(errorMsg);
    }

    // Use camera placeholder for testing
    useCameraPlaceholder(cameraId, videoElementId) {
        try {
            const videoElement = document.getElementById(videoElementId);
            if (videoElement) {
                // Hide video element
                videoElement.style.display = 'none';
                
                // Find parent preview element
                let previewElementId = '';
                if (videoElementId === 'camera1Video') {
                    previewElementId = 'camera1Preview';
                } else if (videoElementId === 'camera2Video') {
                    previewElementId = 'camera2Preview';
                } else if (videoElementId === 'camera3Video') {
                    previewElementId = 'camera3Preview';
                }
                
                const previewElement = document.getElementById(previewElementId);
                if (previewElement) {
                    // Create placeholder image
                    const resolution = '1280x720';
                    const [width, height] = resolution.split('x').map(Number);
                    const color = this.getRandomColor();
                    
                    // Use placeholder image with camera ID
                    const placeholderImage = document.createElement('img');
                    placeholderImage.src = `https://via.placeholder.com/${width}x${height}/${color}/ffffff?text=Camera+${cameraId}`;
                    placeholderImage.alt = `Camera ${cameraId} Placeholder`;
                    placeholderImage.className = 'w-full h-full object-cover';
                    
                    // Clear preview and add placeholder
                    previewElement.innerHTML = '';
                    
                    // Add status indicator
                    const statusIndicator = document.createElement('span');
                    statusIndicator.id = `${cameraId === '0' ? 'camera1Status' : cameraId === '1' ? 'camera2Status' : 'camera3Status'}`;
                    statusIndicator.className = 'absolute top-1 right-1 text-white text-xs px-1 rounded bg-gray-700';
                    statusIndicator.textContent = 'Placeholder';
                    
                    // Add elements to preview
                    previewElement.appendChild(statusIndicator);
                    previewElement.appendChild(placeholderImage);
                    
                    // Update camera status
                    this.updateCameraInfoPanel(true);
                }
            }
            
            return { success: true, isPlaceholder: true };
        } catch (error) {
            console.error('Error creating camera placeholder:', error);
            return { success: false, error: error.message };
        }
    }

    // Get random color for placeholder
    getRandomColor() {
        const colors = ['333333', '555555', '777777', '999999', 'AAAAAA'];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    // Take snapshot from all active cameras
    async takeAllSnapshots() {
        try {
            // Create temporary canvas
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            
            // Get all camera video elements
            const cameraVideos = ['camera1Video', 'camera2Video', 'camera3Video']
                .map(id => document.getElementById(id))
                .filter(video => video && video.style.display !== 'none');
            
            if (cameraVideos.length === 0) {
                alert('No active cameras found.');
                return;
            }
            
            // Take snapshot from each camera
            for (const video of cameraVideos) {
                // Set canvas dimensions
                canvas.width = video.videoWidth || 1280;
                canvas.height = video.videoHeight || 720;
                
                // Draw video frame to canvas
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Convert to data URL
                const dataUrl = canvas.toDataURL('image/jpeg');
                
                // Create download link
                const link = document.createElement('a');
                link.href = dataUrl;
                link.download = `camera_snapshot_${Date.now()}.jpg`;
                link.click();
            }
            
            return { success: true };
        } catch (error) {
            console.error('Error taking snapshots:', error);
            alert('Failed to take snapshots: ' + error.message);
            return { success: false, error: error.message };
        }
    }

    // Camera control functions
    zoomIn() {
        this.cameraSettings.zoom = Math.min(this.cameraSettings.zoom + 0.1, 3.0);
        this.applyCameraSettings();
        console.log('Camera zoom in:', this.cameraSettings.zoom);
    }

    zoomOut() {
        this.cameraSettings.zoom = Math.max(this.cameraSettings.zoom - 0.1, 1.0);
        this.applyCameraSettings();
        console.log('Camera zoom out:', this.cameraSettings.zoom);
    }

    autoFocus() {
        console.log('Auto focus triggered');
        // In a real implementation, this would send focus command to camera
        alert('Auto focus triggered.');
    }

    autoWhiteBalance() {
        console.log('Auto white balance triggered');
        // In a real implementation, this would send white balance command to camera
        alert('Auto white balance triggered.');
    }

    // Apply camera settings to active cameras
    applyCameraSettings() {
        // In a real implementation, this would apply settings to camera devices
        // For now, we'll just log the settings
        console.log('Applying camera settings:', this.cameraSettings);
    }

    // Open camera settings dialog
    openCameraSettings() {
        // In a real implementation, this would open a settings dialog
        alert('Camera settings dialog would open here.\n\nCurrent settings:\n- Zoom: ' + this.cameraSettings.zoom.toFixed(1) + 'x\n- Brightness: ' + this.cameraSettings.brightness + '%\n- Contrast: ' + this.cameraSettings.contrast + '%\n- Saturation: ' + this.cameraSettings.saturation + '%');
    }

    // Update vision processing options
    updateVisionOptions(options) {
        this.visionOptions = { ...this.visionOptions, ...options };
        console.log('Updated vision options:', this.visionOptions);
        
        // If depth calculation is enabled and cameras are active, start processing
        if (this.visionOptions.depthCalculation && Object.keys(this.activeStreams).length >= 2) {
            this.startDepthMapProcessing();
        } else {
            this.stopDepthMapProcessing();
        }
    }

    // Cleanup
    destroy() {
        this.stopAllCameras();
        this.activeStreams = {};
        this.depthMapProcessor = null;
    }
}

// Export the CameraManager class
window.CameraManager = CameraManager;

// Initialize camera manager when the page loads
if (window.selfBrain) {
    document.addEventListener('DOMContentLoaded', function() {
        // Wait for selfBrain to be initialized
        const waitForSelfBrain = setInterval(() => {
            if (window.selfBrain && window.selfBrain.hardware) {
                clearInterval(waitForSelfBrain);
                // Initialize camera manager and attach to selfBrain
                window.cameraManager = new CameraManager();
                window.selfBrain.hardware.cameraManager = window.cameraManager;
            }
        }, 100);
    });
}