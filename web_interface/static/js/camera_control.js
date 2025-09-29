// Camera Control Module with Multi-Camera and Stereo Vision Support
// Multi-Camera Control Module for Self Brain AGI
// Enhanced with three-camera support and advanced depth visualization
(function(window) {
    // Check if CameraControl already exists
    if (window.CameraControl) {
        console.log('CameraControl is already initialized');
        return;
    }
    
    // Create CameraControl object
    const CameraControl = {
        // API endpoints
        API_ENDPOINTS: {
            INPUTS: '/api/cameras',
            START: '/api/cameras/',
            STOP: '/api/cameras/',
            SNAPSHOT: '/api/cameras/',
            GET_SETTINGS: '/api/cameras/',
            UPDATE_SETTINGS: '/api/cameras/',
            // Stereo vision related API endpoints
            STEREO_PAIRS: '/api/stereo/pairs/',
            STEREO_PROCESS: '/api/stereo/process/',
            DEPTH_DATA: '/api/stereo/process/',
            // Three-camera system related API endpoints
            ENABLE_THREE_CAM: '/api/camera/enable-three-camera/',
            GET_CAMERA_INFO: '/api/camera/info/',
            PROCESSING_OPTIONS: '/api/camera/processing-options/'
        },
        
        // Store active camera streams state
        activeStreams: {},
        
        // Store stereo vision pairs configuration
        stereoPairs: [],
        
        // Active stereo vision processing sessions
        activeStereoSessions: {},
        
        // Active three-camera session
        activeThreeCameraSession: null,
        
        // Camera processing options
        processingOptions: {
            depthCalculation: true,
            objectDetection: false,
            reconstruction: false,
            motionTracking: false
        },
        
        // Camera settings dialog
        cameraSettingsDialog: null,
        
        // Frame processing interval
        frameProcessingInterval: null,
        
        // Initialize camera control
        init: function() {
            console.log('Enhanced Camera Control Module with Three-Camera Support initialized');
            
            // Test API endpoints connectivity
            this.testCameraAPI();
            
            // Load stereo vision pairs configuration
            this.loadStereoPairs();
            
            // Initialize UI elements and event listeners
            this.initUIElements();
        },
        
        // Initialize UI elements and event listeners
        initUIElements: function() {
            // Add event listeners for processing options toggles
            const toggles = [
                'depthCalculationToggle', 
                'objectDetectionToggle', 
                'reconstructionToggle', 
                'motionTrackingToggle'
            ];
            
            toggles.forEach(toggleId => {
                const toggle = document.getElementById(toggleId);
                if (toggle) {
                    toggle.addEventListener('change', (e) => {
                        const optionName = toggleId.replace('Toggle', '');
                        this.updateProcessingOption(optionName, e.target.checked);
                    });
                }
            });
            
            // Initialize FPS counter
            this.fpsCounter = {
                frameCount: 0,
                lastTime: performance.now(),
                fps: 0
            };
            
            // Create camera settings dialog if not exists
            this.createCameraSettingsDialog();
        },
        
        // Update processing option
        updateProcessingOption: function(optionName, value) {
            this.processingOptions[optionName] = value;
            console.log(`Processing option ${optionName} updated to ${value}`);
            
            // Send updated options to server
            this.updateProcessingOptionsOnServer();
        },
        
        // Send processing options to server
        updateProcessingOptionsOnServer: async function() {
            try {
                const response = await fetch(this.API_ENDPOINTS.PROCESSING_OPTIONS, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(this.processingOptions)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log('Processing options updated on server:', data);
            } catch (error) {
                console.error('Error updating processing options on server:', error);
            }
        },
        
        // Load stereo vision pairs configuration
        loadStereoPairs: async function() {
            try {
                const response = await fetch(this.API_ENDPOINTS.STEREO_PAIRS);
                const data = await response.json();
                
                if (data.status === 'success') {
                    this.stereoPairs = data.stereo_pairs || [];
                    console.log(`Loaded ${this.stereoPairs.length} stereo camera pairs`);
                }
            } catch (error) {
                console.error('Error loading stereo pairs:', error);
            }
        },
        
        // Enable stereo vision for a specific pair
        enableStereoVision: async function(pairName) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.STEREO_PAIRS}${pairName}/enable`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Stereo vision enabled for pair ${pairName}`, data);
                return data;
            } catch (error) {
                console.error(`Error enabling stereo vision for pair ${pairName}:`, error);
                throw error;
            }
        },
        
        // Disable stereo vision for a specific pair
        disableStereoVision: async function(pairName) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.STEREO_PAIRS}${pairName}/disable`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Stereo vision disabled for pair ${pairName}`, data);
                return data;
            } catch (error) {
                console.error(`Error disabling stereo vision for pair ${pairName}:`, error);
                throw error;
            }
        },
        
        // Process stereo vision for a specific pair
        processStereoVision: async function(pairName, options = {}) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.STEREO_PROCESS}${pairName}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(options)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Stereo vision processed for pair ${pairName}`, data);
                return data;
            } catch (error) {
                console.error(`Error processing stereo vision for pair ${pairName}:`, error);
                throw error;
            }
        },
        
        // Get specific stereo pair configuration
        getStereoPair: async function(pairName) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.STEREO_PAIRS}${pairName}`);
                const data = await response.json();
                
                console.log(`Retrieved stereo pair ${pairName}`, data);
                return data;
            } catch (error) {
                console.error(`Error retrieving stereo pair ${pairName}:`, error);
                throw error;
            }
        },
        
        // Set stereo pair configuration
        setStereoPair: async function(pairName, configuration) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.STEREO_PAIRS}${pairName}`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(configuration)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Stereo pair ${pairName} configuration updated`, data);
                return data;
            } catch (error) {
                console.error(`Error updating stereo pair ${pairName} configuration:`, error);
                throw error;
            }
        },
        
        // Test camera API endpoints
        testCameraAPI: async function() {
            try {
                const response = await fetch(this.API_ENDPOINTS.INPUTS);
                const data = await response.json();
                console.log('Camera API test response:', data);
            } catch (error) {
                console.error('Camera API test failed:', error);
            }
        },
        
        // Test function to verify module is loaded
        testAPI: async function() {
            try {
                console.log('Testing Camera API connection...');
                const response = await fetch(this.API_ENDPOINTS.INPUTS);
                const data = await response.json();
                console.log('Camera API test successful:', data);
                return { success: true, data: data };
            } catch (error) {
                console.error('Camera API test failed:', error);
                return { success: false, error: error.message };
            }
        },
        
        // Get all active camera streams
        getActiveStreams: function() {
            return this.activeStreams;
        },
        
        // Check if camera is active
        isCameraActive: function(cameraId) {
            return this.activeStreams.hasOwnProperty(cameraId);
        },
        
        // Get active camera inputs from API
        getActiveCameraInputs: async function() {
            try {
                const response = await fetch(this.API_ENDPOINTS.INPUTS);
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                const data = await response.json();
                
                if (data.status === 'success') {
                    console.log(`Successfully retrieved ${data.camera_count} camera inputs`);
                } else {
                    console.error('Failed to get camera inputs:', data.message);
                }
                
                return data;
            } catch (error) {
                console.error('Error retrieving camera inputs:', error);
                return { status: 'error', message: error.message };
            }
        },
        
        // Start camera through API
        startCamera: async function(cameraId) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.START}${cameraId}/start`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({})
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Camera ${cameraId} started successfully`, data);
                
                // Update active streams state
                if (data.status === 'success') {
                    this.activeStreams[cameraId] = {
                        started: true,
                        lastUpdated: new Date()
                    };
                    
                    // Update camera status UI
                    this.updateCameraStatusUI(cameraId, 'Active');
                }
                
                return data;
            } catch (error) {
                console.error(`Error starting camera ${cameraId}:`, error);
                
                // Update camera status UI with error
                this.updateCameraStatusUI(cameraId, 'Error', error.message);
                
                throw error;
            }
        },
        
        // Start multiple cameras
        startMultipleCameras: async function(cameraIds) {
            try {
                const promises = cameraIds.map(id => this.startCamera(id));
                const results = await Promise.all(promises);
                console.log(`Started ${cameraIds.length} cameras`);
                return results;
            } catch (error) {
                console.error('Error starting multiple cameras:', error);
                throw error;
            }
        },
        
        // Stop camera through API
        stopCamera: async function(cameraId) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.STOP}${cameraId}/stop`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Camera ${cameraId} stopped successfully`, data);
                
                // Update active streams state
                if (data.status === 'success' && this.activeStreams.hasOwnProperty(cameraId)) {
                    delete this.activeStreams[cameraId];
                    
                    // Update camera status UI
                    this.updateCameraStatusUI(cameraId, 'Inactive');
                }
                
                return data;
            } catch (error) {
                console.error(`Error stopping camera ${cameraId}:`, error);
                throw error;
            }
        },
        
        // Stop all active cameras
        stopAllCameras: async function() {
            try {
                const cameraIds = Object.keys(this.activeStreams);
                const promises = cameraIds.map(id => this.stopCamera(id));
                await Promise.all(promises);
                console.log('All cameras stopped');
                this.activeStreams = {};
                
                // Update status indicators
                this.updateCameraStatusUI('all', 'Inactive');
                this.updateCameraInfoPanel('Not connected', '--x--', 0);
                
                // Clear depth map if visible
                this.clearDepthMap();
                
                // Stop frame processing loop
                if (this.frameProcessingInterval) {
                    clearInterval(this.frameProcessingInterval);
                    this.frameProcessingInterval = null;
                }
                
                return { status: 'success', message: 'All cameras stopped' };
            } catch (error) {
                console.error('Error stopping all cameras:', error);
                return { status: 'error', message: error.message };
            }
        },
        
        // Take snapshot through API
        takeCameraSnapshot: async function(cameraId) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.SNAPSHOT}${cameraId}/snapshot`, {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Snapshot taken from camera ${cameraId}`, data);
                return data;
            } catch (error) {
                console.error(`Error taking snapshot from camera ${cameraId}:`, error);
                throw error;
            }
        },
        
        // Get camera settings through API
        getCameraSettings: async function(cameraId) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.GET_SETTINGS}${cameraId}/settings`, {
                    method: 'GET',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Settings retrieved for camera ${cameraId}`, data);
                return data;
            } catch (error) {
                console.error(`Error retrieving settings for camera ${cameraId}:`, error);
                throw error;
            }
        },
        
        // Update camera settings through API
        updateCameraSettings: async function(cameraId, settings) {
            try {
                const response = await fetch(`${this.API_ENDPOINTS.UPDATE_SETTINGS}${cameraId}/settings`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(settings)
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log(`Settings updated for camera ${cameraId}`, data);
                return data;
            } catch (error) {
                console.error(`Error updating settings for camera ${cameraId}:`, error);
                throw error;
            }
        },
        
        // Get available cameras using navigator.mediaDevices.enumerateDevices
        getAvailableCameras: async function() {
            try {
                if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) {
                    console.log('enumerateDevices not supported');
                    return [];
                }
                
                const devices = await navigator.mediaDevices.enumerateDevices();
                const cameras = devices.filter(device => device.kind === 'videoinput');
                
                console.log(`Found ${cameras.length} cameras`);
                return cameras;
            } catch (error) {
                console.error('Error enumerating cameras:', error);
                return [];
            }
        },
        
        // Enable three-camera system
        enableThreeCameraSystem: async function(leftCameraId, rightCameraId, topCameraId) {
            try {
                const response = await fetch(this.API_ENDPOINTS.ENABLE_THREE_CAM, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        left_camera_id: leftCameraId,
                        right_camera_id: rightCameraId,
                        top_camera_id: topCameraId,
                        processing_options: this.processingOptions
                    })
                });
                
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                
                const data = await response.json();
                console.log('Three-camera system enabled', data);
                
                if (data.status === 'success') {
                    this.activeThreeCameraSession = {
                        leftCameraId: leftCameraId,
                        rightCameraId: rightCameraId,
                        topCameraId: topCameraId,
                        started: true,
                        lastUpdated: new Date(),
                        resolution: data.resolution || '1280x720',
                        fps: data.fps || 30
                    };
                    
                    // Update status indicators
                    this.updateCameraInfoPanel('Connected', this.activeThreeCameraSession.resolution, this.activeThreeCameraSession.fps);
                    
                    // Start frame processing loop
                    this.startFrameProcessingLoop();
                }
                
                return data;
            } catch (error) {
                console.error('Error enabling three-camera system:', error);
                this.updateCameraInfoPanel('Connection Error', '--x--', 0);
                throw error;
            }
        },
        
        // Start frame processing loop for depth visualization and FPS calculation
        startFrameProcessingLoop: function() {
            if (this.frameProcessingInterval) {
                clearInterval(this.frameProcessingInterval);
            }
            
            this.frameProcessingInterval = setInterval(() => {
                try {
                    // Update FPS counter
                    this.updateFPSCounter();
                    
                    // If depth map is visible and depth calculation is enabled
                    if (this.isDepthMapVisible() && this.processingOptions.depthCalculation) {
                        this.updateDepthMap();
                    }
                } catch (error) {
                    console.error('Error in frame processing loop:', error);
                }
            }, 100); // Update every 100ms
        },
        
        // Update FPS counter
        updateFPSCounter: function() {
            const currentTime = performance.now();
            const elapsedTime = currentTime - this.fpsCounter.lastTime;
            
            this.fpsCounter.frameCount++;
            
            // Update FPS every second
            if (elapsedTime >= 1000) {
                this.fpsCounter.fps = Math.round((this.fpsCounter.frameCount * 1000) / elapsedTime);
                this.fpsCounter.frameCount = 0;
                this.fpsCounter.lastTime = currentTime;
                
                // Update FPS display in UI
                const fpsElement = document.getElementById('cameraFps');
                if (fpsElement && Object.keys(this.activeStreams).length > 0) {
                    fpsElement.textContent = this.fpsCounter.fps;
                }
            }
        },
        
        // Start camera stream
        startCameraStream: async function(deviceId, constraints = {}) {
            try {
                // Default constraints if not provided
                const defaultConstraints = {
                    video: {
                        width: { ideal: 1280 },
                        height: { ideal: 720 },
                        frameRate: { ideal: 30 }
                    }
                };
                
                // Add device ID to constraints if provided
                if (deviceId) {
                    defaultConstraints.video.deviceId = deviceId;
                }
                
                const mergedConstraints = { ...defaultConstraints, ...constraints };
                const stream = await navigator.mediaDevices.getUserMedia(mergedConstraints);
                
                console.log(`Camera stream for device ${deviceId || 'default'} started successfully`);
                
                // Store stream reference
                if (deviceId) {
                    this.activeStreams[deviceId] = {
                        stream: stream,
                        started: true,
                        lastUpdated: new Date()
                    };
                }
                
                return stream;
            } catch (error) {
                console.error('Error starting camera stream:', error);
                
                // Handle common errors
                if (error.name === 'NotAllowedError') {
                    alert('Camera access was denied. Please allow camera access in your browser settings.');
                } else if (error.name === 'NotFoundError') {
                    alert('No camera found on this device.');
                } else if (error.name === 'NotReadableError') {
                    alert('Camera is already in use by another application.');
                }
                
                throw error;
            }
        },
        
        // Stop camera stream
        stopCameraStream: function(deviceId) {
            // If device ID provided, stop specific stream
            if (deviceId && this.activeStreams[deviceId] && this.activeStreams[deviceId].stream) {
                const stream = this.activeStreams[deviceId].stream;
                if (stream.getTracks) {
                    stream.getTracks().forEach(track => track.stop());
                }
                delete this.activeStreams[deviceId];
                console.log(`Camera stream for device ${deviceId} stopped`);
            } else if (arguments.length === 0) {
                // If no device ID provided, stop all streams
                this.stopAllCameras();
            }
        },
        
        // Enhanced depth map visualization with color coding
        displayDepthMap: function(canvasId, depthData, width, height, options = {}) {
            const canvas = document.getElementById(canvasId);
            if (!canvas) {
                console.error(`Canvas element with id ${canvasId} not found`);
                return false;
            }
            
            try {
                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                const imageData = ctx.createImageData(width, height);
                
                // Default options
                const displayOptions = {
                    colorMode: 'grayscale', // 'grayscale', 'heatmap', 'rainbow'
                    invert: false,
                    ...options
                };
                
                // Create gradient for color modes
                const getColorForDepth = (depthValue) => {
                    // Normalize depth value if needed
                    let normalizedDepth = depthValue;
                    if (normalizedDepth < 0) normalizedDepth = 0;
                    if (normalizedDepth > 1) normalizedDepth = 1;
                    
                    // Invert depth if needed
                    if (displayOptions.invert) {
                        normalizedDepth = 1 - normalizedDepth;
                    }
                    
                    // Return color based on mode
                    switch (displayOptions.colorMode) {
                        case 'heatmap':
                            // Red (near) to blue (far)
                            return {
                                r: Math.floor(normalizedDepth * 255),
                                g: Math.floor((1 - Math.abs(2 * normalizedDepth - 1)) * 255),
                                b: Math.floor((1 - normalizedDepth) * 255)
                            };
                        case 'rainbow':
                            // ROYGBIV color mapping
                            const hue = normalizedDepth * 360;
                            return this.hslToRgb(hue, 100, 50);
                        case 'grayscale':
                        default:
                            const gray = Math.floor(normalizedDepth * 255);
                            return { r: gray, g: gray, b: gray };
                    }
                };
                
                // Process depth data
                for (let i = 0; i < depthData.length && i < width * height; i++) {
                    const pixelIndex = i * 4;
                    const depthValue = depthData[i];
                    
                    // Get color for this depth value
                    const color = getColorForDepth(depthValue);
                    
                    imageData.data[pixelIndex] = color.r;     // R
                    imageData.data[pixelIndex + 1] = color.g; // G
                    imageData.data[pixelIndex + 2] = color.b; // B
                    imageData.data[pixelIndex + 3] = 255;     // A
                }
                
                ctx.putImageData(imageData, 0, 0);
                
                // Add depth scale legend
                if (options.showLegend) {
                    this.drawDepthLegend(ctx, width, height);
                }
                
                console.log(`Enhanced depth map displayed on canvas ${canvasId}`);
                return true;
            } catch (error) {
                console.error('Error displaying depth map:', error);
                return false;
            }
        },
        
        // Helper function: HSL to RGB conversion
        hslToRgb: function(h, s, l) {
            h /= 360;
            s /= 100;
            l /= 100;
            
            let r, g, b;
            
            if (s === 0) {
                r = g = b = l; // achromatic
            } else {
                const hue2rgb = (p, q, t) => {
                    if (t < 0) t += 1;
                    if (t > 1) t -= 1;
                    if (t < 1/6) return p + (q - p) * 6 * t;
                    if (t < 1/2) return q;
                    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
                    return p;
                };
                
                const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
                const p = 2 * l - q;
                r = hue2rgb(p, q, h + 1/3);
                g = hue2rgb(p, q, h);
                b = hue2rgb(p, q, h - 1/3);
            }
            
            return {
                r: Math.round(r * 255),
                g: Math.round(g * 255),
                b: Math.round(b * 255)
            };
        },
        
        // Draw depth legend on canvas
        drawDepthLegend: function(ctx, canvasWidth, canvasHeight) {
            const legendWidth = 20;
            const legendHeight = canvasHeight * 0.8;
            const legendX = canvasWidth - legendWidth - 10;
            const legendY = (canvasHeight - legendHeight) / 2;
            
            // Create gradient
            const gradient = ctx.createLinearGradient(0, legendY, 0, legendY + legendHeight);
            gradient.addColorStop(0, 'rgb(255, 0, 0)'); // Near (red)
            gradient.addColorStop(1, 'rgb(0, 0, 255)'); // Far (blue)
            
            // Draw gradient rectangle
            ctx.fillStyle = gradient;
            ctx.fillRect(legendX, legendY, legendWidth, legendHeight);
            
            // Draw border
            ctx.strokeStyle = 'white';
            ctx.lineWidth = 1;
            ctx.strokeRect(legendX, legendY, legendWidth, legendHeight);
            
            // Draw labels
            ctx.fillStyle = 'white';
            ctx.font = '10px Arial';
            ctx.textAlign = 'right';
            ctx.fillText('Near', legendX - 5, legendY + 10);
            ctx.fillText('Far', legendX - 5, legendY + legendHeight);
        },
        
        // Update depth map with latest data
        updateDepthMap: async function() {
            if (!this.activeThreeCameraSession) {
                return;
            }
            
            try {
                // For demo purposes, we'll generate synthetic depth data
                const syntheticDepthData = this.generateSyntheticDepthData(640, 480);
                
                // Display the depth map
                this.displayDepthMap('depthMapCanvas', syntheticDepthData, 640, 480, {
                    colorMode: 'heatmap',
                    showLegend: true,
                    invert: false
                });
            } catch (error) {
                console.error('Error updating depth map:', error);
            }
        },
        
        // Generate synthetic depth data for demo purposes
        generateSyntheticDepthData: function(width, height) {
            const data = [];
            const centerX = width / 2;
            const centerY = height / 2;
            const maxRadius = Math.min(width, height) / 2;
            
            for (let y = 0; y < height; y++) {
                for (let x = 0; x < width; x++) {
                    // Calculate distance from center
                    const distance = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2)) / maxRadius;
                    
                    // Add some random noise
                    const noise = (Math.random() - 0.5) * 0.1;
                    
                    // Clamp between 0 and 1
                    const depthValue = Math.max(0, Math.min(1, distance + noise));
                    
                    data.push(depthValue);
                }
            }
            
            return data;
        },
        
        // Clear depth map
        clearDepthMap: function() {
            const canvas = document.getElementById('depthMapCanvas');
            if (!canvas) return;
            
            try {
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            } catch (error) {
                console.error('Error clearing depth map:', error);
            }
        },
        
        // Toggle depth map visibility
        toggleDepthMapVisibility: function() {
            const container = document.getElementById('depthMapContainer');
            if (!container) return;
            
            if (container.classList.contains('hidden')) {
                container.classList.remove('hidden');
                // Start updating depth map if cameras are active
                if (this.activeThreeCameraSession) {
                    this.updateDepthMap();
                }
            } else {
                container.classList.add('hidden');
            }
        },
        
        // Check if depth map is visible
        isDepthMapVisible: function() {
            const container = document.getElementById('depthMapContainer');
            return container && !container.classList.contains('hidden');
        },
        
        // Attach camera stream to video element
        attachStreamToVideo: function(stream, videoElementId) {
            const videoElement = document.getElementById(videoElementId);
            if (!videoElement) {
                console.error(`Video element with id ${videoElementId} not found`);
                return false;
            }
            
            try {
                videoElement.srcObject = stream;
                
                videoElement.onloadedmetadata = function() {
                    console.log(`Camera stream attached to video element ${videoElementId}`);
                    
                    // Get camera ID from element ID (assuming naming convention)
                    const cameraId = videoElementId.replace('Video', '');
                    
                    // Update resolution in info panel
                    if (videoElement.videoWidth && videoElement.videoHeight) {
                        const resolution = `${videoElement.videoWidth}x${videoElement.videoHeight}`;
                        
                        // Only update if this is the main resolution source
                        if (cameraId === 'camera1') {
                            const resolutionElement = document.getElementById('cameraResolution');
                            if (resolutionElement) {
                                resolutionElement.textContent = resolution;
                            }
                        }
                    }
                };
                
                return true;
            } catch (error) {
                console.error('Error attaching stream to video element:', error);
                return false;
            }
        },
        
        // Take snapshot from camera stream with timestamp
        takeSnapshot: function(videoElementId) {
            const videoElement = document.getElementById(videoElementId);
            if (!videoElement || !videoElement.videoWidth) {
                console.error(`Video element with id ${videoElementId} not found or not ready`);
                return null;
            }
            
            try {
                const canvas = document.createElement('canvas');
                canvas.width = videoElement.videoWidth;
                canvas.height = videoElement.videoHeight;
                
                const context = canvas.getContext('2d');
                context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                
                // Add timestamp to snapshot
                const timestamp = new Date().toLocaleString();
                context.font = '12px Arial';
                context.fillStyle = 'rgba(255, 255, 255, 0.7)';
                context.fillText(timestamp, 10, canvas.height - 10);
                
                // Convert canvas to data URL
                const imageUrl = canvas.toDataURL('image/png');
                console.log('Camera snapshot taken with timestamp');
                
                return imageUrl;
            } catch (error) {
                console.error('Error taking camera snapshot:', error);
                return null;
            }
        },
        
        // Update camera status UI
        updateCameraStatusUI: function(cameraId, status, message = '') {
            // Update status text
            let statusElements;
            let indicatorElements;
            
            if (cameraId === 'all') {
                // Update all cameras
                statusElements = document.querySelectorAll('[id$="Status"]');
                indicatorElements = document.querySelectorAll('.camera-status');
            } else {
                // Update specific camera
                statusElements = [document.getElementById(`${cameraId}Status`)];
                indicatorElements = [document.getElementById(`${cameraId}StatusIndicator`)];
            }
            
            // Update status text elements
            statusElements.forEach(element => {
                if (element) {
                    element.textContent = status;
                    
                    // Update status badge style
                    element.className = 'absolute top-1 right-1 text-white text-xs px-1 rounded';
                    
                    switch (status.toLowerCase()) {
                        case 'active':
                            element.classList.add('bg-green-600');
                            break;
                        case 'inactive':
                            element.classList.add('bg-gray-700');
                            break;
                        case 'error':
                            element.classList.add('bg-red-600');
                            if (message) {
                                element.setAttribute('title', message);
                            }
                            break;
                        default:
                            element.classList.add('bg-blue-600');
                    }
                }
            });
            
            // Update status indicators
            indicatorElements.forEach(element => {
                if (element) {
                    element.className = 'camera-status';
                    
                    switch (status.toLowerCase()) {
                        case 'active':
                            element.classList.add('status-active');
                            break;
                        case 'inactive':
                            element.classList.add('status-inactive');
                            break;
                        case 'error':
                            element.classList.add('status-error');
                            break;
                        default:
                            element.classList.add('status-unknown');
                    }
                }
            });
        },
        
        // Update camera info panel
        updateCameraInfoPanel: function(connectionStatus, resolution, fps) {
            const statusElement = document.getElementById('cameraConnectionStatus');
            const resolutionElement = document.getElementById('cameraResolution');
            const fpsElement = document.getElementById('cameraFps');
            
            if (statusElement) {
                statusElement.textContent = connectionStatus;
            }
            
            if (resolutionElement) {
                resolutionElement.textContent = resolution;
            }
            
            if (fpsElement) {
                fpsElement.textContent = fps;
            }
        },
        
        // Create camera settings dialog
        createCameraSettingsDialog: function() {
            // Check if dialog already exists
            if (this.cameraSettingsDialog) {
                return;
            }
            
            // This would be implemented with a proper modal dialog
            // For now, we'll just log that the function is called
            console.log('Camera settings dialog creation triggered');
        },
        
        // Open camera settings dialog
        openCameraSettingsDialog: function() {
            console.log('Opening camera settings dialog');
            // Implementation would show the camera settings modal
        }
    };
    
    // Add CameraControl object to window
    window.CameraControl = CameraControl;
    
    // Initialize CameraControl when DOM is loaded
    document.addEventListener('DOMContentLoaded', function() {
        if (window.CameraControl) {
            window.CameraControl.init();
        }
    });
    
    // Global functions for UI interaction
    window.listAllCameras = function() {
        if (window.CameraControl) {
            console.log('Listing all cameras');
            // Implementation would populate camera selectors
            window.CameraControl.getAvailableCameras().then(cameras => {
                // Populate camera selectors
                const selectors = ['camera1Selector', 'camera2Selector', 'camera3Selector'];
                
                selectors.forEach(selectorId => {
                    const selector = document.getElementById(selectorId);
                    if (selector) {
                        // Clear existing options
                        selector.innerHTML = '<option value="default">Select Camera</option>';
                        
                        // Add camera options
                        cameras.forEach(camera => {
                            const option = document.createElement('option');
                            option.value = camera.deviceId;
                            option.textContent = camera.label || `Camera ${camera.deviceId.substring(0, 8)}`;
                            selector.appendChild(option);
                        });
                    }
                });
            });
        }
    };
    
    window.startStereoVision = async function() {
        if (window.CameraControl) {
            console.log('Starting three-camera vision system');
            
            // Get selected camera IDs
            const camera1Id = document.getElementById('camera1Selector').value;
            const camera2Id = document.getElementById('camera2Selector').value;
            const camera3Id = document.getElementById('camera3Selector').value;
            
            // Validate selections
            if (camera1Id === 'default' || camera2Id === 'default') {
                alert('Please select at least left and right cameras');
                return;
            }
            
            try {
                // Start selected cameras
                await window.CameraControl.startMultipleCameras([camera1Id, camera2Id]);
                
                // If top camera is selected, start it too
                if (camera3Id !== 'default') {
                    await window.CameraControl.startCamera(camera3Id);
                }
                
                // Enable three-camera system
                await window.CameraControl.enableThreeCameraSystem(camera1Id, camera2Id, camera3Id);
                
                // Start local camera streams for preview
                if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                    // Start left camera stream
                    const leftStream = await window.CameraControl.startCameraStream(camera1Id);
                    window.CameraControl.attachStreamToVideo(leftStream, 'camera1Video');
                    
                    // Start right camera stream
                    const rightStream = await window.CameraControl.startCameraStream(camera2Id);
                    window.CameraControl.attachStreamToVideo(rightStream, 'camera2Video');
                    
                    // Start top camera stream if selected
                    if (camera3Id !== 'default') {
                        const topStream = await window.CameraControl.startCameraStream(camera3Id);
                        window.CameraControl.attachStreamToVideo(topStream, 'camera3Video');
                    }
                }
            } catch (error) {
                console.error('Error starting vision system:', error);
                alert('Failed to start vision system: ' + error.message);
            }
        }
    };
    
    window.stopAllCameras = function() {
        if (window.CameraControl) {
            console.log('Stopping all cameras');
            window.CameraControl.stopAllCameras();
        }
    };
    
    window.takeAllSnapshots = function() {
        if (window.CameraControl) {
            console.log('Taking snapshots from all active cameras');
            
            // Take snapshots from all camera video elements
            const snapshots = [];
            const cameraIds = ['camera1', 'camera2', 'camera3'];
            
            cameraIds.forEach(id => {
                const snapshot = window.CameraControl.takeSnapshot(`${id}Video`);
                if (snapshot) {
                    snapshots.push({
                        cameraId: id,
                        imageUrl: snapshot,
                        timestamp: new Date().toISOString()
                    });
                    
                    // Create temporary link to download snapshot
                    const link = document.createElement('a');
                    link.href = snapshot;
                    link.download = `${id}_snapshot_${Date.now()}.png`;
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                }
            });
            
            // Log snapshots info
            if (snapshots.length > 0) {
                console.log(`Successfully took ${snapshots.length} snapshots`);
            }
        }
    };
    
    window.toggleDepthMap = function() {
        if (window.CameraControl) {
            console.log('Toggling depth map display');
            window.CameraControl.toggleDepthMapVisibility();
        }
    };
    
    window.openCameraSettings = function() {
        if (window.CameraControl) {
            console.log('Opening camera settings');
            window.CameraControl.openCameraSettingsDialog();
        }
    };
})(window);