// Camera Control Module
const CameraControl = {
    // API endpoints
    API_ENDPOINTS: {
        INPUTS: '/api/camera/inputs',
        START: '/api/camera/start/',
        STOP: '/api/camera/stop/',
        SNAPSHOT: '/api/camera/take-snapshot/',
        GET_SETTINGS: '/api/camera/settings/',
        UPDATE_SETTINGS: '/api/camera/settings/'
    },
    
    // Initialize camera control
    init: function() {
        console.log('Camera Control Module initialized');
        
        // Test API endpoint connectivity
        this.testCameraAPI();
    },
    
    // Test camera API endpoint
    testCameraAPI: async function() {
        try {
            const response = await fetch(this.API_ENDPOINTS.INPUTS);
            const data = await response.json();
            console.log('Camera API test response:', data);
        } catch (error) {
            console.error('Camera API test failed:', error);
        }
    },
    
    // Test function to verify the module is loaded
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
    
    // Start camera via API
    startCamera: async function(cameraId) {
        try {
            const response = await fetch(`${this.API_ENDPOINTS.START}${cameraId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log(`Camera ${cameraId} started successfully`, data);
            return data;
        } catch (error) {
            console.error(`Error starting camera ${cameraId}:`, error);
            throw error;
        }
    },
    
    // Stop camera via API
    stopCamera: async function(cameraId) {
        try {
            const response = await fetch(`${this.API_ENDPOINTS.STOP}${cameraId}`, {
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
            return data;
        } catch (error) {
            console.error(`Error stopping camera ${cameraId}:`, error);
            throw error;
        }
    },
    
    // Take snapshot via API
    takeCameraSnapshot: async function(cameraId) {
        try {
            const response = await fetch(`${this.API_ENDPOINTS.SNAPSHOT}${cameraId}`, {
                method: 'POST',
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
    
    // Get camera settings via API
    getCameraSettings: async function(cameraId) {
        try {
            const response = await fetch(`${this.API_ENDPOINTS.GET_SETTINGS}${cameraId}`, {
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
    
    // Update camera settings via API
    updateCameraSettings: async function(cameraId, settings) {
        try {
            const response = await fetch(`${this.API_ENDPOINTS.UPDATE_SETTINGS}${cameraId}`, {
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
    
    // Start a camera stream
    startCameraStream: async function(constraints = {}) {
        try {
            // Default constraints if none provided
            const defaultConstraints = {
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    frameRate: { ideal: 30 }
                }
            };
            
            const mergedConstraints = { ...defaultConstraints, ...constraints };
            const stream = await navigator.mediaDevices.getUserMedia(mergedConstraints);
            
            console.log('Camera stream started successfully');
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
    
    // Stop a camera stream
    stopCameraStream: function(stream) {
        if (stream && stream.getTracks) {
            stream.getTracks().forEach(track => track.stop());
            console.log('Camera stream stopped');
        }
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
                // videoElement.play(); // Autoplay may be blocked by browser policies
            };
            
            return true;
        } catch (error) {
            console.error('Error attaching stream to video element:', error);
            return false;
        }
    },
    
    // Create a canvas snapshot from camera stream
    takeSnapshot: function(videoElementId) {
        const videoElement = document.getElementById(videoElementId);
        if (!videoElement) {
            console.error(`Video element with id ${videoElementId} not found`);
            return null;
        }
        
        try {
            const canvas = document.createElement('canvas');
            canvas.width = videoElement.videoWidth;
            canvas.height = videoElement.videoHeight;
            
            const context = canvas.getContext('2d');
            context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            
            // Convert canvas to data URL
            const imageUrl = canvas.toDataURL('image/png');
            console.log('Camera snapshot taken');
            
            return imageUrl;
        } catch (error) {
            console.error('Error taking camera snapshot:', error);
            return null;
        }
    }
};

// Initialize CameraControl when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    CameraControl.init();
});

// Export the module for use in other scripts
if (typeof window !== 'undefined') {
    window.CameraControl = CameraControl;
}