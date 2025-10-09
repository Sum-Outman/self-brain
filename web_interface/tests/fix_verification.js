/**
 * Fix Verification Script - Tests all the fixes we've implemented
 * This script should be run in the browser console to verify fixes are working
 */

// Test 1: Check if DeviceCommunication is available
testDeviceCommunication();

// Test 2: Check camera control functionality
testCameraControl();

// Test 3: Check API endpoints
testApiEndpoints();

/**
 * Test DeviceCommunication availability and functionality
 */
function testDeviceCommunication() {
    console.log('\n=== Testing DeviceCommunication ===');
    
    // Check if DeviceCommunication is available
    if (window.DeviceCommunication) {
        console.log('✅ DeviceCommunication is available');
        
        // Test basic functionality
        try {
            const status = window.DeviceCommunication.getConnectionStatus();
            console.log('DeviceCommunication status:', status);
            
            // Test initialization
            window.DeviceCommunication.initialize()
                .then(result => {
                    console.log('DeviceCommunication initialize result:', result);
                })
                .catch(err => {
                    console.error('DeviceCommunication initialize error:', err);
                });
        } catch (error) {
            console.error('❌ Error accessing DeviceCommunication methods:', error);
        }
    } else {
        console.error('❌ DeviceCommunication is not available');
    }
}

/**
 * Test CameraControl functionality
 */
function testCameraControl() {
    console.log('\n=== Testing CameraControl ===');
    
    // Check if CameraControl is available
    if (window.CameraControl) {
        console.log('✅ CameraControl is available');
        
        // Test basic functionality
        try {
            // Try to get active camera inputs without causing errors
            if (typeof window.CameraControl.getActiveCameraInputs === 'function') {
                window.CameraControl.getActiveCameraInputs()
                    .then(result => {
                        console.log('CameraControl.getActiveCameraInputs result:', result);
                    })
                    .catch(err => {
                        console.error('CameraControl.getActiveCameraInputs error:', err);
                    });
            } else {
                console.log('CameraControl.getActiveCameraInputs method not available');
            }
        } catch (error) {
            console.error('❌ Error accessing CameraControl methods:', error);
        }
    } else {
        console.error('❌ CameraControl is not available');
    }
}

/**
 * Test critical API endpoints
 */
function testApiEndpoints() {
    console.log('\n=== Testing API Endpoints ===');
    
    // Test /api/cameras endpoint
    fetch('/api/cameras')
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
        })
        .then(data => {
            console.log('✅ /api/cameras endpoint is working:', data);
        })
        .catch(error => {
            console.error('❌ /api/cameras endpoint error:', error);
        });
    
    // Test /api/inputs endpoint (for camera inputs)
    fetch('/api/inputs')
        .then(response => {
            if (response.ok) {
                return response.json();
            } else {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
        })
        .then(data => {
            console.log('✅ /api/inputs endpoint is working:', data);
        })
        .catch(error => {
            console.error('❌ /api/inputs endpoint error:', error);
        });
    
    // Test device communication API
    fetch('/api/device/ping')
        .then(response => {
            if (response.ok) {
                return response.text();
            } else {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
        })
        .then(data => {
            console.log('✅ /api/device/ping endpoint is working:', data);
        })
        .catch(error => {
            console.error('❌ /api/device/ping endpoint error:', error);
        });
}

/**
 * Helper function to verify all DOM elements that were causing issues
 */
function checkDomElements() {
    console.log('\n=== Checking Critical DOM Elements ===');
    
    // Check if toggle elements exist
    const toggles = document.querySelectorAll('.toggle');
    if (toggles.length > 0) {
        console.log(`✅ Found ${toggles.length} toggle elements`);
        toggles.forEach((toggle, index) => {
            console.log(`Toggle ${index + 1}:`, toggle);
        });
    } else {
        console.log('No toggle elements found');
    }
}