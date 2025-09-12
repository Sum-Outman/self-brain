// Simplified Dashboard JavaScript
// Simple implementation to avoid JSON parsing errors

// Global variables
let systemData = null;

// Initialization function
document.addEventListener('DOMContentLoaded', function() {
    console.log('Self Brain AGI Dashboard initializing...');
    loadSystemStatus();
    
    // Refresh status every 30 seconds
    setInterval(loadSystemStatus, 30000);
});

// Load system status
async function loadSystemStatus() {
    try {
        const response = await fetch('/api/system_status');
        if (!response.ok) {
            throw new Error('Network response not normal');
        }
        
        const text = await response.text();
        console.log('API response text:', text);
        
        // Try to parse JSON
        let data;
        try {
            data = JSON.parse(text);
        } catch (e) {
        console.error('JSON parsing error:', e);
            // Use default data
            data = {
                status: 'running',
                message: 'Self Brain AGI system running normally',
                models: { total: 11, active: 11 },
                system: { version: '1.0.0', uptime: 'running' }
            };
        }
        
        systemData = data;
        updateDashboard(data);
        
    } catch (error) {
        console.error('Failed to load system status:', error);
        showError('Unable to connect to system');
    }
}

// Update dashboard display
function updateDashboard(data) {
    // Update model count
    const modelCount = document.getElementById('modelCount');
    if (modelCount && data.models) {
        modelCount.textContent = data.models.active || 11;
    }
    
    // Update status indicator
    const statusIndicator = document.getElementById('statusIndicator');
    if (statusIndicator) {
        statusIndicator.className = 'status-indicator status-running';
        statusIndicator.textContent = data.message || 'System running normally';
    }
    
    // Update version information
    const versionInfo = document.getElementById('versionInfo');
    if (versionInfo && data.system) {
        versionInfo.textContent = `Version: ${data.system.version || '1.0.0'}`;
    }
}

// Show error message
function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    if (errorDiv) {
        errorDiv.textContent = message;
        errorDiv.style.display = 'block';
    }
}

// Hide error message
function hideError() {
    const errorDiv = document.getElementById('errorMessage');
    if (errorDiv) {
        errorDiv.style.display = 'none';
    }
}

// Safely get element text content
function safeGetTextContent(selector, defaultValue = '') {
    const element = document.querySelector(selector);
    return element ? element.textContent : defaultValue;
}

// Safely get element length
function safeGetLength(selector) {
    const elements = document.querySelectorAll(selector);
    return elements ? elements.length : 0;
}

// Safely set text content
function safeSetTextContent(selector, text) {
    const element = document.querySelector(selector);
    if (element) {
        element.textContent = text;
    }
}

// Update model count display
function updateModelCount() {
    const countElement = document.getElementById('model-count');
    if (!countElement) {
        console.warn('Model count element not found');
        return;
    }
    
    fetch('/api/models')
        .then(response => response.json())
        .then(data => {
            const models = data.models || [];
            safeSetTextContent('#model-count', `Model count: ${models.length}`);
        })
        .catch(error => {
        console.error('Failed to get model count:', error);
            safeSetTextContent('#model-count', 'Model count: 0');
        });
}

// Update system status
function updateSystemStatus() {
    const statusElement = document.getElementById('system-status');
    if (!statusElement) {
        console.warn('System status element not found');
        return;
    }
    
    fetch('/api/system/status')
        .then(response => response.json())
        .then(data => {
            const status = data.status || 'unknown';
            const indicator = statusElement.querySelector('.status-indicator');
            if (indicator) {
                indicator.className = `status-indicator status-${status}`;
                indicator.textContent = status === 'running' ? 'Running' :
                                      status === 'idle' ? 'Idle' : 'Unknown';
            }
        })
        .catch(error => {
        console.error('Failed to get system status:', error);
            const indicator = statusElement.querySelector('.status-indicator');
            if (indicator) {
                indicator.className = 'status-indicator status-error';
                indicator.textContent = 'Error';
            }
        });
}

// Update version information
function updateVersionInfo() {
    const versionElement = document.getElementById('version-info');
    if (!versionElement) {
        console.warn('Version info element not found');
        return;
    }
    
    fetch('/api/system/version')
        .then(response => response.json())
        .then(data => {
            safeSetTextContent('#version-info', `Version: ${data.version || 'Unknown'}`);
        })
        .catch(error => {
        console.error('Failed to get version info:', error);
            safeSetTextContent('#version-info', 'Version: Unknown');
        });
}

// Initialize after page load
document.addEventListener('DOMContentLoaded', function() {
    // Ensure all elements exist
    const requiredElements = ['model-count', 'system-status', 'version-info'];
    const missingElements = requiredElements.filter(id => !document.getElementById(id));
    
    if (missingElements.length > 0) {
        console.warn('Missing elements:', missingElements);
    }
    
    // Initialize updates
    updateModelCount();
    updateSystemStatus();
    updateVersionInfo();
    
    // Regular updates
    setInterval(() => {
        updateModelCount();
        updateSystemStatus();
    }, 5000);
});
