// Retry function for fetch operations
async function fetchWithRetry(url, options = {}, retries = 3, delay = 1000) {
    try {
        const response = await fetch(url, options);
        if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
        return await response.json();
    } catch (error) {
        if (retries > 0) {
            console.warn(`Request failed, retrying (${retries} retries left)...`, error);
            await new Promise(resolve => setTimeout(resolve, delay));
            return fetchWithRetry(url, options, retries - 1, delay * 2);
        }
        throw error;
    }
}

// Form submission handler
function handleFormSubmission(event, formId) {
    event.preventDefault();
    const form = document.getElementById(formId);
    const formData = new FormData(form);
    const data = Object.fromEntries(formData.entries());
    
    // Data validation
    if (formId === 'generalSettingsForm') {
        const autoSaveInterval = parseInt(data.autoSaveInterval);
        if (isNaN(autoSaveInterval) || autoSaveInterval < 5 || autoSaveInterval > 120) {
            showNotification('Invalid auto-save interval. Must be between 5 and 120 seconds.', 'error');
            return;
        }
    }
    
    // Submit data via API
    fetch('/api/system/settings', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => {
        if (!response.ok) throw new Error('Failed to save settings');
        return response.json();
    })
    .then(result => {
        showNotification('Settings saved successfully!', 'success');
        // Refresh page or update UI if needed
    })
    .catch(error => {
        showNotification('Error saving settings: ' + error.message, 'error');
        console.error('Error saving settings:', error);
    });
}

// Notification display function
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type} fixed bottom-4 right-4 p-4 shadow-lg rounded-lg z-50 transition-opacity duration-300 ease-in-out`;
    notification.textContent = message;
    
    // Add to document
    document.body.appendChild(notification);
    
    // Show notification with animation
    setTimeout(() => {
        notification.classList.add('opacity-100');
    }, 10);
    
    // Auto hide after 5 seconds
    setTimeout(() => {
        notification.classList.remove('opacity-100');
        setTimeout(() => {
            document.body.removeChild(notification);
        }, 300);
    }, 5000);
}

// Sliders initialization
function initializeSliders() {
    const memorySlider = document.getElementById('memoryUsageSlider');
    const cpuSlider = document.getElementById('cpuUsageSlider');
    const memoryValue = document.getElementById('memoryUsageValue');
    const cpuValue = document.getElementById('cpuUsageValue');
    
    if (memorySlider && memoryValue) {
        memoryValue.textContent = `${memorySlider.value}%`;
        memorySlider.addEventListener('input', () => {
            memoryValue.textContent = `${memorySlider.value}%`;
        });
    }
    
    if (cpuSlider && cpuValue) {
        cpuValue.textContent = `${cpuSlider.value}%`;
        cpuSlider.addEventListener('input', () => {
            cpuValue.textContent = `${cpuSlider.value}%`;
        });
    }
}

// API connection test
function testApiConnection(apiProvider) {
    const button = document.getElementById(`test${apiProvider}Connection`);
    const statusElement = document.getElementById(`${apiProvider}ConnectionStatus`);
    
    if (!button || !statusElement) return;
    
    // Change button to loading state
    button.disabled = true;
    button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Testing...';
    statusElement.textContent = 'Testing connection...';
    
    // Test API connection
    fetch(`/api/external/test-connection?provider=${apiProvider}`, {
        method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        if (data.connected) {
            statusElement.textContent = 'Connected';
            showNotification(`${apiProvider} API connection successful!`, 'success');
        } else {
            statusElement.textContent = 'Connection Failed';
            showNotification(`${apiProvider} API connection failed: ${data.error || 'Unknown error'}`, 'error');
        }
    })
    .catch(error => {
        statusElement.textContent = 'Connection Error';
        showNotification(`${apiProvider} API connection error: ${error.message}`, 'error');
    })
    .finally(() => {
        // Reset button state
        button.disabled = false;
        button.innerHTML = 'Test Connection';
    });
}

// Initialize sliders when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeSliders);
} else {
    initializeSliders();
}