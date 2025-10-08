// External API Configuration Handler

// API Provider default endpoints
const API_PROVIDERS = {
    openai: {
        endpoint: 'https://api.openai.com/v1',
        defaultModel: 'gpt-4o'
    },
    anthropic: {
        endpoint: 'https://api.anthropic.com/v1',
        defaultModel: 'claude-3-opus-20240229'
    },
    google: {
        endpoint: 'https://generativelanguage.googleapis.com/v1',
        defaultModel: 'gemini-pro'
    },
    huggingface: {
        endpoint: 'https://api-inference.huggingface.co/models',
        defaultModel: ''
    },
    custom: {
        endpoint: '',
        defaultModel: ''
    }
};

// Show External API Configuration Modal
function showExternalApiModal(modelId) {
    // Get model details
    const modelElement = document.querySelector(`tr:has(td:nth-child(1):contains('${modelId}'))`);
    const modelName = modelElement ? modelElement.querySelector('td:nth-child(2)').textContent : modelId;
    
    // Set modal title and hidden input
    document.getElementById('currentModelId').textContent = modelName;
    document.getElementById('modelIdInput').value = modelId;
    
    // Reset form
    document.getElementById('externalApiForm').reset();
    document.getElementById('apiTimeout').value = 30;
    
    // Show modal
    const externalApiModal = new bootstrap.Modal(document.getElementById('externalApiModal'));
    externalApiModal.show();
    
    // Focus on API Key field
    setTimeout(() => {
        document.getElementById('apiKey').focus();
    }, 300);
}

// Handle API Provider selection change
function onApiProviderChange() {
    const provider = document.getElementById('apiProvider').value;
    const endpointInput = document.getElementById('apiEndpoint');
    const modelInput = document.getElementById('apiModel');
    
    if (API_PROVIDERS[provider]) {
        endpointInput.value = API_PROVIDERS[provider].endpoint;
        modelInput.value = API_PROVIDERS[provider].defaultModel;
    }
    
    // Adjust placeholder based on provider
    if (provider === 'custom') {
        endpointInput.placeholder = 'https://your-custom-api.com/v1';
    } else {
        endpointInput.placeholder = API_PROVIDERS[provider].endpoint;
    }
}

// Test External API Connection
async function testExternalApi() {
    const modelId = document.getElementById('modelIdInput').value;
    const apiProvider = document.getElementById('apiProvider').value;
    const apiKey = document.getElementById('apiKey').value;
    const apiEndpoint = document.getElementById('apiEndpoint').value;
    const apiModel = document.getElementById('apiModel').value;
    const apiTimeout = document.getElementById('apiTimeout').value;
    
    // Validate required fields
    if (!apiKey || !apiEndpoint) {
        showNotification('error', 'API Key and Endpoint are required for testing');
        return;
    }
    
    // Show loading state
    showNotification('info', 'Testing API connection...', true);
    
    try {
        const response = await fetchWithRetry('/api/models/' + modelId + '/test-connection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                api_provider: apiProvider,
                api_key: apiKey,
                api_endpoint: apiEndpoint,
                api_model: apiModel,
                timeout: parseInt(apiTimeout)
            }),
            timeout: parseInt(apiTimeout) * 1000
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Connection test failed');
        }
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('success', 'API connection successful! Response time: ' + data.response_time + 'ms');
        } else {
            showNotification('error', data.message || 'API connection test failed');
        }
    } catch (error) {
        console.error('API test error:', error);
        showNotification('error', 'Connection failed: ' + error.message);
    }
}

// Save External API Configuration
async function saveExternalApiConfig() {
    const modelId = document.getElementById('modelIdInput').value;
    const apiProvider = document.getElementById('apiProvider').value;
    const apiKey = document.getElementById('apiKey').value;
    const apiEndpoint = document.getElementById('apiEndpoint').value;
    const apiModel = document.getElementById('apiModel').value;
    const apiTimeout = document.getElementById('apiTimeout').value;
    
    // Validate required fields
    if (!apiKey || !apiEndpoint) {
        showNotification('error', 'API Key and Endpoint are required');
        return;
    }
    
    // Show loading state
    showNotification('info', 'Saving API configuration...', true);
    
    try {
        const response = await fetchWithRetry('/api/models/' + modelId + '/switch-external', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                api_provider: apiProvider,
                api_key: apiKey,
                api_endpoint: apiEndpoint,
                api_model: apiModel,
                timeout: parseInt(apiTimeout)
            })
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to save configuration');
        }
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('success', 'API configuration saved successfully');
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('externalApiModal'));
            modal.hide();
            
            // Reload models list
            setTimeout(() => {
                loadModelsList();
            }, 1000);
        } else {
            showNotification('error', data.message || 'Failed to save configuration');
        }
    } catch (error) {
        console.error('Save configuration error:', error);
        showNotification('error', 'Failed to save configuration: ' + error.message);
    }
}

// Switch Model to Local Implementation
async function switchToLocal(modelId) {
    // Show confirmation
    if (!confirm(`Are you sure you want to switch model ${modelId} back to local implementation?`)) {
        return;
    }
    
    // Show loading state
    showNotification('info', 'Switching to local implementation...', true);
    
    try {
        const response = await fetchWithRetry('/api/models/' + modelId + '/switch-local', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Failed to switch to local');
        }
        
        const data = await response.json();
        
        if (data.success) {
            showNotification('success', 'Successfully switched to local implementation');
            
            // Reload models list
            setTimeout(() => {
                loadModelsList();
            }, 1000);
        } else {
            showNotification('error', data.message || 'Failed to switch to local');
        }
    } catch (error) {
        console.error('Switch to local error:', error);
        showNotification('error', 'Failed to switch to local: ' + error.message);
    }
}

// Update UI with connection status
function updateConnectionStatus() {
    // This function would be called after API operations to update UI elements
    // It's already handled by loadModelsList()
}

// Show Notification
function showNotification(type, message, isLoading = false) {
    // Create notification element if it doesn't exist
    let notificationContainer = document.getElementById('notificationContainer');
    if (!notificationContainer) {
        notificationContainer = document.createElement('div');
        notificationContainer.id = 'notificationContainer';
        document.body.appendChild(notificationContainer);
    }
    
    // Create notification
    const notification = document.createElement('div');
    notification.className = `notification notification-${type} ${isLoading ? 'notification-loading' : ''}`;
    
    // Add icon based on type
    let icon = '';
    switch (type) {
        case 'success':
            icon = '<i class="bi bi-check-circle"></i>';
            break;
        case 'error':
            icon = '<i class="bi bi-exclamation-circle"></i>';
            break;
        case 'info':
            icon = '<i class="bi bi-info-circle"></i>';
            break;
    }
    
    notification.innerHTML = `
        <div class="notification-icon">${icon}</div>
        <div class="notification-message">${message}</div>
        ${isLoading ? '<div class="notification-spinner"><i class="bi bi-spinner bi-spin"></i></div>' : ''}
    `;
    
    // Add to container
    notificationContainer.appendChild(notification);
    
    // Trigger animation
    setTimeout(() => {
        notification.classList.add('show');
    }, 10);
    
    // Auto remove notification unless it's a loading state
    if (!isLoading) {
        setTimeout(() => {
            notification.classList.remove('show');
            notification.classList.add('hide');
            
            setTimeout(() => {
                if (notificationContainer.contains(notification)) {
                    notificationContainer.removeChild(notification);
                }
            }, 300);
        }, 5000);
    }
    
    // Return notification element for manual removal
    return notification;
}

// Remove All Notifications
function removeAllNotifications() {
    const notificationContainer = document.getElementById('notificationContainer');
    if (notificationContainer) {
        notificationContainer.innerHTML = '';
    }
}

// Helper function for fetch with retry
async function fetchWithRetry(url, options = {}, retries = 3, retryDelay = 1000) {
    try {
        return await fetch(url, options);
    } catch (error) {
        if (retries > 0 && !error.toString().includes('AbortError')) {
            console.warn(`Request failed, retrying (${retries} attempts left)...`);
            await new Promise(resolve => setTimeout(resolve, retryDelay));
            return fetchWithRetry(url, options, retries - 1, retryDelay * 2);
        }
        throw error;
    }
}

// Initialize notification styles
function initializeNotificationStyles() {
    // Check if styles already exist
    if (document.getElementById('notificationStyles')) {
        return;
    }
    
    // Create style element
    const style = document.createElement('style');
    style.id = 'notificationStyles';
    style.textContent = `
        #notificationContainer {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 9999;
            display: flex;
            flex-direction: column;
            gap: 10px;
            min-width: 300px;
        }
        
        .notification {
            background-color: #fff;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
            padding: 12px 16px;
            display: flex;
            align-items: center;
            gap: 10px;
            opacity: 0;
            transform: translateX(100%);
            transition: opacity 0.3s ease, transform 0.3s ease;
        }
        
        .notification.show {
            opacity: 1;
            transform: translateX(0);
        }
        
        .notification.hide {
            opacity: 0;
            transform: translateX(100%);
        }
        
        .notification-icon {
            font-size: 18px;
        }
        
        .notification-message {
            flex: 1;
            font-size: 14px;
            line-height: 1.4;
        }
        
        .notification-spinner {
            font-size: 16px;
        }
        
        .notification-success {
            border-left: 4px solid #28a745;
        }
        
        .notification-success .notification-icon {
            color: #28a745;
        }
        
        .notification-error {
            border-left: 4px solid #dc3545;
        }
        
        .notification-error .notification-icon {
            color: #dc3545;
        }
        
        .notification-info {
            border-left: 4px solid #17a2b8;
        }
        
        .notification-info .notification-icon {
            color: #17a2b8;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .bi-spin {
            animation: spin 1s linear infinite;
        }
    `;
    
    // Add to document head
    document.head.appendChild(style);
}

// Initialize the module
(function() {
    // Add notification styles
    initializeNotificationStyles();
    
    // Add event listener for DOMContentLoaded to ensure the page is fully loaded
    document.addEventListener('DOMContentLoaded', function() {
        // Add click handler for modal close buttons
        document.querySelectorAll('.modal .btn-close').forEach(button => {
            button.addEventListener('click', removeAllNotifications);
        });
        
        // Add click handler for modal dismiss buttons
        document.querySelectorAll('.modal [data-bs-dismiss="modal"]').forEach(button => {
            button.addEventListener('click', removeAllNotifications);
        });
    });
})();