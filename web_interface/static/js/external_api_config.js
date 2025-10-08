// External API Configuration Functions
let currentModelId = null;

function showExternalApiModal(modelId) {
    currentModelId = modelId;
    document.getElementById('currentModelId').textContent = modelId;
    document.getElementById('modelIdInput').value = modelId;
    
    // Reset form
    document.getElementById('externalApiForm').reset();
    
    // Set default endpoint based on provider
    onApiProviderChange();
    
    const modal = new bootstrap.Modal(document.getElementById('externalApiModal'));
    modal.show();
}

function onApiProviderChange() {
    const provider = document.getElementById('apiProvider').value;
    const endpointInput = document.getElementById('apiEndpoint');
    const modelInput = document.getElementById('apiModel');
    
    // Set default endpoints and model names
    switch(provider) {
        case 'openai':
            endpointInput.value = 'https://api.openai.com/v1';
            modelInput.value = 'gpt-4o';
            break;
        case 'anthropic':
            endpointInput.value = 'https://api.anthropic.com/v1';
            modelInput.value = 'claude-3-opus-20240229';
            break;
        case 'google':
            endpointInput.value = 'https://generativelanguage.googleapis.com/v1';
            modelInput.value = 'gemini-pro';
            break;
        case 'huggingface':
            endpointInput.value = 'https://api-inference.huggingface.co/models';
            modelInput.value = 'meta-llama/Llama-3-70b-chat-hf';
            break;
        case 'custom':
            endpointInput.value = '';
            modelInput.value = '';
            break;
    }
}

async function testExternalApi() {
    const provider = document.getElementById('apiProvider').value;
    const apiKey = document.getElementById('apiKey').value;
    const endpoint = document.getElementById('apiEndpoint').value;
    
    if (!apiKey || !endpoint) {
        alert('Please fill in all required fields');
        return;
    }
    
    try {
        const result = await fetchWithRetry('/api/models/test-connection', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                provider: provider,
                api_key: apiKey,
                endpoint: endpoint
            })
        });
        
        const data = await result.json();
        if (data.success) {
            alert('Connection test successful!');
        } else {
            alert('Connection failed: ' + data.error);
        }
    } catch (error) {
        alert('Error during connection test: ' + error.message);
    }
}

async function saveExternalApiConfig() {
    const modelId = document.getElementById('modelIdInput').value;
    const provider = document.getElementById('apiProvider').value;
    const apiKey = document.getElementById('apiKey').value;
    const apiModel = document.getElementById('apiModel').value;
    const endpoint = document.getElementById('apiEndpoint').value;
    const timeout = document.getElementById('apiTimeout').value;
    
    if (!modelId || !apiKey || !apiModel || !endpoint) {
        alert('Please fill in all required fields');
        return;
    }
    
    try {
        showNotification('info', 'Switching to external API. This may take a few moments...');
        
        const result = await fetchWithRetry(`/api/models/${modelId}/switch-external`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                provider: provider,
                api_key: apiKey,
                model_name: apiModel,
                endpoint: endpoint,
                timeout: parseInt(timeout)
            })
        });
        
        const data = await result.json();
        if (data.success) {
            // Update UI to reflect API connection
            updateModelConnectionStatus(modelId, true, provider);
            showNotification('success', 'Successfully switched to external API');
            
            // Close modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('externalApiModal'));
            modal.hide();
        } else {
            showNotification('error', 'Failed to switch to external API: ' + data.error);
        }
    } catch (error) {
        showNotification('error', 'Error during switch: ' + error.message);
    }
}

async function switchToLocal(modelId) {
    if (!confirm(`Are you sure you want to switch model ${modelId} back to local implementation?`)) {
        return;
    }
    
    try {
        showNotification('info', 'Switching back to local implementation...');
        
        const result = await fetchWithRetry(`/api/models/${modelId}/switch-local`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        const data = await result.json();
        if (data.success) {
            // Update UI to reflect local connection
            updateModelConnectionStatus(modelId, false);
            showNotification('success', 'Successfully switched to local implementation');
        } else {
            showNotification('error', 'Failed to switch to local: ' + data.error);
        }
    } catch (error) {
        showNotification('error', 'Error during switch: ' + error.message);
    }
}

function updateModelConnectionStatus(modelId, isExternal, provider = '') {
    // Find the model row
    const rows = document.querySelectorAll('#modelsTableBody tr');
    let targetRow = null;
    
    for (let row of rows) {
        const idCell = row.querySelector('td:first-child strong');
        if (idCell && idCell.textContent === modelId) {
            targetRow = row;
            break;
        }
    }
    
    if (!targetRow) return;
    
    // Update connection status
    const cells = targetRow.querySelectorAll('td');
    if (cells.length >= 7) {
        cells[6].innerHTML = isExternal 
            ? `<span class="badge bg-primary">API (${provider})</span>` 
            : '<span class="badge bg-success">Local</span>';
    }
    
    // Update actions buttons
    if (cells.length >= 9) {
        const actionsCell = cells[8];
        if (isExternal) {
            // Replace "Switch to API" with "Switch to Local"
            actionsCell.innerHTML = actionsCell.innerHTML.replace(
                /Switch to API/g, 'Switch to Local'
            ).replace(
                /showExternalApiModal/g, 'switchToLocal'
            ).replace(
                /btn-outline-primary/g, 'btn-outline-success'
            );
        } else {
            // Replace "Switch to Local" with "Switch to API"
            actionsCell.innerHTML = actionsCell.innerHTML.replace(
                /Switch to Local/g, 'Switch to API'
            ).replace(
                /switchToLocal/g, 'showExternalApiModal'
            ).replace(
                /btn-outline-success/g, 'btn-outline-primary'
            );
        }
    }
}

// Helper function to show notifications
function showNotification(type, message) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification position-fixed top-5 end-5 p-4 rounded shadow-lg z-50 animate-fade-in transition-all duration-300 ${getNotificationClass(type)}`;
    notification.innerHTML = `
        <div class="d-flex align-items-center">
            <i class="${getNotificationIcon(type)} me-3"></i>
            <span>${message}</span>
            <button class="ml-auto btn-close" onclick="this.parentElement.parentElement.remove()"></button>
        </div>
    `;
    
    // Add to body
    document.body.appendChild(notification);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.classList.add('animate-fade-out');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 5000);
}

function getNotificationClass(type) {
    switch(type) {
        case 'success': return 'bg-success text-white';
        case 'error': return 'bg-danger text-white';
        case 'warning': return 'bg-warning text-dark';
        case 'info': return 'bg-info text-white';
        default: return 'bg-dark text-white';
    }
}

function getNotificationIcon(type) {
    switch(type) {
        case 'success': return 'bi bi-check-circle';
        case 'error': return 'bi bi-exclamation-circle';
        case 'warning': return 'bi bi-exclamation-triangle';
        case 'info': return 'bi bi-info-circle';
        default: return 'bi bi-bell';
    }
}

// Initialize external API features when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    try {
        // Add Connection column to Models table
        const modelsTableBody = document.querySelector('#modelsTableBody');
        if (!modelsTableBody) {
            console.log('Models table not found. Skipping external API configuration.');
            return;
        }
        
        const modelsTable = modelsTableBody.parentElement;
        const theadRow = modelsTable.querySelector('thead tr');
    
        // Check if Connection column already exists
        const connectionColumnExists = Array.from(theadRow.children).some(th => 
            th.textContent.trim() === 'Connection'
        );
    
        if (!connectionColumnExists) {
            // Insert Connection column before Last Updated column
            const lastUpdatedIndex = Array.from(theadRow.children).findIndex(th => 
                th.textContent.trim() === 'Last Updated'
            );
            
            if (lastUpdatedIndex > 0) {
                const connectionHeader = document.createElement('th');
                connectionHeader.textContent = 'Connection';
                theadRow.insertBefore(connectionHeader, theadRow.children[lastUpdatedIndex]);
                
                // Add Connection cell to each row
                const rows = document.querySelectorAll('#modelsTableBody tr');
                rows.forEach(row => {
                    const cell = document.createElement('td');
                    cell.innerHTML = '<span class="badge bg-success">Local</span>';
                    row.insertBefore(cell, row.children[lastUpdatedIndex]);
                    
                    // Add Switch to API button to Actions column
                    const actionsCell = row.lastElementChild;
                    const switchButton = document.createElement('button');
                    switchButton.className = 'btn btn-sm btn-outline-primary ml-1';
                    switchButton.textContent = 'Switch to API';
                    const modelId = row.querySelector('td:first-child strong').textContent;
                    switchButton.onclick = function() { showExternalApiModal(modelId); };
                    actionsCell.appendChild(switchButton);
                });
            }
        }
    } catch (error) {
        console.error('Error initializing external API configuration:', error);
    }
});

// Add CSS for notifications
const style = document.createElement('style');
style.textContent = `
    .notification {
        min-width: 300px;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeOut {
        from { opacity: 1; transform: translateY(0); }
        to { opacity: 0; transform: translateY(-10px); }
    }
    
    .animate-fade-in {
        animation: fadeIn 0.3s ease-out;
    }
    
    .animate-fade-out {
        animation: fadeOut 0.3s ease-in;
    }
`;
document.head.appendChild(style);