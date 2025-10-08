// Models Management System
// Handles model operations, external API configuration, and system integration

// Initialize models management when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize model counts and data
    initializeModels();
    
    // Set up event listeners for tab switching
    document.getElementById('models-tab')?.addEventListener('shown.bs.tab', function() {
        updateModelCounts();
        refreshModels(true); // Force refresh when tab becomes active
    });
});

// Initialize models data
function initializeModels() {
    // In a real system, this would fetch data from the backend
    // Here we just update the counts
    updateModelCounts();
}

// Show add model modal
function showAddModelModal() {
    const modal = new bootstrap.Modal(document.getElementById('addModelModal'));
    modal.show();
}

// Add a new model
function addNewModel() {
    const modelId = document.getElementById('newModelId').value;
    const modelName = document.getElementById('newModelName').value;
    const modelType = document.getElementById('newModelType').value;

    // Validate input
    if (!modelId || !modelName) {
        showNotification('Please fill in all required fields', 'error');
        return;
    }

    showNotification(`Adding model ${modelName}...`, 'info');
    
    // Call backend API to add model
    fetch('/api/models/add', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            model_id: modelId,
            model_type: modelType,
            config: {
                name: modelName,
                is_local: modelType === 'local',
                status: 'loading'
            }
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // Add new row to table
            const tbody = document.getElementById('modelsTableBody');
            const newRow = tbody.insertRow();
            newRow.innerHTML = `
                <td><strong>${modelId}</strong></td>
                <td>${modelName}</td>
                <td>${modelType.charAt(0).toUpperCase() + modelType.slice(1)}</td>
                <td><span class="badge bg-warning">Loading</span></td>
                <td>0.0 GB</td>
                <td>${modelType === 'local' ? 'Local' : 'API'}</td>
                <td>${modelType === 'local' ? 'Local' : 'External'}</td>
                <td>${new Date().toISOString().split('T')[0]}</td>
                <td>
                    <button class="btn btn-sm btn-outline-dark" onclick="viewModelDetails('${modelId}', '${modelName}', '${modelType.charAt(0).toUpperCase() + modelType.slice(1)}', 'Loading', '0.0 GB', '0%', '${modelType === 'local' ? 'Local' : 'API'}', 'REST API', 'localhost', '0', 'Disconnected', '-', '${modelId.toLowerCase()}_model', '1.0.0', '${new Date().toISOString().split('T')[0]}')">
                        View Details
                    </button>
                    <button class="btn btn-sm btn-outline-dark" onclick="showExternalApiModal('${modelId}', '${modelName}')">
                        Switch API
                    </button>
                    <button class="btn btn-sm btn-outline-dark" onclick="restartModel('${modelId}')">
                        Restart
                    </button>
                    <button class="btn btn-sm btn-outline-danger" onclick="deleteModel('${modelId}', '${modelName}')">
                        Delete
                    </button>
                </td>
            `;

            // Update model counts
            updateModelCounts();

            // Close modal and reset form
            const modal = bootstrap.Modal.getInstance(document.getElementById('addModelModal'));
            modal.hide();
            document.getElementById('addModelForm').reset();

            showNotification(`Model ${modelName} added successfully!`, 'success');
        } else {
            showNotification(`Failed to add model: ${data.message || 'Unknown error'}`, 'error');
        }
    })
    .catch(error => {
        showNotification(`Error adding model: ${error.message}`, 'error');
    });
}

// Refresh models status
function refreshModels(force = false) {
    showNotification('Refreshing model status...', 'info');
    
    // Call API to get latest model status
    fetch('/api/models', {
        method: 'GET',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success' && data.models) {
            // Update the models table
            updateModelsTable(data.models);
            // Update model counts
            updateModelCounts();
            showNotification('Model status refreshed successfully', 'success');
        } else {
            // Fallback to simulation if API call fails
            simulateModelRefresh();
        }
    })
    .catch(error => {
        console.error('Error refreshing models:', error);
        // Fallback to simulation
        simulateModelRefresh();
    });
}

// Update models table with data from API
function updateModelsTable(models) {
    const tbody = document.getElementById('modelsTableBody');
    if (!tbody) return;
    
    // Clear existing rows
    tbody.innerHTML = '';
    
    // Add new rows
    models.forEach(model => {
        // Check if model is valid
        if (!model || typeof model !== 'object') {
            console.warn('Invalid model data:', model);
            return;
        }
        
        const newRow = tbody.insertRow();
        const statusClass = getStatusClass(model.status || 'Unknown');
        const isExternal = model.api_source === 'external' || model.provider !== 'local';
        const modelId = model.id || 'unknown';
        
        newRow.innerHTML = `
            <td><strong>${modelId}</strong></td>
            <td>${model.name || 'Unnamed Model'}</td>
            <td>${model.type || 'Unknown'}</td>
            <td><span class="badge ${statusClass}">${model.status || 'Unknown'}</span></td>
            <td>${model.memory || '0.0 GB'}</td>
            <td>${isExternal ? 'External' : 'Local'}</td>
            <td>${model.provider || 'Local'}</td>
            <td>${model.last_updated || new Date().toISOString().split('T')[0]}</td>
            <td>
                <button class="btn btn-sm btn-outline-dark" onclick="viewModelDetails('${modelId}', '${model.name || 'Unnamed Model'}', '${model.type || 'Unknown'}', '${model.status || 'Unknown'}', '${model.memory || '0.0 GB'}', '${model.cpu_usage || '0%'}', '${isExternal ? 'External' : 'Local'}', 'REST API', 'localhost', '${model.port || '0'}', '${model.connection_status || 'Unknown'}', '${model.latency || '-'}', '${model.component || modelId.toLowerCase()}_model', '${model.version || '1.0.0'}', '${model.last_updated || new Date().toISOString().split('T')[0]}')">
                    View Details
                </button>
                <button class="btn btn-sm btn-outline-dark" onclick="showExternalApiModal('${modelId}', '${model.name || 'Unnamed Model'}')">
                    ${isExternal ? 'Switch to Local' : 'Switch API'}
                </button>
                <button class="btn btn-sm btn-outline-dark" onclick="restartModel('${modelId}')">
                    Restart
                </button>
                <button class="btn btn-sm btn-outline-danger" onclick="deleteModel('${modelId}', '${model.name || 'Unnamed Model'}')">
                    Delete
                </button>
            </td>
        `;
    });
}

// Simulate model refresh when API is not available
function simulateModelRefresh() {
    // Simulate refresh with random updates
    const badges = document.querySelectorAll('#modelsTableBody .badge');
    badges.forEach(badge => {
        const statuses = ['Active', 'Loading', 'Error', 'Training'];
        const colors = ['bg-dark', 'bg-warning', 'bg-danger', 'bg-info'];
        const randomIndex = Math.floor(Math.random() * statuses.length);
        badge.textContent = statuses[randomIndex];
        badge.className = `badge ${colors[randomIndex]}`;
    });

    updateModelCounts();
    showNotification('Model status refreshed successfully', 'success');
}

// Update model counts display
function updateModelCounts() {
    const rows = document.querySelectorAll('#modelsTableBody tr');
    const totalModels = rows.length;
    let activeModels = 0;
    let trainingModels = 0;
    let errorModels = 0;

    rows.forEach(row => {
        const badge = row.querySelector('.badge');
        // Check if badge exists before accessing textContent
        if (badge) {
            const status = badge.textContent;
            if (status === 'Active') activeModels++;
            else if (status === 'Training') trainingModels++;
            else if (status === 'Error') errorModels++;
        }
    });

    // Update the display
    if (document.getElementById('totalModels')) {
        document.getElementById('totalModels').textContent = totalModels;
    }
    if (document.getElementById('activeModels')) {
        document.getElementById('activeModels').textContent = activeModels;
    }
    if (document.getElementById('trainingModels')) {
        document.getElementById('trainingModels').textContent = trainingModels;
    }
    if (document.getElementById('errorModels')) {
        document.getElementById('errorModels').textContent = errorModels;
    }
}

// Export models configuration
function exportModels() {
    const models = [];
    const rows = document.querySelectorAll('#modelsTableBody tr');
    
    rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        models.push({
            id: cells[0].textContent,
            name: cells[1].textContent,
            type: cells[2].textContent,
            status: cells[3].textContent,
            memory: cells[4].textContent,
            api_source: cells[5].textContent,
            provider: cells[6].textContent,
            last_updated: cells[7].textContent
        });
    });

    // Create and download the JSON file
    const blob = new Blob([JSON.stringify(models, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'models-config.json';
    a.click();
    URL.revokeObjectURL(url);
    
    showNotification('Models configuration exported successfully', 'success');
}

// Import models configuration
function importModels() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                try {
                    const models = JSON.parse(e.target.result);
                    showNotification(`Importing ${models.length} models...`, 'info');
                    
                    // Call API to import models
                    fetch('/api/models/import', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ models: models })
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'success') {
                            // Refresh the models table
                            refreshModels(true);
                            showNotification(`Successfully imported ${models.length} models`, 'success');
                        } else {
                            showNotification(`Failed to import models: ${data.message}`, 'error');
                        }
                    })
                    .catch(error => {
                        showNotification(`Error importing models: ${error.message}`, 'error');
                    });
                } catch (error) {
                    showNotification('Error importing file: Invalid JSON format', 'error');
                }
            };
            reader.readAsText(file);
        }
    };
    input.click();
}

// View model details
function viewModelDetails(id, name, type, status, memory, cpuUsage, provider, apiType, host, port, connectionStatus, latency, component, version, lastUpdated) {
    // Set the model details in the modal
    if (document.getElementById('modelId')) document.getElementById('modelId').textContent = id;
    if (document.getElementById('modelName')) document.getElementById('modelName').textContent = name;
    if (document.getElementById('modelType')) document.getElementById('modelType').textContent = type;
    if (document.getElementById('modelStatus')) document.getElementById('modelStatus').textContent = status;
    if (document.getElementById('modelMemory')) document.getElementById('modelMemory').textContent = memory;
    if (document.getElementById('modelCpuUsage')) document.getElementById('modelCpuUsage').textContent = cpuUsage;
    if (document.getElementById('modelProvider')) document.getElementById('modelProvider').textContent = provider;
    if (document.getElementById('modelApiType')) document.getElementById('modelApiType').textContent = apiType;
    if (document.getElementById('modelHost')) document.getElementById('modelHost').textContent = host;
    if (document.getElementById('modelPort')) document.getElementById('modelPort').textContent = port;
    if (document.getElementById('modelConnectionStatus')) document.getElementById('modelConnectionStatus').textContent = connectionStatus;
    if (document.getElementById('modelLatency')) document.getElementById('modelLatency').textContent = latency;
    if (document.getElementById('modelComponent')) document.getElementById('modelComponent').textContent = component;
    if (document.getElementById('modelVersion')) document.getElementById('modelVersion').textContent = version;
    if (document.getElementById('modelLastUpdated')) document.getElementById('modelLastUpdated').textContent = lastUpdated;
    
    // Show the modal
    const modelDetailsModal = new bootstrap.Modal(document.getElementById('modelDetailsModal'));
    modelDetailsModal.show();
}

// Show external API configuration modal
function showExternalApiModal(modelId, modelName) {
    // Set model ID and name
    if (document.getElementById('externalApiModelId')) {
        document.getElementById('externalApiModelId').value = modelId;
        document.getElementById('externalApiModelId').textContent = modelId;
    }
    if (document.getElementById('externalApiModelName')) {
        document.getElementById('externalApiModelName').textContent = modelName;
    }
    
    // Reset form
    if (document.getElementById('externalApiForm')) {
        document.getElementById('externalApiForm').reset();
        onApiProviderChange(); // Set default values
    }
    
    // Check if model is already using external API
    const rows = document.querySelectorAll('#modelsTableBody tr');
    rows.forEach(row => {
        if (row.querySelector('td:first-child strong').textContent === modelId) {
            const apiSourceCell = row.querySelector('td:nth-child(6)');
            const providerCell = row.querySelector('td:nth-child(7)');
            
            if (apiSourceCell && apiSourceCell.textContent === 'External' && providerCell) {
                // Set provider if model is already using external API
                const provider = providerCell.textContent.toLowerCase();
                if (document.getElementById('externalApiProvider')) {
                    document.getElementById('externalApiProvider').value = provider;
                    onApiProviderChange(); // Update other fields based on provider
                }
            }
        }
    });
    
    // Show the modal
    const externalApiModal = new bootstrap.Modal(document.getElementById('externalApiModal'));
    externalApiModal.show();
}

// Restart a model
function restartModel(modelId) {
    if (confirm(`Are you sure you want to restart model ${modelId}?`)) {
        showNotification(`Restarting model ${modelId}...`, 'info');
        
        // Update UI to show restarting status
        const rows = document.querySelectorAll('#modelsTableBody tr');
        rows.forEach(row => {
            if (row.querySelector('td:first-child strong').textContent === modelId) {
                const statusBadge = row.querySelector('.badge');
                statusBadge.className = 'badge bg-warning';
                statusBadge.textContent = 'Restarting';
            }
        });
        
        // Call API to restart the model
        fetch(`/api/models/${modelId}/restart`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        }).then(response => {
            if (response.ok) {
                return response.json();
            }
            throw new Error('Network response was not ok');
        }).then(data => {
            if (data.status === 'success') {
                // Update UI to show active status after successful restart
                rows.forEach(row => {
                    if (row.querySelector('td:first-child strong').textContent === modelId) {
                        const statusBadge = row.querySelector('.badge');
                        statusBadge.className = 'badge bg-dark';
                        statusBadge.textContent = 'Active';
                    }
                });
                showNotification(`Model ${modelId} restarted successfully`, 'success');
            } else {
                showNotification(`Failed to restart model ${modelId}: ${data.message}`, 'error');
                // Revert status if failed
                rows.forEach(row => {
                    if (row.querySelector('td:first-child strong').textContent === modelId) {
                        const statusBadge = row.querySelector('.badge');
                        statusBadge.className = 'badge bg-danger';
                        statusBadge.textContent = 'Error';
                    }
                });
            }
        }).catch(error => {
            console.error('Error restarting model:', error);
            showNotification(`Failed to restart model ${modelId}: ${error.message}`, 'error');
            // Revert status if failed
            rows.forEach(row => {
                if (row.querySelector('td:first-child strong').textContent === modelId) {
                    const statusBadge = row.querySelector('.badge');
                    statusBadge.className = 'badge bg-danger';
                    statusBadge.textContent = 'Error';
                }
            });
        });
    }
}

// Handle API provider change
function onApiProviderChange() {
    const provider = document.getElementById('externalApiProvider')?.value;
    const modelInput = document.getElementById('externalApiModel');
    const baseUrlInput = document.getElementById('externalApiBaseUrl');
    
    if (!modelInput || !baseUrlInput) return;
    
    // Set default values based on provider
    switch(provider) {
        case 'openai':
            modelInput.value = 'gpt-4o';
            baseUrlInput.value = 'https://api.openai.com/v1';
            break;
        case 'anthropic':
            modelInput.value = 'claude-3-opus-20240229';
            baseUrlInput.value = 'https://api.anthropic.com/v1';
            break;
        case 'google':
            modelInput.value = 'gemini-1.5-pro-latest';
            baseUrlInput.value = 'https://generativelanguage.googleapis.com/v1beta';
            break;
        case 'huggingface':
            modelInput.value = 'meta-llama/Llama-3-70b-chat-hf';
            baseUrlInput.value = 'https://api-inference.huggingface.co/models';
            break;
        case 'custom':
            modelInput.value = '';
            baseUrlInput.value = '';
            break;
    }
}

// Save external API configuration
function saveExternalApiConfig() {
    const modelId = document.getElementById('externalApiModelId')?.value;
    const provider = document.getElementById('externalApiProvider')?.value;
    const apiKey = document.getElementById('externalApiKey')?.value;
    const apiModel = document.getElementById('externalApiModel')?.value;
    const baseUrl = document.getElementById('externalApiBaseUrl')?.value;
    const timeout = document.getElementById('externalApiTimeout')?.value;
    
    // Validate input
    if (!modelId || !apiKey || !apiModel || !baseUrl) {
        showNotification('Please fill in all required fields', 'error');
        return;
    }
    
    // Validate timeout parameter
    const timeoutValue = timeout ? parseInt(timeout) : 30;
    if (isNaN(timeoutValue) || timeoutValue <= 0 || timeoutValue > 120) {
        showNotification('Timeout must be between 1 and 120 seconds', 'error');
        return;
    }
    
    showNotification(`Saving external API configuration for model ${modelId}...`, 'info');
    
    // Send configuration to server
    fetch(`/api/models/${modelId}/api-config`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            provider: provider,
            api_key: apiKey,
            model: apiModel,
            base_url: baseUrl,
            timeout: timeoutValue
        })
    }).then(response => {
        if (response.ok) {
            return response.json();
        }
        throw new Error('Network response was not ok');
    }).then(data => {
        if (data.status === 'success') {
            showNotification(`External API configuration saved successfully for model ${modelId}`, 'success');
            
            // Update the model's provider in the table
            const rows = document.querySelectorAll('#modelsTableBody tr');
            rows.forEach(row => {
                if (row.querySelector('td:first-child strong').textContent === modelId) {
                    // Update API source column
                    const apiSourceCell = row.querySelector('td:nth-child(6)');
                    if (apiSourceCell) {
                        apiSourceCell.textContent = 'External';
                        apiSourceCell.className = 'text-success';
                    }
                    
                    // Update provider column
                    const providerCell = row.querySelector('td:nth-child(7)');
                    if (providerCell) {
                        providerCell.textContent = provider;
                    }
                    
                    // Change status to indicate API is active
                    const statusBadge = row.querySelector('.badge');
                    if (statusBadge) {
                        statusBadge.className = 'badge bg-dark';
                        statusBadge.textContent = 'Active';
                    }
                }
            });
            
            // Close the modal
            const externalApiModal = bootstrap.Modal.getInstance(document.getElementById('externalApiModal'));
            if (externalApiModal) {
                externalApiModal.hide();
            }
        } else {
            showNotification(`Failed to save API configuration: ${data.message || 'Unknown error'}`, 'error');
        }
    }).catch(error => {
        console.error('Error saving API configuration:', error);
        showNotification(`Failed to save API configuration: ${error.message}`, 'error');
    });
}

// Test external API connection
function testExternalApiConnection() {
    const modelId = document.getElementById('externalApiModelId')?.value;
    const provider = document.getElementById('externalApiProvider')?.value;
    const apiKey = document.getElementById('externalApiKey')?.value;
    const apiModel = document.getElementById('externalApiModel')?.value;
    const baseUrl = document.getElementById('externalApiBaseUrl')?.value;
    const timeout = document.getElementById('externalApiTimeout')?.value;
    
    // Validate input
    if (!apiKey || !apiModel || !baseUrl) {
        showNotification('Please fill in all required fields', 'error');
        return;
    }
    
    // Validate timeout parameter
    const timeoutValue = timeout ? parseInt(timeout) : 30;
    if (isNaN(timeoutValue) || timeoutValue <= 0 || timeoutValue > 120) {
        showNotification('Timeout must be between 1 and 120 seconds', 'error');
        return;
    }
    
    showNotification(`Testing external API connection for model ${modelId}...`, 'info');
    
    // Test connection to external API
    fetch('/api/models/test-connection', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            provider: provider,
            api_key: apiKey,
            model: apiModel,
            base_url: baseUrl,
            timeout: timeoutValue
        })
    }).then(response => {
        if (response.ok) {
            return response.json();
        }
        throw new Error('Network response was not ok');
    }).then(data => {
        if (data.status === 'success') {
            showNotification(`External API connection test successful for model ${modelId}! Latency: ${data.latency}ms`, 'success');
        } else {
            showNotification(`External API connection test failed: ${data.message || 'Unknown error'}`, 'error');
        }
    }).catch(error => {
        console.error('Error testing API connection:', error);
        showNotification(`Failed to test API connection: ${error.message}`, 'error');
    });
}

// Switch to local mode
function switchToLocal(modelId) {
    if (confirm(`Are you sure you want to switch model ${modelId} back to local implementation?`)) {
        showNotification(`Switching model ${modelId} to local implementation...`, 'info');
        
        // Call API to switch to local
        fetch(`/api/models/${modelId}/switch-local`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        }).then(response => {
            if (response.ok) {
                return response.json();
            }
            throw new Error('Network response was not ok');
        }).then(data => {
            if (data.status === 'success') {
                // Update the model's provider in the table
                const rows = document.querySelectorAll('#modelsTableBody tr');
                rows.forEach(row => {
                    if (row.querySelector('td:first-child strong').textContent === modelId) {
                        // Update API source column
                        const apiSourceCell = row.querySelector('td:nth-child(6)');
                        if (apiSourceCell) {
                            apiSourceCell.textContent = 'Local';
                            apiSourceCell.className = '';
                        }
                        
                        // Update provider column
                        const providerCell = row.querySelector('td:nth-child(7)');
                        if (providerCell) {
                            providerCell.textContent = 'Local';
                        }
                        
                        // Change status to indicate local is active
                        const statusBadge = row.querySelector('.badge');
                        if (statusBadge) {
                            statusBadge.className = 'badge bg-dark';
                            statusBadge.textContent = 'Active';
                        }
                    }
                });
                
                showNotification(`Model ${modelId} switched to local implementation successfully`, 'success');
            } else {
                showNotification(`Failed to switch to local: ${data.message}`, 'error');
            }
        }).catch(error => {
            console.error('Error switching to local:', error);
            showNotification(`Failed to switch to local: ${error.message}`, 'error');
        });
    }
}

// Delete a model
function deleteModel(modelId, modelName) {
    if (confirm(`Are you sure you want to delete model ${modelId} - ${modelName}? This action cannot be undone.`)) {
        showNotification(`Deleting model ${modelId}...`, 'info');
        
        // Call API to delete the model
        fetch(`/api/models/${modelId}/delete`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        }).then(response => {
            if (response.ok) {
                return response.json();
            }
            throw new Error('Network response was not ok');
        }).then(data => {
            if (data.status === 'success') {
                // Remove the model row from the table
                const rows = document.querySelectorAll('#modelsTableBody tr');
                rows.forEach(row => {
                    if (row.querySelector('td:first-child strong').textContent === modelId) {
                        row.remove();
                    }
                });
                
                // Update model counts
                updateModelCounts();
                
                showNotification(`Model ${modelId} deleted successfully`, 'success');
            } else {
                showNotification(`Failed to delete model ${modelId}: ${data.message}`, 'error');
            }
        }).catch(error => {
            console.error('Error deleting model:', error);
            showNotification(`Failed to delete model ${modelId}: ${error.message}`, 'error');
        });
    }
}

// Helper function to get status class
function getStatusClass(status) {
    switch(status.toLowerCase()) {
        case 'active': return 'bg-dark';
        case 'loading': 
        case 'restarting': return 'bg-warning';
        case 'error': return 'bg-danger';
        case 'training': return 'bg-info';
        default: return 'bg-secondary';
    }
}

// Show notification
function showNotification(message, type) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-dark alert-dismissible fade show position-fixed`;
    notification.style.top = '20px';
    notification.style.right = '20px';
    notification.style.zIndex = '9999';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    // Add notification to body
    document.body.appendChild(notification);
    
    // Remove after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}