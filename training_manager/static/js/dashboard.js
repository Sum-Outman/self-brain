// Dashboard JavaScript

// DOM Elements
const taskSearchInput = document.getElementById('task-search');
const refreshTasksBtn = document.getElementById('refresh-tasks');
const saveResourceLimitsBtn = document.getElementById('save-resource-limits');
const cpuLimitSlider = document.getElementById('cpu-limit-slider');
const memoryLimitSlider = document.getElementById('memory-limit-slider');
const cpuLimitValue = document.getElementById('cpu-limit-value');
const memoryLimitValue = document.getElementById('memory-limit-value');
const createTaskForm = document.getElementById('create-task-form');
const scheduleTaskCheckbox = document.getElementById('schedule-task');
const scheduleOptionsDiv = document.getElementById('schedule-options');
const resetFormBtn = document.getElementById('reset-form');
const taskModal = document.getElementById('task-modal');
const closeModalBtn = document.getElementById('close-modal');
const modalCloseBtn = document.getElementById('modal-close-btn');
const modalStopBtn = document.getElementById('modal-stop-btn');
const notification = document.getElementById('notification');
const notificationMessage = document.getElementById('notification-message');
const notificationIcon = document.getElementById('notification-icon');
const notificationCloseBtn = document.getElementById('close-notification');
const mobileMenuBtn = document.getElementById('mobile-menu-btn');
const sidebar = document.querySelector('aside');
const refreshResourcesBtn = document.getElementById('refresh-resources');
const clearHistoryBtn = document.getElementById('clear-history');

// Current task ID for modal
let currentTaskId = null;

// Initialize dashboard
function initDashboard() {
    // Set current time
    updateCurrentTime();
    setInterval(updateCurrentTime, 60000);

    // Event listeners
    if (taskSearchInput) taskSearchInput.addEventListener('input', searchTasks);
    if (refreshTasksBtn) refreshTasksBtn.addEventListener('click', refreshTasks);
    if (cpuLimitSlider) cpuLimitSlider.addEventListener('input', updateCpuLimitValue);
    if (memoryLimitSlider) memoryLimitSlider.addEventListener('input', updateMemoryLimitValue);
    if (saveResourceLimitsBtn) saveResourceLimitsBtn.addEventListener('click', saveResourceLimits);
    if (createTaskForm) createTaskForm.addEventListener('submit', handleCreateTask);
    if (scheduleTaskCheckbox) scheduleTaskCheckbox.addEventListener('change', toggleScheduleOptions);
    if (resetFormBtn) resetFormBtn.addEventListener('click', resetForm);
    if (closeModalBtn) closeModalBtn.addEventListener('click', closeTaskModal);
    if (modalCloseBtn) modalCloseBtn.addEventListener('click', closeTaskModal);
    if (modalStopBtn) modalStopBtn.addEventListener('click', stopCurrentTask);
    if (notificationCloseBtn) notificationCloseBtn.addEventListener('click', hideNotification);
    if (mobileMenuBtn) mobileMenuBtn.addEventListener('click', toggleMobileMenu);
    if (refreshResourcesBtn) refreshResourcesBtn.addEventListener('click', refreshResources);
    if (clearHistoryBtn) clearHistoryBtn.addEventListener('click', clearHistory);

    // Add event listeners to view task buttons
    document.querySelectorAll('.action-btn.details-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const taskId = btn.getAttribute('data-task-id');
            openTaskModal(taskId);
        });
    });

    // Add event listeners to stop task buttons
    document.querySelectorAll('.action-btn.stop-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const taskId = btn.getAttribute('data-task-id');
            stopTask(taskId);
        });
    });

    // Set up periodic refresh
    setInterval(refreshDashboard, 10000); // Refresh every 10 seconds
}

// Update current time
function updateCurrentTime() {
    const now = new Date();
    const timeElement = document.getElementById('current-time');
    if (timeElement) {
        timeElement.textContent = now.toLocaleTimeString();
    }
}

// Search tasks
function searchTasks() {
    const searchTerm = taskSearchInput.value.toLowerCase();
    const taskRows = document.querySelectorAll('#active-tasks-table tr');
    
    taskRows.forEach(row => {
        const taskId = row.querySelector('td:first-child')?.textContent || '';
        const models = row.querySelector('td:nth-child(2)')?.textContent || '';
        
        if (taskId.toLowerCase().includes(searchTerm) || models.toLowerCase().includes(searchTerm)) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

// Refresh tasks
function refreshTasks() {
    showLoadingIndicator();
    
    // Simulate API call
    setTimeout(() => {
        hideLoadingIndicator();
        showNotification('Tasks refreshed successfully', 'success');
        // In a real app, you would fetch updated task data here
    }, 800);
}

// Update CPU limit value
function updateCpuLimitValue() {
    cpuLimitValue.textContent = `${cpuLimitSlider.value}%`;
}

// Update memory limit value
function updateMemoryLimitValue() {
    memoryLimitValue.textContent = `${memoryLimitSlider.value}%`;
}

// Save resource limits
function saveResourceLimits() {
    const cpuLimit = cpuLimitSlider.value;
    const memoryLimit = memoryLimitSlider.value;
    
    // Simulate API call
    fetch('/api/settings/resource-limits', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            cpu_limit: cpuLimit,
            memory_limit: memoryLimit
        })
    })
    .then(response => {
        if (!response.ok) throw new Error('Failed to save limits');
        return response.json();
    })
    .then(data => {
        showNotification('Resource limits saved successfully', 'success');
    })
    .catch(error => {
        showNotification('Failed to save resource limits', 'error');
        console.error('Error saving resource limits:', error);
    });
}

// Handle create task form submission
function handleCreateTask(event) {
    event.preventDefault();
    
    const formData = new FormData(createTaskForm);
    const selectedModels = Array.from(formData.getAll('models'));
    const trainingType = formData.get('training-type');
    const priority = formData.get('priority');
    const epochs = formData.get('epochs');
    const batchSize = formData.get('batch-size');
    const learningRate = formData.get('learning-rate');
    const knowledgeAssisted = formData.get('knowledge-assisted') === 'on';
    const scheduledTask = formData.get('schedule-task') === 'on';
    const scheduledTime = formData.get('scheduled-time');
    
    if (selectedModels.length === 0) {
        showNotification('Please select at least one model', 'error');
        return;
    }
    
    const taskData = {
        models: selectedModels,
        training_type: trainingType,
        priority: parseInt(priority),
        epochs: parseInt(epochs),
        batch_size: parseInt(batchSize),
        learning_rate: parseFloat(learningRate),
        knowledge_assisted: knowledgeAssisted,
        scheduled_task: scheduledTask
    };
    
    if (scheduledTask && scheduledTime) {
        taskData.scheduled_time = scheduledTime;
    }
    
    // Show loading indicator
    showLoadingIndicator();
    
    // Simulate API call
    fetch('/api/tasks/create', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(taskData)
    })
    .then(response => {
        if (!response.ok) throw new Error('Failed to create task');
        return response.json();
    })
    .then(data => {
        hideLoadingIndicator();
        resetForm();
        refreshTasks();
        showNotification('Training task created successfully', 'success');
    })
    .catch(error => {
        hideLoadingIndicator();
        showNotification('Failed to create training task', 'error');
        console.error('Error creating task:', error);
    });
}

// Toggle schedule options
function toggleScheduleOptions() {
    if (scheduleTaskCheckbox.checked) {
        scheduleOptionsDiv.classList.remove('hidden');
    } else {
        scheduleOptionsDiv.classList.add('hidden');
    }
}

// Reset form
function resetForm() {
    createTaskForm.reset();
    toggleScheduleOptions(); // Ensure schedule options are hidden
}

// Open task modal
function openTaskModal(taskId) {
    currentTaskId = taskId;
    
    // Set modal title
    const modalTitle = document.getElementById('modal-title');
    if (modalTitle) {
        modalTitle.textContent = `Task Details: ${taskId}`;
    }
    
    // Show loading indicator in modal
    const modalContent = document.getElementById('modal-content');
    if (modalContent) {
        modalContent.innerHTML = '<div class="flex items-center justify-center py-8"><i class="fa fa-spinner fa-spin text-2xl text-primary"></i></div>';
    }
    
    // Show modal
    taskModal.classList.remove('hidden');
    
    // Simulate API call to get task details
    fetch(`/api/tasks/${taskId}`)
    .then(response => {
        if (!response.ok) throw new Error('Failed to get task details');
        return response.json();
    })
    .then(taskDetails => {
        renderTaskDetails(taskDetails);
    })
    .catch(error => {
        if (modalContent) {
            modalContent.innerHTML = '<div class="text-center py-8 text-danger"><i class="fa fa-exclamation-circle text-2xl mb-2"></i><p>Failed to load task details</p></div>';
        }
        console.error('Error fetching task details:', error);
    });
}

// Render task details in modal
function renderTaskDetails(taskDetails) {
    const modalContent = document.getElementById('modal-content');
    if (!modalContent) return;
    
    modalContent.innerHTML = `
        <div class="space-y-6">
            <!-- Basic Information -->
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                    <h4 class="text-sm font-medium text-gray-500 mb-1">Task ID</h4>
                    <p class="font-mono text-sm">${taskDetails.id}</p>
                </div>
                <div>
                    <h4 class="text-sm font-medium text-gray-500 mb-1">Status</h4>
                    <span class="px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                        ${taskDetails.status}
                    </span>
                </div>
                <div>
                    <h4 class="text-sm font-medium text-gray-500 mb-1">Created At</h4>
                    <p class="text-sm">${new Date(taskDetails.created_at).toLocaleString()}</p>
                </div>
                <div>
                    <h4 class="text-sm font-medium text-gray-500 mb-1">Priority</h4>
                    <span class="px-2 py-1 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                        ${getPriorityLabel(taskDetails.priority)}
                    </span>
                </div>
            </div>

            <!-- Progress -->
            <div>
                <h4 class="text-sm font-medium text-gray-500 mb-2">Training Progress</h4>
                <div class="w-full h-4 bg-gray-200 rounded-full overflow-hidden mb-2">
                    <div class="progress-bar h-full bg-gray-600 rounded-full" style="width: ${taskDetails.progress}%"></div>
                </div>
                <div class="flex items-center justify-between text-sm">
                    <span>Epoch: ${taskDetails.current_epoch}/${taskDetails.total_epochs}</span>
                    <span>Progress: ${taskDetails.progress}%</span>
                </div>
            </div>

            <!-- Models -->
            <div>
                <h4 class="text-sm font-medium text-gray-500 mb-2">Models</h4>
                <div class="flex flex-wrap gap-2">
                    ${taskDetails.models.map(model => `
                        <span class="px-3 py-1 bg-gray-100 text-gray-600 rounded-md text-sm">
                            ${model}
                        </span>
                    `).join('')}
                </div>
            </div>

            <!-- Metrics -->
            <div>
                <h4 class="text-sm font-medium text-gray-500 mb-2">Performance Metrics</h4>
                <div class="grid grid-cols-2 gap-4">
                    ${Object.entries(taskDetails.metrics || {}).map(([key, value]) => `
                        <div>
                            <p class="text-xs text-gray-500">${formatMetricName(key)}</p>
                            <p class="text-lg font-bold">${value.toFixed(4)}</p>
                        </div>
                    `).join('')}
                </div>
            </div>

            <!-- Parameters -->
            <div class="bg-gray-50 p-4 rounded-lg">
                <h4 class="text-sm font-medium text-gray-500 mb-2">Training Parameters</h4>
                <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <p class="text-xs text-gray-500">Training Type</p>
                        <p class="text-sm">${taskDetails.training_type}</p>
                    </div>
                    <div>
                        <p class="text-xs text-gray-500">Batch Size</p>
                        <p class="text-sm">${taskDetails.batch_size}</p>
                    </div>
                    <div>
                        <p class="text-xs text-gray-500">Learning Rate</p>
                        <p class="text-sm">${taskDetails.learning_rate}</p>
                    </div>
                    <div>
                        <p class="text-xs text-gray-500">Knowledge Assisted</p>
                        <p class="text-sm">${taskDetails.knowledge_assisted ? 'Yes' : 'No'}</p>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Get priority label
function getPriorityLabel(priority) {
    if (priority === 0) return 'Critical';
    if (priority === 1) return 'High';
    if (priority <= 5) return 'Medium';
    if (priority <= 8) return 'Low';
    return 'Background';
}

// Format metric name
function formatMetricName(name) {
    return name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
}

// Close task modal
function closeTaskModal() {
    taskModal.classList.add('hidden');
    currentTaskId = null;
}

// Stop current task
function stopCurrentTask() {
    if (!currentTaskId) return;
    stopTask(currentTaskId);
}

// Stop specific task
function stopTask(taskId) {
    if (!confirm(`Are you sure you want to stop task ${taskId}?`)) {
        return;
    }
    
    // Show loading indicator
    showLoadingIndicator();
    
    // Simulate API call
    fetch(`/api/tasks/${taskId}/stop`, {
        method: 'POST'
    })
    .then(response => {
        if (!response.ok) throw new Error('Failed to stop task');
        return response.json();
    })
    .then(data => {
        hideLoadingIndicator();
        closeTaskModal();
        refreshTasks();
        showNotification('Task stopped successfully', 'success');
    })
    .catch(error => {
        hideLoadingIndicator();
        showNotification('Failed to stop task', 'error');
        console.error('Error stopping task:', error);
    });
}

// Show notification
function showNotification(message, type = 'info') {
    const notificationTitle = document.getElementById('notification-title');
    if (notificationTitle) {
        if (type === 'success') {
            notificationTitle.textContent = 'Success';
        } else if (type === 'error') {
            notificationTitle.textContent = 'Error';
        } else if (type === 'warning') {
            notificationTitle.textContent = 'Warning';
        } else {
            notificationTitle.textContent = 'Info';
        }
    }
    
    notificationMessage.textContent = message;
    
    // Show notification
    notification.classList.remove('hidden');
    
    // Hide notification after 5 seconds
    setTimeout(hideNotification, 5000);
}

// Hide notification
function hideNotification() {
    notification.classList.add('hidden');
}

// Toggle mobile menu
function toggleMobileMenu() {
    sidebar.classList.toggle('hidden');
}

// Refresh resources
function refreshResources() {
    // Show loading indicator
    showLoadingIndicator();
    
    // Simulate API call
    setTimeout(() => {
        // Update resource usage values
        const cpuUsageValue = document.getElementById('cpu-usage-value');
        const memoryUsageValue = document.getElementById('memory-usage-value');
        const cpuProgressBar = document.getElementById('cpu-progress-bar');
        const memoryProgressBar = document.getElementById('memory-progress-bar');
        
        if (cpuUsageValue && memoryUsageValue && cpuProgressBar && memoryProgressBar) {
            // Simulate new values
            const newCpuUsage = Math.floor(Math.random() * 30) + 10; // 10-40%
            const newMemoryUsage = Math.floor(Math.random() * 40) + 20; // 20-60%
            
            cpuUsageValue.textContent = `${newCpuUsage}%`;
            memoryUsageValue.textContent = `${newMemoryUsage}%`;
            cpuProgressBar.style.width = `${newCpuUsage}%`;
            memoryProgressBar.style.width = `${newMemoryUsage}%`;
        }
        
        hideLoadingIndicator();
        showNotification('Resource usage refreshed', 'success');
    }, 800);
}

// Clear history
function clearHistory() {
    if (!confirm('Are you sure you want to clear all training history?')) {
        return;
    }
    
    // Show loading indicator
    showLoadingIndicator();
    
    // Simulate API call
    setTimeout(() => {
        const trainingHistoryDiv = document.getElementById('training-history');
        if (trainingHistoryDiv) {
            trainingHistoryDiv.innerHTML = `
                <div class="text-center py-8 text-gray-500">
                    <i class="fa fa-history text-2xl mb-2"></i>
                    <p>No recent activity</p>
                </div>
            `;
        }
        
        hideLoadingIndicator();
        showNotification('Training history cleared', 'success');
    }, 800);
}

// Refresh dashboard
function refreshDashboard() {
    refreshTasks();
    refreshResources();
}

// Show loading indicator
function showLoadingIndicator() {
    // In a real app, you would show a global loading indicator
    document.body.style.cursor = 'wait';
}

// Hide loading indicator
function hideLoadingIndicator() {
    // In a real app, you would hide a global loading indicator
    document.body.style.cursor = 'default';
}

// Initialize dashboard when DOM is loaded
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initDashboard);
} else {
    initDashboard();
}