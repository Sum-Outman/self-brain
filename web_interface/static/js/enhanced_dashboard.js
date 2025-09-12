// Enhanced Dashboard JavaScript - Includes GPU usage display
// Enhanced Dashboard JavaScript - Includes GPU usage display

// Global variables
let systemData = null;
let enhancedResourceChart = null;
let enhancedUpdateInterval = null;

// Initialization function
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Self Brain AGI Enhanced Dashboard initializing...');
    
    // Ensure Chart.js is loaded
    if (typeof Chart === 'undefined') {
        console.error('❌ Chart.js not loaded');
        return;
    }
    
    // Initialize charts
    initEnhancedCharts();
    
    // Load data immediately
    loadSystemResources();
    loadTrainingStatus();
    
    // Refresh every 5 seconds
    enhancedUpdateInterval = setInterval(() => {
        loadSystemResources();
        loadTrainingStatus();
    }, 5000);
});

// Load system resource information
async function loadSystemResources() {
    try {
        const response = await fetch('/api/system/resources');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        if (data.status === 'success' && data.resources) {
            updateResourceDisplay(data.resources);
            updateEnhancedCharts(data.resources);
        }
        
    } catch (error) {
        console.error('❌ Failed to load system resources:', error);
    }
}

// Load training status
async function loadTrainingStatus() {
    try {
        const response = await fetch('/api/training/status');
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        updateTrainingDisplay(data);
        
    } catch (error) {
        console.error('❌ Failed to load training status:', error);
    }
}

// Update resource display
function updateResourceDisplay(resources) {
    const system = resources.system || {};
    const training = resources.training || {};
    
    // CPU usage
    const cpuUsage = system.cpu_usage_percent || 0;
    updateProgressBar('cpuUsage', 'cpuProgress', cpuUsage, '%');
    
    // GPU usage - newly added
    const gpuUsage = system.gpu_usage_percent || 0;
    const gpuModel = system.gpu_model || 'Unknown';
    updateProgressBar('gpuUsage', 'gpuProgress', gpuUsage, '%', gpuModel);
    
    // Memory usage
    const memoryUsage = system.memory_usage_percent || 0;
    const memoryUsed = system.memory_total_mb - system.memory_available_mb;
    updateProgressBar('memoryUsage', 'memoryProgress', memoryUsage, '%', 
                     `${Math.round(memoryUsed)}/${Math.round(system.memory_total_mb)} MB`);
    
    // Disk usage
    const diskUsage = system.disk_usage_percent || 0;
    updateProgressBar('diskUsage', 'diskProgress', diskUsage, '%');
    
    // Network traffic
    const networkSent = system.network_bytes_sent_mb || 0;
    const networkRecv = system.network_bytes_recv_mb || 0;
    const networkTotal = networkSent + networkRecv;
    updateProgressBar('networkIO', 'networkProgress', Math.min(networkTotal, 100), 'MB/s');
    
    // Update quick stats
    updateQuickStats(training);
}

// Update progress bar
function updateProgressBar(valueId, progressId, value, unit, tooltip = null) {
    const valueElement = document.getElementById(valueId);
    const progressElement = document.getElementById(progressId);
    
    if (valueElement) {
        valueElement.textContent = `${value}${unit}`;
        if (tooltip) {
            valueElement.title = tooltip;
        }
    }
    
    if (progressElement) {
        progressElement.style.width = `${Math.min(value, 100)}%`;
        
        // Set color based on usage rate
        if (value < 50) {
            progressElement.className = 'progress-bar bg-success';
        } else if (value < 80) {
            progressElement.className = 'progress-bar bg-warning';
        } else {
            progressElement.className = 'progress-bar bg-danger';
        }
    }
}

// Update quick stats
function updateQuickStats(training) {
    const sessions = training.sessions || {};
    
    updateStatNumber('totalSessions', sessions.total || 0);
    updateStatNumber('activeSessions', sessions.active || 0);
    updateStatNumber('completedSessions', sessions.completed || 0);
    updateStatNumber('failedSessions', sessions.failed || 0);
    updateStatNumber('knowledgeItems', training.knowledge_items || 0);
}

// Update stat numbers
function updateStatNumber(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        // Add animation effect
        const currentValue = parseInt(element.textContent) || 0;
        if (currentValue !== value) {
            animateNumber(element, currentValue, value);
        }
    }
}

// Number animation
function animateNumber(element, from, to) {
    const duration = 1000;
    const startTime = performance.now();
    
    function updateNumber(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const current = Math.floor(from + (to - from) * progress);
        element.textContent = current;
        
        if (progress < 1) {
            requestAnimationFrame(updateNumber);
        }
    }
    
    requestAnimationFrame(updateNumber);
}

// Initialize charts
function initEnhancedCharts() {
    // Resource usage trend chart
    const resourceCtx = document.getElementById('resourceChart');
    if (resourceCtx && !enhancedResourceChart) {
        try {
            enhancedResourceChart = new Chart(resourceCtx.getContext('2d'), {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'CPU Usage',
                            data: [],
                            borderColor: 'rgb(255, 99, 132)',
                            backgroundColor: 'rgba(255, 99, 132, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'GPU Usage',
                            data: [],
                            borderColor: 'rgb(54, 162, 235)',
                            backgroundColor: 'rgba(54, 162, 235, 0.1)',
                            tension: 0.4
                        },
                        {
                            label: 'Memory Usage',
                            data: [],
                            borderColor: 'rgb(75, 192, 192)',
                            backgroundColor: 'rgba(75, 192, 192, 0.1)',
                            tension: 0.4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100,
                            title: {
                                display: true,
                                text: 'Usage (%)'
                            }
                        }
                    }
                }
            });
        } catch (error) {
            console.error('❌ Failed to initialize resource chart:', error);
        }
    }
}

// Update chart data
function updateEnhancedCharts(resources) {
    if (!enhancedResourceChart) return;
    
    const system = resources.system || {};
    const now = new Date().toLocaleTimeString();
    
    // Add new data points
    enhancedResourceChart.data.labels.push(now);
    enhancedResourceChart.data.datasets[0].data.push(system.cpu_usage_percent || 0);
    enhancedResourceChart.data.datasets[1].data.push(system.gpu_usage_percent || 0);
    enhancedResourceChart.data.datasets[2].data.push(system.memory_usage_percent || 0);
    
    // Keep maximum 20 data points
    const maxPoints = 20;
    if (enhancedResourceChart.data.labels.length > maxPoints) {
        enhancedResourceChart.data.labels.shift();
        enhancedResourceChart.data.datasets.forEach(dataset => {
            dataset.data.shift();
        });
    }
    
    enhancedResourceChart.update('none');
}

// Update training display
function updateTrainingDisplay(data) {
    // Training status update logic can be added here
    console.log('📊 Training status updated:', data);
}

// Cleanup function
function cleanup() {
    if (enhancedUpdateInterval) {
        clearInterval(enhancedUpdateInterval);
        enhancedUpdateInterval = null;
    }
    
    if (enhancedResourceChart) {
        enhancedResourceChart.destroy();
        enhancedResourceChart = null;
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', cleanup);

// Export functions for global use
window.DashboardController = {
    loadSystemResources,
    loadTrainingStatus,
    cleanup
};
