// System Monitoring Module
// Self Brain AGI

class SystemMonitor {
    constructor() {
        this.updateInterval = null;
        this.performanceData = {
            cpu: [],
            memory: [],
            gpu: [],
            timestamps: []
        };
        this.maxDataPoints = 120; // Store 2 minutes of data (120 seconds)
        this.lastUpdateTime = 0;
    }

    // Initialize the system monitor
    init() {
        console.log('System Monitor initialized');
        this.startMonitoring();
    }

    // Start monitoring system resources
    startMonitoring() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        // Update system resources every second
        this.updateInterval = setInterval(() => {
            this.updateSystemResources();
        }, 1000);
    }

    // Stop monitoring
    stopMonitoring() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    // Update system resource usage
    updateSystemResources() {
        const currentTime = Date.now();
        
        // Skip if less than 1 second has passed
        if (currentTime - this.lastUpdateTime < 1000) return;
        
        this.lastUpdateTime = currentTime;

        // Simulate resource usage data
        const cpuUsage = Math.min(100, Math.max(5, selfBrain.state.resources?.cpu || 30 + Math.random() * 20));
        const memoryUsage = Math.min(100, Math.max(10, selfBrain.state.resources?.memory || 40 + Math.random() * 15));
        const gpuUsage = Math.min(100, Math.max(0, selfBrain.state.resources?.gpu || 25 + Math.random() * 30));

        // Update data arrays
        this.performanceData.cpu.push(cpuUsage);
        this.performanceData.memory.push(memoryUsage);
        this.performanceData.gpu.push(gpuUsage);
        this.performanceData.timestamps.push(new Date().toLocaleTimeString());

        // Keep only the latest data points
        if (this.performanceData.cpu.length > this.maxDataPoints) {
            this.performanceData.cpu.shift();
            this.performanceData.memory.shift();
            this.performanceData.gpu.shift();
            this.performanceData.timestamps.shift();
        }

        // Update UI elements
        this.updateUI(cpuUsage, memoryUsage, gpuUsage);
    }

    // Update UI elements with resource usage
    updateUI(cpuUsage, memoryUsage, gpuUsage) {
        // Update CPU usage
        const cpuElement = document.getElementById('cpu-usage');
        const cpuProgress = document.querySelector('#cpu-usage ~ .progress-bar');
        if (cpuElement && cpuProgress) {
            cpuElement.textContent = cpuUsage.toFixed(1) + '%';
            cpuProgress.style.width = cpuUsage + '%';
            this.updateProgressBarColor(cpuProgress, cpuUsage);
        }

        // Update memory usage
        const memoryElement = document.getElementById('memory-usage');
        const memoryProgress = document.querySelector('#memory-usage ~ .progress-bar');
        if (memoryElement && memoryProgress) {
            memoryElement.textContent = memoryUsage.toFixed(1) + '%';
            memoryProgress.style.width = memoryUsage + '%';
            this.updateProgressBarColor(memoryProgress, memoryUsage);
        }

        // Update GPU usage
        const gpuElement = document.getElementById('gpu-usage');
        const gpuProgress = document.querySelector('#gpu-usage ~ .progress-bar');
        if (gpuElement && gpuProgress) {
            gpuElement.textContent = gpuUsage.toFixed(1) + '%';
            gpuProgress.style.width = gpuUsage + '%';
            this.updateProgressBarColor(gpuProgress, gpuUsage);
        }

        // Update system status indicator
        const systemStatusElement = document.getElementById('system-status');
        if (systemStatusElement) {
            if (cpuUsage > 80 || memoryUsage > 80 || gpuUsage > 80) {
                systemStatusElement.className = 'px-2 py-1 text-xs font-medium bg-red-100 text-red-800 rounded';
                systemStatusElement.textContent = 'High Load';
            } else if (cpuUsage > 60 || memoryUsage > 60 || gpuUsage > 60) {
                systemStatusElement.className = 'px-2 py-1 text-xs font-medium bg-yellow-100 text-yellow-800 rounded';
                systemStatusElement.textContent = 'Medium Load';
            } else {
                systemStatusElement.className = 'px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded';
                systemStatusElement.textContent = 'Active';
            }
        }

        // Update hardware status indicator
        const gpuStatusElement = document.getElementById('gpu-status');
        if (gpuStatusElement) {
            if (gpuUsage > 70) {
                gpuStatusElement.className = 'px-2 py-1 text-xs font-medium bg-yellow-100 text-yellow-800 rounded';
                gpuStatusElement.textContent = 'GPU Busy';
            } else {
                gpuStatusElement.className = 'px-2 py-1 text-xs font-medium bg-green-100 text-green-800 rounded';
                gpuStatusElement.textContent = 'GPU Active';
            }
        }

        // Update model count display
        const modelCountElement = document.getElementById('model-count');
        if (modelCountElement) {
            const activeModels = selfBrain.state.activeModels?.length || 11;
            modelCountElement.textContent = `${activeModels}/11`;
        }

        // Update response time display
        const responseTimeElement = document.getElementById('response-time');
        if (responseTimeElement) {
            // Calculate response time based on resource usage
            let responseTime = 100; // Base response time in ms
            if (cpuUsage > 60) responseTime += (cpuUsage - 60) * 5;
            if (memoryUsage > 70) responseTime += (memoryUsage - 70) * 3;
            
            const displayTime = responseTime < 200 ? '< 200ms' : `${Math.round(responseTime)}ms`;
            responseTimeElement.textContent = displayTime;
        }
    }

    // Update progress bar color based on usage level
    updateProgressBarColor(progressBar, usage) {
        if (usage > 80) {
            progressBar.className = 'progress-bar bg-danger';
        } else if (usage > 60) {
            progressBar.className = 'progress-bar bg-warning';
        } else {
            progressBar.className = 'progress-bar bg-success';
        }
    }

    // Get performance history data
    getPerformanceHistory() {
        return this.performanceData;
    }

    // Cleanup
    destroy() {
        this.stopMonitoring();
        this.performanceData = {
            cpu: [],
            memory: [],
            gpu: [],
            timestamps: []
        };
    }
}

// Export the SystemMonitor class
window.SystemMonitor = SystemMonitor;

// Initialize the system monitor when selfBrain is available
if (window.selfBrain && window.selfBrain.init) {
    // Wait for selfBrain to be initialized
    const waitForSelfBrain = setInterval(() => {
        if (window.selfBrain && window.selfBrain.state) {
            clearInterval(waitForSelfBrain);
            window.systemMonitor = new SystemMonitor();
            window.systemMonitor.init();
        }
    }, 100);
}