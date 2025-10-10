// 音频处理模型训练控制JavaScript
// Audio Processing Model Training Control JavaScript

class AudioTrainingControl {
    constructor() {
        this.apiBaseUrl = 'http://localhost:5002/api';
        this.isTraining = false;
        this.currentEpoch = 0;
        this.totalEpochs = 0;
        this.trainingInterval = null;
        this.lossChart = null;
        this.accuracyChart = null;
        this.trainingHistory = [];
        
        this.initializeEventListeners();
        this.initializeCharts();
        this.loadTrainingHistory();
        this.updateSystemInfo();
        
        // 设置定期更新系统信息
        setInterval(() => this.updateSystemInfo(), 5000);
    }

    // 初始化事件监听器
    initializeEventListeners() {
        // 训练控制按钮
        document.getElementById('btnStartTraining').addEventListener('click', () => this.startTraining());
        document.getElementById('btnStopTraining').addEventListener('click', () => this.stopTraining());
        
        // 系统信息刷新
        document.getElementById('btnRefreshSystemInfo').addEventListener('click', () => this.updateSystemInfo());
        
        // 配置保存
        document.getElementById('btnSaveConfig').addEventListener('click', () => this.saveConfig());
        
        // 数据目录浏览
        document.getElementById('btnBrowseData').addEventListener('click', () => this.browseDataDirectory());
        
        // 语言切换
        document.querySelectorAll('.language-switcher button').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchLanguage(e.target.dataset.lang));
        });
    }

    // 初始化图表
    initializeCharts() {
        const lossCtx = document.getElementById('lossChart').getContext('2d');
        const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
        
        this.lossChart = new Chart(lossCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '训练损失 | Training Loss',
                        data: [],
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        tension: 0.1
                    },
                    {
                        label: '验证损失 | Validation Loss',
                        data: [],
                        borderColor: 'rgb(54, 162, 235)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '损失曲线 | Loss Curve'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });

        this.accuracyChart = new Chart(accuracyCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: '训练准确率 | Training Accuracy',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        tension: 0.1
                    },
                    {
                        label: '验证准确率 | Validation Accuracy',
                        data: [],
                        borderColor: 'rgb(153, 102, 255)',
                        backgroundColor: 'rgba(153, 102, 255, 0.2)',
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: '准确率曲线 | Accuracy Curve'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                }
            }
        });
    }

    // 开始训练
    async startTraining() {
        if (this.isTraining) {
            this.showAlert('训练正在进行中 | Training is already in progress', 'warning');
            return;
        }

        const config = this.getTrainingConfig();
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/training/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });

            if (response.ok) {
                const result = await response.json();
                this.isTraining = true;
                this.totalEpochs = config.epochs;
                this.updateTrainingUI(true);
                this.showAlert('训练开始成功 | Training started successfully', 'success');
                
                // 开始轮询训练状态
                this.startPollingTrainingStatus();
            } else {
                const error = await response.json();
                this.showAlert(`训练启动失败: ${error.message} | Training failed to start: ${error.message}`, 'danger');
            }
        } catch (error) {
            this.showAlert(`网络错误: ${error.message} | Network error: ${error.message}`, 'danger');
        }
    }

    // 停止训练
    async stopTraining() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/training/stop`, {
                method: 'POST'
            });

            if (response.ok) {
                this.isTraining = false;
                this.updateTrainingUI(false);
                this.showAlert('训练停止成功 | Training stopped successfully', 'info');
                
                if (this.trainingInterval) {
                    clearInterval(this.trainingInterval);
                    this.trainingInterval = null;
                }
            } else {
                const error = await response.json();
                this.showAlert(`训练停止失败: ${error.message} | Training failed to stop: ${error.message}`, 'danger');
            }
        } catch (error) {
            this.showAlert(`网络错误: ${error.message} | Network error: ${error.message}`, 'danger');
        }
    }

    // 开始轮询训练状态
    startPollingTrainingStatus() {
        if (this.trainingInterval) {
            clearInterval(this.trainingInterval);
        }

        this.trainingInterval = setInterval(async () => {
            await this.updateTrainingStatus();
        }, 2000);
    }

    // 更新训练状态
    async updateTrainingStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/training/status`);
            
            if (response.ok) {
                const status = await response.json();
                this.updateTrainingProgress(status);
                
                if (status.status === 'completed' || status.status === 'error') {
                    this.isTraining = false;
                    this.updateTrainingUI(false);
                    
                    if (this.trainingInterval) {
                        clearInterval(this.trainingInterval);
                        this.trainingInterval = null;
                    }
                    
                    if (status.status === 'completed') {
                        this.showAlert('训练完成 | Training completed', 'success');
                        this.loadTrainingHistory();
                    } else {
                        this.showAlert(`训练错误: ${status.message} | Training error: ${status.message}`, 'danger');
                    }
                }
            }
        } catch (error) {
            console.error('获取训练状态失败 | Failed to get training status:', error);
        }
    }

    // 更新训练进度UI
    updateTrainingProgress(status) {
        if (status.current_epoch && status.total_epochs) {
            this.currentEpoch = status.current_epoch;
            const progress = (status.current_epoch / status.total_epochs) * 100;
            
            document.getElementById('currentEpoch').textContent = status.current_epoch;
            document.getElementById('totalEpochs').textContent = status.total_epochs;
            document.getElementById('trainingProgress').style.width = `${progress}%`;
            document.getElementById('progressText').textContent = 
                `${progress.toFixed(1)}% 完成 | ${progress.toFixed(1)}% Complete`;
        }

        if (status.current_loss !== undefined) {
            document.getElementById('currentLoss').textContent = status.current_loss.toFixed(4);
        }

        if (status.current_accuracy !== undefined) {
            const accuracyPercent = (status.current_accuracy * 100).toFixed(2);
            document.getElementById('currentAccuracy').textContent = `${accuracyPercent}%`;
        }

        // 更新状态徽章
        const statusBadge = document.getElementById('trainingStatus');
        statusBadge.textContent = this.isTraining ? 
            '训练中 | Training' : '空闲 | Idle';
        statusBadge.className = `status-badge ${this.isTraining ? 'bg-success' : 'bg-secondary'}`;

        // 更新图表
        if (status.epoch_history && status.epoch_history.length > 0) {
            this.updateCharts(status.epoch_history);
        }
    }

    // 更新图表数据
    updateCharts(history) {
        const epochs = history.map(item => item.epoch);
        const trainLoss = history.map(item => item.train_loss);
        const valLoss = history.map(item => item.val_loss || 0);
        const trainAccuracy = history.map(item => item.train_metrics?.accuracy || 0);
        const valAccuracy = history.map(item => item.val_metrics?.accuracy || 0);

        this.lossChart.data.labels = epochs;
        this.lossChart.data.datasets[0].data = trainLoss;
        this.lossChart.data.datasets[1].data = valLoss;
        this.lossChart.update();

        this.accuracyChart.data.labels = epochs;
        this.accuracyChart.data.datasets[0].data = trainAccuracy;
        this.accuracyChart.data.datasets[1].data = valAccuracy;
        this.accuracyChart.update();
    }

    // 更新训练UI状态
    updateTrainingUI(isTraining) {
        document.getElementById('btnStartTraining').disabled = isTraining;
        document.getElementById('btnStopTraining').disabled = !isTraining;
        
        const statusBadge = document.getElementById('trainingStatus');
        statusBadge.textContent = isTraining ? 
            '训练中 | Training' : '空闲 | Idle';
        statusBadge.className = `status-badge ${isTraining ? 'bg-success' : 'bg-secondary'}`;
    }

    // 获取训练配置
    getTrainingConfig() {
        return {
            model_type: document.getElementById('modelType').value,
            pretrained_model: document.getElementById('pretrainedModel').value,
            epochs: parseInt(document.getElementById('epochs').value),
            batch_size: parseInt(document.getElementById('batchSize').value),
            learning_rate: parseFloat(document.getElementById('learningRate').value),
            optimizer: document.getElementById('optimizer').value,
            data_dir: document.getElementById('dataDir').value,
            validation_ratio: parseFloat(document.getElementById('valRatio').value),
            data_augmentation: document.getElementById('dataAugmentation').checked
        };
    }

    // 保存配置
    async saveConfig() {
        const config = this.getTrainingConfig();
        
        try {
            const response = await fetch(`${this.apiBaseUrl}/config/save`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(config)
            });

            if (response.ok) {
                this.showAlert('配置保存成功 | Configuration saved successfully', 'success');
            } else {
                const error = await response.json();
                this.showAlert(`配置保存失败: ${error.message} | Configuration save failed: ${error.message}`, 'danger');
            }
        } catch (error) {
            this.showAlert(`网络错误: ${error.message} | Network error: ${error.message}`, 'danger');
        }
    }

    // 浏览数据目录
    browseDataDirectory() {
        // 在实际应用中，这里可能会打开一个文件选择对话框
        // 由于浏览器安全限制，这里只是提示用户手动输入路径
        this.showAlert('请手动输入数据目录路径 | Please manually enter the data directory path', 'info');
    }

    // 加载训练历史
    async loadTrainingHistory() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/training/history`);
            
            if (response.ok) {
                const history = await response.json();
                this.displayTrainingHistory(history);
            }
        } catch (error) {
            console.error('加载训练历史失败 | Failed to load training history:', error);
        }
    }

    // 显示训练历史
    displayTrainingHistory(history) {
        const historyContainer = document.getElementById('trainingHistory');
        
        if (!history || history.length === 0) {
            historyContainer.innerHTML = `
                <div class="text-center text-muted py-4">
                    <i class="bi bi-inbox display-4"></i>
                    <p>暂无训练历史 | No training history yet</p>
                </div>
            `;
            return;
        }

        let html = '';
        history.forEach((item, index) => {
            const accuracy = (item.final_accuracy * 100).toFixed(2);
            const loss = item.final_loss.toFixed(4);
            const date = new Date(item.timestamp).toLocaleString();
            
            html += `
                <div class="history-item">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h6>训练任务 #${index + 1} | Training Task #${index + 1}</h6>
                            <p class="mb-1">准确率: ${accuracy}% | Accuracy: ${accuracy}%</p>
                            <p class="mb-1">损失: ${loss} | Loss: ${loss}</p>
                            <small class="text-muted">${date}</small>
                        </div>
                        <div>
                            <button class="btn btn-sm btn-outline-primary view-details" data-id="${item.id}">
                                <i class="bi bi-eye"></i> 详情 | Details
                            </button>
                        </div>
                    </div>
                </div>
            `;
        });

        historyContainer.innerHTML = html;
        
        // 添加详情按钮事件监听
        document.querySelectorAll('.view-details').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const taskId = e.target.closest('.view-details').dataset.id;
                this.viewTrainingDetails(taskId);
            });
        });
    }

    // 查看训练详情
    viewTrainingDetails(taskId) {
        // 在实际应用中，这里会打开一个模态框显示详细训练信息
        this.showAlert(`查看训练任务 ${taskId} 的详情 | View details for training task ${taskId}`, 'info');
    }

    // 更新系统信息
    async updateSystemInfo() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/system/info`);
            
            if (response.ok) {
                const systemInfo = await response.json();
                this.displaySystemInfo(systemInfo);
            }
        } catch (error) {
            console.error('获取系统信息失败 | Failed to get system info:', error);
        }
    }

    // 显示系统信息
    displaySystemInfo(info) {
        if (info.cpu_usage !== undefined) {
            document.getElementById('cpuUsageBar').style.width = `${info.cpu_usage}%`;
            document.getElementById('cpuUsageText').textContent = `${info.cpu_usage}%`;
        }

        if (info.memory_usage !== undefined) {
            document.getElementById('memoryUsageBar').style.width = `${info.memory_usage}%`;
            const memoryText = info.memory_total ? 
                `${info.memory_usage}% (${(info.memory_used / 1024).toFixed(1)}/${(info.memory_total / 1024).toFixed(1)} GB)` :
                `${info.memory_usage}%`;
            document.getElementById('memoryUsageText').textContent = memoryText;
        }

        if (info.gpu_info) {
            document.getElementById('gpuStatus').textContent = info.gpu_info.available ?
                `GPU可用: ${info.gpu_info.name} | GPU Available: ${info.gpu_info.name}` :
                '未检测到GPU | No GPU detected';
        }
    }

    // 切换语言
    switchLanguage(lang) {
        // 在实际应用中，这里会切换界面语言
        // 这里只是切换按钮的激活状态
        document.querySelectorAll('.language-switcher button').forEach(btn => {
            btn.classList.remove('active');
            if (btn.dataset.lang === lang) {
                btn.classList.add('active');
            }
        });
        
        this.showAlert(`语言已切换到: ${lang === 'zh' ? '中文' : 'English'} | Language switched to: ${lang === 'zh' ? 'Chinese' : 'English'}`, 'info');
    }

    // 显示提示信息
    showAlert(message, type) {
        // 创建一个Bootstrap提示框
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.style.position = 'fixed';
        alertDiv.style.top = '20px';
        alertDiv.style.right = '20px';
        alertDiv.style.zIndex = '1050';
        alertDiv.style.minWidth = '300px';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(alertDiv);
        
        // 5秒后自动消失
        setTimeout(() => {
            if (alertDiv.parentNode) {
                alertDiv.parentNode.removeChild(alertDiv);
            }
        }, 5000);
    }
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    window.audioTrainingControl = new AudioTrainingControl();
    console.log('音频训练控制系统初始化完成 | Audio training control system initialized');
});

// 工具函数
function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

function formatPercentage(value) {
    return (value * 100).toFixed(1) + '%';
}
