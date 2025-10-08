// Training Control Frontend JavaScript

class TrainingController {
    constructor() {
        this.models = [];
        this.sessions = [];
        this.currentSessionId = null;
        this.trainingMode = 'individual';
        this.selectedModels = [];
        this.realtimeMetrics = {};
        this.knowledgeBaseData = {};
        this.collaborationStats = {};
        this.socket = null;
        
        this.init();
    }

    init() {
        this.loadModels();
        this.setupEventListeners();
        this.loadSessions();
        this.loadKnowledgeBase();
        this.setupWebSocket();
        
        // Refresh session list every 5 seconds
        setInterval(() => this.loadSessions(), 5000);
        // Refresh knowledge base data every 10 seconds
        setInterval(() => this.loadKnowledgeBase(), 10000);
        // Refresh real-time monitoring data every 3 seconds
        setInterval(() => this.updateRealtimeDashboard(), 3000);
    }

    // Load model list
    async loadModels() {
        try {
            const response = await fetch('/api/models');
            const data = await response.json();
            
            if (data.status === 'success') {
                // Map model names to backend identifiers
                this.models = data.models.map(model => ({
                    ...model,
                    backend_id: `${model.name}_${model.model_type}`
                }));
                this.renderModelSelector();
            }
        } catch (error) {
            console.error('Failed to load models:', error);
        }
    }

    // Render model selector
    renderModelSelector() {
        const container = document.getElementById('modelSelector');
        if (!container) return;

        container.innerHTML = '';
        
        this.models.forEach(model => {
            const div = document.createElement('div');
            div.className = 'form-check';
            div.innerHTML = `
                <input class="form-check-input model-checkbox" type="checkbox" value="${model.name}" 
                       id="model-${model.name}" name="selectedModels" data-model-type="${model.model_type}">
                <label class="form-check-label" for="model-${model.name}">
                    ${model.name} (${model.model_type})
                    ${model.is_local ? '<span class="badge bg-success ms-1">Local</span>' : '<span class="badge bg-info ms-1">External</span>'}
                </label>
                <small class="form-text text-muted d-block">${model.description}</small>
            `;
            container.appendChild(div);
        });

        // Add model selection event listeners
        document.querySelectorAll('.model-checkbox').forEach(checkbox => {
            checkbox.addEventListener('change', () => this.updateModelSelection());
        });
    }

    // Update model selection
    updateModelSelection() {
        const checkboxes = document.querySelectorAll('.model-checkbox:checked');
        this.selectedModels = Array.from(checkboxes).map(cb => cb.value);
        
        // Validate selection based on training mode
        this.validateModelSelection();
    }

    // Validate model selection
    validateModelSelection() {
        const mode = document.getElementById('trainingMode').value;
        const modelCount = this.selectedModels.length;
        
        let isValid = true;
        let message = '';

        switch (mode) {
            case 'individual':
                if (modelCount !== 1) {
                    isValid = false;
                    message = 'Individual training mode requires exactly one model';
                }
                break;
            case 'joint':
                if (modelCount < 2) {
                    isValid = false;
                    message = 'Joint training mode requires at least two models';
                }
                break;
            case 'transfer':
                if (modelCount < 1) {
                    isValid = false;
                    message = 'Transfer learning mode requires at least one model';
                }
                break;
            case 'fine_tuning':
                if (modelCount < 1) {
                    isValid = false;
                    message = 'Fine-tuning mode requires at least one model';
                }
                break;
        }

        // Show validation result
        const submitBtn = document.querySelector('button[type="submit"]');
        if (submitBtn) {
            submitBtn.disabled = !isValid;
            if (!isValid) {
                submitBtn.title = message;
            } else {
                submitBtn.title = '';
            }
        }

        // Show alert message
        const alertDiv = document.getElementById('modelSelectionAlert');
        if (alertDiv) {
            if (!isValid) {
                alertDiv.innerHTML = `<div class="alert alert-warning py-1">${message}</div>`;
            } else {
                alertDiv.innerHTML = '';
            }
        }
    }

    // Set up event listeners
    setupEventListeners() {
        // Training mode selection change
        const modeSelect = document.getElementById('trainingMode');
        if (modeSelect) {
            modeSelect.addEventListener('change', (e) => {
                this.trainingMode = e.target.value;
                this.updateModeSpecificUI();
                this.validateModelSelection();
            });
        }

        // Form submission
        const form = document.getElementById('trainingForm');
        if (form) {
            form.addEventListener('submit', (e) => this.handleTrainingStart(e));
        }

        // Parameter sliders
        this.setupParameterSliders();
    }

    // Update mode-specific UI
    updateModeSpecificUI() {
        const mode = this.trainingMode;
        const modeInfo = document.getElementById('modeInfo');
        
        if (modeInfo) {
            const modeDescriptions = {
                'individual': {
                    title: 'Individual Training',
                    description: 'Single model independent training, suitable for deep optimization of specific tasks',
                    icon: 'bi-person-fill',
                    color: 'primary'
                },
                'joint': {
                    title: 'Joint Training',
                    description: 'Multiple models training collaboratively to achieve knowledge sharing and performance complementarity',
                    icon: 'bi-people-fill',
                    color: 'success'
                },
                'transfer': {
                    title: 'Transfer Learning',
                    description: 'Utilize knowledge from pre-trained models to quickly adapt to new tasks',
                    icon: 'bi-arrow-left-right',
                    color: 'info'
                },
                'fine_tuning': {
                    title: 'Fine-tuning',
                    description: 'Fine-tune existing models to optimize specific performance metrics',
                    icon: 'bi-tools',
                    color: 'warning'
                },
                'pretraining': {
                    title: 'Pre-training',
                    description: 'Pre-train on large-scale data to establish a general knowledge foundation',
                    icon: 'bi-lightning-charge-fill',
                    color: 'danger'
                }
            };

            const info = modeDescriptions[mode];
            modeInfo.innerHTML = `
                <div class="alert alert-${info.color} d-flex align-items-center">
                    <i class="bi ${info.icon} me-2"></i>
                    <div>
                        <strong>${info.title}</strong>
                        <p class="mb-0 small">${info.description}</p>
                    </div>
                </div>
            `;
        }
    }

    // Set up parameter sliders
    setupParameterSliders() {
        const sliders = [
            { id: 'learningRate', displayId: 'learningRateValue', multiplier: 10000, suffix: '' },
            { id: 'batchSize', displayId: 'batchSizeValue', multiplier: 1, suffix: '' },
            { id: 'epochs', displayId: 'epochsValue', multiplier: 1, suffix: '' }
        ];

        sliders.forEach(slider => {
            const element = document.getElementById(slider.id);
            const display = document.getElementById(slider.displayId);
            
            if (element && display) {
                element.addEventListener('input', (e) => {
                    const value = parseFloat(e.target.value);
                    display.textContent = slider.multiplier > 1 ? 
                        (value * slider.multiplier).toFixed(slider.id === 'learningRate' ? 4 : 0) + slider.suffix : 
                        value + slider.suffix;
                });
            }
        });
    }

    // Handle training start
    async handleTrainingStart(e) {
        e.preventDefault();
        
        const mode = document.getElementById('trainingMode').value;
        const epochs = parseInt(document.getElementById('epochs').value);
        const learningRate = parseFloat(document.getElementById('learningRate').value);
        const batchSize = parseInt(document.getElementById('batchSize').value);

        const trainingData = {
            model_ids: this.selectedModels,
            mode: mode,
            epochs: epochs,
            learning_rate: learningRate,
            batch_size: batchSize,
            training_type: 'supervised',
            compute_device: 'auto'
        };

        try {
            const response = await fetch('/api/training/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(trainingData)
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                this.showMessage('success', `Training started successfully`);
                this.loadSessions();
                this.resetForm();
            } else {
                this.showMessage('error', result.message || 'Failed to start training');
            }
        } catch (error) {
            this.showMessage('error', 'Network error, please try again later');
            console.error('Training start error:', error);
        }
    }

    // Load training sessions
    async loadSessions() {
        try {
            const response = await fetch('/api/training/sessions');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.sessions = data.sessions || [];
                this.renderSessions();
                this.updateSessionsCount();
            }
        } catch (error) {
            console.error('Failed to load sessions:', error);
        }
    }

    // Render sessions list
    renderSessions() {
        const container = document.getElementById('sessionsList');
        if (!container) return;

        if (this.sessions.length === 0) {
            container.innerHTML = '<div class="text-center text-muted">No training sessions</div>';
            return;
        }

        container.innerHTML = '';
        
        this.sessions.forEach(session => {
            const sessionDiv = this.createSessionElement(session);
            container.appendChild(sessionDiv);
        });
    }

    // Create session element
    createSessionElement(session) {
        const div = document.createElement('div');
        div.className = 'session-item mb-3 p-3 border rounded';
        div.dataset.sessionId = session.session_id;
        
        const modeNames = {
            'individual': 'Individual Training',
            'joint': 'Joint Training',
            'transfer': 'Transfer Learning',
            'fine_tuning': 'Fine-tuning'
        };

        const statusClass = {
            'training': 'warning',
            'completed': 'success',
            'paused': 'secondary',
            'stopped': 'danger'
        };

        const statusIcon = {
            'training': 'bi-play-circle',
            'completed': 'bi-check-circle',
            'paused': 'bi-pause-circle',
            'stopped': 'bi-stop-circle'
        };

        div.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
                <div>
                    <h6 class="mb-1">${session.session_id}</h6>
                    <small class="text-muted">
                        <i class="bi ${statusIcon[session.status]}"></i>
                        ${modeNames[session.mode] || session.mode}
                    </small>
                </div>
                <span class="badge bg-${statusClass[session.status]}">${session.status}</span>
            </div>
            <div class="mt-2">
                <small class="text-muted">
                    Models: ${session.models.map(m => m.name).join(', ')}
                </small>
            </div>
            <div class="mt-2">
                <div class="progress" style="height: 4px;">
                    <div class="progress-bar" style="width: ${session.progress * 100}%"></div>
                </div>
                <small class="text-muted">${Math.round(session.progress * 100)}% complete</small>
            </div>
            <div class="mt-2 d-flex gap-1">
                ${this.getSessionActions(session)}
            </div>
        `;

        // Add click event
        div.addEventListener('click', (e) => {
            if (!e.target.classList.contains('btn')) {
                this.selectSession(session.session_id);
            }
        });

        return div;
    }

    // Get session action buttons
    getSessionActions(session) {
        let actions = '';
        
        if (session.status === 'training') {
            actions += `
                <button class="btn btn-sm btn-warning" onclick="trainingController.pauseSession('${session.session_id}')">
                    <i class="bi bi-pause"></i>
                </button>
                <button class="btn btn-sm btn-danger" onclick="trainingController.stopSession('${session.session_id}')">
                    <i class="bi bi-stop"></i>
                </button>
            `;
        } else if (session.status === 'paused') {
            actions += `
                <button class="btn btn-sm btn-success" onclick="trainingController.resumeSession('${session.session_id}')">
                    <i class="bi bi-play"></i>
                </button>
                <button class="btn btn-sm btn-danger" onclick="trainingController.stopSession('${session.session_id}')">
                    <i class="bi bi-stop"></i>
                </button>
            `;
        } else if (session.status === 'completed') {
            actions += `
                <button class="btn btn-sm btn-info" onclick="trainingController.viewResults('${session.session_id}')">
                    <i class="bi bi-eye"></i>
                </button>
                <button class="btn btn-sm btn-danger" onclick="trainingController.deleteSession('${session.session_id}')">
                    <i class="bi bi-trash"></i>
                </button>
            `;
        }

        return actions;
    }

    // Select session
    selectSession(sessionId) {
        this.currentSessionId = sessionId;
        this.updateSessionDetails(sessionId);
        
        // Highlight selected session
        document.querySelectorAll('.session-item').forEach(item => {
            item.classList.remove('border-primary');
            if (item.dataset.sessionId === sessionId) {
                item.classList.add('border-primary');
            }
        });
    }

    // Update session details
    updateSessionDetails(sessionId) {
        const session = this.sessions.find(s => s.session_id === sessionId);
        if (!session) return;

        const detailsContainer = document.getElementById('sessionDetails');
        if (!detailsContainer) return;

        const modeNames = {
            'individual': 'Individual Training',
            'joint': 'Joint Training',
            'transfer': 'Transfer Learning',
            'fine_tuning': 'Fine-tuning'
        };

        detailsContainer.innerHTML = `
            <div class="card">
                <div class="card-body">
                    <h6>${session.session_id}</h6>
                    <p><strong>Mode:</strong> ${modeNames[session.mode] || session.mode}</p>
                    <p><strong>Models:</strong> ${session.models.map(m => m.name).join(', ')}</p>
                    <p><strong>Epoch:</strong> ${session.metrics.epoch}/${session.metrics.total_epochs}</p>
                    <p><strong>Learning Rate:</strong> ${session.metrics.learning_rate}</p>
                    <p><strong>Loss:</strong> ${session.metrics.loss.toFixed(4)}</p>
                    <p><strong>Accuracy:</strong> ${(session.metrics.accuracy * 100).toFixed(2)}%</p>
                    <p><strong>Duration:</strong> ${this.formatDuration(session.duration)}</p>
                    ${session.mode_config ? this.renderModeConfig(session.mode_config) : ''}
                </div>
            </div>
        `;

        // Update real-time metrics
        this.updateTrainingMetrics(session);
    }

    // Render mode configuration
    renderModeConfig(config) {
        let html = '<div class="mt-3"><strong>Mode Configuration:</strong><ul class="list-unstyled">';
        
        if (config.description) {
            html += `<li><small>${config.description}</small></li>`;
        }
        
        if (config.collaboration_efficiency !== undefined) {
            html += `<li><small>Collaboration Efficiency: ${(config.collaboration_efficiency * 100).toFixed(1)}%</small></li>`;
        }
        
        if (config.transfer_efficiency !== undefined) {
            html += `<li><small>Transfer Efficiency: ${(config.transfer_efficiency * 100).toFixed(1)}%</small></li>`;
        }
        
        html += '</ul></div>';
        return html;
    }

    // Update training metrics
    updateTrainingMetrics(session) {
        const progressBar = document.getElementById('trainingProgress');
        const lossValue = document.getElementById('lossValue');
        const accuracyValue = document.getElementById('accuracyValue');

        if (progressBar) {
            progressBar.style.width = `${session.progress * 100}%`;
            progressBar.textContent = `${Math.round(session.progress * 100)}%`;
        }

        if (lossValue) {
            lossValue.textContent = session.metrics.loss.toFixed(4);
        }

        if (accuracyValue) {
            accuracyValue.textContent = `${(session.metrics.accuracy * 100).toFixed(2)}%`;
        }
    }

    // Update sessions count
    updateSessionsCount() {
        const countElement = document.getElementById('sessionsCount');
        if (countElement) {
            countElement.textContent = this.sessions.length;
        }
    }

    // Session operations
    async pauseSession(sessionId) {
        try {
            const response = await fetch(`/api/training/pause/${sessionId}`, { method: 'POST' });
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showMessage('success', result.message);
                this.loadSessions();
            } else {
                this.showMessage('warning', result.message);
            }
        } catch (error) {
            this.showMessage('error', 'Operation failed');
        }
    }

    async resumeSession(sessionId) {
        try {
            const response = await fetch(`/api/training/resume/${sessionId}`, { method: 'POST' });
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showMessage('success', result.message);
                this.loadSessions();
            } else {
                this.showMessage('warning', result.message);
            }
        } catch (error) {
            this.showMessage('error', 'Operation failed');
        }
    }

    async stopSession(sessionId) {
        try {
            const response = await fetch(`/api/training/stop/${sessionId}`, { method: 'POST' });
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showMessage('success', result.message);
                this.loadSessions();
            } else {
                this.showMessage('error', result.message);
            }
        } catch (error) {
            this.showMessage('error', 'Operation failed');
        }
    }

    async deleteSession(sessionId) {
        if (!confirm('Are you sure you want to delete this training session?')) return;

        try {
            const response = await fetch(`/api/training/delete/${sessionId}`, { method: 'DELETE' });
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showMessage('success', result.message);
                this.loadSessions();
                if (this.currentSessionId === sessionId) {
                    this.currentSessionId = null;
                    this.clearSessionDetails();
                }
            } else {
                this.showMessage('error', result.message);
            }
        } catch (error) {
            this.showMessage('error', 'Delete failed');
        }
    }

    // Utility methods
    formatDuration(seconds) {
        if (seconds < 60) return `${seconds} seconds`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)} minutes`;
        return `${Math.floor(seconds / 3600)} hours ${Math.floor((seconds % 3600) / 60)} minutes`;
    }

    showMessage(type, message) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        const container = document.querySelector('.container-fluid');
        if (container) {
            container.insertBefore(alertDiv, container.firstChild);
            setTimeout(() => alertDiv.remove(), 5000);
        }
    }

    resetForm() {
        document.querySelectorAll('.model-checkbox').forEach(cb => cb.checked = false);
        this.selectedModels = [];
        this.validateModelSelection();
    }

    clearSessionDetails() {
        const detailsContainer = document.getElementById('sessionDetails');
        if (detailsContainer) {
        detailsContainer.innerHTML = '<div class="text-center text-muted">Select a training session to view details</div>';
        }
    }

    // Set up WebSocket connection
    setupWebSocket() {
        try {
            this.socket = new WebSocket(`ws://${window.location.host}/ws/training`);
            
            this.socket.onopen = () => {
                console.log('WebSocket connection established');
                this.showMessage('success', 'Real-time monitoring connection established');
            };
            
            this.socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleRealtimeData(data);
            };
            
            this.socket.onclose = () => {
                console.log('WebSocket connection closed');
                this.showMessage('warning', 'Real-time monitoring connection lost, attempting to reconnect...');
                setTimeout(() => this.setupWebSocket(), 3000);
            };
            
            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.error('WebSocket initialization failed:', error);
        }
    }

    // Handle real-time data
    handleRealtimeData(data) {
        if (data.type === 'training_metrics') {
            this.realtimeMetrics = data.metrics;
            this.updateRealtimeDashboard();
        } else if (data.type === 'collaboration_stats') {
            this.collaborationStats = data.stats;
            this.updateCollaborationDashboard();
        } else if (data.type === 'knowledge_update') {
            this.knowledgeBaseData = data.knowledge;
            this.updateKnowledgeDashboard();
        }
    }

    // Load knowledge base data
    async loadKnowledgeBase() {
        try {
            const response = await fetch('/api/knowledge/stats');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.knowledgeBaseData = data.stats;
                this.updateKnowledgeDashboard();
            }
        } catch (error) {
            console.error('Failed to load knowledge base data:', error);
        }
    }

    // Update real-time monitoring dashboard
    updateRealtimeDashboard() {
        const dashboard = document.getElementById('realtimeDashboard');
        if (!dashboard || !this.realtimeMetrics) return;

        const metrics = this.realtimeMetrics;
        dashboard.innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <div class="card bg-primary text-white">
                        <div class="card-body text-center">
                            <h6>CPU Usage</h6>
                            <h3>${metrics.cpu_usage || 0}%</h3>
                            <div class="progress bg-dark" style="height: 4px;">
                                <div class="progress-bar bg-light" style="width: ${metrics.cpu_usage || 0}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-info text-white">
                        <div class="card-body text-center">
                            <h6>Memory Usage</h6>
                            <h3>${Math.round((metrics.memory_usage || 0) / 1024 / 1024)}MB</h3>
                            <small>${Math.round((metrics.memory_usage || 0) / metrics.total_memory * 100)}% usage</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-success text-white">
                        <div class="card-body text-center">
                            <h6>GPU Usage</h6>
                            <h3>${metrics.gpu_usage || 0}%</h3>
                            <div class="progress bg-dark" style="height: 4px;">
                                <div class="progress-bar bg-light" style="width: ${metrics.gpu_usage || 0}%"></div>
                            </div>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-warning text-dark">
                        <div class="card-body text-center">
                            <h6>Training Speed</h6>
                            <h3>${metrics.samples_per_second || 0}/s</h3>
                            <small>${metrics.epoch_progress || 0}% complete</small>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Update collaboration statistics dashboard
    updateCollaborationDashboard() {
        const dashboard = document.getElementById('collaborationDashboard');
        if (!dashboard || !this.collaborationStats) return;

        const stats = this.collaborationStats;
        dashboard.innerHTML = `
            <div class="row">
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body text-center">
                            <h6>Model Collaboration Efficiency</h6>
                            <h3 class="text-success">${(stats.collaboration_efficiency * 100).toFixed(1)}%</h3>
                            <small>Knowledge Sharing Effectiveness</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body text-center">
                            <h6>Data Transfer Rate</h6>
                            <h3 class="text-info">${stats.data_transfer_rate.toFixed(1)}MB/s</h3>
                            <small>Inter-model Communication</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card">
                        <div class="card-body text-center">
                            <h6>Task Coordination Success Rate</h6>
                            <h3 class="text-primary">${(stats.task_coordination_rate * 100).toFixed(1)}%</h3>
                            <small>Multi-model Collaboration</small>
                        </div>
                    </div>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-12">
                    <div class="card">
                        <div class="card-body">
                            <h6>Model Collaboration Network</h6>
                            <div class="d-flex justify-content-around">
                                ${Object.entries(stats.model_interactions || {}).map(([model, count]) => `
                                    <div class="text-center">
                                        <div class="bg-light p-2 rounded">
                                            <strong>${model}</strong><br>
                                            <span class="text-primary">${count} interactions</span>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Update knowledge base dashboard
    updateKnowledgeDashboard() {
        const dashboard = document.getElementById('knowledgeDashboard');
        if (!dashboard || !this.knowledgeBaseData) return;

        const knowledge = this.knowledgeBaseData;
        dashboard.innerHTML = `
            <div class="row">
                <div class="col-md-3">
                    <div class="card bg-dark text-white">
                        <div class="card-body text-center">
                            <h6>Total Knowledge</h6>
                            <h3>${knowledge.total_knowledge_items || 0}</h3>
                            <small>Items</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-secondary text-white">
                        <div class="card-body text-center">
                            <h6>Knowledge Domains</h6>
                            <h3>${knowledge.domain_count || 0}</h3>
                            <small>Covered Domains</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-info text-white">
                        <div class="card-body text-center">
                            <h6>Knowledge Usage Rate</h6>
                            <h3>${(knowledge.usage_rate * 100 || 0).toFixed(1)}%</h3>
                            <small>Applied in Training</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-success text-white">
                        <div class="card-body text-center">
                            <h6>Knowledge Updates</h6>
                            <h3>${knowledge.recent_updates || 0}</h3>
                            <small>Recent Updates</small>
                        </div>
                    </div>
                </div>
            </div>
            <div class="row mt-3">
                <div class="col-12">
                    <div class="card">
                        <div class="card-body">
                            <h6>Knowledge Domain Distribution</h6>
                            <div class="d-flex flex-wrap gap-2">
                                ${Object.entries(knowledge.domain_distribution || {}).map(([domain, count]) => `
                                    <span class="badge bg-primary">${domain}: ${count}</span>
                                `).join('')}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // View training results
    async viewResults(sessionId) {
        try {
            const response = await fetch(`/api/training/results/${sessionId}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.showTrainingResults(data.results);
            } else {
                this.showMessage('error', 'Failed to get training results');
            }
        } catch (error) {
            this.showMessage('error', 'Failed to get training results');
        }
    }

    // Show training results
    showTrainingResults(results) {
        const modal = new bootstrap.Modal(document.getElementById('resultsModal'));
        const modalBody = document.getElementById('resultsModalBody');
        
        if (modalBody) {
            modalBody.innerHTML = `
                <div class="row">
                    <div class="col-md-6">
                        <h6>Training Metrics</h6>
                        <ul class="list-group">
                            <li class="list-group-item">Final Loss: ${results.final_loss.toFixed(4)}</li>
                            <li class="list-group-item">Final Accuracy: ${(results.final_accuracy * 100).toFixed(2)}%</li>
                            <li class="list-group-item">Training Time: ${this.formatDuration(results.training_time)}</li>
                            <li class="list-group-item">Total Epochs: ${results.total_epochs}</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6>Performance Improvement</h6>
                        <ul class="list-group">
                            <li class="list-group-item">Loss Reduction: ${((results.initial_loss - results.final_loss) / results.initial_loss * 100).toFixed(1)}%</li>
                            <li class="list-group-item">Accuracy Improvement: ${((results.final_accuracy - results.initial_accuracy) * 100).toFixed(1)}%</li>
                            <li class="list-group-item">Training Efficiency: ${results.training_efficiency.toFixed(2)} samples/sec</li>
                        </ul>
                    </div>
                </div>
                ${results.knowledge_contributions ? `
                <div class="row mt-3">
                    <div class="col-12">
                        <h6>Knowledge Base Contributions</h6>
                        <div class="table-responsive">
                            <table class="table table-sm">
                                <thead>
                                    <tr>
                                        <th>Knowledge Domain</th>
                                        <th>Usage Count</th>
                                        <th>Contribution</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${Object.entries(results.knowledge_contributions).map(([domain, stats]) => `
                                        <tr>
                                            <td>${domain}</td>
                                            <td>${stats.usage_count}</td>
                                            <td>${(stats.contribution * 100).toFixed(1)}%</td>
                                        </tr>
                                    `).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                ` : ''}
            `;
        }
        
        modal.show();
    }

    // Import knowledge base file
    async importKnowledgeFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        try {
            const response = await fetch('/api/knowledge/import', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.status === 'success') {
                this.showMessage('success', 'Knowledge base file imported successfully');
                this.loadKnowledgeBase();
            } else {
                this.showMessage('error', result.message || 'Import failed');
            }
        } catch (error) {
            this.showMessage('error', 'Import failed');
        }
    }

    // Handle knowledge base file upload
    handleKnowledgeUpload(event) {
        const file = event.target.files[0];
        if (file) {
            this.importKnowledgeFile(file);
        }
        event.target.value = ''; // Reset file input
    }
}

// Initialize after page load
let trainingController;
document.addEventListener('DOMContentLoaded', () => {
    trainingController = new TrainingController();
});

// Global functions
function refreshDashboard() {
    if (trainingController) {
        trainingController.updateRealtimeDashboard();
        trainingController.updateCollaborationDashboard();
        trainingController.updateKnowledgeDashboard();
    }
}

function clearCommandLog() {
    const log = document.getElementById('commandLog');
    if (log) log.innerHTML = '';
}

function toggleAutoScroll() {
    const log = document.getElementById('commandLog');
    if (log) {
        const shouldAutoScroll = log.dataset.autoScroll !== 'true';
        log.dataset.autoScroll = shouldAutoScroll;
        
        if (shouldAutoScroll) {
            log.scrollTop = log.scrollHeight;
            setInterval(() => {
                if (log.dataset.autoScroll === 'true') {
                    log.scrollTop = log.scrollHeight;
                }
            }, 100);
        }
    }
}

function handleCommandInput(event) {
    if (event.key === 'Enter') {
        sendCommand();
    }
}

function sendCommand() {
    const input = document.getElementById('commandInput');
    const log = document.getElementById('commandLog');
    
    if (input && input.value.trim()) {
        const command = input.value.trim();
        input.value = '';
        
        if (log) {
            log.innerHTML += `<div class="text-light">> ${command}</div>`;
            if (log.dataset.autoScroll === 'true') {
                log.scrollTop = log.scrollHeight;
            }
        }
        
        // Send command to backend
        if (trainingController && trainingController.socket && trainingController.socket.readyState === WebSocket.OPEN) {
            trainingController.socket.send(JSON.stringify({
                type: 'command',
                command: command
            }));
        }
    }
}

// Global wrapper functions for TrainingController instance methods
function loadKnowledgeBase() {
    if (trainingController) {
        trainingController.loadKnowledgeBase();
    }
}

function importKnowledgeFile() {
    const fileInput = document.getElementById('knowledgeFileInput');
    if (fileInput && fileInput.files.length > 0) {
        if (trainingController) {
            trainingController.importKnowledgeFile(fileInput.files[0]);
        }
    } else {
        alert('Please select a knowledge base file to import');
    }
}
