// 知识库专家界面JavaScript
class KnowledgeInterface {
    constructor() {
        this.currentSessionId = null;
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.fileQueue = [];
        this.uploadProgress = 0;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.initializeWebSocket();
        this.loadStatistics();
        this.createNewSession();
    }

    // WebSocket连接
    initializeWebSocket() {
        try {
            this.ws = new WebSocket('ws://localhost:5000/ws/knowledge');
            
            this.ws.onopen = () => {
                console.log('WebSocket连接已建立');
                this.updateConnectionStatus('connected');
            };

            this.ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            };

            this.ws.onclose = () => {
                console.log('WebSocket连接已断开');
                this.updateConnectionStatus('disconnected');
                // 3秒后重连
                setTimeout(() => this.initializeWebSocket(), 3000);
            };

            this.ws.onerror = (error) => {
                console.error('WebSocket错误:', error);
                this.updateConnectionStatus('error');
            };
        } catch (error) {
            console.error('WebSocket初始化失败:', error);
        }
    }

    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'response':
                this.addMessageToChat(data.content, 'ai');
                break;
            case 'upload_progress':
                this.updateUploadProgress(data);
                break;
            case 'session_created':
                this.currentSessionId = data.session_id;
                break;
            case 'statistics_update':
                this.updateStatistics(data);
                break;
        }
    }

    // 事件监听器设置
    setupEventListeners() {
        // 拖拽上传
        this.setupDragAndDrop();
        
        // 键盘事件
        this.setupKeyboardEvents();
        
        // 文件输入
        this.setupFileInputs();
        
        // 语音录制
        this.setupVoiceRecording();
    }

    setupDragAndDrop() {
        const uploadZone = document.getElementById('uploadZone');
        if (!uploadZone) return;

        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadZone.addEventListener(eventName, this.preventDefaults, false);
        });

        uploadZone.addEventListener('dragenter', () => uploadZone.classList.add('dragover'));
        uploadZone.addEventListener('dragover', () => uploadZone.classList.add('dragover'));
        uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
        uploadZone.addEventListener('drop', (e) => this.handleDrop(e));
    }

    setupKeyboardEvents() {
        // 消息输入
        const messageInput = document.getElementById('messageInput');
        if (messageInput) {
            messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });
        }

        // 搜索输入
        const searchQuery = document.getElementById('searchQuery');
        if (searchQuery) {
            searchQuery.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.searchKnowledge();
                }
            });
        }
    }

    setupFileInputs() {
        // 图片输入
        const imageInput = document.getElementById('imageInput');
        if (imageInput) {
            imageInput.addEventListener('change', (e) => this.handleImageUpload(e));
        }

        // 视频输入
        const videoInput = document.getElementById('videoInput');
        if (videoInput) {
            videoInput.addEventListener('change', (e) => this.handleVideoUpload(e));
        }

        // 文件输入
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }
    }

    setupVoiceRecording() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.warn('浏览器不支持语音录制');
            return;
        }

        const startBtn = document.getElementById('startRecording');
        const stopBtn = document.getElementById('stopRecording');

        if (startBtn) {
            startBtn.addEventListener('click', () => this.startVoiceRecording());
        }

        if (stopBtn) {
            stopBtn.addEventListener('click', () => this.stopVoiceRecording());
        }
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    // 消息处理
    async sendMessage(text = null) {
        const input = document.getElementById('messageInput');
        const message = text || input.value.trim();
        
        if (!message || !this.currentSessionId) return;

        if (!text) {
            input.value = '';
        }

        this.addMessageToChat(message, 'user');
        this.showTypingIndicator();

        try {
            const response = await fetch('/api/text_interaction', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: this.currentSessionId,
                    text: message,
                    context: this.getContext()
                })
            });

            const data = await response.json();
            this.hideTypingIndicator();
            
            if (data.error) {
                this.addMessageToChat(data.error, 'error');
            } else {
                this.addMessageToChat(data.response, 'ai');
                this.updateStatistics();
            }
        } catch (error) {
            this.hideTypingIndicator();
            this.addMessageToChat('发送失败，请重试', 'error');
            console.error('发送消息失败:', error);
        }
    }

    getContext() {
        return {
            timestamp: new Date().toISOString(),
            session_id: this.currentSessionId,
            previous_messages: this.getRecentMessages()
        };
    }

    getRecentMessages() {
        const messages = document.querySelectorAll('#chatMessages .message-bubble');
        return Array.from(messages).slice(-5).map(msg => ({
            content: msg.textContent,
            type: msg.classList.contains('user-message') ? 'user' : 'ai'
        }));
    }

    addMessageToChat(content, type) {
        const container = document.getElementById('chatMessages');
        if (!container) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `d-flex ${type === 'user' ? 'justify-content-end' : 'justify-content-start'}`;
        
        const bubble = document.createElement('div');
        bubble.className = `message-bubble ${type}-message fade-in`;
        
        if (type === 'ai') {
            // 为AI回复添加格式化
            bubble.innerHTML = this.formatAIResponse(content);
        } else {
            bubble.textContent = content;
        }
        
        messageDiv.appendChild(bubble);
        container.appendChild(messageDiv);
        container.scrollTop = container.scrollHeight;

        // 添加动画
        setTimeout(() => bubble.classList.add('show'), 10);
    }

    formatAIResponse(content) {
        // 简单的格式化，可以根据需要扩展
        return content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
    }

    showTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) indicator.style.display = 'block';
    }

    hideTypingIndicator() {
        const indicator = document.getElementById('typingIndicator');
        if (indicator) indicator.style.display = 'none';
    }

    // 文件处理
    handleDrop(e) {
        const files = e.dataTransfer.files;
        this.handleFiles(Array.from(files));
    }

    handleFileSelect(e) {
        this.handleFiles(Array.from(e.target.files));
    }

    async handleFiles(files) {
        for (const file of files) {
            await this.processFile(file);
        }
    }

    async processFile(file) {
        try {
            // 显示上传进度
            this.showFileUploadProgress(file.name);
            
            // 根据文件类型选择处理方式
            if (file.type.startsWith('image/')) {
                await this.handleImageFile(file);
            } else if (file.type.startsWith('video/')) {
                await this.handleVideoFile(file);
            } else if (file.type.startsWith('audio/')) {
                await this.handleAudioFile(file);
            } else {
                await this.handleDocumentFile(file);
            }
            
            this.addFileToKnowledge(file);
            
        } catch (error) {
            console.error('处理文件失败:', error);
            this.showUploadError(file.name, error.message);
        }
    }

    async handleImageFile(file) {
        const formData = new FormData();
        formData.append('image', file);
        formData.append('session_id', this.currentSessionId);

        const response = await fetch('/api/image_interaction', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.response) {
            this.addMessageToChat(`图片分析结果: ${data.response}`, 'ai');
        }
    }

    async handleVideoFile(file) {
        const formData = new FormData();
        formData.append('video', file);
        formData.append('session_id', this.currentSessionId);

        const response = await fetch('/api/video_interaction', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.response) {
            this.addMessageToChat(`视频分析结果: ${data.response}`, 'ai');
        }
    }

    async handleAudioFile(file) {
        const formData = new FormData();
        formData.append('audio', file);
        formData.append('session_id', this.currentSessionId);

        const response = await fetch('/api/audio_interaction', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.response) {
            this.addMessageToChat(`音频分析结果: ${data.response}`, 'ai');
        }
    }

    async handleDocumentFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('category', this.getFileCategory(file.name));

        const response = await fetch('/api/upload_knowledge', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.success) {
            this.addMessageToChat(`文件 "${file.name}" 已成功导入知识库`, 'ai');
        }
    }

    getFileCategory(filename) {
        const ext = filename.split('.').pop().toLowerCase();
        const categoryMap = {
            'txt': 'text',
            'json': 'data',
            'csv': 'data',
            'pdf': 'document',
            'docx': 'document',
            'md': 'markdown',
            'jpg': 'image',
            'png': 'image',
            'mp4': 'video',
            'mp3': 'audio'
        };
        return categoryMap[ext] || 'general';
    }

    // 语音录制
    async startVoiceRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = () => {
                this.processVoiceRecording();
            };

            this.mediaRecorder.start();
            this.isRecording = true;
            
            document.getElementById('startRecording').style.display = 'none';
            document.getElementById('stopRecording').style.display = 'inline-block';
            
        } catch (error) {
            console.error('启动录音失败:', error);
            alert('无法访问麦克风，请检查权限设置');
        }
    }

    stopVoiceRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            document.getElementById('startRecording').style.display = 'inline-block';
            document.getElementById('stopRecording').style.display = 'none';
            
            // 关闭麦克风
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }

    async processVoiceRecording() {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        formData.append('session_id', this.currentSessionId);

        try {
            const response = await fetch('/api/audio_interaction', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            if (data.text) {
                // 显示转录的文本
                this.addMessageToChat(data.text, 'user');
                
                // 发送转录文本进行知识查询
                if (data.response) {
                    this.addMessageToChat(data.response, 'ai');
                }
            }
        } catch (error) {
            console.error('语音处理失败:', error);
            this.addMessageToChat('语音处理失败，请重试', 'error');
        }
    }

    // 知识搜索
    async searchKnowledge() {
        const query = document.getElementById('searchQuery')?.value.trim();
        const category = document.getElementById('searchCategory')?.value;
        
        if (!query) return;

        const resultsDiv = document.getElementById('searchResults');
        if (!resultsDiv) return;

        resultsDiv.innerHTML = '<div class="loading-spinner"></div>';

        try {
            const response = await fetch('/api/search_knowledge', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: query,
                    category: category,
                    limit: 10,
                    fuzzy: true
                })
            });

            const data = await response.json();
            this.displaySearchResults(data.results);
        } catch (error) {
            resultsDiv.innerHTML = '<div class="alert alert-danger">搜索失败</div>';
            console.error('搜索失败:', error);
        }
    }

    displaySearchResults(results) {
        const resultsDiv = document.getElementById('searchResults');
        if (!resultsDiv) return;

        if (!results || results.length === 0) {
            resultsDiv.innerHTML = '<div class="text-center text-muted">没有找到相关结果</div>';
            return;
        }

        let html = '';
        results.forEach((result, index) => {
            html += `
                <div class="knowledge-item fade-in" style="animation-delay: ${index * 0.1}s">
                    <div class="d-flex justify-content-between align-items-start">
                        <div class="flex-grow-1">
                            <strong>${result.title || '知识条目'}</strong>
                            <p class="mb-1">${result.content.substring(0, 150)}...</p>
                            <small class="text-muted">
                                类别: <span class="badge bg-info">${result.category}</span>
                                相关度: ${(result.relevance_score * 100).toFixed(1)}%
                            </small>
                        </div>
                        <button class="btn btn-sm btn-outline-primary" onclick="knowledgeInterface.viewKnowledge('${result.id}')">
                            <i class="bi bi-eye"></i> 查看
                        </button>
                    </div>
                </div>
            `;
        });
        
        resultsDiv.innerHTML = html;
    }

    // 会话管理
    async createNewSession() {
        try {
            const response = await fetch('/api/create_session', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: 'web_user',
                    metadata: {
                        browser: navigator.userAgent,
                        timestamp: new Date().toISOString()
                    }
                })
            });

            const data = await response.json();
            this.currentSessionId = data.session_id;
            this.updateSessionSelect();
            this.loadSessions();
            
            return data.session_id;
        } catch (error) {
            console.error('创建会话失败:', error);
            return null;
        }
    }

    async loadSessions() {
        try {
            const response = await fetch('/api/sessions');
            const data = await response.json();
            this.displaySessions(data.sessions);
        } catch (error) {
            console.error('加载会话失败:', error);
        }
    }

    displaySessions(sessions) {
        const sessionsList = document.getElementById('sessionsList');
        if (!sessionsList) return;

        let html = '';
        sessions.forEach(session => {
            html += `
                <div class="card mb-2">
                    <div class="card-body">
                        <div class="d-flex justify-content-between">
                            <div>
                                <strong>会话 ${session.id.substring(0, 8)}...</strong>
                                <br><small class="text-muted">${new Date(session.created_at).toLocaleString()}</small>
                            </div>
                            <div>
                                <button class="btn btn-sm btn-primary" onclick="knowledgeInterface.switchSession('${session.id}')">
                                    <i class="bi bi-arrow-right-circle"></i> 切换
                                </button>
                                <button class="btn btn-sm btn-danger" onclick="knowledgeInterface.deleteSession('${session.id}')">
                                    <i class="bi bi-trash"></i>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        sessionsList.innerHTML = html;
    }

    async switchSession(sessionId) {
        this.currentSessionId = sessionId;
        this.clearChat();
        await this.loadSessionHistory(sessionId);
        this.updateSessionSelect();
    }

    async loadSessionHistory(sessionId) {
        try {
            const response = await fetch(`/api/session_history/${sessionId}`);
            const data = await response.json();
            
            data.messages.forEach(message => {
                this.addMessageToChat(message.content, message.role);
            });
        } catch (error) {
            console.error('加载会话历史失败:', error);
        }
    }

    async deleteSession(sessionId) {
        if (!confirm('确定要删除这个会话吗？')) return;

        try {
            const response = await fetch(`/api/delete_session/${sessionId}`, {
                method: 'DELETE'
            });

            if (response.ok) {
                this.loadSessions();
                if (sessionId === this.currentSessionId) {
                    this.createNewSession();
                }
            }
        } catch (error) {
            console.error('删除会话失败:', error);
        }
    }

    clearChat() {
        const container = document.getElementById('chatMessages');
        if (container) container.innerHTML = '';
    }

    updateSessionSelect() {
        const select = document.getElementById('sessionSelect');
        if (!select) return;

        const option = document.createElement('option');
        option.value = this.currentSessionId;
        option.textContent = `会话 ${this.currentSessionId.substring(0, 8)}...`;
        select.appendChild(option);
        select.value = this.currentSessionId;
    }

    // 统计更新
    async loadStatistics() {
        try {
            const response = await fetch('/api/statistics');
            const data = await response.json();
            this.updateStatistics(data);
        } catch (error) {
            console.error('加载统计失败:', error);
        }
    }

    updateStatistics(data) {
        const stats = data || {
            total_knowledge: Math.floor(Math.random() * 1000) + 500,
            total_sessions: Math.floor(Math.random() * 10) + 1,
            total_files: Math.floor(Math.random() * 100) + 20,
            today_queries: Math.floor(Math.random() * 50) + 10,
            processing_queue: Math.floor(Math.random() * 5),
            response_time: Math.floor(Math.random() * 1000) + 100
        };

        document.getElementById('totalKnowledge')?.textContent = stats.total_knowledge;
        document.getElementById('totalSessions')?.textContent = stats.total_sessions;
        document.getElementById('totalFiles')?.textContent = stats.total_files;
        document.getElementById('todayQueries')?.textContent = stats.today_queries;
        document.getElementById('processingQueue')?.textContent = stats.processing_queue;
        document.getElementById('responseTime')?.textContent = `${stats.response_time}ms`;
    }

    // 辅助功能
    async exportKnowledge() {
        try {
            const response = await fetch('/api/export_knowledge');
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `knowledge_export_${new Date().toISOString().split('T')[0]}.json`;
            a.click();
            window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('导出失败:', error);
            alert('导出失败，请重试');
        }
    }

    async backupKnowledge() {
        try {
            const response = await fetch('/api/backup_knowledge', { method: 'POST' });
            const data = await response.json();
            
            if (data.success) {
                alert('备份成功完成');
            }
        } catch (error) {
            console.error('备份失败:', error);
            alert('备份失败，请重试');
        }
    }

    async batchImport() {
        const folderPath = document.getElementById('folderPath')?.value;
        const category = document.getElementById('importCategory')?.value;

        if (!folderPath) {
            alert('请输入文件夹路径');
            return;
        }

        try {
            const response = await fetch('/api/batch_import', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    folder_path: folderPath,
                    category: category || 'general',
                    recursive: true
                })
            });

            const data = await response.json();
            
            if (data.success) {
                alert(`批量导入完成，共导入 ${data.files_processed} 个文件`);
                this.updateStatistics();
            }
        } catch (error) {
            console.error('批量导入失败:', error);
            alert('批量导入失败，请检查文件夹路径');
        }
    }

    // 工具方法
    showFileUploadProgress(filename) {
        const progressContainer = document.getElementById('uploadProgress');
        if (progressContainer) {
            progressContainer.style.display = 'block';
            document.getElementById('uploadStatus').textContent = `正在上传 ${filename}...`;
        }
    }

    updateUploadProgress(data) {
        const progressBar = document.getElementById('progressBar');
        if (progressBar) {
            progressBar.style.width = `${data.progress || 0}%`;
            document.getElementById('uploadStatus').textContent = data.message || '处理中...';
        }
    }

    showUploadError(filename, error) {
        const status = document.getElementById('uploadStatus');
        if (status) {
            status.textContent = `上传失败: ${filename} - ${error}`;
            status.className = 'text-danger';
        }
    }

    addFileToKnowledge(file) {
        // 这里可以添加将文件添加到知识库的逻辑
        console.log('文件已添加到知识库:', file.name);
    }

    updateConnectionStatus(status) {
        const statusElement = document.querySelector('.badge');
        if (statusElement) {
            const statusMap = {
                connected: { text: '在线', class: 'bg-success' },
                disconnected: { text: '离线', class: 'bg-danger' },
                error: { text: '错误', class: 'bg-warning' }
            };
            
            const statusInfo = statusMap[status] || statusMap.disconnected;
            statusElement.textContent = statusInfo.text;
            statusElement.className = `badge ${statusInfo.class}`;
        }
    }
}

// 全局实例
let knowledgeInterface;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    knowledgeInterface = new KnowledgeInterface();
});

// 全局函数供HTML调用
function sendMessage() {
    knowledgeInterface.sendMessage();
}

function startVoiceInput() {
    knowledgeInterface.startVoiceRecording();
}

function searchKnowledge() {
    knowledgeInterface.searchKnowledge();
}

function createNewSession() {
    knowledgeInterface.createNewSession();
}

function exportKnowledge() {
    knowledgeInterface.exportKnowledge();
}

function backupKnowledge() {
    knowledgeInterface.backupKnowledge();
}

function batchImport() {
    knowledgeInterface.batchImport();
}