// Enhanced AI Chat System - Self Brain AGI

class EnhancedChatManager {
    constructor() {
        this.currentConversation = null;
        this.messages = [];
        this.attachments = [];
        this.isLoading = false;
        this.socket = null;
        this.isVoiceMode = false;
        this.recognition = null;
        this.synthesis = null;
        this.currentModel = 'a_management';
        this.currentKnowledgeBase = 'all';
        this.responseSettings = {
            style: 'detailed',
            temperature: 0.7,
            maxTokens: 1000,
            autoSpeak: true
        };
        
        // Initialize the chat system
        document.addEventListener('DOMContentLoaded', () => {
            this.init();
        });
    }

    async init() {
        this.setupEventListeners();
        this.setupSocketIO();
        this.setupVoiceRecognition();
        await this.loadConversations();
        await this.loadKnowledgeBases();
        this.updateAPIStatus();
        
        // Setup notification system
        this.setupNotificationSystem();
    }

    setupEventListeners() {
        // Message input events
        const messageInput = document.getElementById('messageInput');
        if (messageInput) {
            messageInput.addEventListener('keydown', this.handleKeyDown.bind(this));
            messageInput.addEventListener('input', this.adjustTextareaHeight.bind(this));
        }

        // Send button
        const sendBtn = document.getElementById('sendBtn');
        if (sendBtn) {
            sendBtn.addEventListener('click', this.sendMessage.bind(this));
        }

        // File upload events
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        if (uploadArea) {
            uploadArea.addEventListener('click', () => fileInput.click());
            uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
            uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
            uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        }
        
        if (fileInput) {
            fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        }

        // Model selector
        const modelSelector = document.getElementById('modelSelector');
        if (modelSelector) {
            modelSelector.addEventListener('change', this.handleModelChange.bind(this));
        }

        // Knowledge base selector
        const knowledgeSelect = document.getElementById('knowledgeSelect');
        if (knowledgeSelect) {
            knowledgeSelect.addEventListener('change', this.handleKnowledgeChange.bind(this));
        }

        // Conversation management
        const newConversationBtn = document.getElementById('newConversationBtn');
        const clearChatBtn = document.getElementById('clearChatBtn');
        const exportChatBtn = document.getElementById('exportChatBtn');
        const toggleVoiceBtn = document.getElementById('toggleVoiceBtn');
        
        if (newConversationBtn) {
            newConversationBtn.addEventListener('click', this.createNewConversation.bind(this));
        }
        
        if (clearChatBtn) {
            clearChatBtn.addEventListener('click', this.clearChat.bind(this));
        }
        
        if (exportChatBtn) {
            exportChatBtn.addEventListener('click', this.exportChat.bind(this));
        }
        
        if (toggleVoiceBtn) {
            toggleVoiceBtn.addEventListener('click', this.toggleVoice.bind(this));
        }

        // Response settings
        const responseStyle = document.getElementById('responseStyle');
        const temperature = document.getElementById('temperature');
        const tempValue = document.getElementById('tempValue');
        const maxTokens = document.getElementById('maxTokens');
        const tokensValue = document.getElementById('tokensValue');
        const autoSpeak = document.getElementById('autoSpeak');
        
        if (responseStyle) {
            responseStyle.addEventListener('change', (e) => {
                this.responseSettings.style = e.target.value;
            });
        }
        
        if (temperature && tempValue) {
            temperature.addEventListener('input', (e) => {
                this.responseSettings.temperature = parseFloat(e.target.value);
                tempValue.textContent = this.responseSettings.temperature;
            });
        }
        
        if (maxTokens && tokensValue) {
            maxTokens.addEventListener('input', (e) => {
                this.responseSettings.maxTokens = parseInt(e.target.value);
                tokensValue.textContent = this.responseSettings.maxTokens;
            });
        }
        
        if (autoSpeak) {
            autoSpeak.addEventListener('change', (e) => {
                this.responseSettings.autoSpeak = e.target.checked;
            });
        }
    }

    setupSocketIO() {
        // Create Socket.IO connection with robust configuration
        try {
            this.socket = io(window.location.origin, {
                transports: ['websocket', 'polling'],
                reconnection: true,
                reconnectionDelay: 1000,
                reconnectionDelayMax: 5000,
                reconnectionAttempts: 10,
                timeout: 20000,
                autoConnect: true,
                upgrade: true,
                rememberUpgrade: true,
                forceNew: true
            });
            
            this.socket.on('connect', () => {
                console.log('Connected to Self Brain AGI System');
                this.updateConnectionStatus(true);
            });

            this.socket.on('disconnect', (reason) => {
                console.log('Disconnected from Self Brain AGI System:', reason);
                this.updateConnectionStatus(false);
            });
            
            this.socket.on('connect_error', (error) => {
                console.error('Socket.IO connection error:', error);
                this.updateConnectionStatus(false);
            });
            
            this.socket.on('reconnect', (attemptNumber) => {
                console.log('Reconnected to system after', attemptNumber, 'attempts');
                this.updateConnectionStatus(true);
            });
            
            this.socket.on('reconnect_attempt', (attemptNumber) => {
                console.log('Attempting to reconnect to system...', attemptNumber);
            });
            
            this.socket.on('reconnect_error', (error) => {
                console.error('Socket.IO reconnection error:', error);
                this.updateConnectionStatus(false);
            });
            
            this.socket.on('reconnect_failed', () => {
                console.error('Failed to reconnect to system');
                this.updateConnectionStatus(false);
            });

            this.socket.on('ai_response', (data) => {
                this.handleAIResponse(data);
            });

            this.socket.on('typing_indicator', (data) => {
                this.showTypingIndicator(data.show);
            });

            this.socket.on('system_message', (data) => {
                this.showNotification(data.message, data.type || 'info');
            });
        } catch (error) {
            console.error('Socket.IO setup failed:', error);
            this.updateConnectionStatus(false);
        }
    }

    // Handle AI response received via Socket.IO
    handleAIResponse(data) {
        if (!data || !data.response) return;
        
        try {
            // Add AI response to messages
            this.addMessage('assistant', data.response, data.attachments || [], true);
            
            // Update tokens used if provided
            if (data.tokens_used) {
                const tokensUsedElement = document.getElementById('tokensUsed');
                if (tokensUsedElement) {
                    tokensUsedElement.textContent = data.tokens_used;
                }
            }
            
            // Update context length if provided
            if (data.context_length) {
                const contextLengthElement = document.getElementById('contextLength');
                if (contextLengthElement) {
                    contextLengthElement.textContent = data.context_length;
                }
            }
            
            // Speak response if enabled
            if (this.responseSettings.autoSpeak && this.synthesis && data.response) {
                this.speakResponse(data.response);
            }
            
            this.isLoading = false;
            this.updateSendButton();
            
            // Hide typing indicator
            this.showTypingIndicator(false);
            
        } catch (error) {
            console.error('Error handling AI response:', error);
            this.showNotification('Error processing AI response', 'error');
        }
    }

    setupVoiceRecognition() {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.recognition = new SpeechRecognition();
            this.recognition.continuous = false;
            this.recognition.interimResults = false;
            this.recognition.lang = 'en-US';

            this.recognition.onresult = (event) => {
                const transcript = event.results[0][0].transcript;
                const messageInput = document.getElementById('messageInput');
                if (messageInput) {
                    messageInput.value = transcript;
                    this.adjustTextareaHeight(messageInput);
                }
            };

            this.recognition.onerror = (event) => {
                console.error('Speech recognition error:', event.error);
                this.showNotification('Speech recognition failed, please try again', 'error');
            };

            this.recognition.onend = () => {
                this.isVoiceMode = false;
                this.updateVoiceButton();
            };
        }

        if ('speechSynthesis' in window) {
            this.synthesis = window.speechSynthesis;
        }
    }

    setupNotificationSystem() {
        // Create notification system if not exists
        if (!window.notificationSystem) {
            window.notificationSystem = {
                show: (message, type, duration) => {
                    this.showNotification(message, type, duration);
                }
            };
        }
    }

    async loadConversations() {
        try {
            this.setProcessingStatus('text', 'processing');
            const response = await fetch('/api/conversations');
            const data = await response.json();
            
            if (data && data.length > 0) {
                this.renderConversations(data);
            }
        } catch (error) {
            console.error('Failed to load conversations:', error);
            this.showNotification('Failed to load conversations', 'error');
        } finally {
            this.setProcessingStatus('text', 'ready');
        }
    }

    async loadKnowledgeBases() {
        try {
            // In a real implementation, this would fetch actual knowledge bases
            const knowledgeBases = [
                { id: 'general', title: 'General Knowledge' },
                { id: 'technical', title: 'Technical Documentation' },
                { id: 'system', title: 'System Information' }
            ];
            this.renderKnowledgeBases(knowledgeBases);
        } catch (error) {
            console.error('Failed to load knowledge bases:', error);
        }
    }

    async updateAPIStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            const apiStatus = document.getElementById('apiStatus');
            if (apiStatus) {
                if (data.status === 'ok') {
                    apiStatus.classList.remove('status-offline');
                    apiStatus.classList.add('status-healthy');
                    apiStatus.innerHTML = '<i class="bi bi-check-circle"></i> API Online';
                } else {
                    apiStatus.classList.remove('status-healthy');
                    apiStatus.classList.add('status-offline');
                    apiStatus.innerHTML = '<i class="bi bi-exclamation-circle"></i> API Error';
                }
            }
        } catch (error) {
            console.error('Failed to check API status:', error);
            const apiStatus = document.getElementById('apiStatus');
            if (apiStatus) {
                apiStatus.classList.remove('status-healthy');
                apiStatus.classList.add('status-offline');
                apiStatus.innerHTML = '<i class="bi bi-exclamation-circle"></i> API Error';
            }
        }
    }

    renderConversations(conversations) {
        const container = document.getElementById('conversationList');
        if (!container) return;

        container.innerHTML = '';
        
        if (conversations && conversations.length > 0) {
            conversations.forEach(conv => {
                const item = document.createElement('div');
                item.className = 'conversation-item' + (this.currentConversation === conv.id ? ' selected' : '');
                item.innerHTML = `
                    <div class="conversation-title">${this.escapeHtml(conv.title)}</div>
                    <div class="conversation-time">${this.formatTime(conv.last_activity)}</div>
                `;
                item.addEventListener('click', () => this.loadConversation(conv.id));
                container.appendChild(item);
            });
        } else {
            container.innerHTML = `
                <div class="text-center text-gray-500 py-4">
                    <i class="bi bi-chat-left"></i> No conversations yet
                </div>
            `;
        }
    }

    renderKnowledgeBases(knowledge) {
        const select = document.getElementById('knowledgeSelect');
        if (!select) return;

        select.innerHTML = '<option value="all">All Knowledge Bases</option>';
        if (knowledge && knowledge.length > 0) {
            knowledge.forEach(item => {
                const option = document.createElement('option');
                option.value = item.id;
                option.textContent = item.title;
                select.appendChild(option);
            });
        }
    }

    async loadConversation(conversationId) {
        try {
            this.setProcessingStatus('text', 'processing');
            this.currentConversation = conversationId;
            this.messages = [];
            this.renderMessages();
            
            // Update UI
            const titleElement = document.getElementById('currentConversationTitle');
            if (titleElement) {
                // In a real implementation, we would fetch the conversation title
                titleElement.textContent = 'Conversation ' + conversationId.substring(0, 8);
            }
            
            // Highlight selected conversation
            const conversationItems = document.querySelectorAll('.conversation-item');
            conversationItems.forEach(item => {
                item.classList.remove('selected');
                // Add data-id attribute if missing
                if (!item.dataset.id) {
                    const title = item.querySelector('.conversation-title')?.textContent;
                    if (title) {
                        item.dataset.id = conversationId;
                    }
                }
            });
            
            // Try to find by data-id first
            let selectedItem = document.querySelector(`.conversation-item[data-id="${conversationId}"]`);
            
            // If not found, just select the first item as a fallback
            if (!selectedItem && conversationItems.length > 0) {
                selectedItem = conversationItems[0];
            }
            
            if (selectedItem) {
                selectedItem.classList.add('selected');
            }
        } catch (error) {
            console.error('Failed to load conversation messages:', error);
            this.showNotification('Failed to load conversation', 'error');
        } finally {
            this.setProcessingStatus('text', 'ready');
        }
    }

    async createNewConversation() {
        try {
            this.setProcessingStatus('text', 'processing');
            this.currentConversation = 'conv_' + Date.now();
            this.messages = [];
            this.renderMessages();
            
            // Update title
            const titleElement = document.getElementById('currentConversationTitle');
            if (titleElement) {
                titleElement.textContent = 'New Conversation';
            }
            
            this.showNotification('New conversation created', 'success');
        } catch (error) {
            console.error('Failed to create new conversation:', error);
            this.showNotification('Failed to create new conversation', 'error');
        } finally {
            this.setProcessingStatus('text', 'ready');
        }
    }

    async sendMessage() {
        if (this.isLoading) return;

        const input = document.getElementById('messageInput');
        const message = input.value.trim();
        
        if (!message && this.attachments.length === 0) {
            this.showNotification('Please enter a message or upload a file', 'warning');
            return;
        }

        // Save attachments before clearing
        const attachmentsToSend = [...this.attachments];

        // Add user message
        this.addMessage('user', message, attachmentsToSend);
        input.value = '';
        this.adjustTextareaHeight(input);
        
        // Clear attachments
        this.attachments = [];
        this.clearFilePreview();

        this.isLoading = true;
        this.updateSendButton();
        input.disabled = true;
        
        // Show typing indicator
        this.showTypingIndicator(true);

        try {
            const startTime = Date.now();
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    conversation_id: this.currentConversation || 'default',
                    knowledge_base: this.currentKnowledgeBase,
                    model_id: this.currentModel,
                    attachments: attachmentsToSend,
                    response_settings: this.responseSettings
                })
            });

            const endTime = Date.now();
            const responseTime = endTime - startTime;
            
            // Update real-time data
            const responseTimeElement = document.getElementById('responseTime');
            if (responseTimeElement) {
                responseTimeElement.textContent = responseTime + 'ms';
            }

            if (!response.ok) {
                throw new Error('HTTP error! status: ' + response.status);
            }

            const data = await response.json();
            if (data.status === 'success') {
                // Update tokens used
                const tokensUsedElement = document.getElementById('tokensUsed');
                if (tokensUsedElement && data.tokens_used) {
                    tokensUsedElement.textContent = data.tokens_used;
                }
                
                // Update context length
                const contextLengthElement = document.getElementById('contextLength');
                if (contextLengthElement && data.context_length) {
                    contextLengthElement.textContent = data.context_length;
                }
                
                // Add AI response with animation
                this.addMessage('assistant', data.response, data.attachments, true);
                
                // Speak response if enabled
                if (this.responseSettings.autoSpeak && this.synthesis && data.response) {
                    this.speakResponse(data.response);
                }
                
                this.showNotification('Message sent successfully', 'success');
            } else {
                throw new Error(data.message || 'Failed to send message');
            }
        } catch (error) {
            console.error('Failed to send message:', error);
            this.showNotification('Failed to send message, please try again', 'error');
        } finally {
            this.isLoading = false;
            this.updateSendButton();
            input.disabled = false;
            input.focus();
            
            // Hide typing indicator
            this.showTypingIndicator(false);
        }
    }

    addMessage(role, content, attachments = [], animate = false) {
        const message = {
            id: 'msg_' + Date.now(),
            role: role,
            content: content,
            timestamp: new Date().toISOString(),
            attachments: attachments
        };

        this.messages.push(message);
        this.renderMessage(message, animate);
    }

    renderMessage(message, animate = false) {
        const container = document.getElementById('messagesContainer');
        if (!container) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = 'message ' + message.role + '-message';
        messageDiv.dataset.messageId = message.id;

        const avatar = message.role === 'user' ? 
            '<div class="message-avatar user-avatar"><i class="bi bi-person"></i></div>' :
            '<div class="message-avatar ai-avatar"><i class="bi bi-robot"></i></div>';

        messageDiv.innerHTML = `
            ${avatar}
            <div class="message-content">
                <div class="message-text">${animate && message.role === 'assistant' ? 
                    '<span class="typing-indicator">' + this.formatMessageContent(message.content) + '</span>' : 
                    this.formatMessageContent(message.content)}</div>
                ${message.attachments && message.attachments.length > 0 ? this.renderAttachments(message.attachments) : ''}
                <div class="message-time">${this.formatTime(message.timestamp)}</div>
            </div>
        `;

        container.appendChild(messageDiv);
        this.scrollToBottom();

        if (animate && message.role === 'assistant') {
            this.typeText(messageDiv.querySelector('.typing-indicator'), message.content);
        }
    }

    renderMessages() {
        const container = document.getElementById('messagesContainer');
        if (!container) return;

        container.innerHTML = '';
        
        if (this.messages.length === 0) {
            // Show welcome message if no messages
            this.showWelcomeMessage();
        } else {
            this.messages.forEach(message => this.renderMessage(message));
        }
    }

    formatMessageContent(content) {
        if (!content) return '';
        
        // Process code blocks
        content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            return '<pre><code class="language-' + (lang || 'plaintext') + '">' + this.escapeHtml(code.trim()) + '</code></pre>';
        });

        // Process inline code
        content = content.replace(/`([^`]+)`/g, '<code>$1</code>');

        // Process bold text
        content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        // Process italic text
        content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');

        // Process line breaks
        content = content.replace(/\n/g, '<br>');

        return content;
    }

    renderAttachments(attachments) {
        return attachments.map(attachment => {
            let iconClass = 'bi bi-file';
            if (attachment.type && attachment.type.startsWith('image/')) {
                iconClass = 'bi bi-file-image';
            } else if (attachment.type && attachment.type.startsWith('audio/')) {
                iconClass = 'bi bi-file-earmark-music';
            } else if (attachment.type && attachment.type.startsWith('video/')) {
                iconClass = 'bi bi-file-earmark-play';
            } else if (attachment.name && (attachment.name.endsWith('.pdf') || attachment.type === 'application/pdf')) {
                iconClass = 'bi bi-file-pdf';
            } else if (attachment.name && (attachment.name.endsWith('.doc') || attachment.name.endsWith('.docx'))) {
                iconClass = 'bi bi-file-word';
            } else if (attachment.name && (attachment.name.endsWith('.xls') || attachment.name.endsWith('.xlsx'))) {
                iconClass = 'bi bi-file-excel';
            }
            
            return '<div class="message-attachment"><i class="' + iconClass + '"></i><span>' + this.escapeHtml(attachment.name) + '</span><small>(' + this.formatFileSize(attachment.size) + ')</small></div>';
        }).join('');
    }

    async handleFileUpload(files) {
        try {
            this.setProcessingStatus('image', 'processing');
            const uploadPromises = Array.from(files).map(file => this.uploadFile(file));
            const results = await Promise.allSettled(uploadPromises);
            
            results.forEach((result, index) => {
                if (result.status === 'fulfilled') {
                    const fileInfo = {
                        name: files[index].name,
                        size: files[index].size,
                        type: files[index].type,
                        url: result.value.file_path || ''
                    };
                    this.attachments.push(fileInfo);
                    this.renderFilePreview(fileInfo);
                    this.showNotification('File ' + files[index].name + ' uploaded successfully', 'success');
                } else {
                    this.showNotification('File ' + files[index].name + ' upload failed', 'error');
                }
            });
        } catch (error) {
            console.error('File upload error:', error);
            this.showNotification('File upload failed', 'error');
        } finally {
            this.setProcessingStatus('image', 'idle');
        }
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('category', 'chat_attachment');

        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.status !== 'success') {
            throw new Error(data.message || 'Upload failed');
        }

        return data;
    }

    renderFilePreview(fileInfo) {
        const filePreview = document.getElementById('filePreview');
        if (!filePreview) return;

        const previewItem = document.createElement('div');
        previewItem.className = 'file-preview-item';
        previewItem.dataset.fileName = fileInfo.name;
        
        let iconClass = 'bi bi-file';
        if (fileInfo.type && fileInfo.type.startsWith('image/')) {
            iconClass = 'bi bi-file-image';
        } else if (fileInfo.type && fileInfo.type.startsWith('audio/')) {
            iconClass = 'bi bi-file-earmark-music';
        } else if (fileInfo.type && fileInfo.type.startsWith('video/')) {
            iconClass = 'bi bi-file-earmark-play';
        } else if (fileInfo.name && (fileInfo.name.endsWith('.pdf') || fileInfo.type === 'application/pdf')) {
            iconClass = 'bi bi-file-pdf';
        } else if (fileInfo.name && (fileInfo.name.endsWith('.doc') || fileInfo.name.endsWith('.docx'))) {
            iconClass = 'bi bi-file-word';
        } else if (fileInfo.name && (fileInfo.name.endsWith('.xls') || fileInfo.name.endsWith('.xlsx'))) {
            iconClass = 'bi bi-file-excel';
        }

        previewItem.innerHTML = '<i class="' + iconClass + '"></i><span>' + this.escapeHtml(fileInfo.name) + '</span><button class="btn-close btn-close-sm" onclick="event.stopPropagation(); chatManager.removeAttachment(\'' + this.escapeHtml(fileInfo.name) + '\')"></button>';

        filePreview.appendChild(previewItem);
    }

    removeAttachment(fileName) {
        this.attachments = this.attachments.filter(attachment => attachment.name !== fileName);
        
        const filePreview = document.getElementById('filePreview');
        if (filePreview) {
            const previewItem = filePreview.querySelector('[data-file-name="' + fileName + '"]');
            if (previewItem) {
                previewItem.remove();
            }
        }
    }

    clearFilePreview() {
        const filePreview = document.getElementById('filePreview');
        if (filePreview) {
            filePreview.innerHTML = '';
        }
    }

    toggleVoice() {
        if (!this.recognition) {
            this.showNotification('Browser does not support speech recognition', 'warning');
            return;
        }

        if (this.isVoiceMode) {
            this.recognition.stop();
            this.isVoiceMode = false;
        } else {
            this.recognition.start();
            this.isVoiceMode = true;
        }

        this.updateVoiceButton();
    }

    speakResponse(text) {
        if (!this.synthesis) return;

        // Cancel any ongoing speech
        this.synthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-US';
        utterance.rate = 1;
        utterance.pitch = 1;
        
        this.synthesis.speak(utterance);
    }

    clearChat() {
        if (confirm('Are you sure you want to clear the current conversation? This action cannot be undone.')) {
            this.messages = [];
            this.renderMessages();
            this.showNotification('Conversation cleared', 'info');
        }
    }

    exportChat() {
        if (this.messages.length === 0) {
            this.showNotification('No conversation content to export', 'warning');
            return;
        }

        try {
            const chatData = {
                conversation_id: this.currentConversation,
                messages: this.messages,
                export_time: new Date().toISOString(),
                model: this.currentModel,
                knowledge_base: this.currentKnowledgeBase
            };

            const blob = new Blob([JSON.stringify(chatData, null, 2)], {
                type: 'application/json'
            });

            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'chat_export_' + new Date().toISOString().split('T')[0] + '.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            this.showNotification('Conversation exported successfully', 'success');
        } catch (error) {
            console.error('Export failed:', error);
            this.showNotification('Export failed, please try again', 'error');
        }
    }

    showWelcomeMessage() {
        const container = document.getElementById('messagesContainer');
        if (!container) return;

        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'message ai-message welcome-message';
        welcomeDiv.innerHTML = '<div class="message-avatar ai-avatar"><i class="bi bi-robot"></i></div><div class="message-content"><p>👋 Hello! I am your Enhanced AI Assistant.</p><p>I can help you with:</p><ul class="mb-0"><li>📚 Answering questions based on knowledge base</li><li>🔍 Searching and organizing information</li><li>📄 Analyzing uploaded documents</li><li>💡 Providing intelligent suggestions</li><li>🎯 Completing complex tasks</li></ul><p class="mt-2 mb-0">Feel free to ask me anything!</p></div>';
        container.appendChild(welcomeDiv);
    }

    showTypingIndicator(show) {
        const container = document.getElementById('messagesContainer');
        if (!container) return;
        
        let existing = container.querySelector('.typing-indicator-container');
        
        if (show && !existing) {
            const indicator = document.createElement('div');
            indicator.className = 'message ai-message typing-indicator-container';
            indicator.innerHTML = '<div class="message-avatar ai-avatar"><i class="bi bi-robot"></i></div><div class="message-content"><div class="typing-dots"><span></span><span></span><span></span></div></div>';
            
            container.appendChild(indicator);
            this.scrollToBottom();
        } else if (!show && existing) {
            existing.remove();
        }
    }

    showNotification(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = 'notification notification-' + type;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, duration);
    }

    handleModelChange(event) {
        this.currentModel = event.target.value;
        const modelNameElement = document.getElementById('modelName');
        if (modelNameElement) {
            modelNameElement.textContent = this.getModelDisplayName(this.currentModel);
        }
        this.showNotification('Switched to ' + this.getModelDisplayName(this.currentModel), 'info');
    }

    handleKnowledgeChange(event) {
        this.currentKnowledgeBase = event.target.value;
        this.showNotification('Knowledge base updated', 'info');
    }

    getModelDisplayName(modelId) {
        const modelNames = {
            'a_management': 'A Management Model',
            'b_language': 'B Language Model',
            'c_audio': 'C Audio Model',
            'd_image': 'D Image Model',
            'e_video': 'E Video Model',
            'f_spatial': 'F Spatial Model',
            'g_sensor': 'G Sensor Model',
            'h_computer': 'H Computer Model',
            'i_motion': 'I Motion Model',
            'j_knowledge': 'J Knowledge Model',
            'k_programming': 'K Programming Model'
        };
        
        return modelNames[modelId] || modelId;
    }

    updateConnectionStatus(isConnected) {
        const connectionStatus = document.getElementById('connectionStatus');
        if (connectionStatus) {
            if (isConnected) {
                connectionStatus.textContent = 'Connected';
                connectionStatus.style.color = 'var(--gray-700)';
            } else {
                connectionStatus.textContent = 'Disconnected';
                connectionStatus.style.color = 'var(--gray-500)';
            }
        }
    }

    setProcessingStatus(type, status) {
        let statusElement;
        
        switch (type) {
            case 'text':
                statusElement = document.getElementById('textStatus');
                break;
            case 'image':
                statusElement = document.getElementById('imageStatus');
                break;
            case 'audio':
                statusElement = document.getElementById('audioStatus');
                break;
            default:
                return;
        }
        
        if (statusElement) {
            // Remove all status classes
            statusElement.className = 'badge';
            
            // Add appropriate status class and text
            switch (status) {
                case 'ready':
                    statusElement.classList.add('bg-gray-700');
                    statusElement.textContent = 'Ready';
                    break;
                case 'processing':
                    statusElement.classList.add('bg-gray-500');
                    statusElement.textContent = 'Processing';
                    break;
                case 'idle':
                    statusElement.classList.add('bg-gray-400');
                    statusElement.textContent = 'Idle';
                    break;
                case 'error':
                    statusElement.classList.add('bg-gray-800');
                    statusElement.textContent = 'Error';
                    break;
            }
        }
    }

    // Event handler functions
    handleKeyDown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }

    handleDragOver(event) {
        event.preventDefault();
        event.currentTarget.classList.add('dragover');
    }

    handleDragLeave(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('dragover');
    }

    handleDrop(event) {
        event.preventDefault();
        event.currentTarget.classList.remove('dragover');
        this.handleFileUpload(event.dataTransfer.files);
    }

    handleFileSelect(event) {
        this.handleFileUpload(event.target.files);
    }

    // Utility functions
    adjustTextareaHeight(textarea) {
        textarea.style.height = 'auto';
        textarea.style.height = Math.min(textarea.scrollHeight, 120) + 'px';
    }

    scrollToBottom() {
        const container = document.getElementById('messagesContainer');
        if (container) {
            container.scrollTop = container.scrollHeight;
        }
    }

    updateSendButton() {
        const sendBtn = document.getElementById('sendBtn');
        if (sendBtn) {
            sendBtn.disabled = this.isLoading;
            sendBtn.style.opacity = this.isLoading ? '0.5' : '1';
        }
    }

    updateVoiceButton() {
        const voiceBtn = document.getElementById('toggleVoiceBtn');
        if (voiceBtn) {
            if (this.isVoiceMode) {
                voiceBtn.classList.add('active');
                voiceBtn.classList.add('btn-primary');
                voiceBtn.classList.remove('btn-outline-secondary');
            } else {
                voiceBtn.classList.remove('active');
                voiceBtn.classList.remove('btn-primary');
                voiceBtn.classList.add('btn-outline-secondary');
            }
        }
    }

    typeText(element, text, speed = 30) {
        let index = 0;
        element.textContent = '';
        
        const typeInterval = setInterval(() => {
            if (index < text.length) {
                element.textContent += text.charAt(index);
                index++;
                this.scrollToBottom();
            } else {
                clearInterval(typeInterval);
                element.classList.remove('typing-indicator');
            }
        }, speed);
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    formatTime(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize the chat manager
document.addEventListener('DOMContentLoaded', () => {
    window.chatManager = new EnhancedChatManager();
});