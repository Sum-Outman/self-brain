// Self Brain AGI System - AI Chat Functionality
// Self Brain AGI System - AI Chat Functionality

class ChatManager {
    constructor() {
        this.currentConversation = null;
        this.messages = [];
        this.isLoading = false;
        this.socket = null;
        this.isVoiceMode = false;
        this.recognition = null;
        this.synthesis = null;
        
        this.init();
    }

    async init() {
        this.setupEventListeners();
        this.setupSocketIO();
        this.setupVoiceRecognition();
        await this.loadConversations();
        await this.loadKnowledgeBases();
    }
    
    // Setup mock socket for development when Socket.IO is not available
    setupMockSocket() {
        // Create a mock socket object that mimics real Socket.IO functionality
        this.socket = {
            connected: false,
            on: (event, callback) => {
                if (event === 'connect') {
                    setTimeout(() => {
                        this.socket.connected = true;
                        callback();
                    }, 500);
                }
            },
            emit: (event, data, callback) => {
                console.log('Mock socket emit:', event, data);
                if (callback) callback();
                
                // Simulate bot responses for development
                if (event === 'message' && data && data.content) {
                    setTimeout(() => {
                        const mockResponses = [
                            "I'm processing your request. In a full environment, I would connect to the A Management Model for a real response.",
                            "This is a mock response. In a production setup, this would come from the actual Self Brain AGI system.",
                            "Thanks for your message! In a complete deployment, I would leverage the full capabilities of the Self Brain system.",
                            "I'm currently running in development mode with a mock connection. The real system would provide more detailed responses.",
                            "This is a placeholder response. For the full experience, please run the complete Self Brain system with Socket.IO server."
                        ];
                        
                        const randomResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)];
                        
                        this.addMessage('bot', randomResponse);
                        this.scrollToBottom();
                    }, 1000 + Math.random() * 1000);
                }
            },
            disconnect: () => {
                this.socket.connected = false;
            }
        };
        
        // Log that we're using a mock socket
        console.log('Development mode: Using mock socket connection.');
    }

    setupEventListeners() {
        // File upload related
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

        // Input box events
        const messageInput = document.getElementById('messageInput');
        if (messageInput) {
            messageInput.addEventListener('keydown', this.handleKeyDown.bind(this));
            messageInput.addEventListener('input', this.adjustTextareaHeight.bind(this));
        }

        // Knowledge base selector
        const knowledgeSelect = document.getElementById('knowledgeSelect');
        if (knowledgeSelect) {
            knowledgeSelect.addEventListener('change', this.handleKnowledgeChange.bind(this));
        }
    }

    setupSocketIO() {
        // Create Socket.IO connection with robust configuration
        try {
            // Check if io object exists
            if (typeof io === 'undefined') {
                console.warn('Socket.IO is not available. Using mock connection for development.');
                this.setupMockSocket();
                return;
            }
            
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
                this.showNotification('System connected', 'success');
            });

            this.socket.on('disconnect', (reason) => {
                console.log('Disconnected from Self Brain AGI System:', reason);
                this.showNotification('System connection lost', 'warning');
            });
            
            this.socket.on('connect_error', (error) => {
                console.error('Socket.IO connection error:', error);
                this.showNotification('Failed to connect to system', 'error');
            });
            
            this.socket.on('reconnect', (attemptNumber) => {
                console.log('Reconnected to system after', attemptNumber, 'attempts');
                this.showNotification('System reconnected', 'success');
            });
            
            this.socket.on('reconnect_attempt', (attemptNumber) => {
                console.log('Attempting to reconnect to system...', attemptNumber);
            });
            
            this.socket.on('reconnect_error', (error) => {
                console.error('Socket.IO reconnection error:', error);
            });
            
            this.socket.on('reconnect_failed', () => {
                console.error('Failed to reconnect to system');
                // In development environment, provide a fallback mock connection
                if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
                    console.log('Switching to mock socket connection for development');
                    this.setupMockSocket();
                }
                this.showNotification('Failed to reconnect to system', 'error');
            });
        } catch (error) {
            console.error('Socket.IO setup failed:', error);
            this.showNotification('System connection initialization failed', 'error');
        }

        this.socket.on('ai_response', (data) => {
            this.handleAIResponse(data);
        });

        this.socket.on('typing_indicator', (data) => {
            this.showTypingIndicator(data.show);
        });
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
                document.getElementById('messageInput').value = transcript;
                this.adjustTextareaHeight(document.getElementById('messageInput'));
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

    async loadConversations() {
        try {
            const response = await fetch('/api/chat/conversations');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.renderConversations(data.conversations);
            }
        } catch (error) {
            console.error('Failed to load conversation:', error);
            this.showNotification('Failed to load conversation', 'error');
        }
    }

    async loadKnowledgeBases() {
        try {
            const response = await fetch('/api/knowledge/list');
            const data = await response.json();
            
            if (data.status === 'success') {
                this.renderKnowledgeBases(data.knowledge);
            }
        } catch (error) {
            console.error('Failed to load knowledge base:', error);
        }
    }

    renderConversations(conversations) {
        const container = document.getElementById('conversationList');
        if (!container) return;

        container.innerHTML = '';
        conversations.forEach(conv => {
            const item = document.createElement('div');
            item.className = 'conversation-item';
            item.innerHTML = `
                <div class="conversation-title">${conv.title}</div>
                <div class="conversation-time">${this.formatTime(conv.last_activity)}</div>
            `;
            item.addEventListener('click', () => this.loadConversation(conv.id));
            container.appendChild(item);
        });
    }

    renderKnowledgeBases(knowledge) {
        const select = document.getElementById('knowledgeSelect');
        if (!select) return;

        select.innerHTML = '<option value="all">All Knowledge Bases</option>';
        knowledge.forEach(item => {
            const option = document.createElement('option');
            option.value = item.id;
            option.textContent = item.title;
            select.appendChild(option);
        });
    }

    async loadConversation(conversationId) {
        try {
            const response = await fetch(`/api/chat/messages/${conversationId}`);
            const data = await response.json();
            
            if (data.status === 'success') {
                this.currentConversation = conversationId;
                this.messages = data.messages;
                this.renderMessages();
            }
        } catch (error) {
            console.error('Failed to load conversation messages:', error);
        }
    }

    async createNewConversation() {
        try {
            const response = await fetch('/api/chat/new_conversation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    title: 'New Conversation',
                    timestamp: new Date().toISOString()
                })
            });
            
            const data = await response.json();
            if (data.status === 'success') {
                this.currentConversation = data.conversation_id;
                this.messages = [];
                this.renderMessages();
                this.loadConversations();
            }
        } catch (error) {
            console.error('Failed to create new conversation:', error);
        }
    }

    async sendMessage() {
        if (this.isLoading) return;

        const input = document.getElementById('messageInput');
        const message = input.value.trim();
        
        if (!message) return;

        // Add user message
        this.addMessage('user', message);
        input.value = '';
        this.adjustTextareaHeight(input);

        this.isLoading = true;
        this.updateSendButton();
        input.disabled = true;

        try {
            const response = await fetch('/api/chat/send', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    message: message,
                    conversation_id: this.currentConversation || 'default',
                    knowledge_base: document.getElementById('knowledgeSelect')?.value || 'all',
                    attachments: this.getCurrentAttachments()
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            if (data.status === 'success') {
                this.addMessage('assistant', data.response);
                if (data.should_speak && this.synthesis) {
                    this.speakResponse(data.response);
                }
                window.notificationSystem.show('Message sent successfully', 'success', 2000);
            } else {
                throw new Error(data.message || 'Failed to send message');
            }
        } catch (error) {
            console.error('Failed to send message:', error);
            this.showNotification('Failed to send message, please try again', 'error');
            window.notificationSystem.show('Sending failed, please check network connection', 'error', 3000);
        } finally {
            this.isLoading = false;
            this.updateSendButton();
            input.disabled = false;
            input.focus();
        }
    }

    addMessage(role, content, attachments = []) {
        const message = {
            id: `msg_${Date.now()}`,
            role: role,
            content: content,
            timestamp: new Date().toISOString(),
            attachments: attachments
        };

        this.messages.push(message);
        this.renderMessage(message);
    }

    renderMessage(message, animate = false) {
        const container = document.getElementById('messagesContainer');
        if (!container) return;

        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${message.role}-message`;
        messageDiv.dataset.messageId = message.id;

        const avatar = message.role === 'user' ? 
            '<div class="message-avatar user-avatar"><i class="fas fa-user"></i></div>' :
            '<div class="message-avatar ai-avatar-msg"><i class="fas fa-robot"></i></div>';

        messageDiv.innerHTML = `
            ${avatar}
            <div class="message-content">
                <div class="message-text">${animate && message.role === 'assistant' ? 
                    `<span class="typing-indicator">${this.formatMessageContent(message.content)}</span>` : 
                    this.formatMessageContent(message.content)}</div>
                ${message.attachments ? this.renderAttachments(message.attachments) : ''}
                <div class="message-time">${this.formatTime(message.timestamp)}</div>
            </div>
        `;

        container.appendChild(messageDiv);
        this.scrollToBottom();

        if (animate && message.role === 'assistant') {
            this.typeText(messageDiv.querySelector('.typing-indicator'), message.content);
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

    renderMessages() {
        const container = document.getElementById('messagesContainer');
        if (!container) return;

        container.innerHTML = '';
        this.messages.forEach(message => this.renderMessage(message));
    }

    formatMessageContent(content) {
        // Process code blocks
        content = content.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
            return `<pre><code class="language-${lang || 'plaintext'}">${this.escapeHtml(code.trim())}</code></pre>`;
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
        return attachments.map(attachment => `
            <div class="message-attachment">
                <i class="fas fa-paperclip"></i>
                <span>${attachment.name}</span>
                <small>(${this.formatFileSize(attachment.size)})</small>
            </div>
        `).join('');
    }

    async handleFileUpload(files) {
        const uploadPromises = Array.from(files).map(file => this.uploadFile(file));
        const results = await Promise.allSettled(uploadPromises);
        
        results.forEach((result, index) => {
            if (result.status === 'fulfilled') {
                this.showNotification(`File ${files[index].name} uploaded successfully`, 'success');
            } else {
                this.showNotification(`File ${files[index].name} upload failed`, 'error');
            }
        });
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('category', 'chat_attachment');

        const response = await fetch('/api/knowledge/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();
        if (data.status !== 'success') {
            throw new Error(data.message || 'Upload failed');
        }

        return data;
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
            this.showWelcomeMessage();
            window.notificationSystem.show('Conversation cleared', 'info', 2000);
        }
    }

    exportChat() {
        if (this.messages.length === 0) {
            window.notificationSystem.show('No conversation content to export', 'warning', 2000);
            return;
        }

        try {
            const chatData = {
                conversation_id: this.currentConversation,
                messages: this.messages,
                export_time: new Date().toISOString()
            };

            const blob = new Blob([JSON.stringify(chatData, null, 2)], {
                type: 'application/json'
            });

            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `chat_export_${new Date().toISOString().split('T')[0]}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            window.notificationSystem.show('Conversation exported successfully', 'success', 2000);
        } catch (error) {
            console.error('Export failed:', error);
            window.notificationSystem.show('Export failed, please try again', 'error', 3000);
        }
    }

    showWelcomeMessage() {
        const container = document.getElementById('messagesContainer');
        if (!container || container.children.length > 0) return;

        const welcomeDiv = document.createElement('div');
        welcomeDiv.className = 'message ai-message';
        welcomeDiv.innerHTML = `
            <div class="message-avatar ai-avatar-msg">
                <i class="fas fa-robot"></i>
            </div>
            <div class="message-content">
                <p>👋 Hello! I am the A Manager Model AI Assistant.</p>
                <p>I can help you with:</p>
                <ul class="mb-0">
                    <li>📚 Answering questions based on knowledge base</li>
                    <li>🔍 Searching and organizing information</li>
                    <li>📄 Analyzing uploaded documents</li>
                    <li>💡 Providing intelligent suggestions</li>
                </ul>
                <p class="mt-2 mb-0">Feel free to ask me anything!</p>
            </div>
        `;
        container.appendChild(welcomeDiv);
    }

    showTypingIndicator(show) {
        const existing = document.querySelector('.typing-indicator');
        if (show && !existing) {
            const indicator = document.createElement('div');
            indicator.className = 'message ai-message typing-indicator';
            indicator.innerHTML = `
                <div class="message-avatar ai-avatar-msg">
                    <i class="fas fa-robot"></i>
                </div>
                <div class="message-content">
                    <div class="typing-dots">
                        <span></span><span></span><span></span>
                    </div>
                </div>
            `;
            
            const container = document.getElementById('messagesContainer');
            if (container) {
                container.appendChild(indicator);
                this.scrollToBottom();
            }
        } else if (!show && existing) {
            existing.remove();
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.classList.add('show');
        }, 100);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
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

    handleKnowledgeChange(event) {
        console.log('Knowledge base switched:', event.target.value);
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
        const sendBtn = document.querySelector('.send-btn');
        if (sendBtn) {
            sendBtn.disabled = this.isLoading;
            sendBtn.style.opacity = this.isLoading ? '0.5' : '1';
        }
    }

    updateVoiceButton() {
        const voiceBtn = document.querySelector('[onclick="toggleVoice()"]');
        if (voiceBtn) {
            voiceBtn.classList.toggle('active', this.isVoiceMode);
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    formatTime(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            month: 'short',
            day: 'numeric'
        });
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    getCurrentAttachments() {
        // Return current attached file list
        return [];
    }
}

// Global functions
function sendMessage() {
    if (window.chatManager) {
        window.chatManager.sendMessage();
    }
}

function clearChat() {
    if (window.chatManager) {
        window.chatManager.clearChat();
    }
}

function exportChat() {
    if (window.chatManager) {
        window.chatManager.exportChat();
    }
}

function toggleVoice() {
    if (window.chatManager) {
        window.chatManager.toggleVoice();
    }
}

function quickAction(type) {
    const input = document.getElementById('messageInput');
    if (!input) return;

        const prompts = {
            summarize: 'Please help me summarize the following content:',
            translate: 'Please translate the following text:',
            code: 'Please help me generate code for the following functionality:',
            analyze: 'Please analyze the following content:',
            brainstorm: 'Please brainstorm ideas for the following topic:',
            improve: 'Please improve the following text:'
        };

    input.value = prompts[type] || '';
    input.focus();
}

// Mobile sidebar functions
function toggleSidebar() {
    const sidebar = document.querySelector('.chat-sidebar');
    const overlay = document.querySelector('.sidebar-overlay') || createSidebarOverlay();
    
    if (sidebar.classList.contains('show')) {
        sidebar.classList.remove('show');
        overlay.style.display = 'none';
    } else {
        sidebar.classList.add('show');
        overlay.style.display = 'block';
    }
}

function createSidebarOverlay() {
    const overlay = document.createElement('div');
    overlay.className = 'sidebar-overlay';
    overlay.onclick = toggleSidebar;
    document.body.appendChild(overlay);
    return overlay;
}

// Close mobile sidebar
function closeSidebar() {
    const sidebar = document.querySelector('.chat-sidebar');
    const overlay = document.querySelector('.sidebar-overlay');
    sidebar.classList.remove('show');
    if (overlay) overlay.style.display = 'none';
}

// Notification system
class NotificationSystem {
    constructor() {
        this.container = this.createContainer();
    }

    createContainer() {
        const container = document.createElement('div');
        container.className = 'notification-container';
        container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 10000;
            max-width: 400px;
        `;
        document.body.appendChild(container);
        return container;
    }

    show(message, type = 'info', duration = 3000) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.style.cssText = `
            background: ${type === 'success' ? '#28a745' : type === 'error' ? '#dc3545' : type === 'warning' ? '#ffc107' : '#17a2b8'};
            color: white;
            padding: 12px 16px;
            margin-bottom: 10px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            animation: slideIn 0.3s ease;
            font-size: 14px;
            max-width: 100%;
        `;
        
        notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 8px;">
                <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : type === 'warning' ? 'exclamation-triangle' : 'info-circle'}"></i>
                <span>${message}</span>
            </div>
        `;

        this.container.appendChild(notification);

        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, duration);
    }
}

// Initialize chat manager
document.addEventListener('DOMContentLoaded', () => {
    window.chatManager = new ChatManager();
    window.notificationSystem = new NotificationSystem();
    
    // Listen for window resize
    window.addEventListener('resize', function() {
        if (window.innerWidth > 576) {
            closeSidebar();
        }
    });

    // Add notification animation styles
    const style = document.createElement('style');
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
});

// Style enhancements
const additionalStyles = `
    .typing-dots {
        display: flex;
        gap: 4px;
    }
    .typing-dots span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #667eea;
        animation: typing 1.4s infinite ease-in-out;
    }
    .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
    .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
        40% { transform: scale(1); opacity: 1; }
    }

    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 1000;
        transform: translateX(100%);
        transition: transform 0.3s ease;
    }
    .notification.show { transform: translateX(0); }
    .notification-success { background: #28a745; }
    .notification-error { background: #dc3545; }
    .notification-warning { background: #ffc107; color: #000; }
    .notification-info { background: #17a2b8; }

    .message-attachment {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 12px;
        background: #f8f9fa;
        border-radius: 8px;
        margin-top: 8px;
        font-size: 14px;
    }
    .message-time {
        font-size: 12px;
        color: #6c757d;
        margin-top: 4px;
    }

    pre {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 12px;
        overflow-x: auto;
        margin: 8px 0;
    }
    code {
        background: #f8f9fa;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.9em;
    }

    .dragover {
        border-color: #667eea !important;
        background: rgba(102, 126, 234, 0.05);
    }

    .action-icon.active {
        background: #667eea;
        color: white;
    }
`;

// Add styles to page
const styleSheet = document.createElement('style');
styleSheet.textContent = additionalStyles;
document.head.appendChild(styleSheet);
