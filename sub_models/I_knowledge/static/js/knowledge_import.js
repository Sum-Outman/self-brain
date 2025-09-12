/**
 * 知识库文件导入系统 JavaScript
 */

class KnowledgeImporter {
    constructor() {
        this.files = [];
        this.uploading = false;
        this.stats = {
            totalFiles: 0,
            totalSize: 0,
            successCount: 0,
            errorCount: 0
        };
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.updateStats();
    }

    setupEventListeners() {
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');

        // 拖拽上传
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', () => {
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            this.handleFiles(e.dataTransfer.files);
        });

        // 文件选择
        fileInput.addEventListener('change', (e) => {
            this.handleFiles(e.target.files);
        });
    }

    handleFiles(fileList) {
        const maxSize = 100 * 1024 * 1024; // 100MB
        const allowedTypes = [
            'text/plain', 'text/markdown', 'text/csv', 'application/json',
            'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            'application/pdf', 'image/jpeg', 'image/png', 'image/gif', 'image/bmp',
            'audio/mpeg', 'audio/wav', 'audio/flac', 'video/mp4', 'video/avi', 'video/quicktime',
            'text/x-python', 'application/javascript', 'text/x-java-source', 'text/x-c++src'
        ];

        Array.from(fileList).forEach(file => {
            if (file.size > maxSize) {
                this.showError(`文件 ${file.name} 超过100MB限制`);
                return;
            }

            if (!allowedTypes.includes(file.type)) {
                this.showError(`文件 ${file.name} 格式不支持`);
                return;
            }

            // 检查是否已存在
            if (!this.files.some(f => f.name === file.name && f.size === file.size)) {
                this.files.push({
                    file: file,
                    status: 'pending',
                    progress: 0,
                    knowledgeId: null,
                    error: null
                });
            }
        });

        this.renderFileList();
    }

    renderFileList() {
        const fileList = document.getElementById('fileList');
        const fileItems = document.getElementById('fileItems');
        const uploadAllBtn = document.getElementById('uploadAllBtn');

        if (this.files.length === 0) {
            fileList.style.display = 'none';
            uploadAllBtn.disabled = true;
            return;
        }

        fileList.style.display = 'block';
        uploadAllBtn.disabled = this.uploading;

        fileItems.innerHTML = this.files.map((item, index) => `
            <div class="file-item">
                <div class="file-info">
                    <i class="fas ${this.getFileIcon(item.file.type)} fa-2x"></i>
                    <div>
                        <strong>${item.file.name}</strong>
                        <br>
                        <small class="text-muted">${this.formatFileSize(item.file.size)}</small>
                    </div>
                </div>
                <div>
                    <span class="file-status status-${item.status}">
                        ${this.getStatusText(item.status)}
                    </span>
                    ${item.status === 'pending' ? `
                        <button class="btn btn-sm btn-danger ms-2" onclick="importer.removeFile(${index})">
                            <i class="fas fa-times"></i>
                        </button>
                    ` : ''}
                </div>
            </div>
        `).join('');
    }

    getFileIcon(fileType) {
        const iconMap = {
            'text/plain': 'fa-file-alt',
            'text/markdown': 'fa-file-alt',
            'text/csv': 'fa-file-csv',
            'application/json': 'fa-file-code',
            'application/msword': 'fa-file-word',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'fa-file-word',
            'application/pdf': 'fa-file-pdf',
            'image/jpeg': 'fa-file-image',
            'image/png': 'fa-file-image',
            'image/gif': 'fa-file-image',
            'image/bmp': 'fa-file-image',
            'audio/mpeg': 'fa-file-audio',
            'audio/wav': 'fa-file-audio',
            'audio/flac': 'fa-file-audio',
            'video/mp4': 'fa-file-video',
            'video/avi': 'fa-file-video',
            'video/quicktime': 'fa-file-video',
            'text/x-python': 'fa-file-code',
            'application/javascript': 'fa-file-code',
            'text/x-java-source': 'fa-file-code',
            'text/x-c++src': 'fa-file-code'
        };
        return iconMap[fileType] || 'fa-file';
    }

    getStatusText(status) {
        const statusMap = {
            'pending': '待上传',
            'uploading': '上传中...',
            'success': '上传成功',
            'error': '上传失败'
        };
        return statusMap[status] || status;
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async uploadFile(index) {
        const item = this.files[index];
        if (item.status !== 'pending') return;

        item.status = 'uploading';
        this.renderFileList();

        const formData = new FormData();
        formData.append('file', item.file);
        formData.append('category', document.getElementById('knowledgeCategory').value);

        try {
            const response = await fetch('http://localhost:8001/api/upload_knowledge', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.success) {
                item.status = 'success';
                item.knowledgeId = result.knowledge_id;
                this.stats.successCount++;
                this.showSuccess(`文件 ${item.file.name} 上传成功`);
            } else {
                item.status = 'error';
                item.error = result.error;
                this.stats.errorCount++;
                this.showError(`文件 ${item.file.name} 上传失败: ${result.error}`);
            }
        } catch (error) {
            item.status = 'error';
            item.error = error.message;
            this.stats.errorCount++;
            this.showError(`文件 ${item.file.name} 上传失败: ${error.message}`);
        }

        this.stats.totalSize += item.file.size;
        this.updateStats();
        this.renderFileList();
    }

    async uploadAllFiles() {
        if (this.uploading) return;
        
        this.uploading = true;
        const pendingFiles = this.files.filter(item => item.status === 'pending');
        
        if (pendingFiles.length === 0) {
            this.showWarning('没有待上传的文件');
            this.uploading = false;
            return;
        }

        document.getElementById('progressContainer').style.display = 'block';
        
        let completed = 0;
        const total = pendingFiles.length;

        for (let i = 0; i < this.files.length; i++) {
            if (this.files[i].status === 'pending') {
                await this.uploadFile(i);
                completed++;
                
                const progress = Math.round((completed / total) * 100);
                document.getElementById('progressBar').style.width = progress + '%';
                document.getElementById('progressBar').textContent = progress + '%';
            }
        }

        document.getElementById('progressContainer').style.display = 'none';
        this.uploading = false;
        this.renderFileList();

        this.showBatchResult();
    }

    removeFile(index) {
        this.files.splice(index, 1);
        this.renderFileList();
    }

    clearAllFiles() {
        this.files = [];
        this.renderFileList();
    }

    updateStats() {
        this.stats.totalFiles = this.files.length;
        
        document.getElementById('totalFiles').textContent = this.stats.totalFiles;
        document.getElementById('totalSize').textContent = this.formatFileSize(this.stats.totalSize);
        document.getElementById('successCount').textContent = this.stats.successCount;
        document.getElementById('errorCount').textContent = this.stats.errorCount;
    }

    showSuccess(message) {
        this.showToast(message, 'success');
    }

    showError(message) {
        this.showToast(message, 'error');
    }

    showWarning(message) {
        this.showToast(message, 'warning');
    }

    showToast(message, type) {
        const toastClass = type === 'success' ? 'alert-success' : 
                          type === 'error' ? 'alert-danger' : 'alert-warning';
        
        const toast = document.createElement('div');
        toast.className = `alert ${toastClass} alert-dismissible fade show position-fixed`;
        toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        toast.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 5000);
    }

    showBatchResult() {
        const modal = new bootstrap.Modal(document.getElementById('resultModal'));
        const modalBody = document.getElementById('modalBody');
        
        const successFiles = this.files.filter(f => f.status === 'success');
        const errorFiles = this.files.filter(f => f.status === 'error');
        
        modalBody.innerHTML = `
            <div class="text-center">
                <h4 class="text-success">上传完成!</h4>
                <p>成功: ${successFiles.length} 个文件</p>
                <p class="text-danger">失败: ${errorFiles.length} 个文件</p>
                
                ${successFiles.length > 0 ? `
                    <h6 class="mt-3">成功文件:</h6>
                    <ul class="list-group">
                        ${successFiles.map(f => `
                            <li class="list-group-item">
                                ${f.file.name} <small class="text-muted">(ID: ${f.knowledgeId})</small>
                            </li>
                        `).join('')}
                    </ul>
                ` : ''}
                
                ${errorFiles.length > 0 ? `
                    <h6 class="mt-3">失败文件:</h6>
                    <ul class="list-group">
                        ${errorFiles.map(f => `
                            <li class="list-group-item text-danger">
                                ${f.file.name}: ${f.error}
                            </li>
                        `).join('')}
                    </ul>
                ` : ''}
            </div>
        `;
        
        modal.show();
    }
}

// 全局函数
function uploadAllFiles() {
    importer.uploadAllFiles();
}

function clearAllFiles() {
    importer.clearAllFiles();
}

// 初始化
const importer = new KnowledgeImporter();

// 键盘快捷键
document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'u') {
        e.preventDefault();
        uploadAllFiles();
    }
    if (e.ctrlKey && e.key === 'k') {
        e.preventDefault();
        clearAllFiles();
    }
});

// 页面加载完成后更新统计信息
window.addEventListener('load', () => {
    importer.updateStats();
});

// 自动刷新统计信息
setInterval(() => {
    fetch('http://localhost:8001/api/statistics')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                document.getElementById('totalFiles').textContent = data.data.total_knowledge;
            }
        })
        .catch(console.error);
}, 5000);