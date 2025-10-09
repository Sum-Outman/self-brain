/**
 * 知识库管理系统 JavaScript
 */

class KnowledgeManager {
    constructor() {
        this.currentPage = 1;
        this.perPage = 20;
        this.totalPages = 1;
        this.knowledgeList = [];
        this.currentDeleteId = null;
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadStatistics();
        this.loadKnowledgeList();
    }

    setupEventListeners() {
        document.getElementById('categoryFilter').addEventListener('change', () => this.loadKnowledgeList());
        document.getElementById('typeFilter').addEventListener('change', () => this.loadKnowledgeList());
        document.getElementById('searchInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.loadKnowledgeList();
            }
        });

        // 删除确认
        document.getElementById('confirmDelete').addEventListener('click', () => {
            this.confirmDelete();
        });
    }

    async loadStatistics() {
        try {
            const response = await fetch('/api/statistics');
            const data = await response.json();
            
            if (data.success) {
                this.updateStatistics(data.data);
            }
        } catch (error) {
            console.error('加载统计信息失败:', error);
        }
    }

    updateStatistics(stats) {
        document.getElementById('totalCount').textContent = stats.total_knowledge;
        document.getElementById('totalSize').textContent = this.formatFileSize(stats.total_size);
        
        const fileTypes = stats.file_types || {};
        document.getElementById('textCount').textContent = fileTypes.text || 0;
        document.getElementById('documentCount').textContent = fileTypes.document || 0;
        document.getElementById('imageCount').textContent = fileTypes.image || 0;
        document.getElementById('audioCount').textContent = fileTypes.audio || 0;
    }

    async loadKnowledgeList(page = 1) {
        this.showLoading(true);
        
        try {
            const params = new URLSearchParams({
                page: page,
                per_page: this.perPage
            });

            const category = document.getElementById('categoryFilter').value;
            const type = document.getElementById('typeFilter').value;
            const search = document.getElementById('searchInput').value;

            if (category) params.append('category', category);
            if (type) params.append('file_type', type);
            if (search) params.append('search', search);

            const response = await fetch(`/api/knowledge_list?${params}`);
            const data = await response.json();

            if (data.success) {
                this.knowledgeList = data.data.items;
                this.totalPages = data.data.pages;
                this.currentPage = data.data.page;
                
                this.renderKnowledgeList();
                this.renderPagination();
            }
        } catch (error) {
            console.error('加载知识列表失败:', error);
            this.showError('加载知识列表失败');
        } finally {
            this.showLoading(false);
        }
    }

    renderKnowledgeList() {
        const container = document.getElementById('knowledgeList');
        
        if (this.knowledgeList.length === 0) {
            document.getElementById('emptyState').style.display = 'block';
            container.innerHTML = '';
            return;
        }

        document.getElementById('emptyState').style.display = 'none';

        container.innerHTML = this.knowledgeList.map(item => `
            <div class="knowledge-item">
                <div class="row align-items-center">
                    <div class="col-md-1">
                        <i class="fas ${this.getFileIcon(item.file_type)} fa-2x ${this.getFileColor(item.file_type)}"></i>
                    </div>
                    <div class="col-md-4">
                        <h6 class="mb-1">${item.title}</h6>
                        <small class="text-muted">
                            ${item.category} • ${this.formatFileSize(item.file_size)}
                        </small>
                    </div>
                    <div class="col-md-3">
                        <small class="text-muted">
                            上传时间: ${item.upload_time}<br>
                            内容长度: ${item.content_length} 字符
                        </small>
                    </div>
                    <div class="col-md-2">
                        <span class="badge bg-secondary">${item.file_type}</span>
                    </div>
                    <div class="col-md-2">
                        <div class="action-buttons">
                            <button class="btn btn-sm btn-info" onclick="manager.viewKnowledge('${item.id}')" title="查看">
                                <i class="fas fa-eye"></i>
                            </button>
                            <button class="btn btn-sm btn-danger" onclick="manager.deleteKnowledge('${item.id}', '${item.title}')" title="删除">
                                <i class="fas fa-trash"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    getFileIcon(fileType) {
        const iconMap = {
            'text': 'fa-file-alt',
            'document': 'fa-file-pdf',
            'image': 'fa-file-image',
            'audio': 'fa-file-audio',
            'video': 'fa-file-video',
            'code': 'fa-file-code',
            'other': 'fa-file'
        };
        return iconMap[fileType] || 'fa-file';
    }

    getFileColor(fileType) {
        const colorMap = {
            'text': 'text-primary',
            'document': 'text-danger',
            'image': 'text-success',
            'audio': 'text-warning',
            'video': 'text-purple',
            'code': 'text-info',
            'other': 'text-secondary'
        };
        return colorMap[fileType] || 'text-secondary';
    }

    renderPagination() {
        const container = document.getElementById('pagination');
        
        if (this.totalPages <= 1) {
            container.innerHTML = '';
            return;
        }

        let html = '';
        
        // 上一页
        if (this.currentPage > 1) {
            html += `<li class="page-item"><a class="page-link" href="#" onclick="manager.goToPage(${this.currentPage - 1})">上一页</a></li>`;
        }

        // 页码
        const startPage = Math.max(1, this.currentPage - 2);
        const endPage = Math.min(this.totalPages, this.currentPage + 2);

        for (let i = startPage; i <= endPage; i++) {
            const active = i === this.currentPage ? 'active' : '';
            html += `<li class="page-item ${active}"><a class="page-link" href="#" onclick="manager.goToPage(${i})">${i}</a></li>`;
        }

        // 下一页
        if (this.currentPage < this.totalPages) {
            html += `<li class="page-item"><a class="page-link" href="#" onclick="manager.goToPage(${this.currentPage + 1})">下一页</a></li>`;
        }

        container.innerHTML = html;
    }

    goToPage(page) {
        this.loadKnowledgeList(page);
    }

    deleteKnowledge(id, filename) {
        this.currentDeleteId = id;
        document.getElementById('deleteFilename').textContent = filename;
        
        const modal = new bootstrap.Modal(document.getElementById('deleteModal'));
        modal.show();
    }

    async confirmDelete() {
        if (!this.currentDeleteId) return;

        try {
            const response = await fetch(`/api/delete_knowledge/${this.currentDeleteId}`, {
                method: 'DELETE'
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showSuccess('删除成功');
                this.loadKnowledgeList();
                this.loadStatistics();
            } else {
                this.showError(data.error || '删除失败');
            }
        } catch (error) {
            console.error('删除失败:', error);
            this.showError('删除失败');
        }

        const modal = bootstrap.Modal.getInstance(document.getElementById('deleteModal'));
        modal.hide();
        this.currentDeleteId = null;
    }

    viewKnowledge(id) {
        const item = this.knowledgeList.find(k => k.id === id);
        if (item) {
            const modal = new bootstrap.Modal(document.getElementById('viewModal'));
            document.getElementById('viewContent').textContent = item.content;
            document.getElementById('viewTitle').textContent = item.title;
            modal.show();
        }
    }

    async exportKnowledge() {
        try {
            const response = await fetch('/api/export_knowledge');
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `knowledge_export_${new Date().toISOString().split('T')[0]}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                
                this.showSuccess('导出成功');
            } else {
                this.showError('导出失败');
            }
        } catch (error) {
            console.error('导出失败:', error);
            this.showError('导出失败');
        }
    }

    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    showLoading(show) {
        document.getElementById('loading').style.display = show ? 'block' : 'none';
    }

    showSuccess(message) {
        this.showToast(message, 'success');
    }

    showError(message) {
        this.showToast(message, 'danger');
    }

    showToast(message, type) {
        const toast = document.createElement('div');
        toast.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
        toast.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;
        
        document.body.appendChild(toast);
        setTimeout(() => toast.remove(), 3000);
    }
}

// 全局实例
const manager = new KnowledgeManager();

// 自动刷新
setInterval(() => {
    manager.loadStatistics();
}, 30000); // 每30秒刷新统计信息