// Data Upload JavaScript for AI Management System
class DataUploader {
    constructor() {
        this.initializeElements();
        this.bindEvents();
    }

    initializeElements() {
        this.uploadArea = document.getElementById('uploadArea');
        this.fileInput = document.getElementById('fileInput');
        this.modelSelect = document.getElementById('modelSelect');
        this.uploadBtn = document.getElementById('uploadBtn');
        this.progressSection = document.querySelector('.progress-section');
        this.progressFill = document.getElementById('progressFill');
        this.progressText = document.getElementById('progressText');
        this.resultsSection = document.getElementById('resultsSection');
        this.uploadResults = document.getElementById('uploadResults');

        this.selectedFiles = [];
    }

    bindEvents() {
        // File selection
        this.uploadArea.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('change', (e) => this.handleFiles(e.target.files));

        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('drag-over');
        });

        this.uploadArea.addEventListener('dragleave', () => {
            this.uploadArea.classList.remove('drag-over');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('drag-over');
            this.handleFiles(e.dataTransfer.files);
        });

        // Upload button
        this.uploadBtn.addEventListener('click', () => this.startUpload());
    }

    handleFiles(files) {
        this.selectedFiles = Array.from(files);
        this.updateUI();
    }

    updateUI() {
        const fileCount = this.selectedFiles.length;
        this.uploadBtn.disabled = fileCount === 0;
        
        if (fileCount > 0) {
            this.uploadArea.querySelector('p').textContent = `${fileCount} file(s) selected`;
        }
    }

    async startUpload() {
        if (this.selectedFiles.length === 0) return;

        this.showProgress();
        
        const model = this.modelSelect.value;
        const jointTraining = document.getElementById('jointTraining').checked;
        const externalApi = document.getElementById('externalApi').checked;

        const formData = new FormData();
        this.selectedFiles.forEach(file => {
            formData.append('files', file);
        });
        formData.append('model', model);
        formData.append('jointTraining', jointTraining);
        formData.append('externalApi', externalApi);

        try {
            const response = await fetch('/api/upload/training-data', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            this.showResults(result);
        } catch (error) {
            this.showError(error.message);
        }
    }

    showProgress() {
        this.progressSection.style.display = 'block';
        this.resultsSection.style.display = 'none';
        
        let progress = 0;
        const interval = setInterval(() => {
            progress += Math.random() * 10;
            if (progress >= 100) {
                progress = 100;
                clearInterval(interval);
            }
            this.progressFill.style.width = `${progress}%`;
            this.progressText.textContent = `Processing... ${Math.round(progress)}%`;
        }, 200);
    }

    showResults(result) {
        this.progressSection.style.display = 'none';
        this.resultsSection.style.display = 'block';

        this.uploadResults.innerHTML = `
            <div class="result-item">
                <h4>Upload Complete</h4>
                <p>Model: ${result.model || 'Unknown'}</p>
                <p>Files processed: ${result.files || 0}</p>
                <p>Training status: ${result.status || 'Started'}</p>
            </div>
        `;
    }

    showError(message) {
        this.progressSection.style.display = 'none';
        this.resultsSection.style.display = 'block';
        this.uploadResults.innerHTML = `<div class="error">Error: ${message}</div>`;
    }
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DataUploader();
});