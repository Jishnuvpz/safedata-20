// Anonify Frontend JavaScript

class SafeDataApp {
    constructor() {
        this.currentFileId = null;
        this.currentResultId = null;
        this.charts = {};
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadDashboardData();
        this.setupTooltips();
    }

    setupEventListeners() {
        // File upload
        const fileInput = document.getElementById('fileInput');
        const uploadArea = document.getElementById('uploadArea');

        if (fileInput && uploadArea) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
            
            // Drag and drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', () => {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                if (files.length > 0) {
                    this.uploadFile(files[0]);
                }
            });

            uploadArea.addEventListener('click', () => {
                fileInput.click();
            });
        }

        // Anonymization form
        const anonymizeForm = document.getElementById('anonymizeForm');
        if (anonymizeForm) {
            anonymizeForm.addEventListener('submit', (e) => this.handleAnonymization(e));
        }

        // Method selection
        const methodCards = document.querySelectorAll('.method-card');
        methodCards.forEach(card => {
            card.addEventListener('click', () => this.selectMethod(card));
        });

        // Parameter sliders
        const epsilonSlider = document.getElementById('epsilonSlider');
        const deltaSlider = document.getElementById('deltaSlider');

        if (epsilonSlider) {
            epsilonSlider.addEventListener('input', (e) => {
                document.getElementById('epsilonValue').textContent = e.target.value;
                this.updatePrivacyEstimate();
            });
        }

        if (deltaSlider) {
            deltaSlider.addEventListener('input', (e) => {
                document.getElementById('deltaValue').textContent = parseFloat(e.target.value).toExponential(2);
                this.updatePrivacyEstimate();
            });
        }

        // Results actions
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-action="download"]')) {
                this.downloadResult(e.target.dataset.resultId, e.target.dataset.format);
            }
            if (e.target.matches('[data-action="simulate-attacks"]')) {
                this.simulateAttacks(e.target.dataset.resultId);
            }
            if (e.target.matches('[data-action="optimize"]')) {
                this.optimizeParameters(e.target.dataset.fileId);
            }
        });
    }

    setupTooltips() {
        // Initialize Bootstrap tooltips
        const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]');
        const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => 
            new bootstrap.Tooltip(tooltipTriggerEl)
        );
    }

    async loadDashboardData() {
        try {
            // Load user files and results
            const response = await fetch('/api/list');
            const data = await response.json();

            this.updateDashboardStats(data);
            this.updateFilesList(data.files);
            this.updateResultsList(data.anonymization_results);

        } catch (error) {
            console.error('Failed to load dashboard data:', error);
            this.showError('Failed to load dashboard data');
        }
    }

    updateDashboardStats(data) {
        // Update statistics cards
        const statsElements = {
            'totalFiles': data.total_files || 0,
            'totalResults': data.total_results || 0,
            'successRate': data.anonymization_results ? 
                Math.round((data.anonymization_results.filter(r => r.privacy_score > 0.7).length / data.total_results) * 100) : 0,
            'avgUtility': data.anonymization_results ?
                Math.round(data.anonymization_results.reduce((acc, r) => acc + (r.utility_score || 0), 0) / data.total_results * 100) : 0
        };

        Object.entries(statsElements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value;
            }
        });
    }

    updateFilesList(files) {
        const container = document.getElementById('filesList');
        if (!container) return;

        if (files.length === 0) {
            container.innerHTML = `
                <div class="text-center py-4">
                    <i data-feather="folder" class="text-muted mb-2" style="width: 48px; height: 48px;"></i>
                    <p class="text-muted">No files uploaded yet</p>
                </div>
            `;
            feather.replace();
            return;
        }

        container.innerHTML = files.map(file => `
            <div class="card mb-3">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h6 class="card-title mb-1">${file.filename}</h6>
                            <small class="text-muted">
                                ${file.rows ? file.rows.toLocaleString() : 'Unknown'} rows × ${Array.isArray(file.columns) ? file.columns.length : (file.column_count || file.columns || 'Unknown')} columns
                                • ${this.formatFileSize(file.size || 0)}
                                • ${file.upload_time ? new Date(file.upload_time).toLocaleDateString() : 'Unknown date'}
                            </small>
                        </div>
                        <div class="btn-group btn-group-sm">
                            <button class="btn btn-outline-primary" onclick="app.selectFile('${file.file_id}')">
                                <i data-feather="upload-cloud" class="me-1" style="width: 14px; height: 14px;"></i>
                                Anonymize
                            </button>
                            <button class="btn btn-outline-secondary" onclick="app.viewFileInfo('${file.file_id}')">
                                <i data-feather="info" style="width: 14px; height: 14px;"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');

        feather.replace();
    }

    updateResultsList(results) {
        const container = document.getElementById('resultsList');
        if (!container) return;

        if (results.length === 0) {
            container.innerHTML = `
                <div class="text-center py-4">
                    <i data-feather="shield" class="text-muted mb-2" style="width: 48px; height: 48px;"></i>
                    <p class="text-muted">No anonymization results yet</p>
                </div>
            `;
            feather.replace();
            return;
        }

        container.innerHTML = results.map(result => `
            <div class="card mb-3">
                <div class="card-body">
                    <div class="d-flex justify-content-between align-items-start">
                        <div>
                            <h6 class="card-title mb-1">
                                ${result.method.toUpperCase()} Anonymization
                                <span class="badge bg-${this.getPrivacyBadgeColor(result.privacy_score)} ms-2">
                                    ${Math.round(result.privacy_score * 100)}% Privacy
                                </span>
                            </h6>
                            <small class="text-muted">
                                ${new Date(result.created_at).toLocaleDateString()}
                                • Utility: ${Math.round(result.utility_score * 100)}%
                            </small>
                        </div>
                        <div class="btn-group btn-group-sm">
                            <div class="dropdown">
                                <button class="btn btn-outline-success dropdown-toggle" type="button" data-bs-toggle="dropdown">
                                    <i data-feather="download" style="width: 14px; height: 14px;"></i>
                                </button>
                                <ul class="dropdown-menu">
                                    <li><a class="dropdown-item" href="#" onclick="app.downloadResult('${result.id}', 'csv')"><i data-feather="file-text" class="me-2" style="width: 16px;"></i>CSV</a></li>
                                    <li><a class="dropdown-item" href="#" onclick="app.downloadResult('${result.id}', 'xlsx')"><i data-feather="file-text" class="me-2" style="width: 16px;"></i>Excel</a></li>
                                    <li><a class="dropdown-item" href="#" onclick="app.downloadResult('${result.id}', 'pdf')"><i data-feather="file" class="me-2" style="width: 16px;"></i>PDF</a></li>
                                    <li><a class="dropdown-item" href="#" onclick="app.downloadResult('${result.id}', 'docx')"><i data-feather="file-text" class="me-2" style="width: 16px;"></i>Word</a></li>
                                </ul>
                            </div>
                            <button class="btn btn-outline-info" onclick="app.viewResults('${result.id}')">
                                <i data-feather="bar-chart-2" style="width: 14px; height: 14px;"></i>
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `).join('');

        feather.replace();
    }

    getPrivacyBadgeColor(score) {
        if (score >= 0.8) return 'success';
        if (score >= 0.6) return 'warning';
        return 'danger';
    }

    formatFileSize(bytes) {
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        if (bytes === 0) return '0 Bytes';
        const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
        return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
    }

    handleFileSelect(event) {
        const file = event.target.files[0];
        if (file) {
            this.uploadFile(file);
        }
    }

    async uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const progressBar = document.getElementById('uploadProgress');
        if (progressBar) {
            progressBar.style.display = 'block';
            progressBar.querySelector('.progress-bar').style.width = '0%';
        }

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            if (progressBar) {
                progressBar.querySelector('.progress-bar').style.width = '100%';
            }

            const result = await response.json();

            if (response.ok) {
                this.currentFileId = result.file_id;
                this.showSuccess(`File "${result.filename}" uploaded successfully`);
                this.showFileInfo(result);
                this.loadDashboardData();
            } else {
                throw new Error(result.detail || 'Upload failed');
            }
        } catch (error) {
            console.error('Upload error:', error);
            this.showError(`Upload failed: ${error.message}`);
        } finally {
            if (progressBar) {
                setTimeout(() => {
                    progressBar.style.display = 'none';
                }, 1000);
            }
        }
    }

    showFileInfo(fileInfo) {
        const modal = new bootstrap.Modal(document.getElementById('fileInfoModal'));
        
        // Update modal content
        document.getElementById('fileInfoName').textContent = fileInfo.filename || 'Unknown';
        document.getElementById('fileInfoSize').textContent = this.formatFileSize(fileInfo.size || 0);
        document.getElementById('fileInfoRows').textContent = fileInfo.rows ? fileInfo.rows.toLocaleString() : 'Unknown';
        document.getElementById('fileInfoColumns').textContent = Array.isArray(fileInfo.columns) ? fileInfo.columns.length : (fileInfo.column_count || fileInfo.columns || 'Unknown');

        // Update columns list - show actual column names if available
        if (Array.isArray(fileInfo.columns) && fileInfo.columns.length > 0) {
            try {
                const columnsContainer = document.createElement('div');
                columnsContainer.className = 'mt-2';
                columnsContainer.innerHTML = '<strong>Columns:</strong><br>' + fileInfo.columns.map(col => 
                    `<span class="badge bg-secondary me-1 mb-1">${String(col || 'Unknown')}</span>`
                ).join('');
                
                // Add after the table
                const modalBody = document.querySelector('#fileInfoModal .modal-body');
                const existingColumns = modalBody.querySelector('.columns-container');
                if (existingColumns) {
                    existingColumns.remove();
                }
                columnsContainer.className += ' columns-container';
                modalBody.appendChild(columnsContainer);
            } catch (error) {
                console.warn('Error displaying column names:', error);
            }
        }

        modal.show();
    }

    selectMethod(cardElement) {
        // Remove previous selection
        document.querySelectorAll('.method-card').forEach(card => {
            card.classList.remove('selected');
        });

        // Add selection to clicked card
        cardElement.classList.add('selected');

        // Update hidden input
        const methodInput = document.getElementById('selectedMethod');
        if (methodInput) {
            methodInput.value = cardElement.dataset.method;
        }

        this.updatePrivacyEstimate();
    }

    updatePrivacyEstimate() {
        const method = document.getElementById('selectedMethod')?.value;
        const epsilon = parseFloat(document.getElementById('epsilonSlider')?.value || 1.0);
        const delta = parseFloat(document.getElementById('deltaSlider')?.value || 1e-5);

        // Simple privacy/utility estimation (in real app, this would call an API)
        let privacyScore = Math.max(0, Math.min(1, 1 - epsilon / 10));
        let utilityScore = Math.max(0, Math.min(1, epsilon / 5));

        // Adjust based on method
        const methodMultipliers = {
            'sdg': { privacy: 0.7, utility: 0.9 },
            'dp': { privacy: 0.9, utility: 0.6 },
            'sdc': { privacy: 0.6, utility: 0.8 },
            'full': { privacy: 0.95, utility: 0.5 }
        };

        if (method && methodMultipliers[method]) {
            privacyScore *= methodMultipliers[method].privacy;
            utilityScore *= methodMultipliers[method].utility;
        }

        // Update display
        const privacyEstimate = document.getElementById('privacyEstimate');
        const utilityEstimate = document.getElementById('utilityEstimate');

        if (privacyEstimate) {
            privacyEstimate.textContent = Math.round(privacyScore * 100) + '%';
            privacyEstimate.className = `metric-value text-${this.getScoreColor(privacyScore)}`;
        }

        if (utilityEstimate) {
            utilityEstimate.textContent = Math.round(utilityScore * 100) + '%';
            utilityEstimate.className = `metric-value text-${this.getScoreColor(utilityScore)}`;
        }
    }

    getScoreColor(score) {
        if (score >= 0.8) return 'success';
        if (score >= 0.6) return 'warning';
        return 'danger';
    }

    async handleAnonymization(event) {
        event.preventDefault();

        if (!this.currentFileId) {
            this.showError('Please upload a file first');
            return;
        }

        const formData = new FormData(event.target);
        const request = {
            file_id: this.currentFileId,
            method: formData.get('method') || 'sdg_dp',
            epsilon: parseFloat(formData.get('epsilon')) || 1.0,
            delta: parseFloat(formData.get('delta')) || 1e-5,
            utility_metrics: ['statistical_similarity', 'ml_utility']
        };

        const submitButton = event.target.querySelector('button[type="submit"]');
        const originalText = submitButton.textContent;
        submitButton.disabled = true;
        submitButton.innerHTML = '<span class="loading-spinner me-2"></span>Processing...';

        try {
            const response = await fetch('/api/anonymize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(request)
            });

            const result = await response.json();

            if (response.ok) {
                this.currentResultId = result.anonymized_data_id;
                this.showSuccess('Data anonymized successfully!');
                this.showAnonymizationResults(result);
                this.loadDashboardData();
            } else {
                throw new Error(result.detail || 'Anonymization failed');
            }
        } catch (error) {
            console.error('Anonymization error:', error);
            this.showError(`Anonymization failed: ${error.message}`);
        } finally {
            submitButton.disabled = false;
            submitButton.textContent = originalText;
        }
    }

    showAnonymizationResults(results) {
        // Update results display
        const resultsContainer = document.getElementById('anonymizationResults');
        if (resultsContainer) {
            resultsContainer.style.display = 'block';
            
            // Update metrics
            this.updateMetricDisplay('finalPrivacyScore', results.privacy_metrics?.re_identification_risk || 0, true);
            this.updateMetricDisplay('finalUtilityScore', results.utility_metrics?.overall_utility_score || 0);
            
            // Show execution time
            const execTime = document.getElementById('executionTime');
            if (execTime) {
                execTime.textContent = `${Math.round(results.execution_time * 100) / 100}s`;
            }
        }

        // Show warnings if any
        if (results.warnings && results.warnings.length > 0) {
            this.showWarning(`Warnings: ${results.warnings.join(', ')}`);
        }
    }

    updateMetricDisplay(elementId, value, isInverse = false) {
        const element = document.getElementById(elementId);
        if (element) {
            const displayValue = isInverse ? (1 - value) : value;
            element.textContent = Math.round(displayValue * 100) + '%';
            element.className = `metric-value text-${this.getScoreColor(displayValue)}`;
        }
    }

    async downloadResult(resultId, format = 'csv') {
        try {
            const response = await fetch(`/api/result/${resultId}/download?format=${format}`);
            
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `anonymized_data_${resultId}.${format}`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                
                this.showSuccess(`Download started (${format.toUpperCase()} format)`);
            } else {
                throw new Error('Download failed');
            }
        } catch (error) {
            console.error('Download error:', error);
            this.showError(`Download failed: ${error.message}`);
        }
    }

    async simulateAttacks(resultId) {
        const modal = new bootstrap.Modal(document.getElementById('attackSimulationModal'));
        modal.show();

        try {
            const response = await fetch('/api/simulate-attacks', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    original_data_id: this.currentFileId,
                    anonymized_data_id: resultId,
                    attack_types: ['linkage', 'membership', 'attribute']
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.displayAttackResults(result);
            } else {
                throw new Error(result.detail || 'Attack simulation failed');
            }
        } catch (error) {
            console.error('Attack simulation error:', error);
            this.showError(`Attack simulation failed: ${error.message}`);
        }
    }

    displayAttackResults(results) {
        const container = document.getElementById('attackResults');
        if (!container) return;

        const riskLevel = results.overall_risk_score;
        const riskColor = riskLevel > 0.7 ? 'danger' : riskLevel > 0.4 ? 'warning' : 'success';

        container.innerHTML = `
            <div class="alert alert-${riskColor}">
                <h6 class="alert-heading">Overall Risk Score: ${Math.round(riskLevel * 100)}%</h6>
                <p class="mb-0">Risk Level: ${riskLevel > 0.7 ? 'High' : riskLevel > 0.4 ? 'Medium' : 'Low'}</p>
            </div>
            
            <div class="row">
                ${Object.entries(results.attack_results).map(([attackType, attackResult]) => {
                    if (typeof attackResult === 'object' && attackResult.success_rate !== undefined) {
                        return `
                            <div class="col-md-4">
                                <div class="card">
                                    <div class="card-body text-center">
                                        <h6 class="card-title">${attackType.replace('_', ' ').toUpperCase()}</h6>
                                        <div class="metric-value text-${this.getScoreColor(1 - attackResult.success_rate)}">
                                            ${Math.round((1 - attackResult.success_rate) * 100)}%
                                        </div>
                                        <small class="text-muted">Protection Level</small>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    return '';
                }).join('')}
            </div>

            <div class="mt-3">
                <h6>Recommendations:</h6>
                <ul class="list-unstyled">
                    ${results.recommendations.map(rec => `<li><i data-feather="arrow-right" class="text-primary me-2" style="width: 16px; height: 16px;"></i>${rec}</li>`).join('')}
                </ul>
            </div>
        `;

        feather.replace();
    }

    async optimizeParameters(fileId) {
        const modal = new bootstrap.Modal(document.getElementById('optimizationModal'));
        modal.show();

        try {
            const response = await fetch('/api/optimize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    file_id: fileId,
                    target_privacy_level: 'medium',
                    utility_weight: 0.5,
                    privacy_weight: 0.5,
                    max_iterations: 20
                })
            });

            const result = await response.json();

            if (response.ok) {
                this.displayOptimizationResults(result);
            } else {
                throw new Error(result.detail || 'Optimization failed');
            }
        } catch (error) {
            console.error('Optimization error:', error);
            this.showError(`Optimization failed: ${error.message}`);
        }
    }

    displayOptimizationResults(results) {
        const container = document.getElementById('optimizationResults');
        if (!container) return;

        container.innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body text-center">
                            <h6 class="card-title">Optimal Privacy Score</h6>
                            <div class="metric-value text-success">${Math.round(results.expected_privacy_score * 100)}%</div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-body text-center">
                            <h6 class="card-title">Expected Utility Score</h6>
                            <div class="metric-value text-info">${Math.round(results.expected_utility_score * 100)}%</div>
                        </div>
                    </div>
                </div>
            </div>

            <div class="mt-3">
                <h6>Optimal Parameters:</h6>
                <ul class="list-unstyled">
                    ${Object.entries(results.optimal_parameters).map(([param, value]) => 
                        `<li><strong>${param}:</strong> ${typeof value === 'number' ? Math.round(value * 1000) / 1000 : value}</li>`
                    ).join('')}
                </ul>
            </div>
        `;
    }

    selectFile(fileId) {
        this.currentFileId = fileId;
        this.showSuccess('File selected for anonymization');
        
        // Scroll to anonymization section
        const anonymizeSection = document.getElementById('anonymizeSection');
        if (anonymizeSection) {
            anonymizeSection.scrollIntoView({ behavior: 'smooth' });
        }
    }

    async viewFileInfo(fileId) {
        try {
            const response = await fetch(`/api/file/${fileId}/info`);
            const fileInfo = await response.json();

            if (response.ok) {
                this.showFileInfo(fileInfo);
            } else {
                throw new Error(fileInfo.detail || 'Failed to load file info');
            }
        } catch (error) {
            console.error('File info error:', error);
            this.showError(`Failed to load file info: ${error.message}`);
        }
    }

    async viewResults(resultId) {
        window.location.href = `/results.html?id=${resultId}`;
    }

    showSuccess(message) {
        this.showToast(message, 'success');
    }

    showError(message) {
        this.showToast(message, 'danger');
    }

    showWarning(message) {
        this.showToast(message, 'warning');
    }

    showToast(message, type = 'info') {
        const toastContainer = document.getElementById('toastContainer') || this.createToastContainer();
        
        const toast = document.createElement('div');
        toast.className = `toast align-items-center text-bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">${message}</div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;

        toastContainer.appendChild(toast);

        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();

        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }

    createToastContainer() {
        const container = document.createElement('div');
        container.id = 'toastContainer';
        container.className = 'toast-container position-fixed top-0 end-0 p-3';
        container.style.zIndex = '1050';
        document.body.appendChild(container);
        return container;
    }
}

// Initialize the application
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new SafeDataApp();
    
    // Initialize Feather icons
    if (typeof feather !== 'undefined') {
        feather.replace();
    }
});

// Global utility functions
window.formatNumber = (num) => {
    return new Intl.NumberFormat().format(num);
};

window.formatPercentage = (num) => {
    return Math.round(num * 100) + '%';
};

window.formatBytes = (bytes) => {
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    if (bytes === 0) return '0 Bytes';
    const i = parseInt(Math.floor(Math.log(bytes) / Math.log(1024)));
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
};
