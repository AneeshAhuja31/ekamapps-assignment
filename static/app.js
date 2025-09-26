// app.js - Frontend JavaScript Application

class NLPQueryApp {
    constructor() {
        this.baseUrl = window.location.origin;
        this.currentJobId = null;
        this.uploadedFiles = [];
        this.isConnected = false;
        this.schema = null;
        
        this.initializeEventListeners();
        this.loadMetrics();
        
        // Auto-refresh metrics every 30 seconds
        setInterval(() => this.loadMetrics(), 30000);
    }

    initializeEventListeners() {
        // Database connection form
        document.getElementById('dbConnectionForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.connectDatabase();
        });

        document.getElementById('testConnectionBtn').addEventListener('click', () => {
            this.testConnection();
        });

        // File upload
        const uploadZone = document.getElementById('uploadZone');
        const fileInput = document.getElementById('fileInput');

        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.classList.add('dragover');
        });

        uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            this.handleFileUpload(e.dataTransfer.files);
        });

        fileInput.addEventListener('change', (e) => {
            this.handleFileUpload(e.target.files);
        });

        // Query form
        document.getElementById('queryForm').addEventListener('submit', (e) => {
            e.preventDefault();
            this.executeQuery();
        });

        // Query input suggestions
        const queryInput = document.getElementById('queryInput');
        queryInput.addEventListener('input', () => {
            this.showQuerySuggestions();
        });

        queryInput.addEventListener('focus', () => {
            this.showQuerySuggestions();
        });

        // Click outside to hide suggestions
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.query-input')) {
                this.hideQuerySuggestions();
            }
        });

        // Sample query buttons
        document.querySelectorAll('.sample-query').forEach(btn => {
            btn.addEventListener('click', () => {
                const query = btn.getAttribute('data-query');
                document.getElementById('queryInput').value = query;
                this.executeQuery();
            });
        });

        // Tab switching
        document.querySelectorAll('[data-bs-toggle="pill"]').forEach(tab => {
            tab.addEventListener('shown.bs.tab', (e) => {
                if (e.target.id === 'metrics-tab') {
                    this.loadMetrics();
                    this.loadQueryHistory();
                }
            });
        });
    }

    async testConnection() {
        const btn = document.getElementById('testConnectionBtn');
        const connectionString = document.getElementById('connectionString').value;
        const statusDiv = document.getElementById('connectionStatus');

        if (!connectionString) {
            this.showError('Please enter a connection string');
            return;
        }

        btn.disabled = true;
        btn.innerHTML = '<span class="loading-spinner me-2"></span>Testing...';
        statusDiv.innerHTML = '';

        try {
            const response = await fetch(`${this.baseUrl}/api/connect-database`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    connection_string: connectionString,
                    test_only: true
                })
            });

            const result = await response.json();

            if (result.success) {
                statusDiv.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle me-2"></i>${result.message}
                    </div>
                `;
            } else {
                statusDiv.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i>${result.message}
                    </div>
                `;
            }
        } catch (error) {
            statusDiv.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>Connection failed: ${error.message}
                </div>
            `;
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-plug me-2"></i>Test Connection';
        }
    }

    async connectDatabase() {
        const btn = document.getElementById('connectBtn');
        const connectionString = document.getElementById('connectionString').value;
        const statusDiv = document.getElementById('connectionStatus');

        if (!connectionString) {
            this.showError('Please enter a connection string');
            return;
        }

        btn.disabled = true;
        btn.innerHTML = '<span class="loading-spinner me-2"></span>Connecting...';
        statusDiv.innerHTML = '';

        try {
            const response = await fetch(`${this.baseUrl}/api/connect-database`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    connection_string: connectionString,
                    test_only: false
                })
            });

            const result = await response.json();

            if (result.success) {
                this.isConnected = true;
                this.schema = result.schema;
                
                statusDiv.innerHTML = `
                    <div class="alert alert-success">
                        <i class="fas fa-check-circle me-2"></i>${result.message}
                    </div>
                `;

                this.displaySchema(result.schema);
                this.showSuccess('Database connected successfully!');
                
                // Enable query tab
                document.getElementById('query-tab').classList.remove('disabled');
            } else {
                statusDiv.innerHTML = `
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i>Connection failed
                    </div>
                `;
                this.showError('Failed to connect to database');
            }
        } catch (error) {
            statusDiv.innerHTML = `
                <div class="alert alert-danger">
                    <i class="fas fa-exclamation-triangle me-2"></i>Connection error: ${error.message}
                </div>
            `;
            this.showError(`Connection error: ${error.message}`);
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-link me-2"></i>Connect & Analyze Schema';
        }
    }

    displaySchema(schema) {
        const schemaSection = document.getElementById('schemaSection');
        const visualization = document.getElementById('schemaVisualization');

        if (!schema || !schema.tables) {
            return;
        }

        let html = `
            <div class="row mb-3">
                <div class="col-md-4">
                    <div class="card border-primary">
                        <div class="card-body text-center">
                            <h3 class="text-primary">${schema.summary?.table_count || 0}</h3>
                            <small class="text-muted">Tables</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card border-success">
                        <div class="card-body text-center">
                            <h3 class="text-success">${schema.summary?.total_columns || 0}</h3>
                            <small class="text-muted">Columns</small>
                        </div>
                    </div>
                </div>
                <div class="col-md-4">
                    <div class="card border-info">
                        <div class="card-body text-center">
                            <h3 class="text-info">${schema.summary?.relationship_count || 0}</h3>
                            <small class="text-muted">Relations</small>
                        </div>
                    </div>
                </div>
            </div>
        `;

        // Display tables
        Object.entries(schema.tables).forEach(([tableName, tableInfo]) => {
            const purposeBadge = this.getPurposeBadge(tableInfo.purpose);
            
            html += `
                <div class="table-item">
                    <div class="d-flex justify-content-between align-items-start mb-2">
                        <h6 class="mb-0">${tableName}</h6>
                        ${purposeBadge}
                    </div>
                    <small class="text-muted">
                        ${tableInfo.columns?.length || 0} columns, 
                        ${tableInfo.row_count || 0} rows
                    </small>
                    <div class="mt-2">
                        ${(tableInfo.columns || []).slice(0, 5).map(col => 
                            `<span class="badge bg-light text-dark me-1">${col.name}</span>`
                        ).join('')}
                        ${tableInfo.columns?.length > 5 ? '<span class="text-muted">...</span>' : ''}
                    </div>
                </div>
            `;
        });

        visualization.innerHTML = html;
        schemaSection.style.display = 'block';
        schemaSection.scrollIntoView({ behavior: 'smooth' });
    }

    getPurposeBadge(purpose) {
        const badges = {
            'employees': 'bg-primary',
            'departments': 'bg-success',
            'documents': 'bg-info',
            'compensation': 'bg-warning text-dark',
            'reviews': 'bg-secondary',
            'general': 'bg-light text-dark'
        };
        
        const badgeClass = badges[purpose] || 'bg-light text-dark';
        return `<span class="badge ${badgeClass}">${purpose}</span>`;
    }

    async handleFileUpload(files) {
        if (files.length === 0) return;

        const progressDiv = document.getElementById('uploadProgress');
        const progressBar = progressDiv.querySelector('.progress-bar');
        const statusText = document.getElementById('uploadStatus');

        progressDiv.style.display = 'block';
        progressBar.style.width = '0%';
        statusText.textContent = 'Preparing upload...';

        const formData = new FormData();
        Array.from(files).forEach(file => {
            formData.append('files', file);
        });

        try {
            const response = await fetch(`${this.baseUrl}/api/upload-documents`, {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            
            if (result.job_id) {
                this.currentJobId = result.job_id;
                statusText.textContent = result.message;
                this.trackUploadProgress(result.job_id);
                this.showSuccess(`Started processing ${files.length} files`);
            } else {
                throw new Error('No job ID received');
            }
        } catch (error) {
            statusText.textContent = 'Upload failed';
            this.showError(`Upload failed: ${error.message}`);
            progressDiv.style.display = 'none';
        }
    }

    async trackUploadProgress(jobId) {
        const progressBar = document.getElementById('uploadProgress').querySelector('.progress-bar');
        const statusText = document.getElementById('uploadStatus');

        const checkProgress = async () => {
            try {
                const response = await fetch(`${this.baseUrl}/api/ingestion-status/${jobId}`);
                const status = await response.json();

                progressBar.style.width = `${status.progress}%`;
                statusText.textContent = status.message;

                if (status.status === 'completed') {
                    this.showSuccess('Documents processed successfully!');
                    this.loadMetrics(); // Refresh metrics
                } else if (status.status === 'failed') {
                    this.showError(`Processing failed: ${status.message}`);
                } else if (status.status === 'processing') {
                    setTimeout(checkProgress, 2000); // Check again in 2 seconds
                }
            } catch (error) {
                this.showError(`Failed to check progress: ${error.message}`);
            }
        };

        checkProgress();
    }

    showQuerySuggestions() {
        const input = document.getElementById('queryInput');
        const suggestions = document.getElementById('querySuggestions');
        const query = input.value.trim().toLowerCase();

        if (query.length < 2) {
            this.hideQuerySuggestions();
            return;
        }

        const sampleQueries = [
            "How many employees do we have?",
            "Average salary by department",
            "List employees hired this year",
            "Show me Python developers",
            "Top 5 highest paid employees",
            "Who reports to the CEO?",
            "Employees with performance reviews",
            "Department with most employees",
            "Latest hiring trends",
            "Employee skill analysis"
        ];

        const filtered = sampleQueries.filter(q => 
            q.toLowerCase().includes(query) && q.toLowerCase() !== query
        );

        if (filtered.length === 0) {
            this.hideQuerySuggestions();
            return;
        }

        suggestions.innerHTML = filtered.map(suggestion => 
            `<div class="suggestion-item" onclick="app.selectSuggestion('${suggestion}')">${suggestion}</div>`
        ).join('');

        suggestions.style.display = 'block';
    }

    hideQuerySuggestions() {
        document.getElementById('querySuggestions').style.display = 'none';
    }

    selectSuggestion(suggestion) {
        document.getElementById('queryInput').value = suggestion;
        this.hideQuerySuggestions();
        this.executeQuery();
    }

    // ...existing code...
async executeQuery() {
    const queryInput = document.getElementById('queryInput');
    const query = queryInput.value.trim();
    const resultsDiv = document.getElementById('queryResults');

    if (!query) {
        this.showError('Please enter a query');
        return;
    }

    if (!this.isConnected) {
        this.showError('Please connect to a database first');
        return;
    }

    // Show loading state
    resultsDiv.innerHTML = `
        <div class="card">
            <div class="card-body text-center">
                <div class="loading-spinner me-2"></div>
                <span>Processing your query...</span>
            </div>
        </div>
    `;

    const startTime = Date.now();

    try {
        const response = await fetch(`${this.baseUrl}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query })
        });

        // Check if the response is successful
        if (!response.ok) {
            // Try to get error details from response
            let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
            try {
                const errorData = await response.json();
                if (errorData.detail) {
                    errorMessage = errorData.detail;
                }
            } catch (e) {
                // If we can't parse JSON, use the status text
            }
            throw new Error(errorMessage);
        }

        const result = await response.json();
        const responseTime = Date.now() - startTime;

        this.displayQueryResults(result, responseTime);
        this.hideQuerySuggestions();

    } catch (error) {
        console.error('Query error:', error);
        resultsDiv.innerHTML = `
            <div class="card">
                <div class="card-body">
                    <div class="alert alert-danger">
                        <i class="fas fa-exclamation-triangle me-2"></i>
                        Query failed: ${error.message}
                    </div>
                </div>
            </div>
        `;
    }
}
// ...existing code...
    displayQueryResults(result, responseTime) {
        const resultsDiv = document.getElementById('queryResults');
        
        let html = `
            <div class="card fade-in">
                <div class="card-header d-flex justify-content-between align-items-center">
                    <h5 class="mb-0">
                        <i class="fas fa-search-plus me-2"></i>Query Results
                    </h5>
                    <div>
                        <span class="badge bg-primary">${result.query_type}</span>
                        <span class="badge bg-info">${responseTime}ms</span>
                        ${result.cached ? '<span class="badge bg-success">Cached</span>' : ''}
                    </div>
                </div>
                <div class="card-body">
        `;

        if (result.query_type === 'sql') {
            html += this.renderSQLResults(result.results);
        } else if (result.query_type === 'document') {
            html += this.renderDocumentResults(result.results);
        } else if (result.query_type === 'hybrid') {
            html += this.renderHybridResults(result.results);
        } else if (result.results.error) {
            html += `<div class="alert alert-danger">${result.results.error}</div>`;
        }

        // Add sources
        if (result.sources && result.sources.length > 0) {
            html += `
                <div class="mt-3">
                    <h6>Sources:</h6>
                    <ul class="list-unstyled">
                        ${result.sources.map(source => `<li><small class="text-muted"><i class="fas fa-link me-1"></i>${source}</small></li>`).join('')}
                    </ul>
                </div>
            `;
        }

        html += `
                </div>
            </div>
        `;

        resultsDiv.innerHTML = html;
        resultsDiv.scrollIntoView({ behavior: 'smooth' });
    }

    renderSQLResults(results) {
        if (results.error) {
            return `<div class="alert alert-danger">${results.error}</div>`;
        }

        if (!results.data || results.data.length === 0) {
            return `<div class="alert alert-info">No results found.</div>`;
        }

        const columns = results.columns || Object.keys(results.data[0]);
        
        let html = `
            <div class="d-flex justify-content-between align-items-center mb-3">
                <span class="text-muted">${results.row_count} rows returned</span>
                <button class="btn btn-sm btn-outline-primary" onclick="app.exportResults('sql', ${JSON.stringify(results.data).replace(/"/g, '&quot;')})">
                    <i class="fas fa-download me-1"></i>Export CSV
                </button>
            </div>
            <div class="table-responsive">
                <table class="table table-striped table-hover">
                    <thead class="table-dark">
                        <tr>
                            ${columns.map(col => `<th>${col}</th>`).join('')}
                        </tr>
                    </thead>
                    <tbody>
        `;

        results.data.slice(0, 100).forEach(row => {
            html += '<tr>';
            columns.forEach(col => {
                const value = row[col];
                html += `<td>${value !== null ? value : '<span class="text-muted">null</span>'}</td>`;
            });
            html += '</tr>';
        });

        html += `
                    </tbody>
                </table>
            </div>
        `;

        if (results.data.length > 100) {
            html += `<div class="alert alert-info">Showing first 100 rows of ${results.data.length} total results.</div>`;
        }

        return html;
    }

    renderDocumentResults(results) {
        if (results.error) {
            return `<div class="alert alert-danger">${results.error}</div>`;
        }

        if (!results.documents || results.documents.length === 0) {
            return `<div class="alert alert-info">No matching documents found.</div>`;
        }

        let html = `
            <div class="mb-3">
                <span class="text-muted">${results.total_matches} matching documents found</span>
            </div>
        `;

        results.documents.forEach((doc, index) => {
            html += `
                <div class="result-card mb-3">
                    <div class="card-header">
                        <div class="d-flex justify-content-between align-items-center">
                            <h6 class="mb-0">
                                <i class="fas fa-file-alt me-2"></i>${doc.filename}
                            </h6>
                            <div>
                                <span class="badge bg-secondary">${doc.file_type.toUpperCase()}</span>
                                <span class="badge bg-success">${Math.round(doc.similarity * 100)}% match</span>
                            </div>
                        </div>
                    </div>
                    <div class="card-body">
                        <p class="card-text">${doc.content}</p>
                        <small class="text-muted">Chunk ${doc.chunk_index + 1}</small>
                    </div>
                </div>
            `;
        });

        return html;
    }

    renderHybridResults(results) {
        let html = '<div class="row">';

        // Database results
        if (results.database_results && !results.database_results.error) {
            html += `
                <div class="col-lg-6 mb-4">
                    <div class="card">
                        <div class="card-header bg-primary text-white">
                            <h6 class="mb-0"><i class="fas fa-database me-2"></i>Database Results</h6>
                        </div>
                        <div class="card-body">
                            ${this.renderSQLResults(results.database_results)}
                        </div>
                    </div>
                </div>
            `;
        }

        // Document results
        if (results.document_results && !results.document_results.error) {
            html += `
                <div class="col-lg-6 mb-4">
                    <div class="card">
                        <div class="card-header bg-success text-white">
                            <h6 class="mb-0"><i class="fas fa-file-text me-2"></i>Document Results</h6>
                        </div>
                        <div class="card-body">
                            ${this.renderDocumentResults(results.document_results)}
                        </div>
                    </div>
                </div>
            `;
        }

        html += '</div>';

        // Show errors if any
        if (results.database_results?.error || results.document_results?.error) {
            html += '<div class="mt-3">';
            if (results.database_results?.error) {
                html += `<div class="alert alert-warning">Database search failed: ${results.database_results.error}</div>`;
            }
            if (results.document_results?.error) {
                html += `<div class="alert alert-warning">Document search failed: ${results.document_results.error}</div>`;
            }
            html += '</div>';
        }

        return html;
    }

    exportResults(type, data) {
        if (type === 'sql' && data && data.length > 0) {
            const csv = this.convertToCSV(data);
            this.downloadCSV(csv, 'query_results.csv');
        }
    }

    convertToCSV(data) {
        if (!data || data.length === 0) return '';
        
        const headers = Object.keys(data[0]);
        const csvHeaders = headers.join(',');
        
        const csvRows = data.map(row => 
            headers.map(header => {
                const value = row[header];
                return value !== null ? `"${String(value).replace(/"/g, '""')}"` : '';
            }).join(',')
        );
        
        return [csvHeaders, ...csvRows].join('\n');
    }

    downloadCSV(csv, filename) {
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        window.URL.revokeObjectURL(url);
    }

    async loadMetrics() {
        try {
            const response = await fetch(`${this.baseUrl}/api/metrics`);
            const metrics = await response.json();

            document.getElementById('cacheHitRate').textContent = `${metrics.cache_stats?.hit_rate || 0}%`;
            document.getElementById('totalDocuments').textContent = metrics.total_documents || 0;
            document.getElementById('totalChunks').textContent = metrics.total_chunks || 0;
            document.getElementById('activeConnections').textContent = metrics.active_connections || 0;
        } catch (error) {
            console.error('Failed to load metrics:', error);
        }
    }

    async loadQueryHistory() {
        try {
            const response = await fetch(`${this.baseUrl}/api/query/history`);
            const history = await response.json();
            
            const historyDiv = document.getElementById('queryHistory');
            
            if (history.queries && history.queries.length > 0) {
                let html = '<div class="list-group list-group-flush">';
                
                history.queries.forEach(query => {
                    const typebadge = this.getQueryTypeBadge(query.query_type);
                    html += `
                        <div class="list-group-item d-flex justify-content-between align-items-start">
                            <div class="ms-2 me-auto">
                                <div class="fw-bold">${query.query}</div>
                                <small class="text-muted">${query.cached_at}</small>
                            </div>
                            ${typeBadge}
                        </div>
                    `;
                });
                
                html += '</div>';
                historyDiv.innerHTML = html;
            } else {
                historyDiv.innerHTML = '<p class="text-muted">No queries yet. Try running some queries to see them here.</p>';
            }
        } catch (error) {
            console.error('Failed to load query history:', error);
        }
    }

    getQueryTypeBadge(queryType) {
        const badges = {
            'sql': 'bg-primary',
            'document': 'bg-success',
            'hybrid': 'bg-info',
            'error': 'bg-danger'
        };
        
        const badgeClass = badges[queryType] || 'bg-secondary';
        return `<span class="badge ${badgeClass}">${queryType}</span>`;
    }

    showSuccess(message) {
        this.showToast(message, 'success');
    }

    showError(message) {
        this.showToast(message, 'danger');
    }

    showToast(message, type = 'info') {
        const toastContainer = document.querySelector('.toast-container');
        const toastId = 'toast-' + Date.now();
        
        const toast = document.createElement('div');
        toast.id = toastId;
        toast.className = `toast align-items-center text-bg-${type} border-0`;
        toast.setAttribute('role', 'alert');
        toast.innerHTML = `
            <div class="d-flex">
                <div class="toast-body">
                    <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'danger' ? 'exclamation-triangle' : 'info-circle'} me-2"></i>
                    ${message}
                </div>
                <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
            </div>
        `;
        
        toastContainer.appendChild(toast);
        
        const bsToast = new bootstrap.Toast(toast);
        bsToast.show();
        
        // Remove toast element after it's hidden
        toast.addEventListener('hidden.bs.toast', () => {
            toast.remove();
        });
    }
}

// Initialize the application
const app = new NLPQueryApp();