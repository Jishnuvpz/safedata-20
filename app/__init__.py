import os
import logging
from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

def create_app():
    # create the app - configure to look for templates and static files in the root directory
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

    # configure the database
    database_url = os.environ.get("DATABASE_URL")
    if database_url and database_url.startswith("postgres://"):
        database_url = database_url.replace("postgres://", "postgresql://", 1)
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url or "sqlite:///safedata.db"
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 300,
        "pool_pre_ping": True,
    }
    
    # initialize the app with the extension
    db.init_app(app)

    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    with app.app_context():
        # Import models here to ensure they're registered
        from app.models import models
        db.create_all()

    # Register blueprints/routes
    register_routes(app)
    
    return app

def register_routes(app):
    """Register all application routes"""
    
    @app.route('/')
    def index():
        """Main dashboard page"""
        return render_template('index.html')

    @app.route('/dashboard')
    def dashboard():
        """Analytics dashboard page"""
        return render_template('dashboard.html')

    @app.route('/results')
    def results():
        """Results page"""
        return render_template('results.html')

    # API Routes for anonymization functionality
    @app.route('/api/health')
    def health_check():
        """Health check endpoint"""
        return jsonify({"status": "healthy", "service": "SafeData 2.0"})

    @app.route('/api/anonymize', methods=['POST'])
    def anonymize_data():
        """Main anonymization endpoint"""
        try:
            # Import privacy services - commented out for initial setup
            # from app.services.anonymization import AnonymizationService
            # from app.services.audit import AuditService
            
            # anonymization_service = AnonymizationService()
            # audit_service = AuditService()
            
            # Get request data
            data = request.get_json()
            method = data.get('method', 'sdg')
            
            # Mock response for now - will be replaced with actual implementation
            result = {
                "anonymized_data_id": "anon_" + str(abs(hash(str(data))))[:8],
                "method": method,
                "status": "completed",
                "execution_time": 2.3,
                "created_at": "2025-08-14T06:50:00Z",
                "parameters": data,
                "privacy_metrics": {
                    "epsilon_used": data.get('epsilon', 1.0),
                    "delta_used": data.get('delta', 1e-5),
                    "re_identification_risk": 0.15
                },
                "utility_metrics": {
                    "statistical_similarity": 0.89,
                    "correlation_preservation": 0.82,
                    "overall_utility_score": 0.785
                },
                "warnings": []
            }
            
            return jsonify(result)
            
        except Exception as e:
            app.logger.error(f"Anonymization error: {str(e)}")
            return jsonify({"error": "Anonymization failed", "details": str(e)}), 500

    @app.route('/api/privacy/assessment', methods=['POST'])
    def privacy_assessment():
        """Privacy risk assessment"""
        try:
            data = request.get_json()
            
            # Mock privacy assessment response
            assessment = {
                "overall_score": 82.7,
                "privacy_level": "High",
                "risk_factors": [
                    {"type": "linkage_attack", "risk": "Low", "score": 95},
                    {"type": "membership_inference", "risk": "Medium", "score": 78},
                    {"type": "attribute_inference", "risk": "Low", "score": 91}
                ],
                "recommendations": [
                    "Consider increasing epsilon value for stronger privacy",
                    "Apply additional noise to sensitive attributes"
                ]
            }
            
            return jsonify(assessment)
            
        except Exception as e:
            app.logger.error(f"Privacy assessment error: {str(e)}")
            return jsonify({"error": "Assessment failed", "details": str(e)}), 500

    @app.route('/api/upload', methods=['POST'])
    def upload_file():
        """File upload endpoint"""
        try:
            if 'file' not in request.files:
                return jsonify({"error": "No file provided"}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No file selected"}), 400
            
            # Read file content
            file_content = file.read()
            file.seek(0)  # Reset file pointer
            
            # Basic file info
            file_size = len(file_content)
            
            # Try to get file structure info
            rows = 0
            columns = 0
            
            try:
                if file.filename.endswith('.csv'):
                    import pandas as pd
                    import io
                    df = pd.read_csv(io.BytesIO(file_content))
                    rows, columns = df.shape
                elif file.filename.endswith(('.xlsx', '.xls')):
                    import pandas as pd
                    import io
                    df = pd.read_excel(io.BytesIO(file_content))
                    rows, columns = df.shape
                else:
                    # For other file types, estimate
                    if file_content:
                        lines = file_content.decode('utf-8', errors='ignore').split('\n')
                        rows = len([line for line in lines if line.strip()])
                        if rows > 0:
                            columns = len(lines[0].split(',')) if lines[0] else 1
            except Exception as parse_error:
                app.logger.warning(f"Could not parse file structure: {parse_error}")
                rows = 1000  # Default estimate
                columns = 10  # Default estimate
            
            # Create file info response
            file_info = {
                "file_id": "file_" + str(abs(hash(file.filename)))[:8],
                "filename": file.filename,
                "size": file_size,
                "rows": rows,
                "columns": columns,
                "type": file.content_type,
                "status": "uploaded",
                "upload_time": "2025-08-14T06:50:00Z"
            }
            
            app.logger.info(f"File uploaded successfully: {file.filename} ({file_size} bytes, {rows}x{columns})")
            return jsonify(file_info)
            
        except Exception as e:
            app.logger.error(f"File upload error: {str(e)}")
            return jsonify({"error": "Upload failed", "details": str(e)}), 500

    @app.route('/api/files')
    def list_files():
        """List uploaded files"""
        # Mock file list
        files = [
            {
                "id": "file_12345",
                "filename": "dataset.csv",
                "size": 2048000,
                "status": "ready",
                "uploaded_at": "2025-08-14T04:35:00Z"
            }
        ]
        return jsonify(files)

    @app.route('/api/results')
    def list_results():
        """List anonymization results"""
        # Mock results list
        results = [
            {
                "id": "anon_67890",
                "method": "SDG + DP",
                "privacy_score": 85.2,
                "utility_score": 78.5,
                "status": "completed",
                "created_at": "2025-08-14T04:38:00Z"
            }
        ]
        return jsonify(results)

    @app.route('/api/audit/logs')
    def audit_logs():
        """Get audit logs"""
        # Mock audit logs
        logs = [
            {
                "timestamp": "2025-08-14T04:40:00Z",
                "event": "file_upload",
                "user": "anonymous",
                "details": "dataset.csv uploaded successfully"
            },
            {
                "timestamp": "2025-08-14T04:38:00Z",
                "event": "anonymization",
                "user": "anonymous",
                "details": "SDG anonymization completed"
            }
        ]
        return jsonify(logs)

    @app.route('/api/list')
    def list_data():
        """List all files and results for dashboard"""
        # Mock combined data response
        files = [
            {
                "file_id": "file_12345",
                "filename": "patient_data.csv",
                "size": 2048000,
                "rows": 15000,
                "columns": 12,
                "status": "ready",
                "upload_time": "2025-08-14T04:35:00Z"
            },
            {
                "file_id": "file_67890",
                "filename": "medical_records.xlsx",
                "size": 1024000,
                "rows": 8500,
                "columns": 18,
                "status": "ready",
                "upload_time": "2025-08-14T04:30:00Z"
            }
        ]
        
        results = [
            {
                "id": "anon_12345",
                "method": "SDG + DP",
                "privacy_score": 85.2,
                "utility_score": 78.5,
                "status": "completed",
                "created_at": "2025-08-14T04:38:00Z"
            },
            {
                "id": "anon_67890",
                "method": "Full Pipeline",
                "privacy_score": 92.1,
                "utility_score": 71.3,
                "status": "completed",
                "created_at": "2025-08-14T04:42:00Z"
            }
        ]
        
        return jsonify({
            "total_files": len(files),
            "total_results": len(results),
            "files": files,
            "anonymization_results": results
        })

# Create the app instance
app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
