import os
import logging
import io
import tempfile
from datetime import datetime
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
            column_names = []
            
            try:
                if file.filename.endswith('.csv'):
                    import pandas as pd
                    import io
                    df = pd.read_csv(io.BytesIO(file_content))
                    rows, columns = df.shape
                    column_names = df.columns.tolist()[:20]  # Limit to first 20 columns
                elif file.filename.endswith(('.xlsx', '.xls')):
                    import pandas as pd
                    import io
                    df = pd.read_excel(io.BytesIO(file_content))
                    rows, columns = df.shape
                    column_names = df.columns.tolist()[:20]  # Limit to first 20 columns
                else:
                    # For other file types, estimate
                    if file_content:
                        lines = file_content.decode('utf-8', errors='ignore').split('\n')
                        rows = len([line for line in lines if line.strip()])
                        if rows > 0 and lines[0]:
                            first_line = lines[0].split(',')
                            columns = len(first_line)
                            column_names = [f"Column_{i+1}" for i in range(min(columns, 20))]
            except Exception as parse_error:
                app.logger.warning(f"Could not parse file structure: {parse_error}")
                rows = 1000  # Default estimate
                columns = 10  # Default estimate
                column_names = [f"Column_{i+1}" for i in range(10)]
            
            # Create file info response
            file_info = {
                "file_id": "file_" + str(abs(hash(file.filename)))[:8],
                "filename": file.filename,
                "size": file_size,
                "rows": rows,
                "columns": column_names,  # Return array of column names instead of count
                "column_count": columns,   # Keep the count as separate field
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

    def generate_mock_anonymized_data(result_id, rows=100):
        """Generate mock anonymized data for download"""
        import pandas as pd
        import numpy as np
        
        # Generate synthetic anonymized data
        np.random.seed(int(result_id.replace('anon_', ''), 16) % 2**32)  # Consistent seed based on result_id
        
        data = {
            'ID': [f'ANON_{i:06d}' for i in range(1, rows + 1)],
            'Age_Group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '56-65', '65+'], rows),
            'Gender': np.random.choice(['M', 'F', 'Other'], rows),
            'Location_Region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], rows),
            'Income_Range': np.random.choice(['Low', 'Medium', 'High'], rows),
            'Education_Level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], rows),
            'Anonymized_Score_1': np.random.normal(75, 15, rows).round(2),
            'Anonymized_Score_2': np.random.normal(85, 10, rows).round(2),
            'Category_A': np.random.choice(['Type1', 'Type2', 'Type3'], rows),
            'Category_B': np.random.choice(['ClassA', 'ClassB', 'ClassC'], rows),
            'Numeric_Value': np.random.exponential(2, rows).round(2),
            'Boolean_Flag': np.random.choice([True, False], rows),
            'Date_Period': pd.date_range('2020-01-01', '2024-12-31', periods=rows).strftime('%Y-%m'),
            'Status': np.random.choice(['Active', 'Inactive', 'Pending'], rows)
        }
        
        return pd.DataFrame(data)

    def generate_csv_file(df, result_id):
        """Generate CSV file"""
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        return io.BytesIO(output.getvalue().encode('utf-8'))

    def generate_excel_file(df, result_id):
        """Generate Excel file"""
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Anonymized_Data', index=False)
            
            # Add a summary sheet
            summary_df = pd.DataFrame({
                'Metric': ['Total Records', 'Columns', 'Anonymization Method', 'Privacy Score', 'Generated'],
                'Value': [len(df), len(df.columns), 'Differential Privacy + SDG', '85%', datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        output.seek(0)
        return output

    def generate_pdf_file(df, result_id):
        """Generate PDF file"""
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                                topMargin=72, bottomMargin=18)
        
        # Container for the 'Flowable' objects
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center alignment
        )
        title = Paragraph(f"Anonymized Dataset - {result_id}", title_style)
        elements.append(title)
        
        # Summary info
        summary_style = styles['Normal']
        summary_text = f"""
        <b>Dataset Summary:</b><br/>
        Records: {len(df)}<br/>
        Columns: {len(df.columns)}<br/>
        Anonymization Method: Differential Privacy + Synthetic Data Generation<br/>
        Privacy Score: 85%<br/>
        Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
        """
        summary = Paragraph(summary_text, summary_style)
        elements.append(summary)
        elements.append(Spacer(1, 20))
        
        # Data table (first 50 rows to keep PDF size manageable)
        display_df = df.head(50)
        data = [list(display_df.columns)]
        for _, row in display_df.iterrows():
            data.append([str(val)[:20] + '...' if len(str(val)) > 20 else str(val) for val in row])
        
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        elements.append(table)
        
        if len(df) > 50:
            note = Paragraph(f"<i>Note: Showing first 50 records out of {len(df)} total records.</i>", styles['Italic'])
            elements.append(Spacer(1, 12))
            elements.append(note)
        
        doc.build(elements)
        buffer.seek(0)
        return buffer

    def generate_word_file(df, result_id):
        """Generate Word document"""
        from docx import Document
        from docx.shared import Inches
        
        doc = Document()
        
        # Title
        title = doc.add_heading(f'Anonymized Dataset - {result_id}', 0)
        
        # Summary
        doc.add_heading('Dataset Summary', level=1)
        summary_para = doc.add_paragraph()
        summary_para.add_run('Records: ').bold = True
        summary_para.add_run(f'{len(df)}\n')
        summary_para.add_run('Columns: ').bold = True
        summary_para.add_run(f'{len(df.columns)}\n')
        summary_para.add_run('Anonymization Method: ').bold = True
        summary_para.add_run('Differential Privacy + Synthetic Data Generation\n')
        summary_para.add_run('Privacy Score: ').bold = True
        summary_para.add_run('85%\n')
        summary_para.add_run('Generated: ').bold = True
        summary_para.add_run(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        # Data table (first 100 rows)
        doc.add_heading('Anonymized Data', level=1)
        display_df = df.head(100)
        
        table = doc.add_table(rows=1, cols=len(display_df.columns))
        table.style = 'Table Grid'
        
        # Header row
        hdr_cells = table.rows[0].cells
        for i, column in enumerate(display_df.columns):
            hdr_cells[i].text = str(column)
        
        # Data rows
        for _, row in display_df.iterrows():
            row_cells = table.add_row().cells
            for i, value in enumerate(row):
                row_cells[i].text = str(value)[:50]  # Limit cell content length
        
        if len(df) > 100:
            doc.add_paragraph(f'Note: Showing first 100 records out of {len(df)} total records.')
        
        # Save to buffer
        buffer = io.BytesIO()
        doc.save(buffer)
        buffer.seek(0)
        return buffer

    @app.route('/api/result/<result_id>/download', methods=['GET'])
    def download_result(result_id):
        """Download anonymized data in specified format"""
        try:
            format_type = request.args.get('format', 'csv').lower()
            
            # Generate mock anonymized data
            df = generate_mock_anonymized_data(result_id, rows=200)
            
            if format_type == 'csv':
                file_buffer = generate_csv_file(df, result_id)
                mimetype = 'text/csv'
                filename = f'anonymized_data_{result_id}.csv'
                
            elif format_type in ['xlsx', 'excel']:
                file_buffer = generate_excel_file(df, result_id)
                mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                filename = f'anonymized_data_{result_id}.xlsx'
                
            elif format_type == 'pdf':
                file_buffer = generate_pdf_file(df, result_id)
                mimetype = 'application/pdf'
                filename = f'anonymized_data_{result_id}.pdf'
                
            elif format_type in ['docx', 'word', 'doc']:
                file_buffer = generate_word_file(df, result_id)
                mimetype = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                filename = f'anonymized_data_{result_id}.docx'
                
            else:
                return jsonify({"error": "Unsupported format", "supported": ["csv", "xlsx", "pdf", "docx"]}), 400
            
            return send_file(
                file_buffer,
                mimetype=mimetype,
                as_attachment=True,
                download_name=filename
            )
            
        except Exception as e:
            app.logger.error(f"Download error: {str(e)}")
            return jsonify({"error": "Download failed", "details": str(e)}), 500

# Create the app instance
app = create_app()

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
