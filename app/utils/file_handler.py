import os
import tempfile
import hashlib
import magic
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
import uuid
import mimetypes
from pathlib import Path

try:
    import camelot
    CAMELOT_AVAILABLE = True
except ImportError:
    CAMELOT_AVAILABLE = False
    logging.warning("Camelot not available. PDF table extraction will be limited.")

from app.core.config import settings
from app.core.security import security_manager

logger = logging.getLogger(__name__)

class FileProcessor:
    """Secure file processing and validation"""
    
    def __init__(self):
        self.upload_dir = "uploads"
        self.processed_dir = "processed"
        self.temp_dir = "temp"
        
        # Create directories
        for directory in [self.upload_dir, self.processed_dir, self.temp_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # File type mappings
        self.supported_types = {
            '.csv': 'text/csv',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.xls': 'application/vnd.ms-excel',
            '.pdf': 'application/pdf'
        }
        
        # Security limits
        self.max_file_size = settings.max_file_size
        self.max_rows = 1000000  # 1M rows
        self.max_columns = 1000
        
    async def validate_and_process_file(
        self,
        file_content: bytes,
        filename: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Validate and process uploaded file
        
        Args:
            file_content: Raw file content
            filename: Original filename
            user_id: User identifier for audit
            
        Returns:
            Processing result with file metadata and data preview
        """
        
        start_time = datetime.now()
        
        try:
            # Security validation
            validation_result = await self._validate_file_security(file_content, filename)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "error_type": "security_validation"
                }
            
            # Save file securely
            file_info = await self._save_file_securely(file_content, filename, user_id)
            
            # Process file based on type
            processing_result = await self._process_file_by_type(file_info)
            
            if not processing_result["success"]:
                return processing_result
            
            # Generate file metadata
            metadata = await self._generate_file_metadata(file_info, processing_result["data"])
            
            # Data quality assessment
            quality_report = await self._assess_data_quality(processing_result["data"])
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "success": True,
                "file_id": file_info["file_id"],
                "metadata": metadata,
                "data_preview": await self._create_data_preview(processing_result["data"]),
                "quality_report": quality_report,
                "processing_time": execution_time,
                "warnings": processing_result.get("warnings", [])
            }
            
        except Exception as e:
            logger.error(f"File processing failed: {str(e)}")
            return {
                "success": False,
                "error": f"File processing failed: {str(e)}",
                "error_type": "processing_error"
            }
    
    async def _validate_file_security(
        self,
        file_content: bytes,
        filename: str
    ) -> Dict[str, Any]:
        """Comprehensive file security validation"""
        
        # Size check
        if len(file_content) > self.max_file_size:
            return {
                "valid": False,
                "error": f"File size ({len(file_content)} bytes) exceeds maximum allowed ({self.max_file_size} bytes)"
            }
        
        if len(file_content) == 0:
            return {
                "valid": False,
                "error": "File is empty"
            }
        
        # Filename validation
        sanitized_filename = security_manager.sanitize_filename(filename)
        if not sanitized_filename:
            return {
                "valid": False,
                "error": "Invalid filename"
            }
        
        # Extension validation
        file_ext = Path(filename).suffix.lower()
        if not security_manager.validate_file_type(filename, settings.allowed_extensions):
            return {
                "valid": False,
                "error": f"File type '{file_ext}' not allowed. Supported types: {settings.allowed_extensions}"
            }
        
        # MIME type validation
        try:
            detected_mime = magic.from_buffer(file_content, mime=True)
            expected_mime = self.supported_types.get(file_ext)
            
            if expected_mime and detected_mime != expected_mime:
                # Allow some flexibility for common variations
                if not self._is_mime_compatible(detected_mime, expected_mime):
                    logger.warning(f"MIME type mismatch: expected {expected_mime}, got {detected_mime}")
        except Exception as e:
            logger.warning(f"MIME type detection failed: {str(e)}")
        
        # Content validation (basic malware checks)
        if await self._contains_suspicious_content(file_content):
            return {
                "valid": False,
                "error": "File contains suspicious content"
            }
        
        return {"valid": True}
    
    def _is_mime_compatible(self, detected: str, expected: str) -> bool:
        """Check if detected MIME type is compatible with expected"""
        
        compatibility_map = {
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': [
                'application/zip',  # XLSX files are ZIP archives
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            ],
            'application/vnd.ms-excel': [
                'application/vnd.ms-excel',
                'application/msexcel'
            ],
            'text/csv': [
                'text/csv',
                'text/plain',
                'application/csv'
            ],
            'application/pdf': [
                'application/pdf'
            ]
        }
        
        compatible_types = compatibility_map.get(expected, [expected])
        return detected in compatible_types
    
    async def _contains_suspicious_content(self, content: bytes) -> bool:
        """Basic check for suspicious content"""
        
        # Check for common malware signatures (simplified)
        suspicious_patterns = [
            b'<script',
            b'javascript:',
            b'eval(',
            b'exec(',
            b'cmd.exe',
            b'powershell'
        ]
        
        content_lower = content.lower()
        for pattern in suspicious_patterns:
            if pattern in content_lower:
                return True
        
        return False
    
    async def _save_file_securely(
        self,
        file_content: bytes,
        filename: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Save file with security measures"""
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Sanitize filename
        sanitized_filename = security_manager.sanitize_filename(filename)
        file_ext = Path(filename).suffix.lower()
        
        # Create secure filename
        secure_filename = f"{file_id}_{sanitized_filename}"
        file_path = os.path.join(self.upload_dir, secure_filename)
        
        # Calculate file hash for integrity
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Save file
        with open(file_path, 'wb') as f:
            f.write(file_content)
        
        # Set restrictive permissions
        os.chmod(file_path, 0o600)
        
        file_info = {
            "file_id": file_id,
            "original_filename": filename,
            "secure_filename": secure_filename,
            "file_path": file_path,
            "file_extension": file_ext,
            "file_size": len(file_content),
            "file_hash": file_hash,
            "upload_time": datetime.now(),
            "user_id": user_id
        }
        
        logger.info(f"File saved securely: {file_id}")
        return file_info
    
    async def _process_file_by_type(self, file_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process file based on its type"""
        
        file_ext = file_info["file_extension"]
        file_path = file_info["file_path"]
        warnings = []
        
        try:
            if file_ext == '.csv':
                data, file_warnings = await self._process_csv(file_path)
            elif file_ext in ['.xlsx', '.xls']:
                data, file_warnings = await self._process_excel(file_path)
            elif file_ext == '.pdf':
                data, file_warnings = await self._process_pdf(file_path)
            else:
                return {
                    "success": False,
                    "error": f"Unsupported file type: {file_ext}"
                }
            
            warnings.extend(file_warnings)
            
            # Validate processed data
            validation_result = await self._validate_processed_data(data)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"]
                }
            
            warnings.extend(validation_result.get("warnings", []))
            
            return {
                "success": True,
                "data": data,
                "warnings": warnings
            }
            
        except Exception as e:
            logger.error(f"File processing failed for {file_path}: {str(e)}")
            return {
                "success": False,
                "error": f"Failed to process {file_ext} file: {str(e)}"
            }
    
    async def _process_csv(self, file_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """Process CSV file"""
        
        warnings = []
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                # Try different delimiters
                for delimiter in [',', ';', '\t', '|']:
                    try:
                        data = pd.read_csv(
                            file_path,
                            encoding=encoding,
                            delimiter=delimiter,
                            low_memory=False,
                            na_values=['', 'NULL', 'null', 'N/A', 'n/a', 'NA', 'na']
                        )
                        
                        # Check if parsing was successful
                        if len(data.columns) > 1 or len(data) > 0:
                            if encoding != 'utf-8':
                                warnings.append(f"File encoding detected as {encoding}")
                            if delimiter != ',':
                                warnings.append(f"Delimiter detected as '{delimiter}'")
                            return data, warnings
                    except:
                        continue
            except:
                continue
        
        # If all attempts fail, try with error handling
        try:
            data = pd.read_csv(
                file_path,
                encoding='utf-8',
                on_bad_lines='skip',
                low_memory=False
            )
            warnings.append("Some lines were skipped due to parsing errors")
            return data, warnings
        except Exception as e:
            raise Exception(f"CSV parsing failed: {str(e)}")
    
    async def _process_excel(self, file_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """Process Excel file"""
        
        warnings = []
        
        try:
            # Read Excel file
            excel_file = pd.ExcelFile(file_path)
            
            # Get sheet names
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) > 1:
                warnings.append(f"Multiple sheets found ({len(sheet_names)}), using first sheet: '{sheet_names[0]}'")
            
            # Read first sheet
            data = pd.read_excel(
                file_path,
                sheet_name=0,
                na_values=['', 'NULL', 'null', 'N/A', 'n/a', 'NA', 'na']
            )
            
            return data, warnings
            
        except Exception as e:
            raise Exception(f"Excel processing failed: {str(e)}")
    
    async def _process_pdf(self, file_path: str) -> Tuple[pd.DataFrame, List[str]]:
        """Process PDF file (table extraction)"""
        
        warnings = []
        
        if not CAMELOT_AVAILABLE:
            raise Exception("PDF processing not available. Camelot library required.")
        
        try:
            # Extract tables using Camelot
            tables = camelot.read_pdf(file_path, pages='all', flavor='lattice')
            
            if len(tables) == 0:
                # Try stream flavor if lattice fails
                tables = camelot.read_pdf(file_path, pages='all', flavor='stream')
            
            if len(tables) == 0:
                raise Exception("No tables found in PDF")
            
            if len(tables) > 1:
                warnings.append(f"Multiple tables found ({len(tables)}), concatenating all tables")
            
            # Combine all tables
            dataframes = []
            for table in tables:
                df = table.df
                
                # Skip empty tables
                if df.empty or len(df.columns) == 0:
                    continue
                
                # Use first row as headers if they look like headers
                if await self._first_row_looks_like_headers(df):
                    df.columns = df.iloc[0]
                    df = df.drop(df.index[0])
                
                dataframes.append(df)
            
            if not dataframes:
                raise Exception("No valid tables extracted from PDF")
            
            # Concatenate all dataframes
            data = pd.concat(dataframes, ignore_index=True)
            
            warnings.append("Data extracted from PDF tables - please verify accuracy")
            
            return data, warnings
            
        except Exception as e:
            raise Exception(f"PDF processing failed: {str(e)}")
    
    async def _first_row_looks_like_headers(self, df: pd.DataFrame) -> bool:
        """Check if first row contains header-like data"""
        
        if df.empty or len(df) < 2:
            return False
        
        first_row = df.iloc[0]
        second_row = df.iloc[1]
        
        # Check if first row has more text and second row has more numbers
        first_row_text_count = sum(1 for val in first_row if isinstance(val, str) and not val.replace('.', '').isdigit())
        second_row_numeric_count = sum(1 for val in second_row if pd.api.types.is_numeric_dtype(type(val)) or 
                                     (isinstance(val, str) and val.replace('.', '').isdigit()))
        
        return first_row_text_count > len(first_row) * 0.5 and second_row_numeric_count > len(second_row) * 0.3
    
    async def _validate_processed_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate processed data"""
        
        warnings = []
        
        # Check data dimensions
        if len(data) == 0:
            return {
                "valid": False,
                "error": "No data rows found in file"
            }
        
        if len(data.columns) == 0:
            return {
                "valid": False,
                "error": "No columns found in file"
            }
        
        # Check size limits
        if len(data) > self.max_rows:
            return {
                "valid": False,
                "error": f"Too many rows ({len(data)}). Maximum allowed: {self.max_rows}"
            }
        
        if len(data.columns) > self.max_columns:
            return {
                "valid": False,
                "error": f"Too many columns ({len(data.columns)}). Maximum allowed: {self.max_columns}"
            }
        
        # Check for completely empty columns
        empty_columns = data.columns[data.isnull().all()].tolist()
        if empty_columns:
            warnings.append(f"Found {len(empty_columns)} completely empty columns")
        
        # Check for very sparse data
        sparsity = data.isnull().sum().sum() / (len(data) * len(data.columns))
        if sparsity > 0.9:
            warnings.append(f"Data is very sparse ({sparsity:.1%} missing values)")
        
        return {
            "valid": True,
            "warnings": warnings
        }
    
    async def _generate_file_metadata(
        self,
        file_info: Dict[str, Any],
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate comprehensive file metadata"""
        
        # Basic metadata
        metadata = {
            "file_id": file_info["file_id"],
            "original_filename": file_info["original_filename"],
            "file_extension": file_info["file_extension"],
            "file_size": file_info["file_size"],
            "file_hash": file_info["file_hash"],
            "upload_time": file_info["upload_time"].isoformat(),
            "user_id": file_info["user_id"]
        }
        
        # Data metadata
        metadata.update({
            "data_shape": {
                "rows": len(data),
                "columns": len(data.columns)
            },
            "column_names": data.columns.tolist(),
            "column_types": data.dtypes.astype(str).to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "memory_usage": data.memory_usage(deep=True).sum(),
            "numeric_columns": data.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": data.select_dtypes(include=['object']).columns.tolist(),
            "datetime_columns": data.select_dtypes(include=['datetime']).columns.tolist()
        })
        
        return metadata
    
    async def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality and provide insights"""
        
        quality_report = {
            "overall_score": 0.0,
            "completeness": {},
            "consistency": {},
            "uniqueness": {},
            "validity": {},
            "recommendations": []
        }
        
        # Completeness assessment
        missing_ratio = data.isnull().sum() / len(data)
        completeness_score = 1 - missing_ratio.mean()
        quality_report["completeness"] = {
            "score": completeness_score,
            "missing_by_column": missing_ratio.to_dict(),
            "columns_with_missing": missing_ratio[missing_ratio > 0].to_dict()
        }
        
        # Uniqueness assessment
        uniqueness_ratios = {}
        duplicate_rows = data.duplicated().sum()
        
        for col in data.columns:
            uniqueness_ratios[col] = data[col].nunique() / len(data)
        
        uniqueness_score = 1 - (duplicate_rows / len(data))
        quality_report["uniqueness"] = {
            "score": uniqueness_score,
            "duplicate_rows": int(duplicate_rows),
            "uniqueness_by_column": uniqueness_ratios
        }
        
        # Consistency assessment (simplified)
        consistency_issues = []
        
        for col in data.select_dtypes(include=['object']).columns:
            # Check for inconsistent casing
            unique_values = data[col].dropna().unique()
            if len(unique_values) != len([v.lower() for v in unique_values if isinstance(v, str)]):
                consistency_issues.append(f"Inconsistent casing in column '{col}'")
        
        consistency_score = max(0, 1 - len(consistency_issues) / len(data.columns))
        quality_report["consistency"] = {
            "score": consistency_score,
            "issues": consistency_issues
        }
        
        # Validity assessment (basic checks)
        validity_issues = []
        
        for col in data.select_dtypes(include=[np.number]).columns:
            # Check for outliers using IQR method
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = data[(data[col] < Q1 - 1.5 * IQR) | (data[col] > Q3 + 1.5 * IQR)][col]
            
            if len(outliers) > len(data) * 0.05:  # More than 5% outliers
                validity_issues.append(f"High number of outliers in column '{col}' ({len(outliers)} values)")
        
        validity_score = max(0, 1 - len(validity_issues) / len(data.columns))
        quality_report["validity"] = {
            "score": validity_score,
            "issues": validity_issues
        }
        
        # Overall score
        quality_report["overall_score"] = np.mean([
            completeness_score,
            uniqueness_score,
            consistency_score,
            validity_score
        ])
        
        # Generate recommendations
        if completeness_score < 0.8:
            quality_report["recommendations"].append("Consider handling missing values before anonymization")
        
        if duplicate_rows > 0:
            quality_report["recommendations"].append(f"Remove {duplicate_rows} duplicate rows")
        
        if consistency_issues:
            quality_report["recommendations"].append("Address data consistency issues")
        
        if not quality_report["recommendations"]:
            quality_report["recommendations"].append("Data quality looks good")
        
        return quality_report
    
    async def _create_data_preview(self, data: pd.DataFrame, max_rows: int = 10) -> Dict[str, Any]:
        """Create a safe data preview"""
        
        preview_data = data.head(max_rows).copy()
        
        # Replace potentially sensitive data with placeholders
        for col in preview_data.select_dtypes(include=['object']).columns:
            # Check if column might contain sensitive data
            if any(keyword in col.lower() for keyword in ['name', 'email', 'phone', 'address', 'id']):
                preview_data[col] = '[REDACTED]'
        
        # Convert to safe format
        preview = {
            "columns": data.columns.tolist(),
            "data_types": data.dtypes.astype(str).to_dict(),
            "sample_rows": preview_data.to_dict('records'),
            "total_rows": len(data),
            "preview_rows": len(preview_data)
        }
        
        return preview
    
    async def get_file_data(self, file_id: str) -> Optional[pd.DataFrame]:
        """Retrieve processed file data"""
        
        # Find file in processed directory
        processed_file = os.path.join(self.processed_dir, f"{file_id}.pkl")
        
        if os.path.exists(processed_file):
            try:
                return pd.read_pickle(processed_file)
            except Exception as e:
                logger.error(f"Failed to load processed file {file_id}: {str(e)}")
        
        return None
    
    async def save_processed_data(self, file_id: str, data: pd.DataFrame) -> bool:
        """Save processed data for future use"""
        
        try:
            processed_file = os.path.join(self.processed_dir, f"{file_id}.pkl")
            data.to_pickle(processed_file)
            
            # Set restrictive permissions
            os.chmod(processed_file, 0o600)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save processed data for {file_id}: {str(e)}")
            return False
    
    async def cleanup_file(self, file_id: str):
        """Clean up file and associated data"""
        
        try:
            # Remove uploaded file
            for file in os.listdir(self.upload_dir):
                if file.startswith(file_id):
                    os.remove(os.path.join(self.upload_dir, file))
            
            # Remove processed file
            processed_file = os.path.join(self.processed_dir, f"{file_id}.pkl")
            if os.path.exists(processed_file):
                os.remove(processed_file)
            
            logger.info(f"Cleaned up files for {file_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup files for {file_id}: {str(e)}")
