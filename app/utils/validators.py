import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, validator, ValidationError
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Custom validation error"""
    pass

class DataValidator:
    """Comprehensive data validation utilities"""
    
    def __init__(self):
        # Common patterns for data validation
        self.email_pattern = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
        self.phone_pattern = re.compile(r'^\+?1?-?\.?\s?\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})$')
        self.ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
        self.date_patterns = [
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
            r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY
            r'^\d{2}-\d{2}-\d{4}$',  # MM-DD-YYYY
        ]
        
        # Sensitive data indicators
        self.sensitive_keywords = [
            'ssn', 'social', 'security', 'passport', 'license', 'credit',
            'card', 'account', 'bank', 'routing', 'pin', 'password',
            'email', 'phone', 'address', 'name', 'birth', 'age'
        ]
        
        # Statistical validation thresholds
        self.outlier_threshold = 3.0  # Standard deviations
        self.missing_threshold = 0.5  # 50% missing values
        self.cardinality_threshold = 0.95  # 95% unique values
    
    async def validate_dataset(
        self,
        data: pd.DataFrame,
        schema: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive dataset validation
        
        Args:
            data: Dataset to validate
            schema: Optional schema definition
            
        Returns:
            Validation result with errors and warnings
        """
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "data_profile": {},
            "recommendations": []
        }
        
        try:
            # Basic structure validation
            structure_validation = await self._validate_structure(data)
            validation_result.update(structure_validation)
            
            # Data type validation
            type_validation = await self._validate_data_types(data)
            validation_result["warnings"].extend(type_validation["warnings"])
            
            # Data quality validation
            quality_validation = await self._validate_data_quality(data)
            validation_result["warnings"].extend(quality_validation["warnings"])
            
            # Privacy sensitivity detection
            privacy_validation = await self._detect_sensitive_data(data)
            validation_result["data_profile"]["sensitive_columns"] = privacy_validation["sensitive_columns"]
            validation_result["warnings"].extend(privacy_validation["warnings"])
            
            # Statistical validation
            stats_validation = await self._validate_statistics(data)
            validation_result["warnings"].extend(stats_validation["warnings"])
            
            # Schema validation if provided
            if schema:
                schema_validation = await self._validate_schema(data, schema)
                validation_result["errors"].extend(schema_validation["errors"])
                validation_result["warnings"].extend(schema_validation["warnings"])
            
            # Generate recommendations
            validation_result["recommendations"] = await self._generate_recommendations(
                data, validation_result
            )
            
            # Final validation status
            validation_result["valid"] = len(validation_result["errors"]) == 0
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            validation_result["valid"] = False
            validation_result["errors"].append(f"Validation failed: {str(e)}")
        
        return validation_result
    
    async def _validate_structure(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate basic dataset structure"""
        
        errors = []
        warnings = []
        data_profile = {}
        
        # Check if dataset is empty
        if data.empty:
            errors.append("Dataset is empty")
            return {"errors": errors, "warnings": warnings, "data_profile": data_profile}
        
        # Check dimensions
        rows, cols = data.shape
        data_profile["shape"] = {"rows": rows, "columns": cols}
        
        if rows < 2:
            warnings.append("Dataset has very few rows (< 2)")
        elif rows < 10:
            warnings.append("Dataset has few rows (< 10) - results may not be reliable")
        
        if cols < 2:
            warnings.append("Dataset has very few columns (< 2)")
        
        # Check for unnamed columns
        unnamed_cols = [col for col in data.columns if str(col).startswith('Unnamed:')]
        if unnamed_cols:
            warnings.append(f"Found {len(unnamed_cols)} unnamed columns")
        
        # Check for duplicate column names
        duplicate_cols = data.columns[data.columns.duplicated()].tolist()
        if duplicate_cols:
            warnings.append(f"Found duplicate column names: {duplicate_cols}")
        
        return {"errors": errors, "warnings": warnings, "data_profile": data_profile}
    
    async def _validate_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate and analyze data types"""
        
        warnings = []
        
        # Check for mixed data types in columns
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if column contains mixed types
                non_null_values = data[col].dropna()
                if len(non_null_values) > 0:
                    sample_types = set(type(val).__name__ for val in non_null_values.head(100))
                    if len(sample_types) > 1:
                        warnings.append(f"Column '{col}' contains mixed data types: {sample_types}")
        
        # Check for potential data type conversions
        for col in data.select_dtypes(include=['object']).columns:
            if await self._could_be_numeric(data[col]):
                warnings.append(f"Column '{col}' might be numeric but stored as text")
            elif await self._could_be_datetime(data[col]):
                warnings.append(f"Column '{col}' might be datetime but stored as text")
        
        return {"warnings": warnings}
    
    async def _could_be_numeric(self, series: pd.Series) -> bool:
        """Check if a text column could be converted to numeric"""
        
        non_null_values = series.dropna().astype(str)
        if len(non_null_values) == 0:
            return False
        
        # Try to convert a sample to numeric
        sample = non_null_values.head(100)
        numeric_count = 0
        
        for val in sample:
            try:
                float(val.replace(',', '').replace('$', '').replace('%', ''))
                numeric_count += 1
            except:
                continue
        
        return numeric_count / len(sample) > 0.8
    
    async def _could_be_datetime(self, series: pd.Series) -> bool:
        """Check if a text column could be converted to datetime"""
        
        non_null_values = series.dropna().astype(str)
        if len(non_null_values) == 0:
            return False
        
        sample = non_null_values.head(100)
        date_count = 0
        
        for val in sample:
            for pattern in self.date_patterns:
                if re.match(pattern, val):
                    date_count += 1
                    break
        
        return date_count / len(sample) > 0.8
    
    async def _validate_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality aspects"""
        
        warnings = []
        
        # Check for excessive missing values
        missing_ratios = data.isnull().sum() / len(data)
        high_missing_cols = missing_ratios[missing_ratios > self.missing_threshold].index.tolist()
        
        if high_missing_cols:
            warnings.append(f"Columns with >50% missing values: {high_missing_cols}")
        
        # Check for constant columns
        constant_cols = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            warnings.append(f"Constant columns (no variation): {constant_cols}")
        
        # Check for high cardinality columns
        high_cardinality_cols = []
        for col in data.columns:
            uniqueness_ratio = data[col].nunique() / len(data)
            if uniqueness_ratio > self.cardinality_threshold:
                high_cardinality_cols.append(col)
        
        if high_cardinality_cols:
            warnings.append(f"High cardinality columns (>95% unique): {high_cardinality_cols}")
        
        # Check for duplicate rows
        duplicate_count = data.duplicated().sum()
        if duplicate_count > 0:
            warnings.append(f"Found {duplicate_count} duplicate rows")
        
        return {"warnings": warnings}
    
    async def _detect_sensitive_data(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect potentially sensitive data"""
        
        sensitive_columns = []
        warnings = []
        
        for col in data.columns:
            col_lower = col.lower()
            
            # Check column name for sensitive keywords
            if any(keyword in col_lower for keyword in self.sensitive_keywords):
                sensitive_columns.append({
                    "column": col,
                    "reason": "sensitive_keyword_in_name",
                    "confidence": "high"
                })
                continue
            
            # Check data patterns for sensitive information
            if data[col].dtype == 'object':
                sample_values = data[col].dropna().head(100).astype(str)
                
                # Check for email patterns
                email_matches = sum(1 for val in sample_values if self.email_pattern.match(val))
                if email_matches > len(sample_values) * 0.5:
                    sensitive_columns.append({
                        "column": col,
                        "reason": "email_pattern",
                        "confidence": "high"
                    })
                    continue
                
                # Check for phone patterns
                phone_matches = sum(1 for val in sample_values if self.phone_pattern.match(val))
                if phone_matches > len(sample_values) * 0.5:
                    sensitive_columns.append({
                        "column": col,
                        "reason": "phone_pattern",
                        "confidence": "high"
                    })
                    continue
                
                # Check for potential IDs (high uniqueness + alphanumeric)
                uniqueness = data[col].nunique() / len(data)
                if uniqueness > 0.9:
                    alphanumeric_pattern = all(
                        str(val).replace('-', '').replace('_', '').isalnum() 
                        for val in sample_values[:10]
                    )
                    if alphanumeric_pattern:
                        sensitive_columns.append({
                            "column": col,
                            "reason": "potential_identifier",
                            "confidence": "medium"
                        })
        
        if sensitive_columns:
            high_confidence = [col for col in sensitive_columns if col["confidence"] == "high"]
            if high_confidence:
                warnings.append(f"Detected {len(high_confidence)} columns with likely sensitive data")
            
            medium_confidence = [col for col in sensitive_columns if col["confidence"] == "medium"]
            if medium_confidence:
                warnings.append(f"Detected {len(medium_confidence)} columns with potentially sensitive data")
        
        return {"sensitive_columns": sensitive_columns, "warnings": warnings}
    
    async def _validate_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate statistical properties"""
        
        warnings = []
        
        # Check for outliers in numeric columns
        for col in data.select_dtypes(include=[np.number]).columns:
            values = data[col].dropna()
            if len(values) < 10:
                continue
            
            # Z-score method
            z_scores = np.abs((values - values.mean()) / values.std())
            outliers = values[z_scores > self.outlier_threshold]
            
            if len(outliers) > len(values) * 0.05:  # More than 5% outliers
                warnings.append(f"Column '{col}' has {len(outliers)} potential outliers")
        
        # Check for skewed distributions
        for col in data.select_dtypes(include=[np.number]).columns:
            values = data[col].dropna()
            if len(values) < 10:
                continue
            
            skewness = values.skew()
            if abs(skewness) > 2:
                warnings.append(f"Column '{col}' is highly skewed (skewness: {skewness:.2f})")
        
        return {"warnings": warnings}
    
    async def _validate_schema(
        self,
        data: pd.DataFrame,
        schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate data against provided schema"""
        
        errors = []
        warnings = []
        
        # Check required columns
        required_columns = schema.get("required_columns", [])
        missing_columns = set(required_columns) - set(data.columns)
        
        if missing_columns:
            errors.append(f"Missing required columns: {list(missing_columns)}")
        
        # Check column types
        expected_types = schema.get("column_types", {})
        for col, expected_type in expected_types.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if not await self._types_compatible(actual_type, expected_type):
                    warnings.append(f"Column '{col}' type mismatch: expected {expected_type}, got {actual_type}")
        
        # Check value constraints
        constraints = schema.get("constraints", {})
        for col, constraint in constraints.items():
            if col not in data.columns:
                continue
            
            if "min_value" in constraint:
                min_violations = data[data[col] < constraint["min_value"]]
                if len(min_violations) > 0:
                    errors.append(f"Column '{col}' has {len(min_violations)} values below minimum {constraint['min_value']}")
            
            if "max_value" in constraint:
                max_violations = data[data[col] > constraint["max_value"]]
                if len(max_violations) > 0:
                    errors.append(f"Column '{col}' has {len(max_violations)} values above maximum {constraint['max_value']}")
            
            if "allowed_values" in constraint:
                invalid_values = data[~data[col].isin(constraint["allowed_values"])]
                if len(invalid_values) > 0:
                    errors.append(f"Column '{col}' has {len(invalid_values)} invalid values")
        
        return {"errors": errors, "warnings": warnings}
    
    async def _types_compatible(self, actual_type: str, expected_type: str) -> bool:
        """Check if data types are compatible"""
        
        type_compatibility = {
            "int64": ["integer", "int", "numeric"],
            "float64": ["float", "numeric", "decimal"],
            "object": ["string", "text", "categorical"],
            "bool": ["boolean", "bool"],
            "datetime64[ns]": ["datetime", "date", "timestamp"]
        }
        
        compatible_types = type_compatibility.get(actual_type, [actual_type])
        return expected_type.lower() in [t.lower() for t in compatible_types]
    
    async def _generate_recommendations(
        self,
        data: pd.DataFrame,
        validation_result: Dict[str, Any]
    ) -> List[str]:
        """Generate data improvement recommendations"""
        
        recommendations = []
        
        # Handle missing values
        if any("missing values" in warning for warning in validation_result["warnings"]):
            recommendations.append("Consider handling missing values using imputation or removal")
        
        # Handle data type issues
        if any("mixed data types" in warning for warning in validation_result["warnings"]):
            recommendations.append("Review and fix mixed data type columns")
        
        # Handle outliers
        if any("outliers" in warning for warning in validation_result["warnings"]):
            recommendations.append("Investigate and handle outlier values")
        
        # Handle sensitive data
        sensitive_columns = validation_result.get("data_profile", {}).get("sensitive_columns", [])
        if sensitive_columns:
            recommendations.append("Review sensitive data columns and apply appropriate anonymization")
        
        # Handle duplicate data
        if any("duplicate" in warning for warning in validation_result["warnings"]):
            recommendations.append("Remove duplicate rows to improve data quality")
        
        # Handle high cardinality
        if any("cardinality" in warning for warning in validation_result["warnings"]):
            recommendations.append("Consider grouping or binning high cardinality columns")
        
        # General recommendation
        if not recommendations:
            recommendations.append("Data validation passed - dataset appears ready for processing")
        
        return recommendations
    
    async def validate_anonymization_parameters(
        self,
        epsilon: float,
        delta: float,
        method: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Validate anonymization parameters"""
        
        errors = []
        warnings = []
        
        # Validate epsilon
        if epsilon <= 0:
            errors.append("Epsilon must be positive")
        elif epsilon > 10:
            warnings.append("High epsilon value may provide weak privacy protection")
        elif epsilon < 0.1:
            warnings.append("Very low epsilon may significantly reduce data utility")
        
        # Validate delta
        if delta <= 0:
            errors.append("Delta must be positive")
        elif delta >= 1:
            errors.append("Delta must be less than 1")
        elif delta > 1e-3:
            warnings.append("High delta value may weaken privacy guarantees")
        
        # Validate method
        valid_methods = ["sdg", "dp", "sdc", "sdg_dp", "sdg_sdc", "dp_sdc", "full"]
        if method not in valid_methods:
            errors.append(f"Invalid method '{method}'. Valid methods: {valid_methods}")
        
        # Validate method-specific parameters
        if method in ["sdg", "sdg_dp", "sdg_sdc", "full"]:
            epochs = kwargs.get("synthetic_epochs", 300)
            batch_size = kwargs.get("synthetic_batch_size", 500)
            
            if epochs < 50:
                warnings.append("Low epoch count may result in poor synthetic data quality")
            elif epochs > 1000:
                warnings.append("High epoch count may lead to overfitting")
            
            if batch_size < 32:
                warnings.append("Small batch size may slow training")
            elif batch_size > 2048:
                warnings.append("Large batch size may cause memory issues")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    async def validate_file_upload(
        self,
        filename: str,
        file_size: int,
        content_type: str
    ) -> Dict[str, Any]:
        """Validate file upload parameters"""
        
        errors = []
        warnings = []
        
        # Validate filename
        if not filename:
            errors.append("Filename is required")
        else:
            # Check for dangerous characters
            dangerous_chars = ['<', '>', ':', '"', '|', '?', '*', '\\', '/']
            if any(char in filename for char in dangerous_chars):
                errors.append("Filename contains invalid characters")
            
            # Check extension
            from app.core.config import settings
            if not any(filename.lower().endswith(ext) for ext in settings.allowed_extensions):
                errors.append(f"File type not supported. Allowed: {settings.allowed_extensions}")
        
        # Validate file size
        if file_size <= 0:
            errors.append("File is empty")
        elif file_size > settings.max_file_size:
            errors.append(f"File too large ({file_size} bytes). Maximum: {settings.max_file_size} bytes")
        elif file_size > settings.max_file_size * 0.8:
            warnings.append("File is close to size limit")
        
        # Validate content type
        allowed_content_types = [
            'text/csv',
            'application/vnd.ms-excel',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            'application/pdf'
        ]
        
        if content_type not in allowed_content_types:
            warnings.append(f"Unexpected content type: {content_type}")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }

# Validation decorators and utilities
def validate_dataframe(func):
    """Decorator to validate DataFrame inputs"""
    async def wrapper(*args, **kwargs):
        # Find DataFrame arguments
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                if arg.empty:
                    raise ValidationError("DataFrame cannot be empty")
        
        for key, value in kwargs.items():
            if isinstance(value, pd.DataFrame):
                if value.empty:
                    raise ValidationError(f"DataFrame '{key}' cannot be empty")
        
        return await func(*args, **kwargs)
    return wrapper

def validate_numeric_range(min_val: float = None, max_val: float = None):
    """Decorator to validate numeric parameters"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            for key, value in kwargs.items():
                if isinstance(value, (int, float)):
                    if min_val is not None and value < min_val:
                        raise ValidationError(f"Parameter '{key}' must be >= {min_val}")
                    if max_val is not None and value > max_val:
                        raise ValidationError(f"Parameter '{key}' must be <= {max_val}")
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Global validator instance
data_validator = DataValidator()
