"""
Database models for SafeData 2.0
"""
import datetime
from app import db
from sqlalchemy import Column, Integer, String, DateTime, Float, JSON, Text, Boolean


class UploadedFile(db.Model):
    """Model for tracking uploaded files"""
    __tablename__ = 'uploaded_files'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(100), nullable=False)
    upload_timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    status = Column(String(50), default='uploaded', nullable=False)
    file_path = Column(String(500))
    file_metadata = Column(JSON)


class AnonymizationResult(db.Model):
    """Model for storing anonymization results"""
    __tablename__ = 'anonymization_results'
    
    id = Column(Integer, primary_key=True)
    result_id = Column(String(100), unique=True, nullable=False)
    original_file_id = Column(Integer, db.ForeignKey('uploaded_files.id'), nullable=False)
    method = Column(String(100), nullable=False)
    parameters = Column(JSON)
    
    # Privacy metrics
    privacy_score = Column(Float)
    utility_score = Column(Float)
    epsilon_used = Column(Float)
    delta_used = Column(Float)
    reidentification_risk = Column(Float)
    
    # Utility metrics
    statistical_similarity = Column(Float)
    correlation_preservation = Column(Float)
    data_completeness = Column(Float)
    
    # Processing info
    execution_time = Column(Float)
    created_timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    status = Column(String(50), default='processing', nullable=False)
    
    # Results storage
    anonymized_data_path = Column(String(500))
    report_data = Column(JSON)


class PrivacyAssessment(db.Model):
    """Model for privacy assessment results"""
    __tablename__ = 'privacy_assessments'
    
    id = Column(Integer, primary_key=True)
    result_id = Column(Integer, db.ForeignKey('anonymization_results.id'), nullable=False)
    assessment_type = Column(String(100), nullable=False)
    overall_score = Column(Float)
    risk_level = Column(String(50))
    
    # Attack simulation results
    linkage_attack_score = Column(Float)
    membership_inference_score = Column(Float)
    attribute_inference_score = Column(Float)
    
    # Assessment data
    assessment_data = Column(JSON)
    recommendations = Column(JSON)
    created_timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)


class AuditLog(db.Model):
    """Model for audit logging"""
    __tablename__ = 'audit_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    event_type = Column(String(100), nullable=False)
    user_id = Column(String(100), default='anonymous')
    session_id = Column(String(200))
    
    # Event details
    action = Column(String(200), nullable=False)
    resource_type = Column(String(100))
    resource_id = Column(String(100))
    
    # Privacy budget tracking
    epsilon_consumed = Column(Float, default=0.0)
    delta_consumed = Column(Float, default=0.0)
    
    # Additional data
    ip_address = Column(String(45))
    user_agent = Column(Text)
    details = Column(JSON)
    success = Column(Boolean, default=True)


class SystemSettings(db.Model):
    """Model for system configuration"""
    __tablename__ = 'system_settings'
    
    id = Column(Integer, primary_key=True)
    key = Column(String(100), unique=True, nullable=False)
    value = Column(Text)
    value_type = Column(String(50), default='string')
    description = Column(Text)
    created_timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    updated_timestamp = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)


class PrivacyBudget(db.Model):
    """Model for tracking privacy budget usage"""
    __tablename__ = 'privacy_budgets'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(200), nullable=False)
    user_id = Column(String(100), default='anonymous')
    
    # Budget limits
    total_epsilon = Column(Float, default=10.0)
    total_delta = Column(Float, default=1e-5)
    
    # Current usage
    used_epsilon = Column(Float, default=0.0)
    used_delta = Column(Float, default=0.0)
    
    # Tracking
    created_timestamp = Column(DateTime, default=datetime.datetime.utcnow, nullable=False)
    last_used_timestamp = Column(DateTime)
    is_active = Column(Boolean, default=True)