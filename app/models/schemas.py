from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime
import pandas as pd

class FileUploadResponse(BaseModel):
    """Response model for file upload"""
    message: str
    file_id: str
    filename: str
    size: int
    columns: List[str]
    rows: int

class AnonymizationRequest(BaseModel):
    """Request model for anonymization"""
    file_id: str
    epsilon: Optional[float] = Field(default=1.0, ge=0.1, le=10.0)
    delta: Optional[float] = Field(default=1e-5, ge=1e-8, le=1e-3)
    method: str = Field(default="sdg_dp", regex="^(sdg|dp|sdc|sdg_dp|sdg_sdc|dp_sdc|full)$")
    quasi_identifiers: Optional[List[str]] = None
    sensitive_attributes: Optional[List[str]] = None
    utility_metrics: Optional[List[str]] = Field(default=["statistical_similarity", "ml_utility"])
    
    @validator('method')
    def validate_method(cls, v):
        valid_methods = ["sdg", "dp", "sdc", "sdg_dp", "sdg_sdc", "dp_sdc", "full"]
        if v not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        return v

class AnonymizationResponse(BaseModel):
    """Response model for anonymization"""
    success: bool
    message: str
    anonymized_data_id: Optional[str] = None
    privacy_metrics: Optional[Dict[str, Any]] = None
    utility_metrics: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    warnings: Optional[List[str]] = None

class PrivacyMetrics(BaseModel):
    """Privacy assessment metrics"""
    epsilon_used: float
    delta_used: float
    privacy_budget_remaining: float
    re_identification_risk: float
    membership_inference_risk: float
    attribute_inference_risk: float

class UtilityMetrics(BaseModel):
    """Data utility metrics"""
    statistical_similarity: float
    ml_utility_score: float
    correlation_preservation: float
    distribution_similarity: float
    data_completeness: float

class AttackSimulationRequest(BaseModel):
    """Request model for attack simulation"""
    original_data_id: str
    anonymized_data_id: str
    attack_types: List[str] = Field(default=["linkage", "membership", "attribute"])
    auxiliary_data_ratio: Optional[float] = Field(default=0.1, ge=0.01, le=0.5)
    
    @validator('attack_types')
    def validate_attack_types(cls, v):
        valid_attacks = ["linkage", "membership", "attribute"]
        for attack in v:
            if attack not in valid_attacks:
                raise ValueError(f"Attack type must be one of {valid_attacks}")
        return v

class AttackSimulationResponse(BaseModel):
    """Response model for attack simulation"""
    success: bool
    attack_results: Dict[str, Dict[str, float]]
    overall_risk_score: float
    recommendations: List[str]

class OptimizationRequest(BaseModel):
    """Request model for privacy-utility optimization"""
    file_id: str
    target_privacy_level: str = Field(default="medium", regex="^(low|medium|high|maximum)$")
    utility_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    privacy_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    max_iterations: Optional[int] = Field(default=50, ge=10, le=200)
    
    @validator('utility_weight', 'privacy_weight')
    def weights_sum_to_one(cls, v, values, field):
        if 'utility_weight' in values and field.name == 'privacy_weight':
            if abs(v + values['utility_weight'] - 1.0) > 0.001:
                raise ValueError("utility_weight + privacy_weight must equal 1.0")
        return v

class OptimizationResponse(BaseModel):
    """Response model for optimization"""
    success: bool
    optimal_parameters: Dict[str, Any]
    expected_privacy_score: float
    expected_utility_score: float
    optimization_history: List[Dict[str, Any]]

class AuditLogEntry(BaseModel):
    """Audit log entry model"""
    timestamp: datetime
    user_id: Optional[str] = None
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: Optional[str] = None
    success: bool = True

class SystemStatus(BaseModel):
    """System status model"""
    status: str
    uptime: float
    active_sessions: int
    processed_files: int
    total_privacy_budget_used: float
    system_load: Dict[str, float]
