from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime

from app.models.schemas import (
    AttackSimulationRequest,
    AttackSimulationResponse,
    OptimizationRequest,
    OptimizationResponse,
    PrivacyMetrics
)
from app.services.attack_simulation import AttackSimulator
from app.services.optimization import PrivacyUtilityOptimizer
from app.services.differential_privacy import DifferentialPrivacyEngine
from app.services.audit import AuditService
from app.utils.file_handler import FileProcessor
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Service instances
attack_simulator = AttackSimulator()
privacy_optimizer = PrivacyUtilityOptimizer()
dp_engine = DifferentialPrivacyEngine()
audit_service = AuditService()
file_processor = FileProcessor()

# External storage references (in production, use proper database)
from app.routers.anonymize import file_storage, anonymization_results

@router.post("/simulate-attacks", response_model=AttackSimulationResponse)
async def simulate_privacy_attacks(
    request: AttackSimulationRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = None
):
    """
    Simulate privacy attacks on anonymized data
    """
    
    start_time = datetime.now()
    
    try:
        # Validate request
        if request.original_data_id not in file_storage:
            raise HTTPException(status_code=404, detail="Original data not found")
        
        if request.anonymized_data_id not in anonymization_results:
            raise HTTPException(status_code=404, detail="Anonymized data not found")
        
        # Load original data
        original_data = await file_processor.get_file_data(request.original_data_id)
        if original_data is None:
            raise HTTPException(status_code=404, detail="Original data not available")
        
        # Load anonymized data
        anonymized_result = anonymization_results[request.anonymized_data_id]
        anonymized_data = anonymized_result["anonymized_data"]
        
        # Check user access
        original_file_info = file_storage[request.original_data_id]
        if user_id and original_file_info.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        if user_id and anonymized_result.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Run attack simulation
        attack_results = await attack_simulator.simulate_attacks(
            original_data=original_data,
            anonymized_data=anonymized_data,
            attack_types=request.attack_types,
            auxiliary_data_ratio=request.auxiliary_data_ratio
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Store attack results
        attack_id = f"attack_{request.anonymized_data_id}_{datetime.now().timestamp()}"
        
        # Audit log
        background_tasks.add_task(
            audit_service.log_event,
            action="attack_simulation",
            resource="privacy_attack",
            details={
                "attack_id": attack_id,
                "original_data_id": request.original_data_id,
                "anonymized_data_id": request.anonymized_data_id,
                "attack_types": request.attack_types,
                "overall_risk_score": attack_results.get("overall_risk_score", 0)
            },
            user_id=user_id,
            execution_time=execution_time
        )
        
        return AttackSimulationResponse(
            success=True,
            attack_results=attack_results,
            overall_risk_score=attack_results.get("overall_risk_score", 0),
            recommendations=attack_results.get("recommendations", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Attack simulation failed: {str(e)}")
        await audit_service.log_event(
            action="attack_simulation_error",
            resource="privacy_attack",
            details={"error": str(e)},
            user_id=user_id,
            success=False
        )
        raise HTTPException(status_code=500, detail=f"Attack simulation failed: {str(e)}")

@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_privacy_utility(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = None
):
    """
    Optimize privacy-utility trade-off using Bayesian optimization
    """
    
    start_time = datetime.now()
    
    try:
        # Validate file exists
        if request.file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = file_storage[request.file_id]
        
        # Check user access
        if user_id and file_info.get("user_id") != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Load data
        data = await file_processor.get_file_data(request.file_id)
        if data is None:
            raise HTTPException(status_code=404, detail="File data not available")
        
        # Run optimization
        optimization_result = await privacy_optimizer.optimize_parameters(
            data=data,
            target_privacy_level=request.target_privacy_level,
            utility_weight=request.utility_weight,
            privacy_weight=request.privacy_weight,
            max_iterations=request.max_iterations
        )
        
        if not optimization_result["success"]:
            raise HTTPException(
                status_code=500, 
                detail=optimization_result.get("error", "Optimization failed")
            )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Store optimization results
        optimization_id = f"opt_{request.file_id}_{datetime.now().timestamp()}"
        
        # Audit log
        background_tasks.add_task(
            audit_service.log_event,
            action="privacy_optimization",
            resource="optimization",
            details={
                "optimization_id": optimization_id,
                "file_id": request.file_id,
                "target_privacy_level": request.target_privacy_level,
                "optimal_epsilon": optimization_result["optimal_parameters"].get("epsilon"),
                "expected_privacy_score": optimization_result["expected_privacy_score"],
                "expected_utility_score": optimization_result["expected_utility_score"]
            },
            user_id=user_id,
            execution_time=execution_time
        )
        
        return OptimizationResponse(
            success=True,
            optimal_parameters=optimization_result["optimal_parameters"],
            expected_privacy_score=optimization_result["expected_privacy_score"],
            expected_utility_score=optimization_result["expected_utility_score"],
            optimization_history=optimization_result.get("optimization_history", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Privacy optimization failed: {str(e)}")
        await audit_service.log_event(
            action="optimization_error",
            resource="optimization",
            details={"error": str(e), "file_id": request.file_id},
            user_id=user_id,
            success=False
        )
        raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")

@router.get("/budget/{session_id}")
async def get_privacy_budget(session_id: str = "default"):
    """
    Get remaining privacy budget for a session
    """
    
    try:
        budget_info = await dp_engine.get_privacy_budget_remaining(session_id)
        
        return {
            "session_id": session_id,
            "budget_remaining": budget_info,
            "budget_limit": {
                "epsilon": settings.max_epsilon,
                "delta": 1e-3
            },
            "usage_percentage": {
                "epsilon": (budget_info["epsilon_used"] / settings.max_epsilon) * 100,
                "delta": (budget_info["delta_used"] / 1e-3) * 100
            }
        }
        
    except Exception as e:
        logger.error(f"Privacy budget retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Budget retrieval failed: {str(e)}")

@router.get("/assess/{data_id}")
async def assess_privacy_risk(
    data_id: str,
    user_id: Optional[str] = None
):
    """
    Assess privacy risk of anonymized data
    """
    
    try:
        # Check if it's a file or anonymization result
        if data_id in anonymization_results:
            # It's an anonymization result
            result = anonymization_results[data_id]
            
            # Check user access
            if user_id and result.get("user_id") != user_id:
                raise HTTPException(status_code=403, detail="Access denied")
            
            privacy_metrics = result["privacy_metrics"]
            attack_results = result["attack_results"]
            
        elif data_id in file_storage:
            # It's a raw file - assess basic privacy risks
            file_info = file_storage[data_id]
            
            # Check user access
            if user_id and file_info.get("user_id") != user_id:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Load data and perform basic privacy assessment
            data = await file_processor.get_file_data(data_id)
            if data is None:
                raise HTTPException(status_code=404, detail="Data not available")
            
            # Basic privacy assessment (no anonymization applied)
            privacy_metrics = {
                "re_identification_risk": 1.0,  # Maximum risk for raw data
                "privacy_budget_remaining": settings.max_epsilon,
                "data_reduction_ratio": 1.0,
                "column_preservation_ratio": 1.0
            }
            
            attack_results = {
                "overall_risk_score": 1.0,  # Maximum risk for raw data
                "recommendations": [
                    "Raw data has maximum privacy risk",
                    "Apply anonymization before sharing or analysis"
                ]
            }
        else:
            raise HTTPException(status_code=404, detail="Data not found")
        
        # Calculate comprehensive privacy assessment
        privacy_assessment = {
            "data_id": data_id,
            "privacy_score": 1 - privacy_metrics.get("re_identification_risk", 1.0),
            "risk_level": _calculate_risk_level(privacy_metrics, attack_results),
            "privacy_metrics": privacy_metrics,
            "attack_vulnerability": attack_results,
            "compliance_status": _assess_compliance(privacy_metrics),
            "recommendations": _generate_privacy_recommendations(privacy_metrics, attack_results)
        }
        
        return privacy_assessment
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Privacy assessment failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Assessment failed: {str(e)}")

def _calculate_risk_level(privacy_metrics: Dict[str, Any], attack_results: Dict[str, Any]) -> str:
    """Calculate overall privacy risk level"""
    
    risk_factors = []
    
    # Re-identification risk
    reident_risk = privacy_metrics.get("re_identification_risk", 0.5)
    risk_factors.append(reident_risk)
    
    # Attack success rates
    overall_attack_risk = attack_results.get("overall_risk_score", 0.5)
    risk_factors.append(overall_attack_risk)
    
    # Calculate average risk
    avg_risk = sum(risk_factors) / len(risk_factors)
    
    if avg_risk >= 0.8:
        return "CRITICAL"
    elif avg_risk >= 0.6:
        return "HIGH"
    elif avg_risk >= 0.4:
        return "MEDIUM"
    elif avg_risk >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"

def _assess_compliance(privacy_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Assess compliance with privacy standards"""
    
    epsilon_used = privacy_metrics.get("epsilon_used", 0)
    
    compliance_status = {
        "gdpr_compliant": epsilon_used <= 1.0,  # Conservative threshold
        "hipaa_suitable": epsilon_used <= 0.5,  # More strict for health data
        "differential_privacy": epsilon_used > 0,
        "k_anonymity": privacy_metrics.get("k_anonymity", {}).get("k_value", 0) >= 5,
        "overall_compliant": epsilon_used <= 1.0 and privacy_metrics.get("k_anonymity", {}).get("k_value", 0) >= 5
    }
    
    return compliance_status

def _generate_privacy_recommendations(
    privacy_metrics: Dict[str, Any], 
    attack_results: Dict[str, Any]
) -> List[str]:
    """Generate privacy improvement recommendations"""
    
    recommendations = []
    
    # Check re-identification risk
    reident_risk = privacy_metrics.get("re_identification_risk", 0.5)
    if reident_risk > 0.7:
        recommendations.append("High re-identification risk detected - apply stronger anonymization")
    elif reident_risk > 0.4:
        recommendations.append("Moderate re-identification risk - consider additional privacy measures")
    
    # Check privacy budget usage
    epsilon_used = privacy_metrics.get("epsilon_used", 0)
    if epsilon_used > settings.max_epsilon * 0.8:
        recommendations.append("Privacy budget nearly exhausted - limit further operations")
    
    # Check attack vulnerability
    attack_risk = attack_results.get("overall_risk_score", 0.5)
    if attack_risk > 0.6:
        recommendations.append("High vulnerability to privacy attacks - strengthen anonymization parameters")
    
    # Add specific attack recommendations
    attack_recommendations = attack_results.get("recommendations", [])
    recommendations.extend(attack_recommendations[:3])  # Limit to top 3
    
    if not recommendations:
        recommendations.append("Privacy protection appears adequate for current threat model")
    
    return recommendations

@router.get("/metrics/summary")
async def get_privacy_metrics_summary(user_id: Optional[str] = None):
    """
    Get summary of privacy metrics across all user data
    """
    
    try:
        # Filter results by user
        if user_id:
            user_results = {
                result_id: result for result_id, result in anonymization_results.items()
                if result.get("user_id") == user_id
            }
        else:
            user_results = anonymization_results
        
        if not user_results:
            return {
                "total_anonymizations": 0,
                "average_privacy_score": 0,
                "average_utility_score": 0,
                "privacy_budget_used": 0,
                "methods_used": {},
                "risk_distribution": {}
            }
        
        # Calculate aggregated metrics
        privacy_scores = []
        utility_scores = []
        epsilon_used = 0
        delta_used = 0
        methods_count = {}
        risk_levels = []
        
        for result in user_results.values():
            # Privacy scores
            privacy_metric = result["privacy_metrics"]
            utility_metric = result["utility_metrics"]
            
            privacy_score = 1 - privacy_metric.get("re_identification_risk", 0.5)
            privacy_scores.append(privacy_score)
            
            utility_score = utility_metric.get("overall_utility_score", 0.5)
            utility_scores.append(utility_score)
            
            # Privacy budget
            epsilon_used += privacy_metric.get("epsilon_used", 0)
            delta_used += privacy_metric.get("delta_used", 0)
            
            # Methods
            method = result["method"]
            methods_count[method] = methods_count.get(method, 0) + 1
            
            # Risk levels
            risk_level = _calculate_risk_level(privacy_metric, result["attack_results"])
            risk_levels.append(risk_level)
        
        # Risk distribution
        risk_distribution = {}
        for risk in risk_levels:
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        
        summary = {
            "total_anonymizations": len(user_results),
            "average_privacy_score": sum(privacy_scores) / len(privacy_scores),
            "average_utility_score": sum(utility_scores) / len(utility_scores),
            "privacy_budget_used": {
                "epsilon": epsilon_used,
                "delta": delta_used
            },
            "methods_used": methods_count,
            "risk_distribution": risk_distribution,
            "compliance_rate": sum(1 for score in privacy_scores if score > 0.7) / len(privacy_scores),
            "recommendations": _generate_summary_recommendations(privacy_scores, utility_scores, risk_levels)
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Privacy metrics summary failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {str(e)}")

def _generate_summary_recommendations(
    privacy_scores: List[float],
    utility_scores: List[float],
    risk_levels: List[str]
) -> List[str]:
    """Generate recommendations based on summary metrics"""
    
    recommendations = []
    
    avg_privacy = sum(privacy_scores) / len(privacy_scores) if privacy_scores else 0
    avg_utility = sum(utility_scores) / len(utility_scores) if utility_scores else 0
    
    high_risk_count = sum(1 for risk in risk_levels if risk in ["CRITICAL", "HIGH"])
    
    if avg_privacy < 0.5:
        recommendations.append("Overall privacy protection is below recommended levels")
    
    if avg_utility < 0.5:
        recommendations.append("Data utility preservation could be improved")
    
    if high_risk_count > len(risk_levels) * 0.3:
        recommendations.append("High proportion of results have elevated privacy risks")
    
    if avg_privacy > 0.8 and avg_utility > 0.8:
        recommendations.append("Excellent balance between privacy and utility achieved")
    
    if not recommendations:
        recommendations.append("Privacy metrics are within acceptable ranges")
    
    return recommendations

@router.get("/standards")
async def get_privacy_standards():
    """
    Get information about privacy standards and compliance requirements
    """
    
    standards = {
        "differential_privacy": {
            "description": "Formal privacy framework providing mathematical guarantees",
            "epsilon_recommendations": {
                "high_privacy": "ε ≤ 0.1",
                "medium_privacy": "0.1 < ε ≤ 1.0",
                "low_privacy": "1.0 < ε ≤ 10.0"
            },
            "delta_recommendations": "δ ≤ 1/n where n is dataset size"
        },
        "k_anonymity": {
            "description": "Each record is indistinguishable from at least k-1 other records",
            "minimum_k": 5,
            "recommended_k": 10,
            "limitations": "Vulnerable to homogeneity and background knowledge attacks"
        },
        "gdpr_compliance": {
            "description": "EU General Data Protection Regulation requirements",
            "key_principles": [
                "Data minimization",
                "Purpose limitation", 
                "Storage limitation",
                "Accuracy",
                "Integrity and confidentiality"
            ],
            "anonymization_requirement": "Data must be truly anonymous - no re-identification possible"
        },
        "hipaa_compliance": {
            "description": "US Health Insurance Portability and Accountability Act",
            "safe_harbor_method": "Remove 18 specific identifiers",
            "expert_determination": "Statistical/scientific analysis to minimize re-identification risk",
            "recommended_threshold": "Very low re-identification risk (< 0.05)"
        },
        "mospi_guidelines": {
            "description": "Ministry of Statistics & Programme Implementation guidelines for India",
            "microdata_release": "Cell suppression for small counts",
            "disclosure_control": "Statistical disclosure control methods required",
            "risk_threshold": "Acceptable risk levels for Indian statistical releases"
        }
    }
    
    return {
        "privacy_standards": standards,
        "current_settings": {
            "default_epsilon": settings.default_epsilon,
            "max_epsilon": settings.max_epsilon,
            "default_delta": settings.default_delta,
            "minimum_k_anonymity": 3
        },
        "recommendations": [
            "Use ε ≤ 1.0 for strong privacy protection",
            "Combine multiple anonymization techniques for enhanced protection",
            "Perform regular privacy audits and attack simulations",
            "Document privacy measures for compliance reporting"
        ]
    }
