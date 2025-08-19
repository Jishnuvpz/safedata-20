from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta

from app.services.audit import AuditService
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Service instance
audit_service = AuditService()

@router.get("/logs")
async def get_audit_logs(
    start_time: Optional[str] = Query(None, description="Start time in ISO format"),
    end_time: Optional[str] = Query(None, description="End time in ISO format"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    action: Optional[str] = Query(None, description="Filter by action type"),
    resource: Optional[str] = Query(None, description="Filter by resource type"),
    session_id: Optional[str] = Query(None, description="Filter by session ID"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records")
):
    """
    Retrieve audit logs with optional filtering
    """
    
    try:
        # Parse datetime strings
        start_dt = None
        end_dt = None
        
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid start_time format. Use ISO format.")
        
        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid end_time format. Use ISO format.")
        
        # Retrieve logs
        logs = await audit_service.get_audit_logs(
            start_time=start_dt,
            end_time=end_dt,
            user_id=user_id,
            action=action,
            resource=resource,
            session_id=session_id,
            limit=limit
        )
        
        return {
            "logs": logs,
            "total_returned": len(logs),
            "filters_applied": {
                "start_time": start_time,
                "end_time": end_time,
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "session_id": session_id
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audit log retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Log retrieval failed: {str(e)}")

@router.get("/privacy-ledger")
async def get_privacy_ledger(session_id: Optional[str] = Query(None, description="Specific session ID")):
    """
    Get privacy budget usage ledger
    """
    
    try:
        ledger = await audit_service.get_privacy_ledger(session_id)
        
        if session_id and not ledger:
            raise HTTPException(status_code=404, detail="Session not found in privacy ledger")
        
        return {
            "privacy_ledger": ledger,
            "session_id": session_id,
            "budget_limits": {
                "epsilon": settings.max_epsilon,
                "delta": 1e-3
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Privacy ledger retrieval failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ledger retrieval failed: {str(e)}")

@router.post("/compliance-report")
async def generate_compliance_report(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    report_type: str = Query("comprehensive", description="Report type")
):
    """
    Generate compliance report for specified date range
    """
    
    try:
        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        
        if start_dt > end_dt:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        if (end_dt - start_dt).days > 365:
            raise HTTPException(status_code=400, detail="Date range cannot exceed 365 days")
        
        # Generate report
        report = await audit_service.generate_compliance_report(
            start_date=start_dt,
            end_date=end_dt,
            report_type=report_type
        )
        
        return report
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Compliance report generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@router.get("/system-status")
async def get_system_status():
    """
    Get current audit system status
    """
    
    try:
        status = await audit_service.get_system_status()
        
        # Add additional system health indicators
        health_indicators = {
            "audit_service": "healthy" if status["audit_service_status"] == "active" else "unhealthy",
            "recent_activity": "normal" if status["logs_last_24h"] > 0 else "low",
            "privacy_tracking": "active" if status["privacy_sessions_active"] > 0 else "inactive",
            "storage_health": "good"  # Could add actual storage checks
        }
        
        overall_health = "healthy" if all(
            indicator in ["healthy", "normal", "active", "good"] 
            for indicator in health_indicators.values()
        ) else "degraded"
        
        return {
            "overall_health": overall_health,
            "health_indicators": health_indicators,
            "system_status": status,
            "uptime_hours": (datetime.utcnow() - datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds() / 3600
        }
        
    except Exception as e:
        logger.error(f"System status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@router.post("/cleanup")
async def cleanup_old_logs(retention_days: int = Query(90, ge=7, le=365, description="Retention period in days")):
    """
    Clean up old audit logs based on retention policy
    """
    
    try:
        await audit_service.cleanup_old_logs(retention_days)
        
        # Log the cleanup action
        await audit_service.log_event(
            action="audit_cleanup",
            resource="audit_logs",
            details={
                "retention_days": retention_days,
                "cleanup_timestamp": datetime.utcnow().isoformat()
            },
            success=True
        )
        
        return {
            "message": f"Audit logs older than {retention_days} days have been cleaned up",
            "retention_policy": f"{retention_days} days",
            "cleanup_completed": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Audit cleanup failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

@router.get("/export")
async def export_audit_data(
    start_date: str = Query(..., description="Start date in YYYY-MM-DD format"),
    end_date: str = Query(..., description="End date in YYYY-MM-DD format"),
    format: str = Query("json", description="Export format (json)")
):
    """
    Export audit data for external analysis
    """
    
    try:
        # Parse dates
        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD.")
        
        if start_dt > end_dt:
            raise HTTPException(status_code=400, detail="Start date must be before end date")
        
        # Export data
        export_data = await audit_service.export_audit_data(
            start_date=start_dt,
            end_date=end_dt,
            format=format
        )
        
        # Log export action
        await audit_service.log_event(
            action="audit_export",
            resource="audit_data",
            details={
                "export_id": export_data["export_id"],
                "start_date": start_date,
                "end_date": end_date,
                "format": format,
                "records_exported": export_data["total_records"]
            },
            success=True
        )
        
        return export_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audit export failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

@router.get("/statistics")
async def get_audit_statistics(
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """
    Get audit statistics for dashboard
    """
    
    try:
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Get logs for the period
        logs = await audit_service.get_audit_logs(
            start_time=start_date,
            end_time=end_date,
            limit=10000  # Large limit for statistics
        )
        
        # Calculate statistics
        total_events = len(logs)
        successful_events = sum(1 for log in logs if log.get("success", True))
        failed_events = total_events - successful_events
        
        # Action distribution
        action_counts = {}
        for log in logs:
            action = log.get("action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Daily activity
        daily_activity = {}
        for log in logs:
            date_str = log.get("timestamp", "")[:10]  # YYYY-MM-DD
            daily_activity[date_str] = daily_activity.get(date_str, 0) + 1
        
        # Privacy budget usage
        privacy_events = [log for log in logs if "privacy" in log.get("action", "").lower()]
        total_epsilon_used = sum(
            log.get("details", {}).get("epsilon", 0) 
            for log in privacy_events
        )
        
        # User activity
        user_activity = {}
        for log in logs:
            user_id = log.get("user_id", "anonymous")
            user_activity[user_id] = user_activity.get(user_id, 0) + 1
        
        statistics = {
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "days": days
            },
            "event_summary": {
                "total_events": total_events,
                "successful_events": successful_events,
                "failed_events": failed_events,
                "success_rate": successful_events / total_events if total_events > 0 else 0
            },
            "action_distribution": action_counts,
            "daily_activity": daily_activity,
            "privacy_metrics": {
                "privacy_events": len(privacy_events),
                "total_epsilon_used": total_epsilon_used,
                "epsilon_budget_remaining": max(0, settings.max_epsilon - total_epsilon_used)
            },
            "user_activity": user_activity,
            "top_actions": sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        
        return statistics
        
    except Exception as e:
        logger.error(f"Audit statistics generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Statistics generation failed: {str(e)}")

@router.get("/actions")
async def get_available_actions():
    """
    Get list of available audit actions for filtering
    """
    
    common_actions = [
        "file_uploaded",
        "file_processing_failed",
        "data_anonymized",
        "anonymization_error",
        "attack_simulation",
        "privacy_optimization",
        "data_downloaded",
        "file_deleted",
        "system_startup",
        "system_shutdown",
        "audit_cleanup",
        "audit_export"
    ]
    
    return {
        "available_actions": common_actions,
        "action_categories": {
            "file_operations": ["file_uploaded", "file_processing_failed", "file_deleted", "data_downloaded"],
            "anonymization": ["data_anonymized", "anonymization_error"],
            "privacy": ["attack_simulation", "privacy_optimization"],
            "system": ["system_startup", "system_shutdown"],
            "audit": ["audit_cleanup", "audit_export"]
        }
    }

@router.get("/resources")
async def get_available_resources():
    """
    Get list of available audit resources for filtering
    """
    
    common_resources = [
        "file",
        "anonymization",
        "privacy_attack",
        "optimization",
        "anonymized_data",
        "audit_logs",
        "audit_data",
        "system"
    ]
    
    return {
        "available_resources": common_resources,
        "resource_descriptions": {
            "file": "File upload and processing operations",
            "anonymization": "Data anonymization processes",
            "privacy_attack": "Privacy attack simulations",
            "optimization": "Privacy-utility optimization processes",
            "anonymized_data": "Anonymized dataset operations",
            "audit_logs": "Audit log management",
            "audit_data": "Audit data exports",
            "system": "System-level operations"
        }
    }
