import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import uuid
import os
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

@dataclass
class AuditLogEntry:
    """Audit log entry data structure"""
    id: str
    timestamp: datetime
    user_id: Optional[str]
    session_id: Optional[str]
    action: str
    resource: str
    details: Dict[str, Any]
    ip_address: Optional[str]
    user_agent: Optional[str]
    success: bool
    execution_time: Optional[float]
    privacy_budget_used: Optional[Dict[str, float]]

class AuditService:
    """Comprehensive audit and compliance logging service"""
    
    def __init__(self):
        self.audit_logs = []
        self.privacy_ledger = {}
        self.session_tracking = {}
        self.compliance_reports = {}
        
        # Create audit directory if it doesn't exist
        self.audit_dir = "audit_logs"
        os.makedirs(self.audit_dir, exist_ok=True)
        
    async def log_event(
        self,
        action: str,
        resource: str,
        details: Dict[str, Any],
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        success: bool = True,
        execution_time: Optional[float] = None,
        privacy_budget_used: Optional[Dict[str, float]] = None
    ) -> str:
        """Log an audit event"""
        
        entry_id = str(uuid.uuid4())
        
        audit_entry = AuditLogEntry(
            id=entry_id,
            timestamp=datetime.utcnow(),
            user_id=user_id,
            session_id=session_id,
            action=action,
            resource=resource,
            details=details,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            execution_time=execution_time,
            privacy_budget_used=privacy_budget_used
        )
        
        # Add to in-memory storage
        self.audit_logs.append(audit_entry)
        
        # Update privacy ledger if applicable
        if privacy_budget_used:
            await self._update_privacy_ledger(session_id or "default", privacy_budget_used)
        
        # Persist to file
        await self._persist_audit_entry(audit_entry)
        
        logger.info(f"Audit event logged: {action} on {resource} (ID: {entry_id})")
        
        return entry_id
    
    async def _update_privacy_ledger(
        self,
        session_id: str,
        privacy_budget_used: Dict[str, float]
    ):
        """Update privacy budget ledger"""
        
        if session_id not in self.privacy_ledger:
            self.privacy_ledger[session_id] = {
                "total_epsilon": 0.0,
                "total_delta": 0.0,
                "operations": []
            }
        
        # Update totals
        self.privacy_ledger[session_id]["total_epsilon"] += privacy_budget_used.get("epsilon", 0.0)
        self.privacy_ledger[session_id]["total_delta"] += privacy_budget_used.get("delta", 0.0)
        
        # Log operation
        self.privacy_ledger[session_id]["operations"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "epsilon": privacy_budget_used.get("epsilon", 0.0),
            "delta": privacy_budget_used.get("delta", 0.0),
            "mechanism": privacy_budget_used.get("mechanism", "unknown")
        })
    
    async def _persist_audit_entry(self, entry: AuditLogEntry):
        """Persist audit entry to file"""
        
        try:
            # Create daily log file
            date_str = entry.timestamp.strftime("%Y-%m-%d")
            log_file = os.path.join(self.audit_dir, f"audit_{date_str}.jsonl")
            
            # Convert entry to dict and handle datetime serialization
            entry_dict = asdict(entry)
            entry_dict["timestamp"] = entry.timestamp.isoformat()
            
            # Append to daily log file
            with open(log_file, "a") as f:
                f.write(json.dumps(entry_dict) + "\n")
                
        except Exception as e:
            logger.error(f"Failed to persist audit entry: {str(e)}")
    
    async def get_audit_logs(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        user_id: Optional[str] = None,
        action: Optional[str] = None,
        resource: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Query audit logs with filters"""
        
        filtered_logs = []
        
        for entry in self.audit_logs:
            # Apply filters
            if start_time and entry.timestamp < start_time:
                continue
            if end_time and entry.timestamp > end_time:
                continue
            if user_id and entry.user_id != user_id:
                continue
            if action and entry.action != action:
                continue
            if resource and entry.resource != resource:
                continue
            if session_id and entry.session_id != session_id:
                continue
            
            # Convert to dict for JSON serialization
            entry_dict = asdict(entry)
            entry_dict["timestamp"] = entry.timestamp.isoformat()
            filtered_logs.append(entry_dict)
            
            if len(filtered_logs) >= limit:
                break
        
        return filtered_logs
    
    async def get_privacy_ledger(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get privacy budget usage ledger"""
        
        if session_id:
            return self.privacy_ledger.get(session_id, {})
        else:
            return self.privacy_ledger
    
    async def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
        report_type: str = "comprehensive"
    ) -> Dict[str, Any]:
        """Generate compliance report for audit purposes"""
        
        logger.info(f"Generating {report_type} compliance report for {start_date} to {end_date}")
        
        # Filter logs for date range
        period_logs = [
            entry for entry in self.audit_logs
            if start_date <= entry.timestamp <= end_date
        ]
        
        report = {
            "report_id": str(uuid.uuid4()),
            "generated_at": datetime.utcnow().isoformat(),
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "report_type": report_type,
            "summary": await self._generate_report_summary(period_logs),
            "privacy_compliance": await self._analyze_privacy_compliance(period_logs),
            "security_events": await self._analyze_security_events(period_logs),
            "data_processing": await self._analyze_data_processing(period_logs),
            "system_performance": await self._analyze_system_performance(period_logs)
        }
        
        # Store report
        self.compliance_reports[report["report_id"]] = report
        
        return report
    
    async def _generate_report_summary(self, logs: List[AuditLogEntry]) -> Dict[str, Any]:
        """Generate summary statistics for compliance report"""
        
        total_events = len(logs)
        successful_events = sum(1 for log in logs if log.success)
        failed_events = total_events - successful_events
        
        # Count by action type
        action_counts = {}
        for log in logs:
            action_counts[log.action] = action_counts.get(log.action, 0) + 1
        
        # Count by resource
        resource_counts = {}
        for log in logs:
            resource_counts[log.resource] = resource_counts.get(log.resource, 0) + 1
        
        # Unique users and sessions
        unique_users = len(set(log.user_id for log in logs if log.user_id))
        unique_sessions = len(set(log.session_id for log in logs if log.session_id))
        
        return {
            "total_events": total_events,
            "successful_events": successful_events,
            "failed_events": failed_events,
            "success_rate": successful_events / total_events if total_events > 0 else 0,
            "unique_users": unique_users,
            "unique_sessions": unique_sessions,
            "action_distribution": action_counts,
            "resource_distribution": resource_counts
        }
    
    async def _analyze_privacy_compliance(self, logs: List[AuditLogEntry]) -> Dict[str, Any]:
        """Analyze privacy compliance from audit logs"""
        
        privacy_events = [
            log for log in logs
            if log.privacy_budget_used or "privacy" in log.action.lower()
        ]
        
        # Calculate total privacy budget usage
        total_epsilon = 0.0
        total_delta = 0.0
        
        for log in privacy_events:
            if log.privacy_budget_used:
                total_epsilon += log.privacy_budget_used.get("epsilon", 0.0)
                total_delta += log.privacy_budget_used.get("delta", 0.0)
        
        # Analyze privacy budget by session
        session_budgets = {}
        for session_id, ledger in self.privacy_ledger.items():
            session_budgets[session_id] = {
                "epsilon_used": ledger["total_epsilon"],
                "delta_used": ledger["total_delta"],
                "operations_count": len(ledger["operations"])
            }
        
        # Privacy compliance status
        from app.core.config import settings
        epsilon_compliance = total_epsilon <= settings.max_epsilon
        
        return {
            "total_privacy_events": len(privacy_events),
            "total_epsilon_used": total_epsilon,
            "total_delta_used": total_delta,
            "epsilon_budget_limit": settings.max_epsilon,
            "epsilon_compliance": epsilon_compliance,
            "session_budgets": session_budgets,
            "privacy_mechanisms_used": self._extract_privacy_mechanisms(privacy_events),
            "compliance_status": "COMPLIANT" if epsilon_compliance else "NON_COMPLIANT"
        }
    
    def _extract_privacy_mechanisms(self, privacy_events: List[AuditLogEntry]) -> Dict[str, int]:
        """Extract privacy mechanisms used from events"""
        
        mechanisms = {}
        for log in privacy_events:
            if log.privacy_budget_used and "mechanism" in log.privacy_budget_used:
                mechanism = log.privacy_budget_used["mechanism"]
                mechanisms[mechanism] = mechanisms.get(mechanism, 0) + 1
        
        return mechanisms
    
    async def _analyze_security_events(self, logs: List[AuditLogEntry]) -> Dict[str, Any]:
        """Analyze security-related events"""
        
        security_events = [
            log for log in logs
            if any(keyword in log.action.lower() for keyword in 
                  ["login", "auth", "upload", "download", "access", "error"])
        ]
        
        failed_security_events = [log for log in security_events if not log.success]
        
        # IP address analysis
        ip_addresses = [log.ip_address for log in logs if log.ip_address]
        unique_ips = len(set(ip_addresses))
        
        # Failed access attempts by IP
        failed_by_ip = {}
        for log in failed_security_events:
            if log.ip_address:
                failed_by_ip[log.ip_address] = failed_by_ip.get(log.ip_address, 0) + 1
        
        return {
            "total_security_events": len(security_events),
            "failed_security_events": len(failed_security_events),
            "unique_ip_addresses": unique_ips,
            "failed_attempts_by_ip": failed_by_ip,
            "suspicious_activity": [
                ip for ip, count in failed_by_ip.items() if count > 10
            ]
        }
    
    async def _analyze_data_processing(self, logs: List[AuditLogEntry]) -> Dict[str, Any]:
        """Analyze data processing activities"""
        
        processing_events = [
            log for log in logs
            if any(keyword in log.action.lower() for keyword in 
                  ["upload", "anonymize", "download", "process", "generate"])
        ]
        
        # File processing statistics
        files_processed = len([
            log for log in processing_events if "upload" in log.action.lower()
        ])
        
        anonymization_jobs = len([
            log for log in processing_events if "anonymize" in log.action.lower()
        ])
        
        # Processing times
        processing_times = [
            log.execution_time for log in processing_events
            if log.execution_time is not None
        ]
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        return {
            "total_processing_events": len(processing_events),
            "files_processed": files_processed,
            "anonymization_jobs": anonymization_jobs,
            "average_processing_time": avg_processing_time,
            "processing_time_distribution": {
                "min": min(processing_times) if processing_times else 0,
                "max": max(processing_times) if processing_times else 0,
                "median": sorted(processing_times)[len(processing_times)//2] if processing_times else 0
            }
        }
    
    async def _analyze_system_performance(self, logs: List[AuditLogEntry]) -> Dict[str, Any]:
        """Analyze system performance metrics"""
        
        # Error rate analysis
        total_events = len(logs)
        failed_events = len([log for log in logs if not log.success])
        error_rate = failed_events / total_events if total_events > 0 else 0
        
        # Response time analysis
        response_times = [
            log.execution_time for log in logs
            if log.execution_time is not None
        ]
        
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # System load indicators
        peak_hour_events = await self._calculate_peak_hour_events(logs)
        
        return {
            "total_system_events": total_events,
            "system_error_rate": error_rate,
            "average_response_time": avg_response_time,
            "peak_hour_events": peak_hour_events,
            "system_health_score": max(0, 1 - error_rate - (avg_response_time / 100)),
            "performance_trend": "stable"  # Could be enhanced with trend analysis
        }
    
    async def _calculate_peak_hour_events(self, logs: List[AuditLogEntry]) -> Dict[str, int]:
        """Calculate events by hour to identify peak usage"""
        
        hourly_events = {}
        for log in logs:
            hour = log.timestamp.hour
            hourly_events[hour] = hourly_events.get(hour, 0) + 1
        
        return hourly_events
    
    async def cleanup_old_logs(self, retention_days: int = 90):
        """Clean up old audit logs based on retention policy"""
        
        cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
        
        # Remove old in-memory logs
        self.audit_logs = [
            log for log in self.audit_logs
            if log.timestamp > cutoff_date
        ]
        
        # Clean up old log files
        try:
            for filename in os.listdir(self.audit_dir):
                if filename.startswith("audit_") and filename.endswith(".jsonl"):
                    file_path = os.path.join(self.audit_dir, filename)
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_mtime < cutoff_date:
                        os.remove(file_path)
                        logger.info(f"Removed old audit log file: {filename}")
        
        except Exception as e:
            logger.error(f"Error during log cleanup: {str(e)}")
    
    async def export_audit_data(
        self,
        start_date: datetime,
        end_date: datetime,
        format: str = "json"
    ) -> Dict[str, Any]:
        """Export audit data for external analysis"""
        
        # Filter logs for date range
        filtered_logs = [
            log for log in self.audit_logs
            if start_date <= log.timestamp <= end_date
        ]
        
        export_data = {
            "export_id": str(uuid.uuid4()),
            "generated_at": datetime.utcnow().isoformat(),
            "period": {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat()
            },
            "total_records": len(filtered_logs),
            "records": []
        }
        
        for log in filtered_logs:
            record = asdict(log)
            record["timestamp"] = log.timestamp.isoformat()
            export_data["records"].append(record)
        
        return export_data
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get current system audit status"""
        
        recent_logs = [
            log for log in self.audit_logs
            if log.timestamp > datetime.utcnow() - timedelta(hours=24)
        ]
        
        return {
            "audit_service_status": "active",
            "total_logs_stored": len(self.audit_logs),
            "logs_last_24h": len(recent_logs),
            "privacy_sessions_active": len(self.privacy_ledger),
            "audit_storage_path": self.audit_dir,
            "last_cleanup": "not_implemented",  # Could track last cleanup time
            "compliance_reports_generated": len(self.compliance_reports)
        }
