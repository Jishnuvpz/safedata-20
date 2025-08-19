from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import io
import json

from app.models.schemas import (
    AnonymizationRequest, 
    AnonymizationResponse,
    FileUploadResponse
)
from app.services.anonymization import AnonymizationEngine
from app.services.audit import AuditService
from app.utils.file_handler import FileProcessor
from app.utils.validators import data_validator
from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Service instances
anonymization_engine = AnonymizationEngine()
audit_service = AuditService()
file_processor = FileProcessor()

# In-memory storage for demo (in production, use database)
file_storage = {}
anonymization_results = {}

@router.post("/upload", response_model=FileUploadResponse)
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None)
):
    """
    Upload and validate a file for anonymization
    """
    
    start_time = datetime.now()
    
    try:
        # Validate file upload
        file_content = await file.read()
        
        validation_result = await data_validator.validate_file_upload(
            filename=file.filename,
            file_size=len(file_content),
            content_type=file.content_type
        )
        
        if not validation_result["valid"]:
            await audit_service.log_event(
                action="file_upload_failed",
                resource="file",
                details={
                    "filename": file.filename,
                    "errors": validation_result["errors"]
                },
                user_id=user_id,
                success=False
            )
            raise HTTPException(status_code=400, detail=validation_result["errors"])
        
        # Process file
        processing_result = await file_processor.validate_and_process_file(
            file_content=file_content,
            filename=file.filename,
            user_id=user_id
        )
        
        if not processing_result["success"]:
            await audit_service.log_event(
                action="file_processing_failed",
                resource="file",
                details={
                    "filename": file.filename,
                    "error": processing_result["error"]
                },
                user_id=user_id,
                success=False
            )
            raise HTTPException(status_code=400, detail=processing_result["error"])
        
        file_id = processing_result["file_id"]
        
        # Store file data (in production, save to database)
        file_storage[file_id] = {
            "metadata": processing_result["metadata"],
            "data_preview": processing_result["data_preview"],
            "quality_report": processing_result["quality_report"],
            "upload_time": datetime.now(),
            "user_id": user_id
        }
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Audit log
        background_tasks.add_task(
            audit_service.log_event,
            action="file_uploaded",
            resource="file",
            details={
                "file_id": file_id,
                "filename": file.filename,
                "file_size": processing_result["metadata"]["file_size"],
                "rows": processing_result["metadata"]["data_shape"]["rows"],
                "columns": processing_result["metadata"]["data_shape"]["columns"]
            },
            user_id=user_id,
            execution_time=execution_time
        )
        
        return FileUploadResponse(
            message="File uploaded and processed successfully",
            file_id=file_id,
            filename=file.filename,
            size=processing_result["metadata"]["file_size"],
            columns=processing_result["metadata"]["column_names"],
            rows=processing_result["metadata"]["data_shape"]["rows"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {str(e)}")
        await audit_service.log_event(
            action="file_upload_error",
            resource="file",
            details={"error": str(e), "filename": file.filename},
            user_id=user_id,
            success=False
        )
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@router.post("/anonymize", response_model=AnonymizationResponse)
async def anonymize_data(
    request: AnonymizationRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = None
):
    """
    Anonymize uploaded data using specified method and parameters
    """
    
    start_time = datetime.now()
    
    try:
        # Validate request parameters
        param_validation = await data_validator.validate_anonymization_parameters(
            epsilon=request.epsilon,
            delta=request.delta,
            method=request.method
        )
        
        if not param_validation["valid"]:
            raise HTTPException(status_code=400, detail=param_validation["errors"])
        
        # Get file data
        if request.file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Load the actual data (in production, load from database/storage)
        data = await file_processor.get_file_data(request.file_id)
        if data is None:
            # If not cached, re-process the file
            raise HTTPException(status_code=404, detail="File data not available")
        
        # Validate dataset
        dataset_validation = await data_validator.validate_dataset(data)
        warnings = dataset_validation.get("warnings", [])
        
        if not dataset_validation["valid"]:
            raise HTTPException(status_code=400, detail=dataset_validation["errors"])
        
        # Perform anonymization
        anonymization_result = await anonymization_engine.anonymize_data(
            data=data,
            method=request.method,
            epsilon=request.epsilon,
            delta=request.delta,
            quasi_identifiers=request.quasi_identifiers,
            sensitive_attributes=request.sensitive_attributes,
            utility_metrics=request.utility_metrics
        )
        
        if not anonymization_result["success"]:
            raise HTTPException(status_code=500, detail=anonymization_result.get("error", "Anonymization failed"))
        
        # Store results
        result_id = f"{request.file_id}_anonymized_{datetime.now().timestamp()}"
        anonymization_results[result_id] = {
            "anonymized_data": anonymization_result["anonymized_data"],
            "original_file_id": request.file_id,
            "method": request.method,
            "parameters": anonymization_result["parameters"],
            "privacy_metrics": anonymization_result["privacy_metrics"],
            "utility_metrics": anonymization_result["utility_metrics"],
            "attack_results": anonymization_result["attack_results"],
            "created_at": datetime.now(),
            "user_id": user_id
        }
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Audit log
        background_tasks.add_task(
            audit_service.log_event,
            action="data_anonymized",
            resource="anonymization",
            details={
                "file_id": request.file_id,
                "result_id": result_id,
                "method": request.method,
                "epsilon": request.epsilon,
                "delta": request.delta,
                "original_rows": len(data),
                "anonymized_rows": len(anonymization_result["anonymized_data"])
            },
            user_id=user_id,
            execution_time=execution_time,
            privacy_budget_used={
                "epsilon": request.epsilon,
                "delta": request.delta,
                "mechanism": request.method
            }
        )
        
        warnings.extend(param_validation.get("warnings", []))
        
        return AnonymizationResponse(
            success=True,
            message="Data anonymized successfully",
            anonymized_data_id=result_id,
            privacy_metrics=anonymization_result["privacy_metrics"],
            utility_metrics=anonymization_result["utility_metrics"],
            execution_time=execution_time,
            warnings=warnings if warnings else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Anonymization failed: {str(e)}")
        await audit_service.log_event(
            action="anonymization_error",
            resource="anonymization",
            details={"error": str(e), "file_id": request.file_id},
            user_id=user_id,
            success=False
        )
        raise HTTPException(status_code=500, detail=f"Anonymization failed: {str(e)}")

@router.get("/file/{file_id}/info")
async def get_file_info(file_id: str, user_id: Optional[str] = None):
    """
    Get information about an uploaded file
    """
    
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = file_storage[file_id]
    
    # Check user access (basic implementation)
    if user_id and file_info.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    return {
        "file_id": file_id,
        "metadata": file_info["metadata"],
        "data_preview": file_info["data_preview"],
        "quality_report": file_info["quality_report"],
        "upload_time": file_info["upload_time"].isoformat()
    }

@router.get("/result/{result_id}")
async def get_anonymization_result(result_id: str, user_id: Optional[str] = None):
    """
    Get anonymization results
    """
    
    if result_id not in anonymization_results:
        raise HTTPException(status_code=404, detail="Anonymization result not found")
    
    result = anonymization_results[result_id]
    
    # Check user access
    if user_id and result.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Return result without raw data (for security)
    return {
        "result_id": result_id,
        "original_file_id": result["original_file_id"],
        "method": result["method"],
        "parameters": result["parameters"],
        "privacy_metrics": result["privacy_metrics"],
        "utility_metrics": result["utility_metrics"],
        "attack_results": result["attack_results"],
        "created_at": result["created_at"].isoformat(),
        "data_shape": {
            "rows": len(result["anonymized_data"]),
            "columns": len(result["anonymized_data"].columns)
        }
    }

@router.get("/result/{result_id}/download")
async def download_anonymized_data(
    result_id: str,
    format: str = "csv",
    user_id: Optional[str] = None
):
    """
    Download anonymized dataset
    """
    
    if result_id not in anonymization_results:
        raise HTTPException(status_code=404, detail="Anonymization result not found")
    
    result = anonymization_results[result_id]
    
    # Check user access
    if user_id and result.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    anonymized_data = result["anonymized_data"]
    
    # Audit log
    await audit_service.log_event(
        action="data_downloaded",
        resource="anonymized_data",
        details={
            "result_id": result_id,
            "format": format,
            "rows": len(anonymized_data),
            "columns": len(anonymized_data.columns)
        },
        user_id=user_id
    )
    
    if format.lower() == "csv":
        # Generate CSV
        output = io.StringIO()
        anonymized_data.to_csv(output, index=False)
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=anonymized_data_{result_id}.csv"}
        )
    
    elif format.lower() == "excel":
        # Generate Excel
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            anonymized_data.to_excel(writer, sheet_name='Anonymized Data', index=False)
        output.seek(0)
        
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=anonymized_data_{result_id}.xlsx"}
        )
    
    elif format.lower() == "json":
        # Generate JSON
        json_data = anonymized_data.to_json(orient='records', indent=2)
        
        return StreamingResponse(
            io.BytesIO(json_data.encode()),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=anonymized_data_{result_id}.json"}
        )
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported format. Use: csv, excel, or json")

@router.get("/result/{result_id}/preview")
async def preview_anonymized_data(
    result_id: str,
    rows: int = 10,
    user_id: Optional[str] = None
):
    """
    Preview anonymized data (limited rows)
    """
    
    if result_id not in anonymization_results:
        raise HTTPException(status_code=404, detail="Anonymization result not found")
    
    result = anonymization_results[result_id]
    
    # Check user access
    if user_id and result.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    anonymized_data = result["anonymized_data"]
    
    # Limit rows for preview
    preview_data = anonymized_data.head(min(rows, 100))
    
    return {
        "result_id": result_id,
        "preview_rows": len(preview_data),
        "total_rows": len(anonymized_data),
        "columns": anonymized_data.columns.tolist(),
        "data": preview_data.to_dict('records')
    }

@router.delete("/file/{file_id}")
async def delete_file(file_id: str, user_id: Optional[str] = None):
    """
    Delete uploaded file and associated data
    """
    
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = file_storage[file_id]
    
    # Check user access
    if user_id and file_info.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        # Remove from storage
        del file_storage[file_id]
        
        # Clean up file system
        await file_processor.cleanup_file(file_id)
        
        # Remove associated anonymization results
        results_to_remove = [
            result_id for result_id, result in anonymization_results.items()
            if result["original_file_id"] == file_id
        ]
        
        for result_id in results_to_remove:
            del anonymization_results[result_id]
        
        # Audit log
        await audit_service.log_event(
            action="file_deleted",
            resource="file",
            details={
                "file_id": file_id,
                "associated_results_removed": len(results_to_remove)
            },
            user_id=user_id
        )
        
        return {"message": "File and associated data deleted successfully"}
        
    except Exception as e:
        logger.error(f"File deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File deletion failed: {str(e)}")

@router.delete("/result/{result_id}")
async def delete_anonymization_result(result_id: str, user_id: Optional[str] = None):
    """
    Delete anonymization result
    """
    
    if result_id not in anonymization_results:
        raise HTTPException(status_code=404, detail="Anonymization result not found")
    
    result = anonymization_results[result_id]
    
    # Check user access
    if user_id and result.get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        # Remove from storage
        del anonymization_results[result_id]
        
        # Audit log
        await audit_service.log_event(
            action="result_deleted",
            resource="anonymization_result",
            details={"result_id": result_id},
            user_id=user_id
        )
        
        return {"message": "Anonymization result deleted successfully"}
        
    except Exception as e:
        logger.error(f"Result deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Result deletion failed: {str(e)}")

@router.get("/list")
async def list_user_files(user_id: Optional[str] = None):
    """
    List files and results for a user
    """
    
    # Filter files by user
    user_files = {}
    user_results = {}
    
    if user_id:
        user_files = {
            file_id: info for file_id, info in file_storage.items()
            if info.get("user_id") == user_id
        }
        user_results = {
            result_id: result for result_id, result in anonymization_results.items()
            if result.get("user_id") == user_id
        }
    else:
        # Return all files if no user_id specified (admin view)
        user_files = file_storage
        user_results = anonymization_results
    
    # Format response
    files_list = []
    for file_id, info in user_files.items():
        files_list.append({
            "file_id": file_id,
            "filename": info["metadata"]["original_filename"],
            "upload_time": info["upload_time"].isoformat(),
            "size": info["metadata"]["file_size"],
            "rows": info["metadata"]["data_shape"]["rows"],
            "columns": info["metadata"]["data_shape"]["columns"]
        })
    
    results_list = []
    for result_id, result in user_results.items():
        results_list.append({
            "result_id": result_id,
            "original_file_id": result["original_file_id"],
            "method": result["method"],
            "created_at": result["created_at"].isoformat(),
            "privacy_score": result["privacy_metrics"].get("re_identification_risk", 0),
            "utility_score": result["utility_metrics"].get("overall_utility_score", 0)
        })
    
    return {
        "files": files_list,
        "anonymization_results": results_list,
        "total_files": len(files_list),
        "total_results": len(results_list)
    }

@router.get("/methods")
async def get_anonymization_methods():
    """
    Get available anonymization methods and their descriptions
    """
    
    methods = {
        "sdg": {
            "name": "Synthetic Data Generation",
            "description": "Generate synthetic dataset using deep learning models (CTGAN/TVAE)",
            "privacy_level": "Medium",
            "utility_preservation": "High",
            "parameters": ["synthetic_epochs", "synthetic_batch_size"]
        },
        "dp": {
            "name": "Differential Privacy",
            "description": "Add calibrated noise to provide formal privacy guarantees",
            "privacy_level": "High",
            "utility_preservation": "Medium",
            "parameters": ["epsilon", "delta", "mechanism"]
        },
        "sdc": {
            "name": "Statistical Disclosure Control",
            "description": "Apply generalization, suppression, and microaggregation",
            "privacy_level": "Medium",
            "utility_preservation": "High",
            "parameters": ["k_anonymity", "suppression_threshold"]
        },
        "sdg_dp": {
            "name": "SDG + Differential Privacy",
            "description": "Combine synthetic data generation with differential privacy",
            "privacy_level": "High",
            "utility_preservation": "Medium-High",
            "parameters": ["epsilon", "delta", "synthetic_epochs", "synthetic_batch_size"]
        },
        "sdg_sdc": {
            "name": "SDG + Statistical Disclosure Control",
            "description": "Apply SDC techniques to synthetic data",
            "privacy_level": "Medium-High",
            "utility_preservation": "High",
            "parameters": ["synthetic_epochs", "k_anonymity", "suppression_threshold"]
        },
        "dp_sdc": {
            "name": "DP + Statistical Disclosure Control",
            "description": "Apply SDC to differentially private data",
            "privacy_level": "High",
            "utility_preservation": "Medium",
            "parameters": ["epsilon", "delta", "k_anonymity"]
        },
        "full": {
            "name": "Full SafeData 2.0 Pipeline",
            "description": "Complete pipeline: SDG + DP + SDC for maximum protection",
            "privacy_level": "Maximum",
            "utility_preservation": "Medium",
            "parameters": ["epsilon", "delta", "synthetic_epochs", "k_anonymity", "suppression_threshold"]
        }
    }
    
    return {
        "methods": methods,
        "default_parameters": {
            "epsilon": settings.default_epsilon,
            "delta": settings.default_delta,
            "synthetic_epochs": settings.synthetic_epochs,
            "synthetic_batch_size": settings.batch_size
        }
    }
