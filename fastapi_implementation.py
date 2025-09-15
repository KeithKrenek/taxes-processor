# api/main.py - FastAPI Application
"""
Financial Transaction Processing REST API

A production-ready FastAPI application demonstrating:
- Async request handling
- File upload processing
- Background task processing
- Comprehensive error handling
- API documentation with OpenAPI/Swagger
- Authentication and rate limiting
- Monitoring and logging
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles

import asyncio
import aiofiles
import logging
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime, timedelta
import uuid
import os
from pathlib import Path
import json
import tempfile

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Monitoring
import time
from prometheus_client import Counter, Histogram, generate_latest
import structlog

# Import our processing pipeline
from processor.pipeline import FinancialPipeline, ProcessingConfig, TransactionRecord

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
PROCESSING_DURATION = Histogram('processing_duration_seconds', 'File processing duration')
PROCESSING_COUNT = Counter('files_processed_total', 'Total files processed', ['status'])

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

# Security
security = HTTPBearer()

# Global state for background tasks
task_results = {}

# Pydantic Models
class TransactionModel(BaseModel):
    """Pydantic model for transaction data."""
    date: datetime
    amount: float
    description: str
    account: str
    category: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    is_duplicate: bool = False
    duplicate_group: Optional[int] = None
    source_sheet: str = ""

    class Config:
        schema_extra = {
            "example": {
                "date": "2024-01-15T00:00:00",
                "amount": 125.50,
                "description": "GROCERY STORE PURCHASE",
                "account": "checking",
                "category": "Food & Dining",
                "confidence": 0.85,
                "is_duplicate": False,
                "duplicate_group": None,
                "source_sheet": "Checking"
            }
        }


class ProcessingConfigModel(BaseModel):
    """Pydantic model for processing configuration."""
    ml_confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    enable_fuzzy_matching: bool = True
    fuzzy_threshold: int = Field(default=85, ge=0, le=100)
    temporal_window_days: int = Field(default=5, ge=1, le=30)
    amount_tolerance: float = Field(default=0.01, ge=0.0)
    similarity_threshold: float = Field(default=0.85, ge=0.0, le=1.0)

    class Config:
        schema_extra = {
            "example": {
                "ml_confidence_threshold": 0.8,
                "enable_fuzzy_matching": True,
                "fuzzy_threshold": 85,
                "temporal_window_days": 5,
                "amount_tolerance": 0.01,
                "similarity_threshold": 0.85
            }
        }


class ProcessingResultModel(BaseModel):
    """Pydantic model for processing results."""
    task_id: str
    status: str
    transactions: List[TransactionModel]
    summary: Dict[str, Any]
    processing_time: float
    created_at: datetime

    class Config:
        schema_extra = {
            "example": {
                "task_id": "123e4567-e89b-12d3-a456-426614174000",
                "status": "completed",
                "transactions": [],
                "summary": {
                    "total_transactions": 100,
                    "unique_transactions": 95,
                    "duplicate_count": 5,
                    "total_amount": 5432.10
                },
                "processing_time": 2.5,
                "created_at": "2024-01-15T10:30:00"
            }
        }


class HealthCheckModel(BaseModel):
    """Health check response model."""
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    checks: Dict[str, str]


class ErrorModel(BaseModel):
    """Error response model."""
    error: str
    message: str
    timestamp: datetime
    request_id: Optional[str] = None


# Create FastAPI application
app = FastAPI(
    title="Financial Transaction Processor API",
    description="""
    A comprehensive financial data processing API with machine learning-powered
    transaction categorization, duplicate detection, and advanced analytics.
    
    ## Features
    
    * **File Upload**: Process Excel files containing financial transactions
    * **ML Categorization**: Intelligent transaction categorization using ensemble methods
    * **Duplicate Detection**: Advanced duplicate identification using multiple similarity metrics
    * **Real-time Processing**: Async processing with background task support
    * **Comprehensive Analytics**: Detailed reports and visualizations
    * **Production Ready**: Rate limiting, monitoring, logging, and error handling
    
    ## Use Cases
    
    * Personal finance management and tax preparation
    * Business expense reporting and audit compliance
    * Financial data consolidation across multiple accounts
    * Transaction pattern analysis and anomaly detection
    """,
    version="1.0.0",
    contact={
        "name": "Portfolio Demonstration",
        "email": "demo@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Startup time for uptime calculation
app.state.startup_time = time.time()

# Static files for documentation
app.mount("/static", StaticFiles(directory="static"), name="static")


# Middleware for request logging and metrics
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests and collect metrics."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Add request ID to logs
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)
    
    logger.info(
        "Request started",
        method=request.method,
        url=str(request.url),
        client_ip=get_remote_address(request)
    )
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(process_time)
    
    logger.info(
        "Request completed",
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        process_time=process_time
    )
    
    response.headers["X-Request-ID"] = request_id
    return response


# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API token (simplified for demo)."""
    # In production, implement proper JWT validation
    token = credentials.credentials
    if token != "demo-api-key":  # Simple token for demo
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


# API Endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Financial Transaction Processor API",
        "version": "1.0.0",
        "documentation": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthCheckModel)
async def health_check():
    """Comprehensive health check endpoint."""
    uptime = time.time() - app.state.startup_time
    
    # Check various system components
    checks = {
        "database": "healthy",  # Would check actual DB in production
        "file_system": "healthy",
        "memory": "healthy",
        "processing_pipeline": "healthy"
    }
    
    return HealthCheckModel(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        uptime_seconds=uptime,
        checks=checks
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/upload", response_model=Dict[str, str])
@limiter.limit("10/minute")
async def upload_file(
    request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    config: Optional[ProcessingConfigModel] = None,
    token: str = Depends(verify_token)
):
    """
    Upload and process a financial data file.
    
    - **file**: Excel file containing financial transactions
    - **config**: Optional processing configuration parameters
    
    Returns a task ID for tracking processing status.
    """
    logger.info("File upload started", filename=file.filename, content_type=file.content_type)
    
    # Validate file type
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only Excel files (.xlsx, .xls) are supported"
        )
    
    # Validate file size (10MB limit)
    file_content = await file.read()
    if len(file_content) > 10 * 1024 * 1024:  # 10MB
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 10MB limit"
        )
    
    # Generate task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    task_results[task_id] = {
        "status": "processing",
        "created_at": datetime.now(),
        "filename": file.filename
    }
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    temp_file.write(file_content)
    temp_file.close()
    
    # Start background processing
    background_tasks.add_task(
        process_file_background,
        task_id,
        temp_file.name,
        config.dict() if config else {}
    )
    
    logger.info("File processing started", task_id=task_id, filename=file.filename)
    
    return {
        "task_id": task_id,
        "status": "processing",
        "message": "File uploaded successfully. Use /status/{task_id} to check progress."
    }


@app.get("/status/{task_id}", response_model=Dict[str, Any])
async def get_processing_status(task_id: str, token: str = Depends(verify_token)):
    """Get the processing status of a file upload task."""
    if task_id not in task_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    result = task_results[task_id]
    
    # Return appropriate response based on status
    if result["status"] == "processing":
        return {
            "task_id": task_id,
            "status": "processing",
            "created_at": result["created_at"],
            "message": "File is being processed. Please check back in a few moments."
        }
    elif result["status"] == "completed":
        return {
            "task_id": task_id,
            "status": "completed",
            "created_at": result["created_at"],
            "completed_at": result.get("completed_at"),
            "summary": result.get("summary"),
            "processing_time": result.get("processing_time"),
            "download_url": f"/download/{task_id}"
        }
    elif result["status"] == "failed":
        return {
            "task_id": task_id,
            "status": "failed",
            "created_at": result["created_at"],
            "error": result.get("error"),
            "message": "Processing failed. Please try again or contact support."
        }


@app.get("/results/{task_id}", response_model=ProcessingResultModel)
async def get_processing_results(task_id: str, token: str = Depends(verify_token)):
    """Get the detailed processing results for a completed task."""
    if task_id not in task_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    result = task_results[task_id]
    
    if result["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Task is not completed. Current status: {result['status']}"
        )
    
    # Convert transactions to Pydantic models
    transactions = [
        TransactionModel(**transaction) for transaction in result["transactions"]
    ]
    
    return ProcessingResultModel(
        task_id=task_id,
        status=result["status"],
        transactions=transactions,
        summary=result["summary"],
        processing_time=result["processing_time"],
        created_at=result["created_at"]
    )


@app.get("/download/{task_id}")
async def download_results(task_id: str, token: str = Depends(verify_token)):
    """Download processed results as Excel file."""
    if task_id not in task_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    result = task_results[task_id]
    
    if result["status"] != "completed":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Results not ready for download"
        )
    
    if "output_file" not in result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Output file not found"
        )
    
    output_file = result["output_file"]
    if not os.path.exists(output_file):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Output file has been removed"
        )
    
    return FileResponse(
        path=output_file,
        filename=f"processed_transactions_{task_id}.xlsx",
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


@app.post("/process-sync", response_model=ProcessingResultModel)
@limiter.limit("5/minute")
async def process_file_sync(
    request,
    file: UploadFile = File(...),
    config: Optional[ProcessingConfigModel] = None,
    token: str = Depends(verify_token)
):
    """
    Synchronously process a file and return results immediately.
    
    Note: Use this only for small files as it blocks the request until processing is complete.
    """
    logger.info("Synchronous processing started", filename=file.filename)
    
    # Validate file
    if not file.filename.endswith(('.xlsx', '.xls')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only Excel files (.xlsx, .xls) are supported"
        )
    
    # Check file size (smaller limit for sync processing)
    file_content = await file.read()
    if len(file_content) > 5 * 1024 * 1024:  # 5MB limit for sync
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 5MB limit for synchronous processing. Use /upload for larger files."
        )
    
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    temp_file.write(file_content)
    temp_file.close()
    
    try:
        # Process file
        start_time = time.time()
        
        # Create configuration
        processing_config = ProcessingConfig(**(config.dict() if config else {}))
        
        # Initialize pipeline
        pipeline = FinancialPipeline()
        pipeline.config = processing_config
        
        # Process file
        results = pipeline.process_file(temp_file.name)
        
        processing_time = time.time() - start_time
        
        # Record metrics
        PROCESSING_DURATION.observe(processing_time)
        PROCESSING_COUNT.labels(status="success").inc()
        
        # Convert transactions to dict format
        transactions_data = [
            {
                "date": t.date,
                "amount": t.amount,
                "description": t.description,
                "account": t.account,
                "category": t.category,
                "confidence": t.confidence,
                "is_duplicate": t.is_duplicate,
                "duplicate_group": t.duplicate_group,
                "source_sheet": t.source_sheet
            }
            for t in results["transactions"]
        ]
        
        # Convert to Pydantic models
        transactions = [TransactionModel(**t) for t in transactions_data]
        
        task_id = str(uuid.uuid4())
        
        logger.info(
            "Synchronous processing completed",
            task_id=task_id,
            processing_time=processing_time,
            transaction_count=len(transactions)
        )
        
        return ProcessingResultModel(
            task_id=task_id,
            status="completed",
            transactions=transactions,
            summary=results["summary"],
            processing_time=processing_time,
            created_at=datetime.now()
        )
        
    except Exception as e:
        PROCESSING_COUNT.labels(status="error").inc()
        logger.error("Synchronous processing failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}"
        )
    finally:
        # Cleanup temporary file
        os.unlink(temp_file.name)


@app.get("/tasks", response_model=List[Dict[str, Any]])
async def list_tasks(token: str = Depends(verify_token)):
    """List all processing tasks."""
    tasks = []
    for task_id, result in task_results.items():
        tasks.append({
            "task_id": task_id,
            "status": result["status"],
            "created_at": result["created_at"],
            "filename": result.get("filename"),
            "completed_at": result.get("completed_at")
        })
    
    # Sort by creation time (newest first)
    tasks.sort(key=lambda x: x["created_at"], reverse=True)
    
    return tasks


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str, token: str = Depends(verify_token)):
    """Delete a processing task and its results."""
    if task_id not in task_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    
    result = task_results[task_id]
    
    # Clean up output file if it exists
    if "output_file" in result and os.path.exists(result["output_file"]):
        os.unlink(result["output_file"])
    
    # Remove from memory
    del task_results[task_id]
    
    logger.info("Task deleted", task_id=task_id)
    
    return {"message": "Task deleted successfully"}


# Background processing function
async def process_file_background(task_id: str, file_path: str, config_dict: Dict[str, Any]):
    """Process file in background task."""
    try:
        logger.info("Background processing started", task_id=task_id)
        
        start_time = time.time()
        
        # Create configuration
        processing_config = ProcessingConfig(**config_dict)
        
        # Initialize pipeline
        pipeline = FinancialPipeline()
        pipeline.config = processing_config
        
        # Process file
        results = pipeline.process_file(file_path)
        
        processing_time = time.time() - start_time
        
        # Create output file
        output_dir = Path("data/output")
        output_dir.mkdir(exist_ok=True, parents=True)
        output_file = output_dir / f"processed_{task_id}.xlsx"
        
        # Generate reports
        pipeline.generate_reports(results, str(output_dir))
        
        # Convert transactions for storage
        transactions_data = [
            {
                "date": t.date.isoformat(),
                "amount": t.amount,
                "description": t.description,
                "account": t.account,
                "category": t.category,
                "confidence": t.confidence,
                "is_duplicate": t.is_duplicate,
                "duplicate_group": t.duplicate_group,
                "source_sheet": t.source_sheet
            }
            for t in results["transactions"]
        ]
        
        # Update task result
        task_results[task_id].update({
            "status": "completed",
            "completed_at": datetime.now(),
            "transactions": transactions_data,
            "summary": results["summary"],
            "processing_time": processing_time,
            "output_file": str(output_file)
        })
        
        # Record metrics
        PROCESSING_DURATION.observe(processing_time)
        PROCESSING_COUNT.labels(status="success").inc()
        
        logger.info(
            "Background processing completed",
            task_id=task_id,
            processing_time=processing_time,
            transaction_count=len(transactions_data)
        )
        
    except Exception as e:
        # Update task with error
        task_results[task_id].update({
            "status": "failed",
            "error": str(e),
            "completed_at": datetime.now()
        })
        
        PROCESSING_COUNT.labels(status="error").inc()
        logger.error("Background processing failed", task_id=task_id, error=str(e))
        
    finally:
        # Cleanup temporary file
        if os.path.exists(file_path):
            os.unlink(file_path)


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with structured logging."""
    logger.warning(
        "HTTP exception",
        status_code=exc.status_code,
        detail=exc.detail,
        url=str(request.url)
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP Error",
            "message": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(
        "Unhandled exception",
        error=str(exc),
        url=str(request.url),
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An unexpected error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )


# Startup/shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    logger.info("Application starting up")
    
    # Create required directories
    Path("data/uploads").mkdir(exist_ok=True, parents=True)
    Path("data/output").mkdir(exist_ok=True, parents=True)
    Path("data/temp").mkdir(exist_ok=True, parents=True)
    Path("logs").mkdir(exist_ok=True, parents=True)
    
    logger.info("Application startup complete")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Application shutting down")
    
    # Cleanup temporary files
    temp_files = Path("data/temp").glob("*")
    for temp_file in temp_files:
        temp_file.unlink()
    
    logger.info("Application shutdown complete")


if __name__ == "__main__":
    import uvicorn
    
    # Development server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
