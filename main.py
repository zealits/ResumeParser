from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
import json
import time
import logging
from dotenv import load_dotenv
from loadjson import extract_text_from_pdf, parse_resume_with_genai

# Import authentication and database modules
from auth import get_current_user, auth_manager
from database import db
from auth_routes import router as auth_router
from middleware import (
    UsageTrackingMiddleware, SecurityHeadersMiddleware, 
    RequestLoggingMiddleware, FileSizeLimitMiddleware, CORSHeadersMiddleware
)
from config import settings
from models import SubscriptionTier, SUBSCRIPTION_LIMITS

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Database initialization with lifespan
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database connection on startup and cleanup on shutdown"""
    # Startup
    try:
        await db.connect()
        logger.info("✅ Database connected successfully")
        
        # Create default admin user if it doesn't exist
        admin_user = await db.get_user_by_username(settings.ADMIN_USERNAME)
        if not admin_user:
            admin_data = {
                "username": settings.ADMIN_USERNAME,
                "email": settings.ADMIN_EMAIL,
                "password": settings.ADMIN_PASSWORD,
                "subscription_tier": SubscriptionTier.ENTERPRISE,
                "api_calls_limit": -1,  # Unlimited
                "company_name": "System Admin",
                "contact_person": "System Administrator",
                "role": "admin",
                "status": "active",
                "is_active": True
            }
            
            result = await auth_manager.create_user_account(admin_data)
            logger.info(f"✅ Default admin user created: {settings.ADMIN_USERNAME}")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize database: {e}")
        raise
    
    yield
    
    # Shutdown
    try:
        await db.disconnect()
        logger.info("✅ Database disconnected successfully")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Add lifespan to FastAPI app
app = FastAPI(
    title="Resume Parser API",
    description="API for parsing resume PDFs and extracting structured information with authentication",
    version="2.0.0",
    docs_url=settings.API_DOCS_URL if settings.API_DOCS_ENABLED else None,
    redoc_url=settings.API_REDOC_URL if settings.API_DOCS_ENABLED else None,
    lifespan=lifespan
)

# Add middleware in order (last added is first executed)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(FileSizeLimitMiddleware, max_file_size=settings.MAX_FILE_SIZE)
app.add_middleware(UsageTrackingMiddleware, track_endpoints=["/parse-resume", "/parse-resume-text"])
app.add_middleware(CORSHeadersMiddleware, allowed_origins=settings.ALLOWED_ORIGINS)

# Include authentication routes
app.include_router(auth_router)

@app.get("/")
async def root():
    return {"message": "Resume Parser API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "resume-parser"}

@app.post("/parse-resume", summary="Parse resume PDF")
async def parse_resume(
    request: Request,
    file: UploadFile = File(..., description="Resume PDF file"),
    current_user: dict = Depends(get_current_user)
):
    """
    Parse a resume PDF file and return structured JSON data.
    
    - **file**: PDF file containing the resume
    """
    # Set user in request state for middleware
    request.state.user = current_user
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file format. Please upload a PDF file."
        )
    
    # Create temporary file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            # Read uploaded file content
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        # Extract text from PDF
        raw_text = extract_text_from_pdf(temp_path)
        if not raw_text:
            raise HTTPException(
                status_code=400, 
                detail="Could not extract text from PDF. The file might be corrupted or contain no text."
            )
        
        # Parse resume with GenAI
        parsed_data = parse_resume_with_genai(raw_text)
        
        # Check for parsing errors
        if "error" in parsed_data:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse resume: {parsed_data['error']}"
            )
        
        return JSONResponse(
            content={
                "success": True,
                "filename": file.filename,
                "parsed_data": parsed_data
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing file: {str(e)}"
        )
    finally:
        # Clean up temporary file
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass

@app.post("/parse-resume-text", summary="Parse resume from text")
async def parse_resume_text(
    request: Request,
    resume_text: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Parse resume information from raw text.
    
    - **resume_text**: Raw text content of the resume
    """
    # Set user in request state for middleware
    request.state.user = current_user
    
    try:
        if not resume_text or not resume_text.strip():
            raise HTTPException(
                status_code=400, 
                detail="Resume text cannot be empty"
            )
        
        # Parse resume with GenAI
        parsed_data = parse_resume_with_genai(resume_text.strip())
        
        # Check for parsing errors
        if "error" in parsed_data:
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to parse resume: {parsed_data['error']}"
            )
        
        return JSONResponse(
            content={
                "success": True,
                "parsed_data": parsed_data
            }
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing resume text: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    
    # Validate configuration
    config_issues = settings.validate_config()
    if config_issues and settings.is_production():
        logger.error("❌ Configuration issues found in production mode:")
        for issue in config_issues:
            logger.error(f"   - {issue}")
        exit(1)
    elif config_issues:
        logger.warning("⚠️  Configuration issues found:")
        for issue in config_issues:
            logger.warning(f"   - {issue}")
    
    # Run the application
    uvicorn.run(
        app, 
        host=settings.HOST, 
        port=settings.PORT,
        ssl_keyfile=settings.SSL_KEY_PATH if settings.SSL_ENABLED else None,
        ssl_certfile=settings.SSL_CERT_PATH if settings.SSL_ENABLED else None
    )