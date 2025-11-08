from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
import json
import time
import logging
import re
from uuid import uuid4
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
from services.resumeParser import extract_text_from_pdf, parse_resume_with_genai
from services.vectoriser import pinecone_vectoriser
from services.retrival import CandidateRetrievalPipeline
from services.project_retrieval import ProjectRetrievalPipeline
from services.testgen import generate_test
from services.evaluate import evaluate_test

# Import authentication and database modules
from auth import get_current_user, auth_manager
from database import db
from auth_routes import router as auth_router
from middleware import (
    UsageTrackingMiddleware, SecurityHeadersMiddleware, 
    RequestLoggingMiddleware, FileSizeLimitMiddleware, CORSHeadersMiddleware
)
from config import settings
from models import SubscriptionTier, SUBSCRIPTION_LIMITS, ProjectRegisterRequest, ProjectUpdateRequest, ProjectResponse

# Load environment variables from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATASET_DIR = os.getenv("DATASET_DIR", "dataset")
os.makedirs(DATASET_DIR, exist_ok=True)

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
    title="Resume Parser API with RAG-based ATS",
    description="API for parsing resume PDFs, extracting structured information, and managing candidates with vector search using Pinecone. Includes authentication and subscription management.",
    version="2.0.0",
    docs_url=settings.API_DOCS_URL if settings.API_DOCS_ENABLED else None,
    redoc_url=settings.API_REDOC_URL if settings.API_DOCS_ENABLED else None,
    lifespan=lifespan
)

# Add middleware in order (last added is first executed)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(FileSizeLimitMiddleware, max_file_size=settings.MAX_FILE_SIZE)
app.add_middleware(UsageTrackingMiddleware, track_endpoints=[
    "/parse-resume", 
    "/parse-resume-text",
    "/register-json",
    "/get-ranked-candidates",
    "/register-project",
    "/project/{project_id}",
    "/candidate/{candidate_id}/relevant-projects"
])
app.add_middleware(CORSHeadersMiddleware, allowed_origins=settings.ALLOWED_ORIGINS)

# Include authentication routes
app.include_router(auth_router)

@app.get("/")
async def root():
    return {"message": "Resume Parser API is running", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "resume-parser"}

# ------------------------------------------------------------
# Helper function: Extract candidate summary
# ------------------------------------------------------------
def extract_candidate_summary(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract candidate summary details from MongoDB document:
    1) Highest qualification (only qualification + category)
    2) Project - description + project_skills
    3) Experience - designation + description + experience_skills
    4) Certifications
    """

    # --- Highest Qualification ---
    education_data = doc.get("education", [])
    highest_qualification = None

    if isinstance(education_data, list) and education_data:
        education_rank = {
            "post graduate": 4, "masters": 4, "master": 4,
            "undergraduate": 3, "bachelor": 3, "be": 3, "b.tech": 3, "b.e.": 3,
            "diploma": 2, "vocational": 2,
            "higher secondary": 1, "hsc": 1, "12th": 1,
            "secondary": 0, "ssc": 0, "10th": 0
        }

        best_edu, best_score = None, -1
        for edu in education_data:
            qual = str(edu.get("qualification", "")).lower()
            for key, score in education_rank.items():
                if key in qual and score > best_score:
                    best_score = score
                    best_edu = edu

        highest_qualification = best_edu or education_data[0]

        # Only keep qualification + category
        highest_qualification = {
            "qualification": highest_qualification.get("qualification", "Unknown"),
            "category": highest_qualification.get("category", "Unknown")
        }
    else:
        highest_qualification = {"qualification": "Not available", "category": "Not available"}

    # --- Projects ---
    projects_data = doc.get("projects", [])
    projects: List[Dict[str, Any]] = []
    if isinstance(projects_data, list):
        for proj in projects_data:
            description = proj.get("description", "No description provided.")
            project_skills = (
                proj.get("project_skills")
                or proj.get("skills")
                or proj.get("technologies")
                or []
            )
            if isinstance(project_skills, str):
                project_skills = [s.strip() for s in project_skills.split(",") if s.strip()]
            projects.append({
                "description": description,
                "project_skills": project_skills
            })

    # --- Experience ---
    experience_data = doc.get("experience", [])
    experience: List[Dict[str, Any]] = []
    if isinstance(experience_data, list):
        for exp in experience_data:
            designation = exp.get("designation", "Unknown Role")
            description = exp.get("description", "No description provided.")
            experience_skills = (
                exp.get("experiance_skills")
                or exp.get("skills")
                or exp.get("technologies")
                or []
            )
            if isinstance(experience_skills, str):
                experience_skills = [s.strip() for s in experience_skills.split(",") if s.strip()]
            experience.append({
                "designation": designation,
                "description": description,
                "experience_skills": experience_skills
            })

    # --- Certifications ---
    certifications = doc.get("certifications", [])
    if not isinstance(certifications, list):
        certifications = []

    return {
        "highest_qualification": highest_qualification,
        "projects": projects,
        "experience": experience,
        "certifications": certifications
    }

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

# ------------------------------------------------------------
# Candidate Management Endpoints 
# ------------------------------------------------------------

@app.post("/register-json", status_code=201, summary="Register candidate JSON")
async def register_json(
    request: Request,
    payload: Dict[str, Any] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Accept confirmed JSON, store in MongoDB, and add to Pinecone incrementally.
    
    - **payload**: Candidate resume data in JSON format
    """
    # Set user in request state for middleware
    request.state.user = current_user
    
    try:
        candidate_id = str(uuid4())
        payload_copy = dict(payload)
        
        # Add to Pinecone FIRST to get vector IDs
        pinecone_result = pinecone_vectoriser.add_candidate(payload_copy, candidate_id)
        
        if not pinecone_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to add candidate to Pinecone: {pinecone_result.get('error', 'Unknown error')}"
            )

        # Now store in MongoDB WITH vector IDs
        payload_copy["_id"] = candidate_id
        payload_copy["created_at"] = datetime.utcnow().isoformat() + "Z"
        payload_copy["vector_ids"] = pinecone_result["vector_ids"]  # Store vector IDs
        payload_copy["pinecone_metadata"] = pinecone_result["metadata"]  # Store Pinecone metadata

        # Insert in MongoDB using async database connection
        await db.candidates_collection.insert_one(payload_copy)
        logger.info(f"Saved candidate to MongoDB with id: {candidate_id}")

        # Save JSON to dataset/ (optional, for backup)
        file_path = Path(DATASET_DIR) / f"{candidate_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload_copy, f, ensure_ascii=False, indent=2)

        logger.info(f"Successfully registered candidate: {candidate_id}")
        logger.info(f"Vector IDs stored: {pinecone_result['vector_ids']}")

        return {
            "success": True, 
            "candidate_id": candidate_id,
            "vector_ids": pinecone_result["vector_ids"],
            "message": "Candidate registered successfully and added to vector database"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to register JSON")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candidate/{candidate_id}", summary="Get candidate by ID")
async def get_candidate(
    request: Request,
    candidate_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieve candidate information by candidate ID.
    
    - **candidate_id**: Unique identifier of the candidate
    """
    # Set user in request state for middleware
    request.state.user = current_user
    
    doc = await db.candidates_collection.find_one({"_id": candidate_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    # Convert ObjectId to string for JSON serialization if present
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    
    return {
        "success": True, 
        "candidate": doc,
        "vector_ids": doc.get("vector_ids", {})
    }

@app.put("/candidate/{candidate_id}", summary="Update candidate")
async def update_candidate(
    request: Request,
    candidate_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Update candidate JSON and update Pinecone.
    
    - **candidate_id**: Unique identifier of the candidate
    - **payload**: Updated candidate resume data in JSON format
    """
    # Set user in request state for middleware
    request.state.user = current_user
    
    try:
        # Check if candidate exists
        existing_doc = await db.candidates_collection.find_one({"_id": candidate_id})
        if not existing_doc:
            raise HTTPException(status_code=404, detail="Candidate not found")

        # Update in Pinecone
        pinecone_result = pinecone_vectoriser.update_candidate(payload, candidate_id)
        
        if not pinecone_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update candidate in Pinecone: {pinecone_result.get('error', 'Unknown error')}"
            )

        # Update MongoDB document
        payload_copy = dict(payload)
        payload_copy["_id"] = candidate_id
        payload_copy["updated_at"] = datetime.utcnow().isoformat() + "Z"
        payload_copy["vector_ids"] = pinecone_result["vector_ids"]  # Update vector IDs
        payload_copy["pinecone_metadata"] = pinecone_result["metadata"]  # Update metadata
        # Preserve created_at if it exists
        if "created_at" in existing_doc:
            payload_copy["created_at"] = existing_doc["created_at"]

        await db.candidates_collection.replace_one({"_id": candidate_id}, payload_copy, upsert=True)

        # Update dataset file (optional backup)
        file_path = Path(DATASET_DIR) / f"{candidate_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(payload_copy, f, ensure_ascii=False, indent=2)

        logger.info(f"Successfully updated candidate: {candidate_id}")
        logger.info(f"Updated vector IDs: {pinecone_result['vector_ids']}")

        return {
            "success": True, 
            "candidate_id": candidate_id,
            "vector_ids": pinecone_result["vector_ids"],
            "message": "Candidate updated successfully in vector database"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update candidate")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/candidate/{candidate_id}", summary="Delete candidate")
async def delete_candidate(
    request: Request,
    candidate_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete candidate and remove from Pinecone.
    
    - **candidate_id**: Unique identifier of the candidate
    """
    # Set user in request state for middleware
    request.state.user = current_user
    
    try:
        # Get candidate first to log vector IDs
        candidate = await db.candidates_collection.find_one({"_id": candidate_id})
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")

        # Delete from MongoDB
        result = await db.candidates_collection.delete_one({"_id": candidate_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Candidate not found")

        # Delete from Pinecone using stored vector IDs
        vector_ids = candidate.get("vector_ids", {})
        logger.info(f"Deleting candidate {candidate_id} with vector IDs: {vector_ids}")
        
        success = pinecone_vectoriser.delete_candidate(candidate_id)
        
        if not success:
            logger.warning(f"Failed to delete candidate {candidate_id} from Pinecone")

        # Delete dataset file (optional)
        file_path = Path(DATASET_DIR) / f"{candidate_id}.json"
        if file_path.exists():
            file_path.unlink()

        return {
            "success": True, 
            "message": "Candidate deleted successfully",
            "deleted_vector_ids": vector_ids
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete candidate")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candidate/{candidate_id}/vectors", summary="Get candidate vector IDs")
async def get_candidate_vectors(
    request: Request,
    candidate_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Get the vector IDs for a candidate.
    
    - **candidate_id**: Unique identifier of the candidate
    """
    # Set user in request state for middleware
    request.state.user = current_user
    
    doc = await db.candidates_collection.find_one({"_id": candidate_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Candidate not found")
    
    return {
        "success": True,
        "candidate_id": candidate_id,
        "vector_ids": doc.get("vector_ids", {}),
        "name": doc.get("name", "Unknown")
    }



@app.get("/candidate/{candidate_id}/summary", summary="Get candidate summary details")
async def get_candidate_summary(
    request: Request,
    candidate_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieve candidate summary details:
    1) Highest qualification (only qualification + category)
    2) Project - description + project_skills
    3) Experience - designation + description + experience_skills
    4) Certifications
    """
    # Set user in request state for middleware
    request.state.user = current_user

    doc = await db.candidates_collection.find_one({"_id": candidate_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Candidate not found")

    summary = extract_candidate_summary(doc)
    return summary

@app.get("/candidate/{candidate_id}/generate-test", summary="Generate candidate test")
async def generate_candidate_test(
    request: Request,
    candidate_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Generate a technical test for a candidate based on their profile summary.
    Takes candidate_id, extracts summary, sends to test generator, and returns generated test JSON.
    """
    # Set user in request state for middleware
    request.state.user = current_user

    try:
        # Get candidate document from MongoDB
        doc = await db.candidates_collection.find_one({"_id": candidate_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Candidate not found")

        # Extract candidate summary
        candidate_summary = extract_candidate_summary(doc)

        logger.info(f"Generating test for candidate: {candidate_id}")

        # Generate test
        test_output = generate_test(candidate_summary)

        # Parse the JSON response (generate_test may return a JSON string with fences)
        def _extract_json_string(s: str) -> str:
            s = s.strip()
            if s.startswith("```"):
                s = s.lstrip("`")
                s = s.split("\n", 1)[-1]
                if s.rstrip().endswith("```"):
                    s = s.rstrip().rstrip("`")
            first = s.find("{")
            last = s.rfind("}")
            if first != -1 and last != -1 and last > first:
                return s[first:last+1]
            return s

        try:
            candidate_json_str = _extract_json_string(test_output)
            test_json = json.loads(candidate_json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse test JSON: {str(e)}")
            logger.error(f"Raw output: {test_output}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to parse generated test JSON: {str(e)}"
            )

        logger.info(f"Successfully generated test for candidate: {candidate_id}")

        return {
            "success": True,
            "candidate_id": candidate_id,
            "test": test_json,
            "message": "Test generated successfully"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to generate test for candidate {candidate_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating test: {str(e)}"
        )

@app.post("/candidate/{candidate_id}/evaluate", summary="Evaluate candidate test submission")
async def evaluate_candidate_test(
    request: Request,
    candidate_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Evaluate a candidate's answers for a given test.
    Body expects:
    {
      "test": { ... },
      "answers": { "mcqs": [...], "theory": [...] }
    }
    Returns only marks scored (total and breakdown) and saves to MongoDB.
    """
    # Set user in request state for middleware
    request.state.user = current_user

    try:
        # Validate candidate exists
        doc = await db.candidates_collection.find_one({"_id": candidate_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Candidate not found")

        test = payload.get("test")
        answers = payload.get("answers")
        if not isinstance(test, dict) or not isinstance(answers, dict):
            raise HTTPException(status_code=400, detail="Invalid payload: 'test' and 'answers' objects are required")

        # Evaluate
        evaluation = evaluate_test(test, answers)

        # Prepare record
        record = {
            "candidate_id": candidate_id,
            "evaluation": evaluation,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }

        # Save evaluation to candidate document (history + latest fields)
        await db.candidates_collection.update_one(
            {"_id": candidate_id},
            {
                "$push": {"test_evaluations": record},
                "$set": {
                    "latest_test_score": evaluation.get("total", {}).get("score", 0),
                    "latest_test_max": evaluation.get("total", {}).get("max", 0),
                    "latest_test_at": record["created_at"],
                }
            }
        )

        # Return only marks to frontend
        return {
            "success": True,
            "candidate_id": candidate_id,
            "marks": evaluation.get("total", {}),
            "breakdown": {
                "mcq": evaluation.get("mcq", {}).get("score", 0),
                "theory": evaluation.get("theory", {}).get("score", 0)
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to evaluate test for candidate {candidate_id}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register-project", status_code=201, summary="Register project", response_model=ProjectResponse)
async def register_project(
    request: Request,
    payload: ProjectRegisterRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Register a project with project_description and project_skills.
    Stores in MongoDB and creates 2 vectors in Pinecone (project_description and project_skills).
    """
    # Set user in request state for middleware
    request.state.user = current_user
    
    try:
        # Convert Pydantic model to dict
        payload_dict = payload.model_dump(exclude_none=True)
        
        # Validate required fields
        project_description = payload_dict.get("project_description", "")
        project_skills = payload_dict.get("project_skills", [])
        project_heading = payload_dict.get("project_heading")  # Optional, MongoDB only
        application_deadline = payload_dict.get("application_deadline")
        
        if not project_description:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="project_description is required"
            )
        
        if not project_skills:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="project_skills is required (list or string)"
            )
        
        # Validate application_deadline if provided
        if application_deadline:
            try:
                # Parse to ensure it's a valid ISO format datetime
                deadline_str = str(application_deadline).strip()
                # Normalize: replace Z with +00:00 for parsing
                if deadline_str.endswith('Z'):
                    deadline_str = deadline_str.replace('Z', '+00:00')
                deadline_dt = datetime.fromisoformat(deadline_str)
                
                # Ensure timezone-aware (assume UTC if naive)
                if deadline_dt.tzinfo is None:
                    deadline_dt = deadline_dt.replace(tzinfo=timezone.utc)
                else:
                    # Convert to UTC if not already
                    deadline_dt = deadline_dt.astimezone(timezone.utc)
                
                # Store as ISO format string in UTC with Z suffix (not +00:00Z)
                # Format: YYYY-MM-DDTHH:MM:SSZ
                payload_dict["application_deadline"] = deadline_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, AttributeError) as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"application_deadline must be in ISO format (e.g., '2024-12-31T23:59:59Z'). Error: {str(e)}"
                )
        
        # Generate project ID
        project_id = str(uuid4())
        payload_copy = dict(payload_dict)
        
        # Add to Pinecone FIRST to get vector IDs
        pinecone_result = pinecone_vectoriser.add_project(payload_copy, project_id)
        
        if not pinecone_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to add project to Pinecone: {pinecone_result.get('error', 'Unknown error')}"
            )
        
        # Now store in MongoDB WITH vector IDs
        payload_copy["_id"] = project_id
        payload_copy["created_at"] = datetime.utcnow().isoformat() + "Z"
        payload_copy["vector_ids"] = pinecone_result["vector_ids"]  # Store vector IDs
        payload_copy["pinecone_metadata"] = pinecone_result["metadata"]  # Store Pinecone metadata
        
        # Insert in MongoDB using async database connection
        await db.projects_collection.insert_one(payload_copy)
        logger.info(f"Saved project to MongoDB with id: {project_id}")
        
        logger.info(f"Successfully registered project: {project_id}")
        logger.info(f"Vector IDs stored: {pinecone_result['vector_ids']}")
        
        return {
            "success": True,
            "project_id": project_id,
            "vector_ids": pinecone_result["vector_ids"],
            "message": "Project registered successfully and added to vector database"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to register project")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/project/{project_id}", summary="Update project", response_model=ProjectResponse)
async def update_project(
    request: Request,
    project_id: str,
    payload: ProjectUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """
    Update project JSON and update Pinecone.
    All fields are optional - only provided fields will be updated.
    """
    # Set user in request state for middleware
    request.state.user = current_user
    
    try:
        # Check if project exists
        existing_doc = await db.projects_collection.find_one({"_id": project_id})
        if not existing_doc:
            raise HTTPException(status_code=404, detail="Project not found")

        # Convert Pydantic model to dict, excluding None values
        payload_dict = payload.model_dump(exclude_none=True)
        
        # Merge with existing document - use provided values or keep existing ones
        project_description = payload_dict.get("project_description", existing_doc.get("project_description", ""))
        project_skills = payload_dict.get("project_skills", existing_doc.get("project_skills", []))
        
        if not project_description:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="project_description is required"
            )
        
        if not project_skills:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="project_skills is required (list or string)"
            )

        # Validate application_deadline if provided
        application_deadline = payload_dict.get("application_deadline")
        if application_deadline:
            try:
                # Parse to ensure it's a valid ISO format datetime
                deadline_str = str(application_deadline).strip()
                # Normalize: replace Z with +00:00 for parsing
                if deadline_str.endswith('Z'):
                    deadline_str = deadline_str.replace('Z', '+00:00')
                deadline_dt = datetime.fromisoformat(deadline_str)
                
                # Ensure timezone-aware (assume UTC if naive)
                if deadline_dt.tzinfo is None:
                    deadline_dt = deadline_dt.replace(tzinfo=timezone.utc)
                else:
                    # Convert to UTC if not already
                    deadline_dt = deadline_dt.astimezone(timezone.utc)
                
                # Store as ISO format string in UTC with Z suffix (not +00:00Z)
                # Format: YYYY-MM-DDTHH:MM:SSZ
                payload_dict["application_deadline"] = deadline_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, AttributeError) as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"application_deadline must be in ISO format (e.g., '2024-12-31T23:59:59Z'). Error: {str(e)}"
                )

        # Merge payload_dict with existing document for complete update
        # Only update fields that were provided, keep others from existing_doc
        update_data = {**existing_doc, **payload_dict}
        # Ensure required fields are present
        update_data["project_description"] = project_description
        update_data["project_skills"] = project_skills

        # Update in Pinecone (only send project_description and project_skills)
        pinecone_payload = {
            "project_description": project_description,
            "project_skills": project_skills
        }
        pinecone_result = pinecone_vectoriser.update_project(pinecone_payload, project_id)
        
        if not pinecone_result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to update project in Pinecone: {pinecone_result.get('error', 'Unknown error')}"
            )

        # Update MongoDB document
        payload_copy = dict(update_data)
        payload_copy["_id"] = project_id
        payload_copy["updated_at"] = datetime.utcnow().isoformat() + "Z"
        payload_copy["vector_ids"] = pinecone_result["vector_ids"]  # Update vector IDs
        payload_copy["pinecone_metadata"] = pinecone_result["metadata"]  # Update metadata
        # Preserve created_at if it exists
        if "created_at" in existing_doc:
            payload_copy["created_at"] = existing_doc["created_at"]

        await db.projects_collection.replace_one({"_id": project_id}, payload_copy, upsert=True)

        logger.info(f"Successfully updated project: {project_id}")
        logger.info(f"Updated vector IDs: {pinecone_result['vector_ids']}")

        return {
            "success": True, 
            "project_id": project_id,
            "vector_ids": pinecone_result["vector_ids"],
            "message": "Project updated successfully in vector database"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to update project")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/project/{project_id}", summary="Get project by ID", response_model=ProjectResponse)
async def get_project(
    request: Request,
    project_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Retrieve project information by project ID.
    
    - **project_id**: Unique identifier of the project
    """
    # Set user in request state for middleware
    request.state.user = current_user
    
    doc = await db.projects_collection.find_one({"_id": project_id})
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found")
    
    # Convert ObjectId to string for JSON serialization if present
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    
    return {
        "success": True, 
        "project": doc,
        "vector_ids": doc.get("vector_ids", {})
    }

@app.delete("/project/{project_id}", summary="Delete project")
async def delete_project(
    request: Request,
    project_id: str,
    current_user: dict = Depends(get_current_user)
):
    """
    Delete project and remove from Pinecone.
    
    - **project_id**: Unique identifier of the project
    """
    # Set user in request state for middleware
    request.state.user = current_user
    
    try:
        # Get project first to log vector IDs
        project = await db.projects_collection.find_one({"_id": project_id})
        if not project:
            raise HTTPException(status_code=404, detail="Project not found")

        # Delete from MongoDB
        result = await db.projects_collection.delete_one({"_id": project_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Project not found")

        # Delete from Pinecone using stored vector IDs
        vector_ids = project.get("vector_ids", {})
        logger.info(f"Deleting project {project_id} with vector IDs: {vector_ids}")
        
        success = pinecone_vectoriser.delete_project(project_id)
        
        if not success:
            logger.warning(f"Failed to delete project {project_id} from Pinecone")

        return {
            "success": True, 
            "message": "Project deleted successfully",
            "deleted_vector_ids": vector_ids
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to delete project")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/candidate/{candidate_id}/relevant-projects", summary="Get relevant projects for candidate")
async def get_relevant_projects_for_candidate(
    request: Request,
    candidate_id: str,
    top_k: int = 100,
    current_user: dict = Depends(get_current_user)
):
    """
    Get relevant projects for a candidate based on their profile.
    
    Matching logic:
    - professional_summary + project_portfolio → project_description index
    - skills_matrix → project_skills index
    
    Returns ranked list of project IDs with scores. Filters out projects with past application deadlines.
    
    - **candidate_id**: Unique identifier of the candidate
    - **top_k**: Number of top projects to return
    """
    # Set user in request state for middleware
    request.state.user = current_user
    
    try:
        # Get candidate document from MongoDB
        candidate_doc = await db.candidates_collection.find_one({"_id": candidate_id})
        if not candidate_doc:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        # Get candidate's vector IDs
        vector_ids = candidate_doc.get("vector_ids", {})
        if not vector_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Candidate vector IDs not found. Candidate may not be properly vectorized."
            )
        
        # Initialize retrieval pipeline
        retrieval_pipeline = ProjectRetrievalPipeline()
        
        # Get relevant projects
        results = retrieval_pipeline.get_relevant_projects_for_candidate(
            candidate_vector_ids=vector_ids,
            top_k=top_k
        )
        
        # Extract project IDs from combined ranked results
        project_results = [
            {
                "project_id": result["project_id"],
                "overall_score": result["overall_score"],
                "description_score": result["description_score"],
                "skills_score": result["skills_score"]
            }
            for result in results["combined_ranked"]
        ]
        
        # Filter out projects with past deadlines
        current_time = datetime.now(timezone.utc)  # Use timezone-aware UTC datetime
        valid_projects = []
        
        for project_result in project_results:
            project_id = project_result["project_id"]
            
            # Fetch project document from MongoDB to check deadline
            project_doc = await db.projects_collection.find_one({"_id": project_id})
            
            if not project_doc:
                # Project doesn't exist in MongoDB, skip it
                continue
            
            # Check application_deadline
            application_deadline = project_doc.get("application_deadline")
            
            if application_deadline:
                try:
                    # Normalize deadline string - handle various formats
                    deadline_str = str(application_deadline).strip()
                    
                    # Handle malformed formats like "2024-12-31T23:59:59+00:00Z"
                    # Remove trailing 'Z' if there's already timezone offset (+XX:XX or -XX:XX)
                    if deadline_str.endswith('Z'):
                        # Check if there's a timezone offset pattern before the Z
                        # Pattern: ends with +HH:MMZ or -HH:MMZ
                        if re.search(r'[+-]\d{2}:\d{2}Z?$', deadline_str):
                            # Has timezone offset, just remove trailing Z
                            deadline_str = deadline_str.rstrip('Z')
                        else:
                            # No timezone offset, replace Z with +00:00
                            deadline_str = deadline_str.replace('Z', '+00:00')
                    
                    # Parse the deadline
                    deadline_dt = datetime.fromisoformat(deadline_str)
                    
                    # Ensure deadline is timezone-aware (if naive, assume UTC)
                    if deadline_dt.tzinfo is None:
                        deadline_dt = deadline_dt.replace(tzinfo=timezone.utc)
                    
                    # Only include if deadline is in the future (strictly greater than current time)
                    if deadline_dt > current_time:
                        valid_projects.append(project_result)
                    # If deadline is in the past or equal to current time, skip this project
                    else:
                        logger.info(f"Skipping project {project_id} - deadline {application_deadline} ({deadline_dt.isoformat()}) is in the past (current: {current_time.isoformat()})")
                except (ValueError, AttributeError) as e:
                    # If deadline format is invalid, log error and EXCLUDE the project
                    # We can't verify if it's valid, so err on the side of caution
                    logger.error(f"Invalid deadline format for project {project_id}: {application_deadline}. Error: {str(e)}. Excluding project.")
                    # Do NOT add to valid_projects - exclude it
            else:
                # No deadline specified, include the project
                valid_projects.append(project_result)
        
        return {
            "success": True,
            "candidate_id": candidate_id,
            "candidate_name": candidate_doc.get("name", "Unknown"),
            "total_projects_matched": len(project_results),
            "total_valid_projects": len(valid_projects),
            "projects": valid_projects
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get relevant projects for candidate {candidate_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving relevant projects: {str(e)}"
        )

@app.post("/get-ranked-candidates", summary="Get ranked candidates for project")
async def get_ranked_candidates(
    request: Request,
    project_id: str = Body(..., description="ID of the project to match candidates against"),
    top_k: int = Body(100, description="Number of top candidates to return"),
    filters: Dict[str, Any] = Body(
        default={
            "has_leadership": None,
            "highest_education": None, 
            "seniority_level": None
        },
        description="Filters for candidate search (use null for any filter to ignore it)"
    ),
    current_user: dict = Depends(get_current_user)
):
    """
    Get ranked candidates based on project ID.
    Fetches project data from MongoDB and uses project_description and project_skills for matching.
    Returns top_k ranked candidates.
    """
    # Set user in request state for middleware
    request.state.user = current_user
    
    try:
        # Fetch project from MongoDB
        project_doc = await db.projects_collection.find_one({"_id": project_id})
        if not project_doc:
            raise HTTPException(status_code=404, detail="Project not found")
        
        # Extract project description and skills
        project_description = project_doc.get("project_description", "")
        project_skills_raw = project_doc.get("project_skills", [])
        
        # Handle project_skills - can be list or string
        if isinstance(project_skills_raw, str):
            # Convert comma-separated string to list
            required_skills = [skill.strip() for skill in project_skills_raw.split(",") if skill.strip()]
        elif isinstance(project_skills_raw, list):
            required_skills = [str(skill).strip() for skill in project_skills_raw if skill]
        else:
            required_skills = []
        
        if not project_description:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Project description not found in project data"
            )
        
        if not required_skills:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Project skills not found in project data"
            )
        
        # Initialize the retrieval pipeline
        retrieval_pipeline = CandidateRetrievalPipeline()
        
        # Retrieve ranked candidates (this returns all candidates, not just top-k)
        results = retrieval_pipeline.retrieve_ranked_candidates(
            project_description=project_description,
            required_skills=required_skills,
            filters=filters
        )
        
        # Limit to top_k results
        combined_results = results["combined_ranked"][:top_k]
        
        # Return only the combined ranked results (top_k)
        return {
            "success": True,
            "project_id": project_id,
            "project_description": project_description,
            "required_skills": required_skills,
            "filters_applied": filters,
            "top_k": top_k,
            "results_count": {
                "professional_summary": len(results["professional_summary_ranked"]),
                "project_portfolio": len(results["project_portfolio_ranked"]),
                "skills_matrix": len(results["skills_matrix_ranked"]),
                "combined_total": len(results["combined_ranked"]),
                "combined_returned": len(combined_results)
            },
            "combined_ranked_results": combined_results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get ranked candidates")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving ranked candidates: {str(e)}"
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