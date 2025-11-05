from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request, Body, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
import json
import time
import logging
from uuid import uuid4
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from dotenv import load_dotenv
from services.resumeParser import extract_text_from_pdf, parse_resume_with_genai
from services.vectoriser import pinecone_vectoriser
from services.retrival import CandidateRetrievalPipeline
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
from models import SubscriptionTier, SUBSCRIPTION_LIMITS

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
    "/get-ranked-candidates"
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
# Candidate Management Endpoints (from main2.py)
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

@app.post("/get-ranked-candidates", summary="Get ranked candidates for project")
async def get_ranked_candidates(
    request: Request,
    project_description: str = Body(..., description="Description of the project for matching"),
    required_skills: List[str] = Body(..., description="List of required technical skills"),
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
    Get ranked candidates based on project description and required skills across all three indexes.
    Returns combined ranked results without saving to files.
    
    - **project_description**: Description of the project for matching
    - **required_skills**: List of required technical skills
    - **filters**: Filters for candidate search
    """
    # Set user in request state for middleware
    request.state.user = current_user
    
    try:
        # Initialize the retrieval pipeline
        retrieval_pipeline = CandidateRetrievalPipeline()
        
        # Retrieve ranked candidates (this returns all candidates, not just top-k)
        results = retrieval_pipeline.retrieve_ranked_candidates(
            project_description=project_description,
            required_skills=required_skills,
            filters=filters
        )
        
        # Return only the combined ranked results
        return {
            "success": True,
            "project_description": project_description,
            "required_skills": required_skills,
            "filters_applied": filters,
            "results_count": {
                "professional_summary": len(results["professional_summary_ranked"]),
                "project_portfolio": len(results["project_portfolio_ranked"]),
                "skills_matrix": len(results["skills_matrix_ranked"]),
                "combined": len(results["combined_ranked"])
            },
            "combined_ranked_results": results["combined_ranked"]
        }

    except Exception as e:
        logger.exception("Failed to get ranked candidates")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving ranked candidates: {str(e)}"
        )

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