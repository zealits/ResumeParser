"""
Pinecone Vectoriser Module with proper ID management and complete metadata
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
import uuid

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Configuration constants
DATASET_DIR = "dataset"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
EMB_MODEL = "text-embedding-3-large"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Pinecone index names
PROFESSIONAL_INDEX = "professional-summary"
SKILLS_INDEX = "skills-matrix" 
PROJECT_INDEX = "project-portfolio"

# Project indexes (for reverse matching)
PROJECT_DESCRIPTION_INDEX = "project-description"
PROJECT_SKILLS_INDEX = "project-skills"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

class PineconeVectoriser:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=EMB_MODEL)
        self._ensure_indexes_exist()
    
    def _normalize_text(self, text: str) -> str:
        """Collapse newlines and excessive spaces into single spaces."""
        if not text:
            return ""
        return " ".join(text.replace("\n", " ").split())

    def _build_base_metadata(self, content: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
        """Build common metadata fields to be shared across all document types."""
        name = content.get("name", "")
        total_experience = content.get("total_experience", 0.0)
        seniority = self.derive_seniority_level(total_experience)
        highest_edu = self.extract_education_level(content.get("education", ""))
        has_lead = self.has_leadership_experience(content.get("experience", ""))
        return {
            "candidate_id": candidate_id,
            "name": name,
            "seniority_level": seniority,
            "highest_education": highest_edu,
            "has_leadership": has_lead,
        }

    def _ensure_indexes_exist(self):
        """Create Pinecone indexes if they don't exist"""
        index_names = [
            PROFESSIONAL_INDEX, 
            SKILLS_INDEX, 
            PROJECT_INDEX,
            PROJECT_DESCRIPTION_INDEX,
            PROJECT_SKILLS_INDEX
        ]
        
        for index_name in index_names:
            if index_name not in pc.list_indexes().names():
                pc.create_index(
                    name=index_name,
                    dimension=3072,  # OpenAI text-embedding-3-large dimension
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=PINECONE_ENVIRONMENT
                    )
                )
                print(f"Created Pinecone index: {index_name}")
    
    def _generate_vector_ids(self, candidate_id: str) -> Dict[str, str]:
        """
        Generate consistent vector IDs for all three document types
        """
        return {
            "professional_summary": f"prof_{candidate_id}",
            "skills_matrix": f"skills_{candidate_id}",
            "project_portfolio": f"project_{candidate_id}"
        }
    
    def _generate_project_vector_ids(self, project_id: str) -> Dict[str, str]:
        """
        Generate consistent vector IDs for project document types
        """
        return {
            "project_description": f"proj_desc_{project_id}",
            "project_skills": f"proj_skills_{project_id}"
        }
    
    def extract_prioritized_content(self, resume_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and prioritize content from resume data
        """
        content = {}
        
        # 1. Experience Section
        content["experience"] = self.format_experience_section(resume_data.get("experience", []))
        
        # 2. Projects Section (without explicit technologies lines)
        content["projects"] = self.format_projects_section(resume_data.get("projects", []))
        
        # 3. High-level Skills
        content["skills"] = self.format_skills_section(resume_data.get("skills", []))

        # Collect raw skill lists for skills_matrix aggregation
        content["skills_list"] = [s.strip() for s in resume_data.get("skills", []) if isinstance(s, str)]

        # Gather project skills
        projects_data = resume_data.get("projects", [])
        project_skills_list = []
        for proj in projects_data:
            skills = []
            for key in ["project_skills", "skills", "technologies", "tech_stack", "tools"]:
                val = proj.get(key)
                if isinstance(val, list):
                    skills.extend([str(x).strip() for x in val if str(x).strip()])
                elif isinstance(val, str) and val.strip():
                    skills.extend([s.strip() for s in val.split(',') if s.strip()])
            project_skills_list.extend(skills)
        content["project_skills_list"] = project_skills_list

        # Gather experience skills
        experience_data = resume_data.get("experience", [])
        experience_skills_list = []
        for exp in experience_data:
            skills = []
            for key in ["experience_skills", "skills", "technologies", "tech_stack", "tools"]:
                val = exp.get(key)
                if isinstance(val, list):
                    skills.extend([str(x).strip() for x in val if str(x).strip()])
                elif isinstance(val, str) and val.strip():
                    skills.extend([s.strip() for s in val.split(',') if s.strip()])
            experience_skills_list.extend(skills)
        content["experience_skills_list"] = experience_skills_list
        
        # 4. Education & Certifications
        content["education"] = self.format_education_section(resume_data.get("education", []))
        content["certifications"] = self.format_certifications_section(
            resume_data.get("certifications", []), 
            resume_data.get("achievements", [])
        )
        
        # Basic info
        content["name"] = resume_data.get("name", "")
        content["total_experience"] = resume_data.get("total_experience_years", 0.0)
        
        return content
    
    def format_experience_section(self, experience_data: List[Dict]) -> str:
        """Format experience data keeping only designation and description, no headers/dates."""
        if not experience_data:
            return ""

        lines = []
        for exp in experience_data:
            role = exp.get("designation", "").strip()
            description = (exp.get("description", "") or "").strip()
            if role and description:
                lines.append(f"- {role}: {description}")
            elif role:
                lines.append(f"- {role}")
            elif description:
                lines.append(f"- {description}")

        return "\n".join(lines)
    
    def format_projects_section(self, projects_data: List[Dict]) -> str:
        """Format projects with technologies and impact"""
        if not projects_data:
            return ""
        
        lines = ["KEY PROJECTS:"]
        for project in projects_data:
            title = project.get("title", "").strip()
            description = project.get("description", "")
            
            scale_info = self.extract_project_scale(description)
            key_achievement = self.extract_key_achievements(description)
            
            lines.append(f"- {title}")
            if key_achievement:
                lines.append(f"  Achievement: {key_achievement}")
            if scale_info:
                lines.append(f"  Scale: {scale_info}")
        
        return "\n".join(lines)
    
    def format_skills_section(self, skills_data: List[str]) -> str:
        """Format high-level skills only"""
        if not skills_data:
            return ""
        
        normalized_skills = [skill.strip().lower() for skill in skills_data if skill and isinstance(skill, str)]
        unique_skills = list(dict.fromkeys(normalized_skills))
        
        return f"TECHNICAL SKILLS:\n{', '.join(unique_skills)}"
    
    def format_education_section(self, education_data: List[Dict]) -> str:
        """Format only highest qualification"""
        if not education_data:
            return ""
        
        highest_edu = self.find_highest_education(education_data)
        if highest_edu:
            qualification = highest_edu.get("qualification", "").strip()
            institution = highest_edu.get("name", "").strip()
            return f"EDUCATION:\n{qualification} from {institution}"
        
        return ""
    
    def format_certifications_section(self, certifications: List[str], achievements: List[str]) -> str:
        """Format certifications and notable achievements"""
        content = []
        
        if certifications:
            content.append("CERTIFICATIONS:")
            content.extend([f"- {cert}" for cert in certifications if cert.strip()])
        
        if achievements:
            content.append("ACHIEVEMENTS:")
            content.extend([f"- {achievement}" for achievement in achievements if achievement.strip()])
        
        return "\n".join(content) if content else ""
    
    def extract_key_achievements(self, description: str) -> str:
        """Use full description"""
        return description
    
    def extract_project_scale(self, description: str) -> str:
        """Extract scale indicators from project description"""
        if not description:
            return ""
        
        scale_indicators = []
        
        if "500+" in description or "500 users" in description.lower():
            scale_indicators.append("500+ users")
        elif "100+" in description or "100 users" in description.lower():
            scale_indicators.append("100+ users")
        
        if "2-member" in description.lower() or "2 member" in description.lower():
            scale_indicators.append("team of 2")
        elif "5+" in description or "5 members" in description.lower():
            scale_indicators.append("team of 5+")
        
        if "94%" in description:
            scale_indicators.append("94% accuracy")
        elif "80%" in description:
            scale_indicators.append("80% components")
        
        return ", ".join(scale_indicators) if scale_indicators else ""
    
    def find_highest_education(self, education_data: List[Dict]) -> Dict[str, Any]:
        """Find the highest education qualification"""
        if not education_data:
            return {}
        
        education_rank = {
            "doctorate": 5, "phd": 5, "post graduate": 4, "masters": 4, "master": 4,
            "undergraduate": 3, "bachelor": 3, "be": 3, "b.tech": 3, "b.e.": 3,
            "diploma": 2, "vocational": 2, "polytechnic": 2,
            "higher secondary": 1, "hsc": 1, "12th": 1,
            "secondary": 0, "ssc": 0, "10th": 0
        }
        
        best_edu = None
        best_score = -1
        
        for edu in education_data:
            qualification = edu.get("qualification", "").lower()
            category = edu.get("category", "").lower()
            
            for key, score in education_rank.items():
                if key in qualification or key in category:
                    if score > best_score:
                        best_score = score
                        best_edu = edu
                    break
            else:
                if best_score < 0:
                    best_score = 0
                    best_edu = edu
        
        return best_edu or education_data[0]
    
    def create_document_types(self, extracted_content: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
        """Create 3 specialized document types"""
        
        name = extracted_content["name"]
        total_experience = extracted_content["total_experience"]
        
        # Generate consistent vector IDs
        vector_ids = self._generate_vector_ids(candidate_id)
        
        # Build base metadata for consistent usage across docs
        base_metadata = self._build_base_metadata(extracted_content, candidate_id)
        
        # Document Type 1: Professional Summary
        professional_summary = self.create_professional_summary(
            extracted_content, candidate_id, vector_ids["professional_summary"], base_metadata
        )
        
        # Document Type 2: Technical Skills Matrix
        skills_matrix = self.create_skills_matrix(
            extracted_content, candidate_id, vector_ids["skills_matrix"], base_metadata
        )
        
        # Document Type 3: Project Portfolio
        project_portfolio = self.create_project_portfolio(
            extracted_content, candidate_id, vector_ids["project_portfolio"], base_metadata
        )
        
        documents = {
            "professional_summary": professional_summary,
            "skills_matrix": skills_matrix,
            "project_portfolio": project_portfolio
        }
        
        return {
            "documents": documents,
            "metadata": base_metadata,
            "vector_ids": vector_ids
        }
    
    def create_professional_summary(self, content: Dict[str, Any], candidate_id: str, vector_id: str, metadata: Dict[str, Any]) -> Document:
        """Create professional summary document with only designation and description entries."""

        # Only include the cleaned experience lines (no name, dates, headers, education, achievements)
        raw_page_content = content.get("experience", "")
        page_content = self._normalize_text(raw_page_content)

        base_meta = self._build_base_metadata(content, candidate_id)
        document_metadata = {
            "candidate_id": base_meta["candidate_id"],
            "document_type": "professional_summary",
            "has_leadership": base_meta["has_leadership"],
            "highest_education": base_meta["highest_education"],
            "name": base_meta["name"],
            "seniority_level": base_meta["seniority_level"],
            "text": page_content,
        }

        return Document(
            page_content=page_content,
            metadata=document_metadata
        )
    
    def create_skills_matrix(self, content: Dict[str, Any], candidate_id: str, vector_id: str, metadata: Dict[str, Any]) -> Document:
        """Create skills matrix document"""
        
        all_skills = self.extract_all_skills(content)
        
        # Build a deduplicated flat technologies list from skills, project skills, and experience skills
        combined = []
        combined.extend(content.get("skills_list", []))
        combined.extend(content.get("project_skills_list", []))
        combined.extend(content.get("experience_skills_list", []))
        # Normalize to lowercase, dedupe preserving order
        seen = set()
        deduped = []
        for s in combined:
            key = s.strip().lower()
            if not key:
                continue
            if key not in seen:
                seen.add(key)
                deduped.append(key)
        # Final text: comma-separated single line
        page_content = self._normalize_text(
            ", ".join(deduped)
        )
        
        base_meta = self._build_base_metadata(content, candidate_id)
        document_metadata = {
            "candidate_id": base_meta["candidate_id"],
            "document_type": "skills_matrix",
            "has_leadership": base_meta["has_leadership"],
            "highest_education": base_meta["highest_education"],
            "name": base_meta["name"],
            "seniority_level": base_meta["seniority_level"],
            "text": page_content,
        }
        
        return Document(
            page_content=page_content,
            metadata=document_metadata
        )
    
    def create_project_portfolio(self, content: Dict[str, Any], candidate_id: str, vector_id: str, metadata: Dict[str, Any]) -> Document:
        """Create project portfolio document"""
        
        projects_text = content.get("projects", "")
        if not projects_text:
            projects_text = "No project experience documented."
        
        # Remove newlines and ensure no technologies lines were added upstream
        page_content = self._normalize_text(f"PROJECT PORTFOLIO: {projects_text}")
        
        base_meta = self._build_base_metadata(content, candidate_id)
        document_metadata = {
            "candidate_id": base_meta["candidate_id"],
            "document_type": "project_portfolio",
            "has_leadership": base_meta["has_leadership"],
            "highest_education": base_meta["highest_education"],
            "name": base_meta["name"],
            "seniority_level": base_meta["seniority_level"],
            "text": page_content,
        }
        
        return Document(
            page_content=page_content,
            metadata=document_metadata
        )
    
    def extract_current_role(self, experience_text: str) -> str:
        """Extract current/most recent role from experience"""
        if not experience_text:
            return "Professional"
        
        lines = experience_text.split('\n')
        for line in lines:
            if line.startswith('-') and ' at ' in line:
                role_part = line.split(' at ')[0]
                return role_part.replace('-', '').strip()
        
        return "Professional"
    
    def extract_all_skills(self, content: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract and categorize all skills from different sections"""
        
        skills_text = content.get("skills", "")
        core_skills = []
        if "TECHNICAL SKILLS:" in skills_text:
            skills_line = skills_text.split("TECHNICAL SKILLS:")[1].strip()
            core_skills = [skill.strip() for skill in skills_line.split(',') if skill.strip()]
        
        projects_text = content.get("projects", "")
        project_technologies = self.extract_technologies_from_text(projects_text)
        
        domains = self.infer_technical_domains(core_skills + project_technologies)
        
        return {
            "core_skills": core_skills[:15],
            "project_technologies": project_technologies[:10],
            "domains": domains
        }
    
    def extract_technologies_from_text(self, text: str) -> List[str]:
        """Extract technology mentions from text"""
        if not text:
            return []
        
        common_techs = [
            'python', 'react', 'javascript', 'java', 'django', 'node.js', 'express.js',
            'html', 'css', 'mongodb', 'mysql', 'sql', 'tensorflow', 'pytorch', 'opencv',
            'aws', 'docker', 'kubernetes', 'git', 'rest', 'api', 'mern', 'blockchain',
            'ethereum', 'web3.js', 'streamlit', 'yolov8', 'llm', 'gen ai', 'agentic ai',
            'zoho', 'creator', 'catalyst', 'deluge', 'rag', 'langchain', 'openai', 'zia'
        ]
        
        found_techs = []
        text_lower = text.lower()
        
        for tech in common_techs:
            if tech in text_lower:
                found_techs.append(tech)
        
        return found_techs
    
    def infer_technical_domains(self, skills: List[str]) -> List[str]:
        """Infer technical domains from skills"""
        domains = set()
        
        domain_mapping = {
            'web_development': ['react', 'javascript', 'html', 'css', 'node.js', 'express.js', 'django'],
            'ai_ml': ['python', 'tensorflow', 'pytorch', 'machine learning', 'deep learning', 'llm', 'gen ai', 'agentic ai', 'rag', 'langchain'],
            'data_science': ['python', 'opencv', 'yolov8', 'data analysis'],
            'blockchain': ['blockchain', 'ethereum', 'web3.js', 'smart contracts'],
            'backend': ['python', 'java', 'django', 'node.js', 'express.js', 'rest', 'api'],
            'frontend': ['react', 'javascript', 'html', 'css'],
            'database': ['mongodb', 'mysql', 'sql'],
            'devops': ['aws', 'docker', 'kubernetes', 'git'],
            'low_code': ['zoho', 'creator', 'catalyst', 'deluge']
        }
        
        for skill in skills:
            skill_lower = skill.lower()
            for domain, domain_skills in domain_mapping.items():
                if any(domain_skill in skill_lower for domain_skill in domain_skills):
                    domains.add(domain.replace('_', ' ').title())
        
        return list(domains)[:5]
    
    def create_comprehensive_metadata(self, content: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
        """Create comprehensive metadata for filtering"""
        
        name = content["name"]
        total_experience = content["total_experience"]
        
        all_skills = self.extract_all_skills(content)
        
        metadata = {
            "candidate_id": candidate_id,
            "name": name,
            "total_experience_years": total_experience,
            "seniority_level": self.derive_seniority_level(total_experience),
            "highest_education": self.extract_education_level(content.get("education", "")),
            "primary_skills": all_skills.get("core_skills", [])[:10],
            "technologies_used": all_skills.get("project_technologies", [])[:8],
            "technical_domains": all_skills.get("domains", [])[:3],
            "project_count": self.count_projects(content.get("projects", "")),
            "has_leadership": self.has_leadership_experience(content.get("experience", "")),
            "created_at": datetime.utcnow().isoformat() + "Z"
        }
        
        return metadata
    
    def derive_seniority_level(self, total_experience: float) -> str:
        """Derive seniority level from total experience"""
        if total_experience >= 5:
            return "senior"
        elif total_experience >= 2:
            return "mid"
        elif total_experience > 0:
            return "junior"
        else:
            return "fresher"
    
    def extract_education_level(self, education_text: str) -> str:
        """Extract simplified education level"""
        if not education_text:
            return "Unknown"
        
        education_text_lower = education_text.lower()
        
        if any(term in education_text_lower for term in ["be", "bachelor", "undergraduate", "b.tech"]):
            return "BE"
        elif any(term in education_text_lower for term in ["diploma", "polytechnic", "vocational"]):
            return "Diploma"
        elif any(term in education_text_lower for term in ["masters", "master", "post graduate"]):
            return "Masters"
        elif any(term in education_text_lower for term in ["phd", "doctorate"]):
            return "PhD"
        else:
            return "Other"
    
    def count_projects(self, projects_text: str) -> int:
        """Count number of projects from projects text"""
        if not projects_text:
            return 0
        
        # Count project entries (lines starting with "-" after KEY PROJECTS)
        lines = projects_text.split('\n')
        count = 0
        in_projects_section = False
        
        for line in lines:
            if "KEY PROJECTS:" in line:
                in_projects_section = True
                continue
            if in_projects_section and line.strip().startswith('-'):
                count += 1
        
        return count
    
    def has_leadership_experience(self, experience_text: str) -> bool:
        """Check if candidate has leadership experience"""
        if not experience_text:
            return False
        
        leadership_indicators = ["led", "managed", "team", "supervised", "mentored", "guided", "coordinated"]
        experience_lower = experience_text.lower()
        
        return any(indicator in experience_lower for indicator in leadership_indicators)
    
    def add_candidate(self, resume_data: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
        """
        Add a single candidate to Pinecone indexes with proper ID management
        """
        try:
            # Extract prioritized content
            extracted_content = self.extract_prioritized_content(resume_data)
            
            # Create document types and metadata
            result = self.create_document_types(extracted_content, candidate_id)
            
            documents = result["documents"]
            metadata = result["metadata"]
            vector_ids = result["vector_ids"]
            
            # Add to professional summary index with custom ID and enhanced metadata
            professional_store = PineconeVectorStore.from_documents(
                documents=[documents["professional_summary"]],
                embedding=self.embeddings,
                index_name=PROFESSIONAL_INDEX,
                ids=[vector_ids["professional_summary"]]
            )
            
            # Add to skills matrix index with custom ID and enhanced metadata
            skills_store = PineconeVectorStore.from_documents(
                documents=[documents["skills_matrix"]],
                embedding=self.embeddings,
                index_name=SKILLS_INDEX,
                ids=[vector_ids["skills_matrix"]]
            )
            
            # Add to project portfolio index with custom ID and enhanced metadata
            project_store = PineconeVectorStore.from_documents(
                documents=[documents["project_portfolio"]],
                embedding=self.embeddings,
                index_name=PROJECT_INDEX,
                ids=[vector_ids["project_portfolio"]]
            )
            
            print(f"Successfully added candidate '{extracted_content['name']}' to Pinecone indexes")
            print(f"Vector IDs: {vector_ids}")
            print(f"Seniority Level: {metadata.get('seniority_level')}")
            print(f"Highest Education: {metadata.get('highest_education')}")
            print(f"Has Leadership: {metadata.get('has_leadership')}")
            
            return {
                "success": True,
                "candidate_id": candidate_id,
                "name": extracted_content["name"],
                "metadata": metadata,
                "vector_ids": vector_ids  # Return vector IDs to store in MongoDB
            }
            
        except Exception as e:
            print(f"Error adding candidate to Pinecone: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_candidate(self, resume_data: Dict[str, Any], candidate_id: str) -> Dict[str, Any]:
        """
        Update a candidate in Pinecone indexes
        """
        try:
            # First delete existing vectors
            self.delete_candidate(candidate_id)
            
            # Then add updated vectors
            return self.add_candidate(resume_data, candidate_id)
            
        except Exception as e:
            print(f"Error updating candidate in Pinecone: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete_candidate(self, candidate_id: str) -> bool:
        """
        Delete all vectors for a candidate from all indexes using known vector IDs
        """
        try:
            vector_ids = self._generate_vector_ids(candidate_id)
            
            # Delete from professional summary index
            professional_index = pc.Index(PROFESSIONAL_INDEX)
            professional_index.delete(ids=[vector_ids["professional_summary"]])
            
            # Delete from skills matrix index
            skills_index = pc.Index(SKILLS_INDEX)
            skills_index.delete(ids=[vector_ids["skills_matrix"]])
            
            # Delete from project portfolio index
            project_index = pc.Index(PROJECT_INDEX)
            project_index.delete(ids=[vector_ids["project_portfolio"]])
            
            print(f"Successfully deleted candidate '{candidate_id}' from Pinecone indexes")
            print(f"Deleted vector IDs: {list(vector_ids.values())}")
            return True
            
        except Exception as e:
            print(f"Error deleting candidate from Pinecone: {e}")
            return False
    
    # ============================================================
    # PROJECT VECTORIZATION METHODS (for reverse matching)
    # ============================================================
    
    def create_project_documents(self, project_data: Dict[str, Any], project_id: str) -> Dict[str, Any]:
        """
        Create 2 document types for a project:
        1. Project Description - for matching with candidate professional_summary + project_portfolio
        2. Project Skills - for matching with candidate skills_matrix
        """
        project_description = project_data.get("project_description", "")
        project_skills = project_data.get("project_skills", [])
        
        # Normalize and format skills
        if isinstance(project_skills, list):
            skills_text = ", ".join([str(skill).strip() for skill in project_skills if skill])
        elif isinstance(project_skills, str):
            skills_text = project_skills
        else:
            skills_text = ""
        
        # Generate vector IDs
        vector_ids = self._generate_project_vector_ids(project_id)
        
        # Create Project Description Document
        description_content = self._normalize_text(project_description)
        description_metadata = {
            "project_id": project_id,
            "document_type": "project_description",
            "text": description_content,
        }
        description_doc = Document(
            page_content=description_content,
            metadata=description_metadata
        )
        
        # Create Project Skills Document
        skills_content = self._normalize_text(skills_text)
        skills_metadata = {
            "project_id": project_id,
            "document_type": "project_skills",
            "text": skills_content,
            "skills_list": project_skills if isinstance(project_skills, list) else []
        }
        skills_doc = Document(
            page_content=skills_content,
            metadata=skills_metadata
        )
        
        return {
            "documents": {
                "project_description": description_doc,
                "project_skills": skills_doc
            },
            "vector_ids": vector_ids
        }
    
    def add_project(self, project_data: Dict[str, Any], project_id: str) -> Dict[str, Any]:
        """
        Add a project to Pinecone indexes (project_description and project_skills)
        """
        try:
            # Create project documents
            result = self.create_project_documents(project_data, project_id)
            documents = result["documents"]
            vector_ids = result["vector_ids"]
            
            # Add to project description index
            description_store = PineconeVectorStore.from_documents(
                documents=[documents["project_description"]],
                embedding=self.embeddings,
                index_name=PROJECT_DESCRIPTION_INDEX,
                ids=[vector_ids["project_description"]]
            )
            
            # Add to project skills index
            skills_store = PineconeVectorStore.from_documents(
                documents=[documents["project_skills"]],
                embedding=self.embeddings,
                index_name=PROJECT_SKILLS_INDEX,
                ids=[vector_ids["project_skills"]]
            )
            
            print(f"Successfully added project '{project_id}' to Pinecone indexes")
            print(f"Vector IDs: {vector_ids}")
            
            return {
                "success": True,
                "project_id": project_id,
                "metadata": {
                    "project_description": project_data.get("project_description", ""),
                    "project_skills": project_data.get("project_skills", [])
                },
                "vector_ids": vector_ids
            }
            
        except Exception as e:
            print(f"Error adding project to Pinecone: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def update_project(self, project_data: Dict[str, Any], project_id: str) -> Dict[str, Any]:
        """
        Update a project in Pinecone indexes
        """
        try:
            # First delete existing vectors
            self.delete_project(project_id)
            
            # Then add updated vectors
            return self.add_project(project_data, project_id)
            
        except Exception as e:
            print(f"Error updating project in Pinecone: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete all vectors for a project from all indexes using known vector IDs
        """
        try:
            vector_ids = self._generate_project_vector_ids(project_id)
            
            # Delete from project description index
            description_index = pc.Index(PROJECT_DESCRIPTION_INDEX)
            description_index.delete(ids=[vector_ids["project_description"]])
            
            # Delete from project skills index
            skills_index = pc.Index(PROJECT_SKILLS_INDEX)
            skills_index.delete(ids=[vector_ids["project_skills"]])
            
            print(f"Successfully deleted project '{project_id}' from Pinecone indexes")
            print(f"Deleted vector IDs: {list(vector_ids.values())}")
            return True
            
        except Exception as e:
            print(f"Error deleting project from Pinecone: {e}")
            return False

# Global instance
pinecone_vectoriser = PineconeVectoriser()