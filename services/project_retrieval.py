"""
Project Retrieval Pipeline for Candidate → Projects Matching
Matches candidate vectors against project indexes to find relevant projects
"""

import os
from typing import List, Dict, Any
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

class ProjectRetrievalPipeline:
    def __init__(self):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Project index names
        self.PROJECT_DESCRIPTION_INDEX = "project-description"
        self.PROJECT_SKILLS_INDEX = "project-skills"
        
        # Candidate index names (to fetch candidate vectors)
        self.PROFESSIONAL_INDEX = "professional-summary"
        self.SKILLS_INDEX = "skills-matrix"
        self.PROJECT_INDEX = "project-portfolio"
        
        # Initialize indexes
        self.project_description_index = self.pc.Index(self.PROJECT_DESCRIPTION_INDEX)
        self.project_skills_index = self.pc.Index(self.PROJECT_SKILLS_INDEX)
        self.professional_index = self.pc.Index(self.PROFESSIONAL_INDEX)
        self.skills_index = self.pc.Index(self.SKILLS_INDEX)
        self.project_portfolio_index = self.pc.Index(self.PROJECT_INDEX)
    
    def get_candidate_vector(self, index, vector_id: str) -> List[float]:
        """Fetch a candidate vector from Pinecone by vector ID"""
        try:
            result = index.fetch(ids=[vector_id])
            if result.vectors and vector_id in result.vectors:
                vector_data = result.vectors[vector_id]
                # Handle both dict and object access
                if hasattr(vector_data, 'values'):
                    return list(vector_data.values)
                elif isinstance(vector_data, dict) and 'values' in vector_data:
                    return list(vector_data['values'])
                elif isinstance(vector_data, list):
                    return vector_data
            return None
        except Exception as e:
            print(f"Error fetching vector {vector_id}: {e}")
            return None
    
    def search_projects_with_query_vector(self, index, query_vector: List[float], 
                                         top_k: int = 100) -> List[Dict[str, Any]]:
        """Search project index using a query vector"""
        try:
            results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            return results.matches
        except Exception as e:
            print(f"Error searching project index: {e}")
            return []
    
    def get_relevant_projects_for_candidate(self, candidate_vector_ids: Dict[str, str], 
                                           top_k: int = 100) -> Dict[str, List[Dict[str, Any]]]:
        """
        Match candidate vectors against project indexes:
        - professional_summary + project_portfolio → project_description index
        - skills_matrix → project_skills index
        
        Args:
            candidate_vector_ids: Dictionary with keys:
                - professional_summary: vector ID
                - skills_matrix: vector ID
                - project_portfolio: vector ID
            top_k: Number of top projects to return per search
        
        Returns:
            Dictionary with:
                - description_matches: Projects matched via description
                - skills_matches: Projects matched via skills
                - combined_ranked: Combined and ranked results
        """
        try:
            # Get candidate vectors
            prof_vector_id = candidate_vector_ids.get("professional_summary")
            project_portfolio_vector_id = candidate_vector_ids.get("project_portfolio")
            skills_vector_id = candidate_vector_ids.get("skills_matrix")
            
            # Fetch candidate vectors
            prof_vector = self.get_candidate_vector(self.professional_index, prof_vector_id) if prof_vector_id else None
            project_portfolio_vector = self.get_candidate_vector(self.project_portfolio_index, project_portfolio_vector_id) if project_portfolio_vector_id else None
            skills_vector = self.get_candidate_vector(self.skills_index, skills_vector_id) if skills_vector_id else None
            
            # Combine professional_summary and project_portfolio vectors (average them)
            description_query_vector = None
            if prof_vector and project_portfolio_vector:
                # Average the two vectors
                description_query_vector = [(a + b) / 2 for a, b in zip(prof_vector, project_portfolio_vector)]
            elif prof_vector:
                description_query_vector = prof_vector
            elif project_portfolio_vector:
                description_query_vector = project_portfolio_vector
            
            # Search project description index
            description_matches = []
            if description_query_vector:
                matches = self.search_projects_with_query_vector(
                    self.project_description_index, 
                    description_query_vector, 
                    top_k
                )
                for match in matches:
                    description_matches.append({
                        "project_id": match.metadata.get("project_id"),
                        "score": match.score,
                        "match_type": "description"
                    })
            
            # Search project skills index using candidate's skills_matrix vector
            skills_matches = []
            if skills_vector:
                matches = self.search_projects_with_query_vector(
                    self.project_skills_index,
                    skills_vector,
                    top_k
                )
                for match in matches:
                    skills_matches.append({
                        "project_id": match.metadata.get("project_id"),
                        "score": match.score,
                        "match_type": "skills"
                    })
            
            # Combine and rank results
            combined_results = self._combine_project_results(description_matches, skills_matches)
            
            return {
                "description_matches": description_matches,
                "skills_matches": skills_matches,
                "combined_ranked": combined_results
            }
            
        except Exception as e:
            print(f"Error retrieving relevant projects: {e}")
            return {
                "description_matches": [],
                "skills_matches": [],
                "combined_ranked": []
            }
    
    def _combine_project_results(self, description_matches: List[Dict], 
                                skills_matches: List[Dict]) -> List[Dict]:
        """
        Combine description and skills matches, calculate weighted score, and rank
        """
        project_scores = {}
        
        # Process description matches
        for match in description_matches:
            project_id = match["project_id"]
            if project_id not in project_scores:
                project_scores[project_id] = {
                    "project_id": project_id,
                    "description_score": 0.0,
                    "skills_score": 0.0,
                    "overall_score": 0.0
                }
            project_scores[project_id]["description_score"] = match["score"]
        
        # Process skills matches
        for match in skills_matches:
            project_id = match["project_id"]
            if project_id not in project_scores:
                project_scores[project_id] = {
                    "project_id": project_id,
                    "description_score": 0.0,
                    "skills_score": 0.0,
                    "overall_score": 0.0
                }
            project_scores[project_id]["skills_score"] = match["score"]
        
        # Calculate overall score (average of description and skills scores)
        # Only consider non-zero scores for average
        for project_id in project_scores:
            scores = [
                project_scores[project_id]["description_score"],
                project_scores[project_id]["skills_score"]
            ]
            valid_scores = [score for score in scores if score > 0]
            project_scores[project_id]["overall_score"] = (
                sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
            )
        
        # Convert to list and sort by overall score
        combined_list = list(project_scores.values())
        combined_list.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return combined_list

