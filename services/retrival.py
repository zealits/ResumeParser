"""
Retrieval Pipeline for Candidate Ranking across Professional Summary, Project Portfolio, and Skills Matrix
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

class CandidateRetrievalPipeline:
    def __init__(self):
        # Verify required API keys are present
        pinecone_key = os.getenv("PINECONE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY environment variable not set. Please set it in your .env file.")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable not set. Please set it in your .env file.")
        
        self.pc = Pinecone(api_key=pinecone_key)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Index names
        self.PROFESSIONAL_INDEX = "professional-summary"
        self.SKILLS_INDEX = "skills-matrix"
        self.PROJECT_INDEX = "project-portfolio"
        
        # Initialize indexes
        self.professional_index = self.pc.Index(self.PROFESSIONAL_INDEX)
        self.skills_index = self.pc.Index(self.SKILLS_INDEX)
        self.project_index = self.pc.Index(self.PROJECT_INDEX)
    
    def generate_query_embedding(self, text: str) -> List[float]:
        """Generate embedding for query text"""
        return self.embeddings.embed_query(text)
    
    def build_filter_conditions(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Build filter conditions for Pinecone query, handling null filters"""
        filter_conditions = {}
        
        if filters.get("has_leadership") is not None:
            filter_conditions["has_leadership"] = {"$eq": filters["has_leadership"]}
        
        if filters.get("highest_education") is not None:
            filter_conditions["highest_education"] = {"$eq": filters["highest_education"]}
        
        if filters.get("seniority_level") is not None:
            filter_conditions["seniority_level"] = {"$eq": filters["seniority_level"]}
        
        return filter_conditions if filter_conditions else None
    
    def search_index(self, index, query_embedding: List[float], filters: Dict[str, Any], 
                    top_k: int = 10000) -> List[Dict[str, Any]]:
        """Search a specific index with filters"""
        filter_conditions = self.build_filter_conditions(filters)
        
        try:
            results = index.query(
                vector=query_embedding,
                filter=filter_conditions,
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            return results.matches
        except Exception as e:
            print(f"Error searching index: {e}")
            return []
    
    def rank_professional_summary(self, project_description: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank candidates based on professional summary relevance to project description"""
        query_embedding = self.generate_query_embedding(project_description)
        results = self.search_index(self.professional_index, query_embedding, filters)
        
        ranked_candidates = []
        for match in results:
            candidate_data = {
                "candidate_id": match.metadata.get("candidate_id"),
                "name": match.metadata.get("name"),
                "score": match.score,
                "document_type": "professional_summary",
                "seniority_level": match.metadata.get("seniority_level"),
                "highest_education": match.metadata.get("highest_education"),
                "has_leadership": match.metadata.get("has_leadership")
            }
            ranked_candidates.append(candidate_data)
        
        return sorted(ranked_candidates, key=lambda x: x["score"], reverse=True)
    
    def rank_project_portfolio(self, project_description: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank candidates based on project portfolio relevance to project description"""
        query_embedding = self.generate_query_embedding(project_description)
        results = self.search_index(self.project_index, query_embedding, filters)
        
        ranked_candidates = []
        for match in results:
            candidate_data = {
                "candidate_id": match.metadata.get("candidate_id"),
                "name": match.metadata.get("name"),
                "score": match.score,
                "document_type": "project_portfolio",
                "seniority_level": match.metadata.get("seniority_level"),
                "highest_education": match.metadata.get("highest_education"),
                "has_leadership": match.metadata.get("has_leadership")
            }
            ranked_candidates.append(candidate_data)
        
        return sorted(ranked_candidates, key=lambda x: x["score"], reverse=True)
    
    def rank_skills_matrix(self, required_skills: List[str], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank candidates based on skills match with required skills"""
        skills_query = ", ".join(required_skills)
        query_embedding = self.generate_query_embedding(skills_query)
        results = self.search_index(self.skills_index, query_embedding, filters)
        
        ranked_candidates = []
        for match in results:
            candidate_data = {
                "candidate_id": match.metadata.get("candidate_id"),
                "name": match.metadata.get("name"),
                "score": match.score,
                "document_type": "skills_matrix",
                "seniority_level": match.metadata.get("seniority_level"),
                "highest_education": match.metadata.get("highest_education"),
                "has_leadership": match.metadata.get("has_leadership")
            }
            ranked_candidates.append(candidate_data)
        
        return sorted(ranked_candidates, key=lambda x: x["score"], reverse=True)
    
    def get_combined_candidates(self, professional_results: List[Dict], project_results: List[Dict], 
                              skills_results: List[Dict]) -> Dict[str, List[Dict]]:
        """Combine results from all three searches and create final ranked lists"""
        
        # Create a mapping of candidate_id to their scores from different searches
        candidate_scores = {}
        
        # Process professional summary results
        for candidate in professional_results:
            candidate_id = candidate["candidate_id"]
            if candidate_id not in candidate_scores:
                candidate_scores[candidate_id] = {
                    "candidate_id": candidate_id,
                    "name": candidate["name"],
                    "professional_score": candidate["score"],
                    "project_score": 0.0,
                    "skills_score": 0.0,
                    "seniority_level": candidate["seniority_level"],
                    "highest_education": candidate["highest_education"],
                    "has_leadership": candidate["has_leadership"]
                }
            else:
                candidate_scores[candidate_id]["professional_score"] = candidate["score"]
        
        # Process project portfolio results
        for candidate in project_results:
            candidate_id = candidate["candidate_id"]
            if candidate_id not in candidate_scores:
                candidate_scores[candidate_id] = {
                    "candidate_id": candidate_id,
                    "name": candidate["name"],
                    "professional_score": 0.0,
                    "project_score": candidate["score"],
                    "skills_score": 0.0,
                    "seniority_level": candidate["seniority_level"],
                    "highest_education": candidate["highest_education"],
                    "has_leadership": candidate["has_leadership"]
                }
            else:
                candidate_scores[candidate_id]["project_score"] = candidate["score"]
        
        # Process skills matrix results
        for candidate in skills_results:
            candidate_id = candidate["candidate_id"]
            if candidate_id not in candidate_scores:
                candidate_scores[candidate_id] = {
                    "candidate_id": candidate_id,
                    "name": candidate["name"],
                    "professional_score": 0.0,
                    "project_score": 0.0,
                    "skills_score": candidate["score"],
                    "seniority_level": candidate["seniority_level"],
                    "highest_education": candidate["highest_education"],
                    "has_leadership": candidate["has_leadership"]
                }
            else:
                candidate_scores[candidate_id]["skills_score"] = candidate["score"]
        
        # Calculate weighted overall score (equal weights for now)
        for candidate_id in candidate_scores:
            scores = [
                candidate_scores[candidate_id]["professional_score"],
                candidate_scores[candidate_id]["project_score"], 
                candidate_scores[candidate_id]["skills_score"]
            ]
            # Only consider non-zero scores for average
            valid_scores = [score for score in scores if score > 0]
            candidate_scores[candidate_id]["overall_score"] = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        # Convert to list and sort by overall score
        combined_candidates = list(candidate_scores.values())
        combined_candidates.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return {
            "professional_summary_ranked": professional_results,
            "project_portfolio_ranked": project_results,
            "skills_matrix_ranked": skills_results,
            "combined_ranked": combined_candidates
        }
    
    def retrieve_ranked_candidates(self, project_description: str, required_skills: List[str], 
                                 filters: Dict[str, Any]) -> Dict[str, List[Dict]]:
        """
        Main retrieval function that ranks candidates across all three indexes
        
        Args:
            project_description: Description of the project for semantic matching
            required_skills: List of required technical skills
            filters: Dictionary with has_leadership, highest_education, seniority_level
                   (use None for any filter to ignore it)
        
        Returns:
            Dictionary with ranked results from all three indexes and combined ranking
        """
        print("Starting candidate retrieval pipeline...")
        
        # Rank candidates from professional summary
        print("Ranking professional summaries...")
        professional_results = self.rank_professional_summary(project_description, filters)
        
        # Rank candidates from project portfolio  
        print("Ranking project portfolios...")
        project_results = self.rank_project_portfolio(project_description, filters)
        
        # Rank candidates from skills matrix
        print("Ranking skills matrix...")
        skills_results = self.rank_skills_matrix(required_skills, filters)
        
        # Combine all results
        print("Combining results...")
        combined_results = self.get_combined_candidates(professional_results, project_results, skills_results)
        
        return combined_results
    
    def save_results_to_files(self, results: Dict[str, List[Dict]], output_dir: str = "retrieval_results"):
        """Save all ranked results to JSON files"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save professional summary results
        with open(f"{output_dir}/professional_summary_results.json", "w") as f:
            json.dump(results["professional_summary_ranked"], f, indent=2)
        
        # Save project portfolio results
        with open(f"{output_dir}/project_portfolio_results.json", "w") as f:
            json.dump(results["project_portfolio_ranked"], f, indent=2)
        
        # Save skills matrix results
        with open(f"{output_dir}/skills_matrix_results.json", "w") as f:
            json.dump(results["skills_matrix_ranked"], f, indent=2)
        
        # Save combined results
        with open(f"{output_dir}/combined_ranked_results.json", "w") as f:
            json.dump(results["combined_ranked"], f, indent=2)
        
        print(f"Results saved to {output_dir}/ directory")
        
        return {
            "professional_summary_file": f"{output_dir}/professional_summary_results.json",
            "project_portfolio_file": f"{output_dir}/project_portfolio_results.json", 
            "skills_matrix_file": f"{output_dir}/skills_matrix_results.json",
            "combined_file": f"{output_dir}/combined_ranked_results.json"
        }

# Usage Example
def main():
    # Initialize the pipeline
    retrieval_pipeline = CandidateRetrievalPipeline()
    
    # Example project description and required skills
    project_description = """
    Looking for collaborators to join in developing an advanced **hospital management platform** built with the **MERN stack**. The project aims to simplify hospital workflows by centralizing patient records, doctor schedules, and administrative operations into a unified system. Features include secure data management, appointment scheduling, and real-time status tracking — all designed to improve efficiency and patient care. If you’re interested in working on a project that blends healthcare and technology for real-world impact, let’s build it together!

    """
    
    required_skills = [
        "MERN stack"
      ]
    
    # Filters (use None for any filter you want to ignore)
    filters = {
        "has_leadership": None,  # Ignore leadership filter
        "highest_education": None,  # Only candidates with BE degree
        "seniority_level": None   # Only junior level candidates
    }
    
    # Alternative: No filters (search all candidates)
    # filters = {
    #     "has_leadership": None,
    #     "highest_education": None, 
    #     "seniority_level": None
    # }
    
    # Retrieve and rank candidates
    results = retrieval_pipeline.retrieve_ranked_candidates(
        project_description=project_description,
        required_skills=required_skills,
        filters=filters
    )
    
    # Save results to files
    file_paths = retrieval_pipeline.save_results_to_files(results)
    
    # Print summary
    print(f"\nRetrieval Complete!")
    print(f"Professional Summary Results: {len(results['professional_summary_ranked'])} candidates")
    print(f"Project Portfolio Results: {len(results['project_portfolio_ranked'])} candidates") 
    print(f"Skills Matrix Results: {len(results['skills_matrix_ranked'])} candidates")
    print(f"Combined Results: {len(results['combined_ranked'])} candidates")
    
    # Show top 3 combined results
    print(f"\nTop 3 Candidates:")
    for i, candidate in enumerate(results["combined_ranked"][:3], 1):
        print(f"{i}. {candidate['name']} (ID: {candidate['candidate_id']}) - Overall Score: {candidate['overall_score']:.4f}")

if __name__ == "__main__":
    main()