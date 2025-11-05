import json
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Configuration constants
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.5
NUM_MCQS = 5
NUM_THEORY_QUESTIONS = 5


def get_openai_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Initialize and return an OpenAI client.
    
    Args:
        api_key: OpenAI API key. If None, reads from environment variable.
        
    Returns:
        OpenAI client instance.
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
    
    return OpenAI(api_key=api_key)


def get_base_prompt_template() -> str:
    """
    Returns the base prompt template for test generation.
    
    Returns:
        String containing the prompt template with placeholder for candidate JSON.
    """
    return """
You are an expert technical assessment designer. Your task is to generate a 10-question test **strictly based on the skills, knowledge depth, and domains implied by this JSON:**

<<CANDIDATE_JSON>>

Follow these rules carefully:

1. 1. Identify the candidate's SKILL AREAS & certifications 
   - Do NOT use ANY specific project, description, or scenario mentioned in the JSON.
   - Do NOT generate questions that reference the candidate's projects, tools they used in projects, or what they built. 


2. **Question Structure**
   - The test must contain exactly:
     - **5 complex MCQs** (Hard level)
     - **5 descriptive theory/coding questions**
   - Output must be in **valid JSON** only.
 
3. **Theory/Coding Questions**
   - These must combine conceptual understanding with problem-solving.
   - Keep difficulty moderate but intellectually demanding.

4. **Question Domains**
   - Use only knowledge areas implied by the JSON.
   - Do **not** refer to any specific project, company, or tool from the candidate's JSON.

5. **Output Format**

{
  "mcqs": [
    {
      "question": "",
      "options": ["", "", "", ""],
      "correct_answer": ""
    }
  ],
  "theory": [
    {
      "category": "theory" | "coding",
      "question": ""
    }
  ]
}

7. **Quality Enforcement**
   - Theory/coding questions must test reasoning, not recall.
   - Answers should require more than one layer of thought to justify.

If any question feels guessable by general intuition or "common sense," regenerate it with more context, ambiguity, or trade-off reasoning.

"""


def build_prompt(candidate_profile: Dict[str, Any]) -> str:
    """
    Builds the complete prompt by inserting candidate profile JSON into the template.
    
    Args:
        candidate_profile: Dictionary containing candidate's profile information.
        
    Returns:
        Complete prompt string ready to send to OpenAI.
    """
    base_prompt = get_base_prompt_template()
    candidate_json = json.dumps(candidate_profile, indent=2)
    return base_prompt.replace("<<CANDIDATE_JSON>>", candidate_json)


def generate_test(
    candidate_profile: Dict[str, Any],
    client: Optional[OpenAI] = None,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE
) -> str:
    """
    Generates a technical test based on the candidate profile.
    
    Args:
        candidate_profile: Dictionary containing candidate's profile information.
        client: OpenAI client instance. If None, creates a new one.
        model: OpenAI model to use for generation.
        temperature: Temperature parameter for the model (0.0 to 2.0).
        
    Returns:
        String containing the generated test in JSON format.
        
    Raises:
        ValueError: If API key is missing.
        Exception: If OpenAI API call fails.
    """
    if client is None:
        client = get_openai_client()
    
    prompt = build_prompt(candidate_profile)
    
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    
    return response.choices[0].message.content


def load_candidate_profile_from_file(file_path: str) -> Dict[str, Any]:
    """
    Loads candidate profile from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing candidate profile.
        
    Returns:
        Dictionary containing candidate profile.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_test_to_file(test_output: str, file_path: str) -> None:
    """
    Saves the generated test to a file.
    
    Args:
        test_output: The generated test content (JSON string).
        file_path: Path where the test should be saved.
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(test_output)


def main():
    """
    Main execution function with example usage.
    """
    # Example candidate profile
    candidate_profile = {
        "highest_qualification": {
            "qualification": "BE Computer Engineering",
            "category": "Undergraduate"
        },
        "projects": [
            {
                "description": "Developed a smart contract escrow system for secure B2B payments, integrating MetaMask for Ethereum transactions and penalty enforcement. Created a bidirectional marketplace for manufacturers and suppliers, with real-time negotiations and supplier ratings. Integrated AI for supplier recommendations and market demand analysis, alongside live commodity price tracking. Built and hard-coded AI agents for decision-making and automation across supply chain tasks. Developed an intelligent logistics system with route optimization, lead time calculation, and vehicle selection.",
                "project_skills": [
                    "Django",
                    "Ethereum",
                    "Smart Contracts",
                    "Google Gemini AI",
                    "Phidata",
                    "SQLite",
                    "Geopy",
                    "Tailwind CSS",
                    "Web3.js"
                ]
            },
            {
                "description": "Developed a real-time chess analysis system where users can snap a photo of any chessboard (2D or 3D) and instantly receive the optimal next move. Built a custom chess dataset from scratch, achieving 94% detection accuracy on real-world board configurations. Employed YOLOv8 for both object detection and segmentation of chess pieces and boards. Designed a perspective-aware mapping algorithm to translate board positions from both white and black views. Improvised AI to convert visual inputs into FEN notation, enabling seamless move prediction using Stockfish.",
                "project_skills": [
                    "Pytorch",
                    "YOLOv8",
                    "OpenCV",
                    "Streamlit",
                    "Stockfish",
                    "HuggingFace"
                ]
            },
            {
                "description": "Built a full-stack platform for automated interview scheduling, real-time candidate analysis, and intelligent evaluation. Leveraged Groq LLM API for contextual interview question generation and domain-based difficulty categorization. Implemented PDF parsing for resume analysis and Zoom API with OAuth2 for seamless scheduling and email automation. Developed real-time video analytics using OpenCV and MediaPipe for gaze tracking, and DeepFace for emotion detection. Built AI agents to evaluate responses, emotions, and engagement using heatmaps and structured reporting.",
                "project_skills": [
                    "Django",
                    "Groq LLM API",
                    "OpenCV",
                    "MediaPipe",
                    "DeepFace",
                    "Zoom API",
                    "OAuth2",
                    "PDF parsing",
                    "Django ORM"
                ]
            }
        ],
        "experience": [
            {
                "designation": "AI Intern",
                "description": "Working on Agentic AI-driven solutions involving autonomous agents for enterprise applications. Built modular AI workflows using LangGraph, integrating complex reasoning paths for scalable agent coordination. Developed robust backend services with Django, ensuring secure and efficient data handling across components. Applied OpenCV for Computer Vision tasks including image preprocessing, object detection, and visual analysis.",
                "experience_skills": [
                    "Agentic AI",
                    "LangGraph",
                    "Django",
                    "OpenCV"
                ]
            },
            {
                "designation": "Machine Learning Intern",
                "description": "Implemented AI-driven computer vision solutions for real-time object recognition and strategic decision-making systems. Built and fine-tuned deep learning models using TensorFlow & Pytorch. Leveraged YOLOv8 for object detection and segmentation. Deployed interactive user interfaces using Streamlit.",
                "experience_skills": [
                    "TensorFlow",
                    "Pytorch",
                    "YOLOv8",
                    "Streamlit"
                ]
            }
        ],
        "certifications": []
    }
    
    # Generate test
    test_output = generate_test(candidate_profile)
    print(test_output)


if __name__ == "__main__":
    main()
