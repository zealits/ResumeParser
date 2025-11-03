from openai import OpenAI
import json
import PyPDF2
import re
import os
from datetime import datetime
from glob import glob

from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
try:
    if not os.environ.get("OPENAI_API_KEY"):
        raise TypeError
    client = OpenAI()
except TypeError:
    print("FATAL ERROR: OPENAI_API_KEY environment variable not set.")
    print("Please set the environment variable before running the script.")
    exit()

# --- PDF Text Extraction ---
def extract_text_from_pdf(pdf_path):
    """Extracts raw text from a PDF file and cleans it."""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            text = re.sub(r'\s*\n\s*', '\n', text).strip()
            return text
    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
        return None
    except Exception as e:
        print(f"Error reading or parsing PDF: {e}")
        return None

# --- BULLETPROOF Date Parser ---
def parse_any_date(date_str):
    """
    Parse ANY date format that appears in resumes - NO EXCUSES
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    # Clean the string aggressively
    date_str = str(date_str).strip()
    
    # Handle empty strings
    if not date_str:
        return None
    
    # Handle "Present", "Current", "Now"
    present_indicators = ['present', 'current', 'now', 'till date', 'till now', 'ongoing']
    if date_str.lower() in present_indicators:
        return datetime.now()
    
    # Replace fancy quotes and apostrophes
    date_str = date_str.replace('’', "'").replace('‘', "'").replace('”', '"').replace('“', '"')
    
    # Handle formats like Dec'24, Aug'24, Jan'23
    match = re.match(r"(\w{3,})['\"]?(\d{2,4})", date_str, re.IGNORECASE)
    if match:
        month_str, year_str = match.groups()
        year = int(year_str) if len(year_str) == 4 else 2000 + int(year_str)
        month_dict = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        month = month_dict.get(month_str.lower()[:3])
        if month:
            return datetime(year, month, 1)
    
    # Handle month-year formats with spaces: "Dec 2024", "August 2023"
    match = re.match(r"(\w+)\s+(\d{4})", date_str, re.IGNORECASE)
    if match:
        month_str, year_str = match.groups()
        month_dict = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4, 'may': 5, 'june': 6,
            'july': 7, 'august': 8, 'september': 9, 'october': 10, 'november': 11, 'december': 12,
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        month = month_dict.get(month_str.lower())
        if month:
            return datetime(int(year_str), month, 1)
    
    # Handle numeric formats: "12-2024", "12/2024", "2024-12", "2024/12"
    match = re.match(r"(\d{1,2})[-/](\d{4})", date_str)  # MM-YYYY or MM/YYYY
    if match:
        month, year = match.groups()
        return datetime(int(year), int(month), 1)
    
    match = re.match(r"(\d{4})[-/](\d{1,2})", date_str)  # YYYY-MM or YYYY/MM
    if match:
        year, month = match.groups()
        return datetime(int(year), int(month), 1)
    
    # Handle full dates: "15-12-2024", "15/12/2024", "2024-12-15"
    date_formats = [
        '%Y-%m-%d', '%d-%m-%Y', '%m-%d-%Y', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
        '%Y.%m.%d', '%d.%m.%Y', '%m.%d.%Y', '%b %d, %Y', '%B %d, %Y'
    ]
    
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    
    # Handle just year: "2024", "2023"
    if re.match(r"^\d{4}$", date_str):
        return datetime(int(date_str), 1, 1)
    
    # Handle month abbreviations with dots: "Dec. 2024", "Aug. 2023"
    match = re.match(r"(\w{3,})\.?\s+(\d{4})", date_str, re.IGNORECASE)
    if match:
        month_str, year_str = match.groups()
        month_dict = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        month = month_dict.get(month_str.lower()[:3])
        if month:
            return datetime(int(year_str), month, 1)
    
    # Handle weird formats like "12.2024", "12.24"
    match = re.match(r"(\d{1,2})\.(\d{2,4})", date_str)
    if match:
        month, year = match.groups()
        year = int(year) if len(year) == 4 else 2000 + int(year)
        return datetime(year, int(month), 1)
    
    # Final fallback - try to extract any numbers that look like dates
    numbers = re.findall(r'\d+', date_str)
    if len(numbers) >= 2:
        # Assume first two numbers are month and year, or year and month
        try:
            num1, num2 = int(numbers[0]), int(numbers[1])
            if 1 <= num1 <= 12 and num2 > 1000:  # month-year
                return datetime(num2, num1, 1)
            elif 1 <= num2 <= 12 and num1 > 1000:  # year-month
                return datetime(num1, num2, 1)
        except:
            pass
    
    # If we STILL can't parse it, return None and log the problematic format
    print(f"⚠️  Could not parse date: '{date_str}'")
    return None


# --- Robust Experience Calculator ---
def calculate_total_experience(experience_data):
    """
    Calculate total experience in years - handles ALL edge cases
    """
    if not experience_data:
        return 0.0
    
    periods = []
    
    for exp in experience_data:
        start_str = exp.get('start', '')
        end_str = exp.get('end', '')
        
        # Skip if no start date
        if not start_str:
            continue
            
        start_date = parse_any_date(start_str)
        end_date = parse_any_date(end_str) if end_str else datetime.now()
        
        # Validate dates
        if not start_date:
            print(f"⚠️  Could not parse start date: '{start_str}'")
            continue
        
        if not end_date:
            print(f"⚠️  Could not parse end date: '{end_str}', using current date")
            end_date = datetime.now()
        
        # Ensure logical date order
        if end_date < start_date:
            print(f"⚠️  End date before start date: {start_str} - {end_str}, swapping")
            start_date, end_date = end_date, start_date
        
        # Add to periods
        periods.append((start_date, end_date))
    
    if not periods:
        return 0.0
    
    # Merge overlapping periods
    merged_periods = merge_date_periods(periods)
    
    # Calculate total days
    total_days = 0
    for start, end in merged_periods:
        total_days += (end - start).days
    
    # Convert to years (accounting for leap years)
    total_years = total_days / 365.25
    
    # Round to 2 decimal places
    return round(total_years, 2)


def merge_date_periods(periods):
    """
    Merge overlapping date periods to avoid double-counting
    """
    if not periods:
        return []
    
    # Sort by start date
    sorted_periods = sorted(periods, key=lambda x: x[0])
    
    merged = []
    current_start, current_end = sorted_periods[0]
    
    for start, end in sorted_periods[1:]:
        if start <= current_end:
            # Overlapping periods, merge them
            current_end = max(current_end, end)
        else:
            # Non-overlapping period
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    
    # Add the last period
    merged.append((current_start, current_end))
    
    return merged


# --- Test the date parser with your problematic cases ---
def test_date_parser():
    """Test the parser with all kinds of crazy date formats"""
    test_cases = [
        "Dec'24", "Aug'24", "Jan'23", 
        "Dec 2024", "August 2023", "Mar 2022",
        "12-2024", "08-2024", "01/2023",
        "2024-12", "2024-08", "2023-01",
        "15-12-2024", "15/12/2024", "2024-12-15",
        "2024", "2023",
        "Dec. 2024", "Aug. 2023",
        "12.2024", "08.24",
        "Present", "Current", "Now",
        "December 2024", "August 2023",
        "12/15/2024", "08/15/2023"
    ]
    



# --- The Core GenAI Parser ---
def parse_resume_with_genai(resume_text):
    """
    Uses a Generative AI model to parse the resume text directly into JSON.
    """
    prompt_template = """
    You are an expert HR data extraction system. Your task is to analyze the provided resume text and extract key information into a structured JSON format.

    Follow these rules strictly:
    1. Extract the information based ONLY on the text provided. Do not infer or add any information that is not present.
    2. If a specific field is not found in the resume, use a blank string "" for strings, or an empty list [] for lists.
    3. For the 'experience' section, the 'description' should be a concise summary of the key responsibilities and achievements from the text.
    4. For education marks, extract ONLY numerical percentage (e.g., "85.5"). If marks are not in percentage format, leave blank.
    5. Categorize education level based on the qualification:
        -Secondary Education: 10th grade/high school
        -Higher Secondary: 12th grade/senior secondary
        -Undergraduate: Bachelor's degrees
        -Post Graduate: Master's degrees, PhD, etc.
        -Diploma / Vocational Education: Diploma courses, ITI, Polytechnic, Vocational or Certification programmes
        -Other / Unknown: Education entries that don't fit clearly into the above categories or require manual review
    6. The output MUST be a valid JSON object. Do not include any text, explanations, or code formatting like ```json before or after the JSON object itself.
    7. If project_skills, experiance_skills are not explicitly mentioned, generate them based on the description.

    IMPORTANT: For experience dates, extract them EXACTLY as they appear in the resume. Don't normalize them.

    JSON Schema to follow:
    {{
      "name": "string",
      "phone": "string",
      "mail": "string",
      "social": {{
        "github": "string (URL)",
        "linkedin": "string (URL)",
        "portfolio": "string (URL)"
      }},
      "education": [
        {{
          "name": "string (Institution Name)",
          "qualification": "string (e.g., Bachelor of Science in Computer Science)",
          "category": "string (Secondary Education, Higher Secondary, Undergraduate, Post Graduate)",
          "start": "string (Start Date as it appears in resume)",
          "end": "string (End Date as it appears in resume, use 'Present' if current)"
        }}
      ],
      "skills": ["string", "string", ...],
      "projects": [
        {{
          "title": "string",
          "description": "string",
          "project_skills": ["string", "string", ...]
        }}
      ],
      "experience": [
        {{
          "company_name": "string",
          "designation": "string (Job Title)",
          "description": "string (Summary of responsibilities and achievements)",
          "experiance_skills": ["string", "string", ...],
          "start": "string (Start Date EXACTLY as it appears)",
          "end": "string (End Date EXACTLY as it appears)"
        }}
      ],
      "certifications": ["string", "string", ...],
      "achievements": ["string", "string", ...]
    }}

    --- RESUME TEXT ---
    {resume_text}
    --- END RESUME TEXT ---
    """

    prompt = prompt_template.format(resume_text=resume_text)

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        content = response.choices[0].message.content if response.choices else ""
        json_string = (content or "").strip().replace("```json", "").replace("```", "").strip()
        parsed_data = json.loads(json_string)
        
        # Calculate total experience PROPERLY
        experience_data = parsed_data.get('experience', [])
        total_experience = calculate_total_experience(experience_data)
        parsed_data['total_experience_years'] = total_experience
        
        print(f"✅ Calculated total experience: {total_experience} years")
        
        return parsed_data
        
    except Exception as e:
        print(f"❌ An error occurred during LLM processing or JSON parsing: {e}")
        try:
            raw = response.choices[0].message.content if response and response.choices else None
            if raw:
                print(f"Raw response from API was: {raw}")
        except Exception:
            pass
        return {"error": "Failed to parse GenAI response into valid JSON."}


# --- Batch Processing Utilities ---
def process_pdf_file(pdf_path, output_dir=None):
    """Extract text from a PDF, parse it via LLM, and write JSON with same basename."""
    try:
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text:
            print(f"Skipping '{pdf_path}': no text extracted.")
            return False
        print(f"Parsing '{pdf_path}'...")
        parsed_data = parse_resume_with_genai(raw_text)
        # Decide output path
        target_dir = output_dir if output_dir else os.path.dirname(pdf_path)
        os.makedirs(target_dir, exist_ok=True)
        json_path = os.path.join(target_dir, f"{base_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=4)
        print(f"Saved: {json_path}")
        return True
    except Exception as e:
        print(f"Failed processing '{pdf_path}': {e}")
        return False


def process_directory(input_dir, output_dir=None):
    """Process all PDFs in a directory (non-recursive)."""
    pattern = os.path.join(input_dir, "*.pdf")
    pdf_files = sorted(glob(pattern))
    if not pdf_files:
        print(f"No PDFs found in '{input_dir}'.")
        return 0, 0
    success_count = 0
    for pdf_path in pdf_files:
        ok = process_pdf_file(pdf_path, output_dir=output_dir)
        success_count += 1 if ok else 0
    print(f"Completed. {success_count}/{len(pdf_files)} files succeeded.")
    return success_count, len(pdf_files)


# --- CLI Entry Point ---
if __name__ == "__main__":
    import argparse

    # Test the parser first
    test_date_parser()
    
    # Test with your specific problematic case

    
    parser = argparse.ArgumentParser(description="Parse resume PDFs into JSON using gpt-4o-mini.")
    parser.add_argument(
        "input_path",
        nargs="?",
        default=".",
        help="Path to a PDF file or a directory containing PDFs (default: current directory)",
    )
    parser.add_argument(
        "--out",
        dest="output_dir",
        default=None,
        help="Optional output directory for JSON files (defaults to input location)",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else None

    if os.path.isdir(input_path):
        process_directory(input_path, output_dir=output_dir)
    elif os.path.isfile(input_path) and input_path.lower().endswith(".pdf"):
        process_pdf_file(input_path, output_dir=output_dir)
    else:
        print("Input path must be a directory or a .pdf file.")