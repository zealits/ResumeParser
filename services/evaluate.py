import os
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

from dotenv import load_dotenv

try:
    from openai import OpenAI  
except Exception:
    OpenAI = None  # type: ignore

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

MCQ_POINTS = 5
THEORY_POINTS = 5


def _normalize_text(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def evaluate_mcqs(test: Dict[str, Any], answers: Dict[str, Any]) -> Tuple[int, List[int]]:
    mcqs: List[Dict[str, Any]] = test.get("mcqs", []) or []
    given: List[Any] = (answers.get("mcqs") or [])

    total_score = 0
    per_question: List[int] = []

    for idx, q in enumerate(mcqs):
        correct = _normalize_text(q.get("correct_answer"))

        # Allow answer to be option index (int) or text (str)
        ans_val = None
        if idx < len(given):
            ans_val = given[idx]

        chosen_norm = ""
        if isinstance(ans_val, int):
            options = q.get("options") or []
            if 0 <= ans_val < len(options):
                chosen_norm = _normalize_text(options[ans_val])
        else:
            chosen_norm = _normalize_text(ans_val)

        score = MCQ_POINTS if chosen_norm == correct else 0
        per_question.append(score)
        total_score += score

    return total_score, per_question


def _get_openai_client() -> Optional["OpenAI"]:
    if OpenAI is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        return OpenAI(api_key=api_key)
    except Exception:
        return None


def _score_theory_with_llm(questions: List[Dict[str, Any]], answers: List[str]) -> Optional[List[int]]:
    client = _get_openai_client()
    if client is None:
        return None

    prompt = {
        "instructions": (
            "You are an expert evaluator. Score each theory/coding answer from 0 to 5 (integers only). "
            "Score strictly on relevance, correctness, clarity, and depth for the given question. "
            "Return ONLY a JSON object with array 'scores' of integers, one per answer."
        ),
        "questions": questions,
        "answers": answers,
        "scoring": {
            "0": "No attempt or entirely irrelevant",
            "1": "Minimal/incorrect",
            "2": "Partially correct, shallow",
            "3": "Mostly correct, some depth",
            "4": "Correct, clear, good depth",
            "5": "Excellent, comprehensive"
        }
    }

    completion = client.chat.completions.create(
        model=os.getenv("EVAL_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": json.dumps(prompt)}],
        temperature=0,
        response_format={"type": "json_object"}
    )

    try:
        content = completion.choices[0].message.content
        data = json.loads(content)
        scores = data.get("scores")
        if not isinstance(scores, list):
            return None
        # Clamp and coerce to int within range
        cleaned = []
        for s in scores:
            try:
                val = int(s)
            except Exception:
                val = 0
            cleaned.append(max(0, min(THEORY_POINTS, val)))
        return cleaned
    except Exception:
        return None


def _score_theory_heuristic(questions: List[Dict[str, Any]], answers: List[str]) -> List[int]:
    scores: List[int] = []
    for idx in range(len(questions)):
        ans = answers[idx] if idx < len(answers) else ""
        word_count = len((_normalize_text(ans)).split())
        if word_count == 0:
            score = 0
        elif word_count < 20:
            score = 2
        elif word_count < 50:
            score = 3
        elif word_count < 100:
            score = 4
        else:
            score = 5
        scores.append(score)
    return scores


def evaluate_theory(test: Dict[str, Any], answers: Dict[str, Any]) -> Tuple[int, List[int]]:
    theory_questions: List[Dict[str, Any]] = test.get("theory", []) or []
    given: List[str] = (answers.get("theory") or [])

    # Try LLM-based scoring first; fallback to heuristic
    llm_scores = _score_theory_with_llm(theory_questions, given)
    if llm_scores is None:
        llm_scores = _score_theory_heuristic(theory_questions, given)

    total_score = sum(max(0, min(THEORY_POINTS, int(s))) for s in llm_scores)
    return total_score, llm_scores


def evaluate_test(test: Dict[str, Any], answers: Dict[str, Any]) -> Dict[str, Any]:
    mcq_score, mcq_breakdown = evaluate_mcqs(test, answers)
    theory_score, theory_breakdown = evaluate_theory(test, answers)

    result = {
        "mcq": {
            "score": mcq_score,
            "max": MCQ_POINTS * len(test.get("mcqs", []) or []),
            "per_question": mcq_breakdown,
        },
        "theory": {
            "score": theory_score,
            "max": THEORY_POINTS * len(test.get("theory", []) or []),
            "per_question": theory_breakdown,
        },
    }

    total = result["mcq"]["score"] + result["theory"]["score"]
    max_total = result["mcq"]["max"] + result["theory"]["max"]
    result["total"] = {"score": total, "max": max_total}
    return result


