import json
import os
from typing import Dict, Any, Optional

# Prefer requests; fall back to http.client if requests is not available
try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except Exception:
    import http.client as _http_client
    from urllib import parse as _urlparse
    _HAS_REQUESTS = False
try:
    from state import OutputState
except Exception:
    # Support importing when running from workspace root as a package
    from question_agent.state import OutputState


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text or not isinstance(text, str):
        return None
    # Try to locate the first JSON object in the text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        candidate = text[start:end+1]
        return json.loads(candidate)
    except Exception:
        return None


def _build_prompt(role: str,
                  difficulty: str,
                  preferred_type: str,
                  question_index: int,
                  previous_answers: Any,
                  time_limit_seconds: int,
                  number_of_questions: Optional[int]) -> str:
    progression = (
        "Q1: fundamentals\n"
        "Q2: tools / methods\n"
        "Q3: applied scenario\n"
        "Q4: problem solving\n"
        "Q5+: real-world or edge cases\n"
    )

    prompt = f"""
You are an interview question generator. Generate EXACTLY ONE JSON object and NOTHING ELSE. No text, no markdown, no explanation.

Constraints:
- Role/topic relevance: {role}
- Difficulty level (force this): {difficulty}
- Preferred question type (guide generation; may be 'mixed'): {preferred_type}
- Question index: {question_index}
- Time limit (seconds) to use as guidance: {time_limit_seconds}
- Number of questions in this interview (for progression context): {number_of_questions}
- Avoid repeating earlier questions or paraphrasing previous answers: {json.dumps(previous_answers)}

Progression rules:
{progression}

Generation rules:
- question_type must be one of: behavioral | technical | situational. If preferred type is 'mixed', choose the best match.
- difficulty must be one of: easy | medium | hard.
- time_limit_seconds must be an integer number of seconds.
- Do NOT include any fields other than: question_id, question_text, question_type, difficulty, time_limit_seconds.
- question_id must be exactly: "q{question_index}".

Output JSON ONLY in this exact shape (example):
{{
  "question_id": "q{question_index}",
  "question_text": "...",
  "question_type": "technical",
  "difficulty": "{difficulty}",
  "time_limit_seconds": {time_limit_seconds}
}}

If you cannot comply, output nothing.
"""
    return prompt


def _call_ollama(prompt: str, timeout: int = 30) -> str:
    payload = {
        "model": os.environ.get("OLLAMA_MODEL", "llama3"),
        "prompt": prompt,
        "stream": False,
        "max_tokens": 400,
        "temperature": 0.2,
    }

    if _HAS_REQUESTS:
        resp = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        # Handle common response shapes
        if isinstance(data, dict):
            if "response" in data and isinstance(data["response"], str):
                return data["response"]
            if "output" in data:
                out = data["output"]
                if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                    # Ollama sometimes returns [{'role':..., 'content': '...'}]
                    return out[0].get("content", "")
                if isinstance(out, str):
                    return out
            if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                c = data["choices"][0]
                if isinstance(c, dict):
                    return c.get("message", c.get("text", ""))

        return resp.text

    # Fallback using http.client when requests is not available
    try:
        conn = _http_client.HTTPConnection("localhost", 11434, timeout=timeout)
        body = json.dumps(payload)
        headers = {"Content-Type": "application/json"}
        conn.request("POST", "/api/generate", body, headers)
        res = conn.getresponse()
        status = res.status
        raw = res.read().decode("utf-8")
        if status >= 400:
            raise Exception(f"Ollama API error: {status} {raw}")
        try:
            data = json.loads(raw)
        except Exception:
            return raw

        if isinstance(data, dict):
            if "response" in data and isinstance(data["response"], str):
                return data["response"]
            if "output" in data:
                out = data["output"]
                if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
                    return out[0].get("content", "")
                if isinstance(out, str):
                    return out
            if "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                c = data["choices"][0]
                if isinstance(c, dict):
                    return c.get("message", c.get("text", ""))

        return raw
    finally:
        try:
            conn.close()
        except Exception:
            pass


def question_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    # Required output fields
    required_keys = {"question_id", "question_text", "question_type", "difficulty", "time_limit_seconds"}

    # Gather locked inputs
    role = state.get("role", "General")
    experience_level = state.get("experience_level", "Mid")
    question_index = int(state.get("question_index", 1))
    previous_answers = state.get("previous_answers", [])

    # Support optional pre-run prerequisites passed in state under 'prerequisites' or top-level keys.
    prereq = state.get("prerequisites", {}) or {}
    # Top-level fallbacks (CLI might pass them directly)
    number_of_questions = prereq.get("number_of_questions") or state.get("number_of_questions")
    preferred_question_type = prereq.get("preferred_question_type") or state.get("preferred_question_type") or prereq.get("preferred_type") or "mixed"
    cli_difficulty = prereq.get("difficulty_level") or state.get("difficulty_level")
    cli_time_limit = prereq.get("time_limit_seconds") or state.get("time_limit_seconds")

    # Map experience_level to difficulty if CLI difficulty not provided
    difficulty_map = {"Fresher": "easy", "Junior": "easy", "Mid": "medium", "Senior": "hard"}
    difficulty = (cli_difficulty or difficulty_map.get(experience_level, "medium")).lower()
    if difficulty not in {"easy", "medium", "hard"}:
        difficulty = "medium"

    # Suggested time limits by difficulty when not provided
    default_time_by_diff = {"easy": 60, "medium": 90, "hard": 120}
    time_limit_seconds = int(cli_time_limit) if cli_time_limit else default_time_by_diff.get(difficulty, 90)

    # Build prompt and call LLM, with one retry on failure
    prompt = _build_prompt(role, difficulty, preferred_question_type, question_index, previous_answers, time_limit_seconds, number_of_questions)

    for attempt in range(2):
        try:
            raw = _call_ollama(prompt)
        except Exception:
            raw = ""

        parsed = _extract_json(raw)
        if not parsed:
            # prepare a stricter prompt for retry
            prompt = """
STRICT MODE: Output ONLY a single JSON object exactly matching the required shape. Do NOT output any text or explanation. If you fail, output nothing.
""" + _build_prompt(role, difficulty, preferred_question_type, question_index, previous_answers, time_limit_seconds, number_of_questions)
            continue

        # Remove any extra keys
        keys = set(parsed.keys())
        if not required_keys.issubset(keys):
            # Missing required keys -> retry
            prompt = """
STRICT MODE: Output ONLY a single JSON object exactly matching the required shape. Do NOT output any text or explanation. If you fail, output nothing.
""" + _build_prompt(role, difficulty, preferred_question_type, question_index, previous_answers, time_limit_seconds, number_of_questions)
            continue

        # Build final validated output, coercing and clamping values
        output: Dict[str, Any] = {}
        output["question_id"] = f"q{question_index}"
        output["question_text"] = str(parsed.get("question_text", ""))[:2000]

        qtype = parsed.get("question_type", "technical").lower()
        if qtype not in {"behavioral", "technical", "situational"}:
            # If preferred is explicit and valid, use it; else default to technical
            if preferred_question_type in {"behavioral", "technical", "situational"}:
                qtype = preferred_question_type
            else:
                qtype = "technical"
        output["question_type"] = qtype

        diff = parsed.get("difficulty", difficulty).lower()
        if diff not in {"easy", "medium", "hard"}:
            diff = difficulty
        output["difficulty"] = diff

        try:
            t = int(parsed.get("time_limit_seconds", time_limit_seconds))
            if t <= 0:
                t = time_limit_seconds
        except Exception:
            t = time_limit_seconds
        output["time_limit_seconds"] = t

        # Ensure exactly the locked output contract (no extra keys)
        final_keys = set(output.keys())
        if final_keys == required_keys:
            return output

        # If something odd happened (shouldn't), coerce to only required keys
        coerced = {k: output[k] for k in ["question_id", "question_text", "question_type", "difficulty", "time_limit_seconds"]}
        return coerced

    # If we reach here, both attempts failed — return a safe fallback matching contract
    return {
        "question_id": f"q{question_index}",
        "question_text": "Describe a challenging problem you faced and how you solved it.",
        "question_type": "behavioral",
        "difficulty": difficulty,
        "time_limit_seconds": time_limit_seconds,
    }