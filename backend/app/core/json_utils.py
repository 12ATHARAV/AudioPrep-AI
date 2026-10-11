"""
Utility for robust JSON extraction from LLM outputs.

Handles:
- Reasoning / thinking models (e.g. <think>...</think> tags from DeepSeek R1 / V3)
- Markdown code blocks (```json ... ``` or ``` ... ```)
- Conversational preambles ("Here is the JSON plan:")
- Trailing text or notes
- Trailing commas before closing braces/brackets
"""

import json
import re
from typing import Any


def extract_json(text: str) -> Any:
    """
    Robustly extract and parse JSON from any LLM response.
    Returns parsed Python dict or list.
    """
    if not text or not text.strip():
        raise ValueError("Empty response text from LLM")

    # 1. Strip <think>...</think> reasoning blocks (DeepSeek, etc.)
    cleaned = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE).strip()

    # 2. Try direct json.loads first (fast path)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 3. Look for ```json ... ``` or ``` ... ``` markdown fences
    fence_matches = re.findall(r'```(?:json)?\s*([\s\S]*?)\s*```', cleaned, flags=re.IGNORECASE)
    for match in fence_matches:
        trimmed = match.strip()
        try:
            return json.loads(trimmed)
        except Exception:
            # Clean trailing commas
            fixed = re.sub(r',\s*([}\]])', r'\1', trimmed)
            try:
                return json.loads(fixed)
            except Exception:
                continue

    # 4. Find outermost JSON object {...} or array [...] based on which opens first
    first_brace = cleaned.find('{')
    last_brace = cleaned.rfind('}')
    first_bracket = cleaned.find('[')
    last_bracket = cleaned.rfind(']')

    candidates = []
    # Check which opens first: array or object
    if first_bracket != -1 and last_bracket != -1 and (first_brace == -1 or first_bracket < first_brace):
        candidates.append(cleaned[first_bracket:last_bracket + 1])
        if first_brace != -1 and last_brace != -1:
            candidates.append(cleaned[first_brace:last_brace + 1])
    elif first_brace != -1 and last_brace != -1:
        candidates.append(cleaned[first_brace:last_brace + 1])
        if first_bracket != -1 and last_bracket != -1:
            candidates.append(cleaned[first_bracket:last_bracket + 1])

    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:
            # Clean common trailing commas before closing braces/brackets
            fixed = re.sub(r',\s*([}\]])', r'\1', cand)
            try:
                return json.loads(fixed)
            except Exception:
                continue

    raise ValueError(f"Could not extract valid JSON from response: {text[:200]}...")
