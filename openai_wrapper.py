"""
openai_wrapper.py
Creates a LIME-compatible prediction function backed by OpenAI GPT-4o-mini.

LIME calls the predict function with a list of *perturbed* texts and expects a
2-D numpy array of shape (n_samples, n_classes).  Because each call can contain
100+ texts we use a ThreadPoolExecutor to fan out requests concurrently.
"""

import json
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_probabilities(content: str, class_names: list[str]) -> list[float]:
    """
    Extract a probability array from the GPT response.
    Handles both clean JSON and lightly-malformed outputs.
    """
    n = len(class_names)

    try:
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?", "", content).strip().rstrip("`").strip()
        data = json.loads(cleaned)

        # Accept {"probabilities": [...]} or {"class_name": prob, ...}
        if "probabilities" in data:
            items = data["probabilities"]
            prob_dict = {}
            for item in items:
                if isinstance(item, dict):
                    cls  = str(item.get("class", "")).lower().strip()
                    prob = float(item.get("probability", 0.0))
                    prob_dict[cls] = prob
        else:
            prob_dict = {str(k).lower().strip(): float(v) for k, v in data.items()}

        arr = [prob_dict.get(c.lower().strip(), 0.0) for c in class_names]

    except Exception:
        # Fall back: try to find any numbers in the text
        numbers = re.findall(r"[-+]?\d*\.?\d+", content)
        if len(numbers) >= n:
            arr = [float(x) for x in numbers[:n]]
        else:
            arr = [1.0 / n] * n

    # Normalise
    total = sum(arr)
    if total <= 0:
        return [1.0 / n] * n
    return [p / total for p in arr]


# ── public API ────────────────────────────────────────────────────────────────

def create_openai_classifier(api_key: str, task: str, class_names: list[str]):
    """
    Returns a predict(texts) function suitable for LIME.

    Parameters
    ----------
    api_key     : OpenAI API key
    task        : Human-readable task description, e.g. "sentiment analysis"
    class_names : Ordered list of class labels

    Returns
    -------
    predict(texts: list[str]) -> np.ndarray  shape (len(texts), len(class_names))
    """
    client = OpenAI(api_key=api_key)
    classes_str  = ", ".join(f'"{c}"' for c in class_names)
    system_prompt = (
        f"You are a {task} classifier. "
        f"Your ONLY job is to return a JSON object with the probability for each class. "
        f"Classes: [{classes_str}]. "
        f"Always respond with ONLY valid JSON — no markdown, no explanation."
    )
    example_json = json.dumps({
        "probabilities": [{"class": c, "probability": round(1 / len(class_names), 4)} for c in class_names]
    })

    def _predict_one(text: str) -> list[float]:
        user_prompt = (
            f'Text: """{text}"""\n\n'
            f"Return probabilities as JSON exactly like this example:\n{example_json}"
        )
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=150,
            )
            content = response.choices[0].message.content or ""
            return _parse_probabilities(content, class_names)
        except Exception:
            return [1.0 / len(class_names)] * len(class_names)

    def predict(texts: list[str], max_workers: int = 8) -> np.ndarray:
        """
        Fan out one GPT call per perturbed text, collect results.
        max_workers=8 keeps latency low without hitting rate limits.
        """
        results: list[list[float] | None] = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {pool.submit(_predict_one, t): i for i, t in enumerate(texts)}
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = [1.0 / len(class_names)] * len(class_names)

        return np.array(results, dtype=float)

    return predict


def validate_openai_key(api_key: str) -> tuple[bool, str]:
    """Quick sanity-check: list models and return (ok, message)."""
    try:
        client = OpenAI(api_key=api_key)
        client.models.list()
        return True, "API key is valid."
    except Exception as exc:
        return False, str(exc)
