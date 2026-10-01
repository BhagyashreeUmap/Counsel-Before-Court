"""Similarity retrieval of synthetic examples, never legal advice or precedent."""

from copy import deepcopy
from functools import lru_cache
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.database import load_cases

RETRIEVAL_NOTICE = (
    "Synthetic demonstration example only; not a real judgment, legal precedent, "
    "or legal rule. For legal information and case preparation, not legal advice "
    "or a substitute for a lawyer."
)
SEARCH_FIELDS = (
    "title", "story_summary", "documents_citizen_had", "key_factors", "lesson"
)


def _as_text(value: Any) -> str:
    """Join list fields and tolerate missing or unexpected JSON values."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return " ".join(_as_text(item) for item in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return ""


@lru_cache(maxsize=1)
def _cases() -> list[dict[str, Any]]:
    return load_cases()


@lru_cache(maxsize=1)
def _search_index():
    """Build once per process. Restart after changing the case dataset."""
    texts = [
        " ".join(_as_text(case.get(field)) for field in SEARCH_FIELDS)
        for case in _cases()
    ]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    # An empty dataset or records without searchable words have no matches.
    analyzer = vectorizer.build_analyzer()
    if not any(analyzer(text) for text in texts):
        return None, None
    return vectorizer, vectorizer.fit_transform(texts)


def _result(case: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(case)
    result["retrieval_notice"] = RETRIEVAL_NOTICE
    return result


def get_case(case_id: str) -> dict[str, Any] | None:
    """Return an independent case copy, or None for an unknown/invalid ID.

    Data loading failures raise CaseDataError so callers can report them.
    """
    if not isinstance(case_id, str) or not case_id.strip():
        return None
    for case in _cases():
        if case["id"] == case_id.strip():
            return _result(case)
    return None


def search_cases(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Return up to k positive TF-IDF matches, highest similarity first.

    Scores measure word/phrase similarity, not legal relevance or likelihood
    of success. Empty queries and nonpositive k return []; invalid argument
    types raise TypeError. Data loading failures raise CaseDataError.
    """
    if not isinstance(query, str):
        raise TypeError("query must be a string")
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError("k must be an integer")
    if not query.strip() or k <= 0:
        return []
    vectorizer, matrix = _search_index()
    if vectorizer is None:
        return []
    scores = cosine_similarity(vectorizer.transform([query]), matrix).ravel()
    ranked = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    results = []
    for index in ranked[:k]:
        if scores[index] <= 0:
            break
        result = _result(_cases()[index])
        result["similarity_score"] = float(scores[index])
        results.append(result)
    return results
