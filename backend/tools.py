"""Similarity retrieval of synthetic examples, never legal advice or precedent."""

from copy import deepcopy
from functools import lru_cache
import json
from pathlib import Path
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


class ReferralDataError(ValueError):
    """Local fictional directory could not be read safely."""


def _directory_text(value):
    return " ".join(value.split()).casefold() if isinstance(value, str) else ""


def _load_referral_directory(filename):
    """Read fresh local synthetic records; no caches or external lookups."""
    path = Path(__file__).resolve().parent.parent / "data" / filename
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError:
        raise ReferralDataError("Could not read the local synthetic referral directory.") from None
    if not text.strip():
        return []
    try:
        records = json.loads(text)
        if not isinstance(records, list):
            raise ValueError()
        seen = set()
        for record in records:
            if (not isinstance(record, dict) or record.get("source") != "synthetic"
                    or not all(isinstance(record.get(field), str) and record[field].strip()
                               for field in ("id", "name", "city"))
                    or record["id"] in seen
                    or not isinstance(record.get("languages"), list)
                    or not all(isinstance(item, str) and item.strip() for item in record["languages"])):
                raise ValueError()
            if filename == "lawyers.json" and (
                    not isinstance(record.get("specializations"), list)
                    or not all(isinstance(item, str) and item.strip() for item in record["specializations"])):
                raise ValueError()
            seen.add(record["id"])
    except (ValueError, TypeError):
        raise ReferralDataError("Invalid local synthetic referral directory.") from None
    return records


def find_lawyers(specialization, city, language):
    """Exact normalized lookup in fictional memory; all three inputs required.

    Languages use listed codes (en/hi/mr), with no inferred translations.
    These records are demo identities, not real referrals or contact services.
    """
    specialization, city, language = map(_directory_text, (specialization, city, language))
    if not specialization or not city or not language:
        return []
    return [deepcopy(record) for record in _load_referral_directory("lawyers.json")
            if _directory_text(record["city"]) == city
            and specialization in {_directory_text(item) for item in record["specializations"]}
            and language in {_directory_text(item) for item in record["languages"]}]


def find_legal_aid(city):
    """Fictional providers for an explicitly supplied city; no eligibility claim."""
    city = _directory_text(city)
    if not city:
        return []
    return [deepcopy(record) for record in _load_referral_directory("legal_aid.json")
            if _directory_text(record["city"]) == city]
