"""Read synthetic demonstration cases; these are not judgments or precedent."""

import json
from pathlib import Path
from typing import Any

CASES_PATH = Path(__file__).resolve().parent.parent / "data" / "cases.json"


class CaseDataError(ValueError):
    """The case data could not be read or has an invalid structure."""


def load_cases(path: str | Path = CASES_PATH) -> list[dict[str, Any]]:
    """Load fresh records, checking the list, record objects, and unique IDs.

    The default path is relative to this module, not the working directory.
    An optional path supports testing or a future alternative data source.
    """
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as file:
            cases = json.load(file)
    except FileNotFoundError as exc:
        raise CaseDataError(f"Case data file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CaseDataError(
            f"Invalid JSON in {path} at line {exc.lineno}, column {exc.colno}."
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise CaseDataError(f"Cannot read case data at {path}: {exc}") from exc

    if not isinstance(cases, list):
        raise CaseDataError(f"Case data in {path} must be a list of case records.")
    seen_ids = set()
    for position, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise CaseDataError(f"Case record {position} in {path} must be an object.")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise CaseDataError(f"Case record {position} in {path} needs a nonempty string ID.")
        if case_id in seen_ids:
            raise CaseDataError(f"Duplicate case ID {case_id!r} in {path}.")
        seen_ids.add(case_id)
    return cases
