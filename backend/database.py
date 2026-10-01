"""Read synthetic demonstration cases; these are not judgments or precedent."""

import json
import os
import tempfile
from threading import RLock
from pathlib import Path
from typing import Any

CASES_PATH = Path(__file__).resolve().parent.parent / "data" / "cases.json"
USER_MEMORY_PATH = CASES_PATH.with_name("user_memory.json")
_MEMORY_LOCK = RLock()


def load_user_memory():
    from backend.memory import MemoryAgentError, validate_memory_record
    try:
        if not USER_MEMORY_PATH.exists():
            return []
        records = json.loads(USER_MEMORY_PATH.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            raise ValueError()
        validated = [validate_memory_record(record) for record in records]
        if len({record["id"] for record in validated}) != len(validated):
            raise ValueError()
        return validated
    except Exception:
        raise MemoryAgentError("Stored user memory is invalid or unreadable; nothing was overwritten.") from None


def save_user_memory(record):
    from backend.memory import MemoryAgentError, validate_memory_record
    record = validate_memory_record(record)
    with _MEMORY_LOCK:
        records = load_user_memory()
        for existing in records:
            if existing["id"] == record["id"]:
                if existing == record:
                    return record, False
                raise MemoryAgentError("Memory ID collision; prepare a new preview.")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=USER_MEMORY_PATH.parent,
                                             delete=False, suffix=".tmp") as file:
                temporary = Path(file.name)
                json.dump(records + [record], file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, USER_MEMORY_PATH)
        except Exception:
            raise MemoryAgentError("Memory write failed; previous memory was preserved.") from None
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return record, True


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
