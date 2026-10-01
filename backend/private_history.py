"""Local prototype profile access and private history, NOT authentication.

Contact digests are deterministic lookup keys, not secrets or proof of identity.
Anyone knowing a contact can recover its profile. Snapshots contain private
citizen data and SQLite is not encrypted. Protect the host/database accordingly.
This module is deliberately absent from Research and shared Memory storage.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3
from uuid import uuid4

from backend.pipeline import CasePipeline

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "private_history.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS profiles (
 id TEXT PRIMARY KEY, display_name TEXT NOT NULL,
 contact_digest TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS private_cases (
 id TEXT PRIMARY KEY, profile_id TEXT NOT NULL REFERENCES profiles(id),
 title TEXT NOT NULL, legal_area TEXT NOT NULL, status TEXT NOT NULL,
 pipeline_snapshot_json TEXT NOT NULL, created_at TEXT NOT NULL,
 updated_at TEXT NOT NULL);
"""


class HistoryError(ValueError):
    """Safe invalid-input, ownership or storage failure."""


def _text(value, limit):
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit or any(ord(c) < 32 for c in value):
        raise HistoryError("Invalid profile or case metadata.")
    return value.strip()


def _contact_digest(contact):
    contact = _text(contact, 254)
    if "@" in contact:
        normalized = contact.lower()
        if not re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+", normalized):
            raise HistoryError("Provide a basic email or phone identifier.")
        value = "email:" + normalized
    else:
        if not re.fullmatch(r"(?:\+|00)?[0-9][0-9 ()-]*", contact):
            raise HistoryError("Provide a basic email or phone identifier.")
        normalized = re.sub(r"[ ()-]", "", contact)
        if normalized.startswith("00"):
            normalized = "+" + normalized[2:]
        if not 7 <= len(normalized.lstrip("+")) <= 15:
            raise HistoryError("Provide a basic email or phone identifier.")
        value = "phone:" + normalized
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _snapshot_json(snapshot):
    # Only persist the existing public snapshot; no storage-specific state.
    try:
        if not isinstance(snapshot, dict) or set(snapshot) != set(CasePipeline().state()):
            raise ValueError()
        template = CasePipeline().state()
        for field, default in template.items():
            if default is not None and type(snapshot[field]) is not type(default):
                raise ValueError()
        if set(snapshot["case_file"]) != set(template["case_file"]):
            raise ValueError()
        for field, default in template["case_file"].items():
            if type(snapshot["case_file"][field]) is not type(default):
                raise ValueError()
        def check(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if not isinstance(key, str) or key.casefold() in {
                        "api_key", "gemini_api_key", "secret", "password", "credentials",
                        "system_prompt", "raw_prompt", "chain_of_thought", "hidden_reasoning"}:
                        raise ValueError()
                    check(item)
            elif isinstance(value, list):
                for item in value:
                    check(item)
            elif value is not None and type(value) not in (str, int, float, bool):
                raise ValueError()
        check(snapshot)
        encoded = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
        CasePipeline(json.loads(encoded))
        return encoded
    except Exception:
        raise HistoryError("Invalid public pipeline snapshot.") from None


class PrivateHistory:
    """Inject a path for tests. Construction is lazy and creates no database.

    All case operations require the owning profile ID. These IDs are prototype
    ownership scopes, not authenticated access tokens. Name changes on recovery
    are ignored: the first registered display name is retained.
    """
    def __init__(self, path=DEFAULT_PATH):
        self.path = Path(path)

    @contextmanager
    def _connection(self):
        connection = None
        try:
            connection = sqlite3.connect(self.path)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.executescript(SCHEMA)
            with connection:
                yield connection
        except sqlite3.Error:
            raise HistoryError("Private history storage operation failed.") from None
        finally:
            if connection is not None:
                connection.close()

    def get_or_create_profile(self, display_name, contact):
        name = _text(display_name, 100)
        digest = _contact_digest(contact)
        with self._connection() as db:
            db.execute("INSERT INTO profiles VALUES (?, ?, ?, ?) ON CONFLICT(contact_digest) DO NOTHING",
                       ("P" + uuid4().hex, name, digest, datetime.now(timezone.utc).isoformat()))
            row = db.execute("SELECT id, display_name, created_at FROM profiles WHERE contact_digest = ?", (digest,)).fetchone()
            return dict(row)

    def create_private_case(self, profile_id, pipeline_snapshot, title):
        encoded = _snapshot_json(pipeline_snapshot)
        title = _text(title, 200)
        area = pipeline_snapshot["case_file"].get("legal_area", "")
        if not isinstance(area, str):
            raise HistoryError("Invalid legal area.")
        case_id = "H" + uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with self._connection() as db:
            if not db.execute("SELECT id FROM profiles WHERE id = ?", (profile_id,)).fetchone():
                raise HistoryError("Profile not found.")
            db.execute("INSERT INTO private_cases VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       (case_id, profile_id, title, area, pipeline_snapshot["current_stage"], encoded, now, now))
        return self.get_private_case(profile_id, case_id)

    def get_profile_cases(self, profile_id):
        with self._connection() as db:
            return [dict(row) for row in db.execute(
                "SELECT id, profile_id, title, legal_area, status, created_at, updated_at FROM private_cases WHERE profile_id = ? ORDER BY updated_at DESC, id DESC", (profile_id,))]

    def get_private_case(self, profile_id, case_id):
        with self._connection() as db:
            row = db.execute("SELECT * FROM private_cases WHERE profile_id = ? AND id = ?", (profile_id, case_id)).fetchone()
        if row is None:
            return None
        result = dict(row)
        try:
            snapshot = json.loads(result.pop("pipeline_snapshot_json"))
            _snapshot_json(snapshot)
        except Exception:
            raise HistoryError("Stored private snapshot could not be restored.") from None
        result["pipeline_snapshot"] = snapshot
        return result

    def save_private_case_snapshot(self, profile_id, case_id, pipeline_snapshot, title=None):
        encoded = _snapshot_json(pipeline_snapshot)
        if title is not None:
            title = _text(title, 200)
        area = pipeline_snapshot["case_file"].get("legal_area", "")
        if not isinstance(area, str):
            raise HistoryError("Invalid legal area.")
        with self._connection() as db:
            cursor = db.execute("UPDATE private_cases SET pipeline_snapshot_json = ?, legal_area = ?, status = ?, title = COALESCE(?, title), updated_at = ? WHERE profile_id = ? AND id = ?",
                (encoded, area, pipeline_snapshot["current_stage"], title, datetime.now(timezone.utc).isoformat(), profile_id, case_id))
            if cursor.rowcount != 1:
                raise HistoryError("Private case not found for this profile.")
        return self.get_private_case(profile_id, case_id)

    def delete_private_case(self, profile_id, case_id):
        with self._connection() as db:
            return db.execute("DELETE FROM private_cases WHERE profile_id = ? AND id = ?", (profile_id, case_id)).rowcount == 1

    def reopen_private_case(self, profile_id, case_id):
        record = self.get_private_case(profile_id, case_id)
        return CasePipeline(record["pipeline_snapshot"]) if record is not None else None
