"""Reusable Gemini JSON transport for legal information and case preparation.

This is not legal advice or a substitute for a lawyer. No prompts or responses
are logged or saved here. Agents should import this module, not Google's SDK.
"""

import json
import os
from functools import lru_cache
from pathlib import Path
import re
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
import httpx
from jsonschema import Draft202012Validator

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
_DEFAULT_MODEL = "gemini-3.5-flash-lite"
_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}


class LLMError(Exception):
    """Base error with a message safe to display without request content."""


class LLMConfigurationError(LLMError):
    """Missing credentials or unusable local configuration."""


class LLMInputError(LLMError):
    """Invalid prompt arguments or response schema; no request was sent."""


class LLMAPIError(LLMError):
    """The provider request failed, possibly after bounded retries."""


class LLMResponseError(LLMError):
    """The response was empty, invalid JSON, or inconsistent with the schema."""


def _reject_references(value: Any) -> None:
    # Inline schemas keep validation local: never fetch schemas from the web.
    if isinstance(value, dict):
        if any(key in value for key in ("$ref", "$dynamicRef", "$recursiveRef")):
            raise LLMInputError("Use an inline response_schema without schema references.")
        for child in value.values():
            _reject_references(child)
    elif isinstance(value, list):
        for child in value:
            _reject_references(child)


def _prepare_schema(response_schema: dict) -> tuple[dict, Draft202012Validator]:
    if not isinstance(response_schema, dict) or response_schema.get("type") != "object":
        raise LLMInputError("response_schema must be a dict with type 'object'.")
    try:
        # Copy before passing to the SDK, and reject non-JSON values/NaN.
        schema = json.loads(json.dumps(response_schema, allow_nan=False))
        _reject_references(schema)
        draft_uri = Draft202012Validator.META_SCHEMA["$id"]
        if schema.get("$schema", draft_uri) != draft_uri:
            raise LLMInputError("Use a Draft 2020-12 schema or omit $schema.")
        Draft202012Validator.check_schema(schema)
        return schema, Draft202012Validator(schema)
    except LLMInputError:
        raise
    except Exception:
        # Validator errors may embed schema descriptions or example values.
        raise LLMInputError("response_schema is not a valid JSON object schema.") from None


def _settings() -> tuple[str, str]:
    try:
        load_dotenv(dotenv_path=_ENV_PATH, override=False)
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        model = os.environ.get("GEMINI_MODEL", _DEFAULT_MODEL).strip()
    except Exception:
        raise LLMConfigurationError("Could not load Gemini configuration.") from None
    if not api_key:
        raise LLMConfigurationError("GEMINI_API_KEY is required in the environment or project .env.")
    if not re.fullmatch(r"(?:models/)?[A-Za-z0-9][A-Za-z0-9._-]*", model):
        raise LLMConfigurationError("GEMINI_MODEL must be a nonempty Gemini model ID.")
    return api_key, model


@lru_cache(maxsize=1)
def _get_client(api_key: str):
    """Reuse the client; disable SDK retries so our attempt cap is real."""
    try:
        return genai.Client(
            api_key=api_key,
            vertexai=False,
            http_options=types.HttpOptions(
                timeout=30_000,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    except Exception:
        raise LLMConfigurationError("Could not initialize the Gemini client.") from None


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON number")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _parse_response(response: Any, validator: Draft202012Validator) -> dict:
    try:
        text = response.text if response is not None else None
    except Exception:
        raise LLMResponseError("Could not read Gemini's structured response.") from None
    if not isinstance(text, str) or not text.strip():
        raise LLMResponseError("Gemini returned no text; the response may have been blocked.")
    try:
        result = json.loads(
            text, parse_constant=_reject_constant, object_pairs_hook=_unique_object
        )
        # Also reject overflow such as 1e999, which Python otherwise parses as inf.
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, RecursionError):
        raise LLMResponseError("Gemini returned malformed JSON.") from None
    if not isinstance(result, dict):
        raise LLMResponseError("Gemini's JSON response must be an object.")
    try:
        validator.validate(result)
    except Exception:
        raise LLMResponseError("Gemini's JSON response does not match response_schema.") from None
    return result


def _ask_llm_json(
    system_prompt: str, user_prompt: str, response_schema: dict, *, max_attempts: int
) -> dict:
    """Shared implementation; the manual smoke test uses one total attempt."""
    for name, prompt in (("system_prompt", system_prompt), ("user_prompt", user_prompt)):
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMInputError(f"{name} must be nonempty text.")
    schema, validator = _prepare_schema(response_schema)
    api_key, model = _settings()
    client = _get_client(api_key)
    try:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            response_mime_type="application/json",
            response_json_schema=schema,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
    except Exception:
        raise LLMInputError("Could not configure Gemini structured output.") from None

    for attempt in range(max_attempts):
        try:
            response = client.models.generate_content(
                model=model, contents=user_prompt, config=config
            )
        except errors.APIError as exc:
            temporary = exc.code in _TRANSIENT_STATUS_CODES
            if not temporary:
                raise LLMAPIError(
                    "Gemini rejected the request; check credentials, model access, and schema support."
                ) from None
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError):
            temporary = True
        except Exception:
            # Never propagate SDK exception text, which can contain private data.
            raise LLMAPIError("Gemini request failed; check the SDK and local configuration.") from None
        else:
            # Invalid/empty output is not retried and is never replaced with defaults.
            return _parse_response(response, validator)
        if temporary and attempt + 1 < max_attempts:
            time.sleep(0.5 * (2 ** attempt))
    raise LLMAPIError("Gemini is temporarily unavailable or rate limited; try again later.")


def ask_llm_json(system_prompt: str, user_prompt: str, response_schema: dict) -> dict:
    """Return validated JSON using separate system and user instructions.

    Accepts inline Draft 2020-12 object schemas. Core schema constraints are
    checked locally; `format` remains an annotation. Gemini supports a subset
    of JSON Schema and may reject complex schemas. At most three attempts are
    made, only for transient HTTP/network failures, with 0.5s then 1s backoff.
    Raises LLMInputError, LLMConfigurationError, LLMAPIError or LLMResponseError.
    """
    return _ask_llm_json(system_prompt, user_prompt, response_schema, max_attempts=3)
