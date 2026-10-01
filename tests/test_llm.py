"""Offline unit tests: no real credentials, .env reads, or Gemini calls."""

from copy import deepcopy
import json
import os
from types import SimpleNamespace
import unittest
from unittest.mock import call, patch
import traceback

from backend import llm

SCHEMA = {
    "type": "object",
    "properties": {
        "legal_area": {"type": "string"},
        "summary": {"type": "string"},
    },
    "required": ["legal_area", "summary"],
    "additionalProperties": False,
}
OUTPUT = {"legal_area": "land_dispute", "summary": "Synthetic demo."}


class LLMTests(unittest.TestCase):
    def setUp(self):
        # Patch the SDK constructor as a final guard against real API calls.
        llm._get_client.cache_clear()
        self.addCleanup(llm._get_client.cache_clear)
        self.env = patch.dict(os.environ, {"GEMINI_API_KEY": "unit-test-placeholder"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        self.dotenv = patch.object(llm, "load_dotenv").start()
        self.constructor = patch.object(llm.genai, "Client").start()
        self.sleep = patch.object(llm.time, "sleep").start()
        self.addCleanup(patch.stopall)
        self.generate = self.constructor.return_value.models.generate_content
        self.generate.return_value = SimpleNamespace(text=json.dumps(OUTPUT))

    def ask(self, schema=None):
        return llm.ask_llm_json("Synthetic system instruction", "Synthetic story", SCHEMA if schema is None else schema)

    def test_missing_key(self):
        for key in (None, "", "  "):
            with self.subTest(key=key):
                if key is None:
                    os.environ.pop("GEMINI_API_KEY", None)
                else:
                    os.environ["GEMINI_API_KEY"] = key
                with self.assertRaises(llm.LLMConfigurationError):
                    self.ask()
        self.constructor.assert_not_called()

    def test_structured_success_and_client_reuse(self):
        before = deepcopy(SCHEMA)
        self.assertEqual(self.ask(), OUTPUT)
        self.assertEqual(self.ask(), OUTPUT)
        self.constructor.assert_called_once()
        self.dotenv.assert_called_with(dotenv_path=llm._ENV_PATH, override=False)
        self.assertTrue(llm._ENV_PATH.is_absolute())
        kwargs = self.generate.call_args.kwargs
        self.assertEqual(kwargs["model"], "gemini-3.5-flash-lite")
        self.assertEqual(kwargs["contents"], "Synthetic story")
        config = kwargs["config"]
        self.assertEqual(config.system_instruction, "Synthetic system instruction")
        self.assertEqual(config.response_mime_type, "application/json")
        self.assertEqual(config.response_json_schema, SCHEMA)
        self.assertTrue(config.automatic_function_calling.disable)
        options = self.constructor.call_args.kwargs["http_options"]
        self.assertEqual(options.retry_options.attempts, 1)
        self.assertEqual(options.timeout, 30_000)
        self.assertEqual(SCHEMA, before)
        self.sleep.assert_not_called()

    def test_model_override(self):
        os.environ["GEMINI_MODEL"] = "demo-model"
        self.ask()
        self.assertEqual(self.generate.call_args.kwargs["model"], "demo-model")

    def test_sdk_response_object(self):
        self.generate.return_value = llm.types.GenerateContentResponse(
            candidates=[llm.types.Candidate(
                content=llm.types.Content(
                    parts=[llm.types.Part(text=json.dumps(OUTPUT))]
                )
            )]
        )
        self.assertEqual(self.ask(), OUTPUT)

    def test_invalid_model(self):
        for model in (" ", "bad model", "https://invalid.example"):
            with self.subTest(model=model):
                os.environ["GEMINI_MODEL"] = model
                with self.assertRaises(llm.LLMConfigurationError):
                    self.ask()
        self.constructor.assert_not_called()

    def test_invalid_prompt_arguments(self):
        for bad in (None, "", " \n ", 7):
            for prompts in ((bad, "story"), ("system", bad)):
                with self.subTest(prompts=prompts), self.assertRaises(llm.LLMInputError):
                    llm.ask_llm_json(*prompts, SCHEMA)
        self.generate.assert_not_called()

    def test_invalid_schema(self):
        schemas = [
            None, [], {}, {"type": "array"},
            {"type": "object", "properties": {"x": {"type": "not-a-type"}}},
            {"type": "object", "required": "summary"},
            {"type": "object", "properties": {"x": {"$ref": "https://invalid.example"}}},
            {"type": "object", "$schema": "unknown-draft"},
            {"type": "object", "default": object()},
        ]
        for schema in schemas:
            with self.subTest(schema=schema), self.assertRaises(llm.LLMInputError):
                llm.ask_llm_json("system", "story", schema)
        self.generate.assert_not_called()
        self.dotenv.assert_not_called()

    def test_empty_response(self):
        for response in (None, SimpleNamespace(text=None), SimpleNamespace(text=" \n ")):
            self.generate.reset_mock()
            self.generate.return_value = response
            with self.subTest(response=response), self.assertRaises(llm.LLMResponseError):
                self.ask()
            self.generate.assert_called_once()
        self.sleep.assert_not_called()

    def test_malformed_json(self):
        for text in ("broken", "```json\n{}\n```", '{"x": NaN}', '{"x": 1e999}', '{"x":1,"x":2}', "[]", "null"):
            self.generate.reset_mock()
            self.generate.return_value = SimpleNamespace(text=text)
            with self.subTest(text=text), self.assertRaises(llm.LLMResponseError):
                self.ask()
            self.generate.assert_called_once()

    def test_schema_violations(self):
        for output in ({}, {"legal_area": 7, "summary": "demo"}, {**OUTPUT, "extra": True}):
            self.generate.return_value = SimpleNamespace(text=json.dumps(output))
            with self.subTest(output=output), self.assertRaises(llm.LLMResponseError):
                self.ask()

    def test_nested_schema_validation(self):
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "minItems": 1, "items": {
                "type": "object", "properties": {"kind": {"enum": ["demo"]}},
                "required": ["kind"], "additionalProperties": False,
            }}},
            "required": ["items"],
        }
        self.generate.return_value = SimpleNamespace(text='{"items":[{"kind":"demo"}]}')
        self.assertEqual(self.ask(schema), {"items": [{"kind": "demo"}]})
        for text in ('{"items":[]}', '{"items":[{"kind":"other"}]}'):
            self.generate.return_value = SimpleNamespace(text=text)
            with self.assertRaises(llm.LLMResponseError):
                self.ask(schema)

    def test_permanent_api_failure_not_retried(self):
        for code in (400, 401, 403, 404):
            self.generate.reset_mock()
            self.generate.side_effect = llm.errors.APIError(code, {"message": "private provider details"})
            with self.subTest(code=code), self.assertRaises(llm.LLMAPIError):
                self.ask()
            self.generate.assert_called_once()
        self.sleep.assert_not_called()

    def test_transient_failure_then_success(self):
        self.generate.side_effect = [
            llm.errors.APIError(503, {"message": "temporary"}),
            SimpleNamespace(text=json.dumps(OUTPUT)),
        ]
        self.assertEqual(self.ask(), OUTPUT)
        self.assertEqual(self.generate.call_count, 2)
        self.sleep.assert_called_once_with(0.5)

    def test_transient_retries_are_bounded(self):
        failures = [llm.errors.APIError(code, {"message": "temporary"}) for code in (429, 503, 500)]
        failures += [llm.httpx.ConnectError("private"), llm.httpx.ReadTimeout("private")]
        for failure in failures:
            self.generate.reset_mock()
            self.sleep.reset_mock()
            self.generate.side_effect = failure
            with self.subTest(kind=type(failure).__name__), self.assertRaises(llm.LLMAPIError):
                self.ask()
            self.assertEqual(self.generate.call_count, 3)
            self.assertEqual(self.sleep.call_args_list, [call(0.5), call(1.0)])

    def test_errors_do_not_expose_sensitive_details(self):
        marker = "PRIVATE_SENTINEL"
        for failure in (llm.errors.APIError(400, {"message": marker}), RuntimeError(marker)):
            self.generate.side_effect = failure
            try:
                self.ask()
            except llm.LLMAPIError as exc:
                self.assertNotIn(marker, "".join(traceback.format_exception(exc)))
            else:
                self.fail("Expected a sanitized error")
        self.generate.side_effect = None
        self.generate.return_value = SimpleNamespace(text=json.dumps({"private": marker}))
        try:
            self.ask()
        except llm.LLMResponseError as exc:
            self.assertNotIn(marker, "".join(traceback.format_exception(exc)))

    def test_client_initialization_error_is_sanitized(self):
        self.constructor.side_effect = ValueError("PRIVATE_SENTINEL")
        with self.assertRaisesRegex(llm.LLMConfigurationError, "initialize") as caught:
            self.ask()
        self.assertNotIn("PRIVATE_SENTINEL", str(caught.exception))
        self.generate.assert_not_called()

    def test_manual_smoke_uses_one_attempt(self):
        # Importing the script does not run it. Even explicit main() is mocked.
        from scripts import smoke_llm
        self.generate.assert_not_called()
        self.generate.side_effect = llm.errors.APIError(503, {"message": "temporary"})
        with patch("builtins.print"):
            self.assertEqual(smoke_llm.main(), 1)
        self.generate.assert_called_once()
        self.sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
