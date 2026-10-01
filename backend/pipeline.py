"""Resumable orchestration of locked agents; no automatic memory writes.

Snapshots are private citizen session data. Store them only in appropriate UI
session storage. Activity events contain operational metadata, never prompts.
"""
from copy import deepcopy
from enum import Enum
import re

from backend import agents, tools


class Stage(str, Enum):
    LISTENING = "LISTENING"
    RESEARCHING = "RESEARCHING"
    ANALYZING_INITIAL = "ANALYZING_INITIAL"
    CHECKING_GAPS = "CHECKING_GAPS"
    WAITING_FOR_GAP_ANSWERS = "WAITING_FOR_GAP_ANSWERS"
    ANALYZING_FINAL = "ANALYZING_FINAL"
    GENERATING_GUIDANCE = "GENERATING_GUIDANCE"
    COMPLETE = "COMPLETE"
    ERROR = "ERROR"


class PipelineInputError(ValueError):
    """Invalid caller input; no agent runs and state remains unchanged."""


class CasePipeline:
    """Create with start(); resume with listener_message() or gap_answers().

    state() returns a detached JSON-compatible snapshot; CasePipeline(snapshot)
    restores a trusted snapshot from this application's session storage.
    Invalid caller input raises PipelineInputError. Agent errors become ERROR.
    ERROR is terminal; automatic retries and automatic safety resets are absent.
    """

    def __init__(self, snapshot=None):
        self._state = deepcopy(snapshot) if snapshot is not None else {
            "current_stage": Stage.LISTENING.value, "conversation": [],
            "question_count": 0, "next_question": "", "listener_ready": False,
            "case_file": agents.create_empty_case_file(), "research_result": None,
            "initial_analysis": None, "gap_result": None, "gap_items": [],
            "gap_answers": [], "final_analysis": None, "guidance_result": None,
            "activity_events": [], "error": None}
        if (not isinstance(self._state, dict)
                or self._state.get("current_stage") not in {stage.value for stage in Stage}):
            raise PipelineInputError("Invalid pipeline snapshot.")

    @classmethod
    def start(cls, message):
        pipeline = cls()
        pipeline.listener_message(message)
        return pipeline

    def state(self):
        return deepcopy(self._state)

    def events(self):
        return deepcopy(self._state["activity_events"])

    def _event(self, agent, status, message, **details):
        event = {"agent": agent, "status": status, "message": message,
                 "stage": self._state["current_stage"]}
        if details:
            event["details"] = details
        self._state["activity_events"].append(event)

    def _call(self, stage, agent, function, *args):
        self._state["current_stage"] = stage.value
        self._event(agent, "started", "Stage started.")
        try:
            result = function(*deepcopy(args))
        except Exception:
            self._state["current_stage"] = Stage.ERROR.value
            self._state["error"] = "A required stage failed. No downstream stages were run."
            self._event(agent, "error", self._state["error"])
            return None
        self._event(agent, "completed", "Stage completed.")
        return result

    def listener_message(self, message):
        s = self._state
        if s["current_stage"] != Stage.LISTENING.value:
            raise PipelineInputError("This case is not waiting for a Listener message.")
        if not isinstance(message, str) or not message.strip():
            raise PipelineInputError("Provide a nonempty citizen message.")
        s["conversation"].append({"role": "user", "content": message})
        result = self._call(Stage.LISTENING, "story_listener", agents.story_listener,
                            s["conversation"], s["case_file"], s["question_count"])
        if s["current_stage"] == Stage.ERROR.value:
            return self.state()
        s["case_file"] = result["updated_case_file"]
        s["next_question"] = result["next_question"]
        if result["danger_flag"]:
            s["current_stage"] = Stage.ERROR.value
            s["error"] = "Intake stopped because of an immediate safety concern."
            self._event("story_listener", "error", s["error"])
        elif not result["ready"]:
            s["question_count"] += 1
            s["conversation"].append({"role": "assistant", "content": s["next_question"]})
            self._event("story_listener", "waiting", "Waiting for a citizen intake answer.")
        else:
            s["listener_ready"] = True
            self.resume()
        return self.state()

    def resume(self):
        """Advance automatic stages from a trusted snapshot, stopping at input gates."""
        s = self._state
        if s["current_stage"] == Stage.LISTENING.value:
            # Only the ready Listener path enters here without a pending question.
            if not s["listener_ready"] or s["research_result"] is not None:
                return self.state()
            s["current_stage"] = Stage.RESEARCHING.value
        while True:
            stage = s["current_stage"]
            if stage == Stage.RESEARCHING.value:
                result = self._call(Stage.RESEARCHING, "research", agents.research_agent, s["case_file"])
                if s["current_stage"] == Stage.ERROR.value:
                    break
                s["research_result"] = result
                ids = [item["id"] for item in result["matches"] if re.fullmatch(r"[CU][0-9]+", item["id"])]
                self._event("research", "completed", "Retrieved case references recorded.", case_ids=ids)
                s["current_stage"] = Stage.ANALYZING_INITIAL.value
            elif stage == Stage.ANALYZING_INITIAL.value:
                result = self._call(Stage.ANALYZING_INITIAL, "analyst_initial", agents.analyst_agent,
                                    s["case_file"], s["research_result"])
                if s["current_stage"] == Stage.ERROR.value:
                    break
                s["initial_analysis"] = result
                s["current_stage"] = Stage.CHECKING_GAPS.value
            elif stage == Stage.CHECKING_GAPS.value:
                result = self._call(Stage.CHECKING_GAPS, "gap", agents.gap_agent,
                                    s["case_file"], s["initial_analysis"])
                if s["current_stage"] == Stage.ERROR.value:
                    break
                s["gap_result"] = result
                # Map only explicit authoritative document labels, never key-factor prose.
                documents = set()
                try:
                    for match in s["research_result"]["matches"]:
                        record = tools.get_case(match["id"])
                        if record:
                            documents.update(agents._gap_canonical(item) for item in record.get("documents_citizen_had", [])
                                             if isinstance(item, str))
                except Exception:
                    s["current_stage"] = Stage.ERROR.value
                    s["error"] = "Could not verify Gap evidence mapping."
                    self._event("gap", "error", s["error"])
                    break
                s["gap_items"] = [{"id": f"G{index + 1:03d}", "evidence": evidence,
                                   "question": question,
                                   "field": "documents_mentioned" if agents._gap_canonical(evidence) in documents else None}
                                  for index, (evidence, question) in enumerate(zip(result["missing_evidence"], result["questions"]))]
                if s["gap_items"]:
                    s["current_stage"] = Stage.WAITING_FOR_GAP_ANSWERS.value
                    self._event("gap", "waiting", "Waiting for structured citizen responses.", question_count=len(s["gap_items"]))
                    break
                s["final_analysis"] = s["initial_analysis"]
                s["current_stage"] = Stage.GENERATING_GUIDANCE.value
            elif stage == Stage.ANALYZING_FINAL.value:
                result = self._call(Stage.ANALYZING_FINAL, "analyst_final", agents.analyst_agent,
                                    s["case_file"], s["research_result"])
                if s["current_stage"] == Stage.ERROR.value:
                    break
                s["final_analysis"] = result
                s["current_stage"] = Stage.GENERATING_GUIDANCE.value
            elif stage == Stage.GENERATING_GUIDANCE.value:
                # Raw explanations stay in audit state. Canonical statuses alone
                # are adapted to the locked Guidance question/answer interface.
                questions = {item["id"]: item["question"] for item in s["gap_items"]}
                answers = [{"question": questions[item["id"]],
                            "answer": {"available": "Citizen explicitly confirms this evidence is available.",
                                       "unavailable": "Citizen reports this evidence is unavailable.",
                                       "unknown": "Citizen is unsure whether this evidence is available."}[item["status"]]}
                           for item in s["gap_answers"]]
                result = self._call(Stage.GENERATING_GUIDANCE, "guidance", agents.guidance_agent,
                                    s["case_file"], s["research_result"], s["final_analysis"], s["gap_result"], answers)
                if s["current_stage"] == Stage.ERROR.value:
                    break
                s["guidance_result"] = result
                s["current_stage"] = Stage.COMPLETE.value
                break
            else:
                break
        return self.state()

    def gap_answers(self, responses):
        """Submit all current items once; optional explanations are audit-only.

        Each response requires id and status; evidence is optional but must
        match exactly if supplied. Omitted explanation defaults to empty text.
        Partial submissions are rejected; callers should select unknown for
        unanswered items. Validation is atomic and never runs an LLM.
        """
        s = self._state
        if s["current_stage"] != Stage.WAITING_FOR_GAP_ANSWERS.value:
            raise PipelineInputError("This case is not waiting for Gap responses.")
        items = {item["id"]: item for item in s["gap_items"]}
        if not isinstance(responses, list) or len(responses) != len(items):
            raise PipelineInputError("Provide one structured response for every Gap item.")
        checked, seen = [], set()
        for response in responses:
            if (not isinstance(response, dict) or set(response) - {"id", "evidence", "status", "answer"}
                    or not isinstance(response.get("id"), str) or response["id"] not in items
                    or response["id"] in seen or response.get("status") not in ("available", "unavailable", "unknown")
                    or not isinstance(response.get("answer", ""), str)):
                raise PipelineInputError("Invalid structured Gap response.")
            item = items[response["id"]]
            if "evidence" in response and response["evidence"] != item["evidence"]:
                raise PipelineInputError("Gap evidence does not match the current question.")
            seen.add(response["id"])
            checked.append({"id": item["id"], "evidence": item["evidence"],
                            "status": response["status"], "answer": response.get("answer", "")})
        case = deepcopy(s["case_file"])
        existing = {agents._gap_canonical(item) for item in case["documents_mentioned"]}
        for response in checked:
            item = items[response["id"]]
            if response["status"] == "available" and item["field"] == "documents_mentioned":
                key = agents._gap_canonical(item["evidence"])
                if key not in existing:
                    case["documents_mentioned"].append(item["evidence"])
                    existing.add(key)
        changed = case != s["case_file"]
        s["case_file"], s["gap_answers"] = case, checked
        self._event("gap", "completed", "Structured citizen responses recorded.", facts_changed=changed)
        if not changed:
            s["final_analysis"] = s["initial_analysis"]
        s["current_stage"] = (Stage.ANALYZING_FINAL if changed else Stage.GENERATING_GUIDANCE).value
        return self.resume()
