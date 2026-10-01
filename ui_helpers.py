"""Small UI adapters; the existing pipeline remains the workflow authority."""
from backend.pipeline import CasePipeline
from copy import deepcopy


def current_activity(snapshot):
    """One latest status per agent; the backend timeline remains unchanged."""
    latest = {}
    for event in snapshot["activity_events"]:
        latest[event["agent"]] = event
    return list(latest.values())


def listener_recovery(snapshot):
    """Reconstruct the trusted pre-call state of a failed intake turn only.

    Listener failures retain prior facts and append a user message followed by
    started/error events. Safety stops, downstream failures and other shapes
    are ineligible. Restoring requires a separate explicit UI action.
    """
    events = snapshot["activity_events"]
    messages = snapshot["conversation"]
    if (snapshot["current_stage"] != "ERROR" or snapshot["case_file"]["danger_flag"]
            or snapshot["listener_ready"] or len(events) < 2 or not messages
            or messages[-1]["role"] != "user"
            or [(e["agent"], e["status"]) for e in events[-2:]] !=
            [("story_listener", "started"), ("story_listener", "error")]):
        return None
    previous = deepcopy(snapshot)
    previous["activity_events"] = previous["activity_events"][:-2]
    previous["conversation"] = previous["conversation"][:-1]
    previous["current_stage"] = "LISTENING"
    previous["error"] = None
    return previous


def workflow_view(snapshot):
    return {"WAITING_FOR_GAP_ANSWERS": "evidence", "COMPLETE": "preparation"}.get(snapshot["current_stage"], "workspace")


def create_case(store, profile_id, title):
    pipeline = CasePipeline()
    record = store.create_private_case(profile_id, pipeline.state(), title)
    return record["id"], pipeline


def gap_responses(items, statuses, explanations):
    return [{"id": item["id"], "evidence": item["evidence"],
             "status": status, "answer": explanation}
            for item, status, explanation in zip(items, statuses, explanations)]


def progress(snapshot):
    names = [("story_listener", "LISTENING"), ("research", "RESEARCHING"),
             ("analyst_initial", "ANALYZING_INITIAL"), ("gap", "CHECKING_GAPS"),
             ("analyst_final", "ANALYZING_FINAL"), ("guidance", "GENERATING_GUIDANCE")]
    events = snapshot["activity_events"]
    result = []
    for agent, stage in names:
        completed = any(e["agent"] == agent and e["status"] == "completed" for e in events)
        active = snapshot["current_stage"] == stage or (agent == "gap" and snapshot["current_stage"] == "WAITING_FOR_GAP_ANSWERS")
        skipped = agent == "analyst_final" and snapshot["current_stage"] == "COMPLETE" and not completed
        result.append((agent, "current" if active else "completed" if completed else "not_required" if skipped else "waiting"))
    return result
