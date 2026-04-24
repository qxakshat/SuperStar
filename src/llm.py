"""Thin LLM client — single interface for all model calls."""

import os
import json
import random
from typing import Optional

_client = None

def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "sk-placeholder"))
    return _client

def chat(
    messages: list[dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
    max_tokens: int = 1024,
    json_mode: bool = False,
) -> str:
    """Send a chat completion request. Falls back to mock if no API key."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key or api_key == "sk-placeholder":
        return _mock_response(messages)
    try:
        client = _get_client()
        kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except Exception as e:
        print(f"[LLM] API error: {e}, using mock")
        return _mock_response(messages)

def chat_json(messages: list[dict], model: str = "gpt-4o-mini", temperature: float = 0.4) -> dict:
    """Chat and parse JSON response."""
    raw = chat(messages, model=model, temperature=temperature, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON from the response
        import re
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            return json.loads(match.group())
        return {}

def _mock_response(messages: list[dict]) -> str:
    """Deterministic mock for testing without API key."""
    last = messages[-1]["content"] if messages else ""
    if "standup" in last.lower() or "update" in last.lower():
        tasks = ["authentication module", "API endpoint", "database migration", "unit tests", "code review"]
        statuses = ["making good progress", "almost done", "hit a blocker but working through it", "completed ahead of schedule"]
        return json.dumps({
            "standup": f"Working on {random.choice(tasks)}. {random.choice(statuses)}.",
            "task_progress": random.randint(10, 40),
            "blockers": random.choice(["none", "waiting on code review", "unclear requirements"]),
            "mood": random.choice(["focused", "motivated", "tired but pushing through", "energized"])
        })
    if "manager" in last.lower() or "feedback" in last.lower() or "assign" in last.lower():
        return json.dumps({
            "visible_feedback": random.choice([
                "Great progress team, keep it up!",
                "Let's focus on the critical path items today.",
                "I've reprioritized some tasks based on yesterday's progress.",
                "Good standup everyone. A few notes on priorities."
            ]),
            "hidden_feedback": random.choice([
                "Agent_2 seems to be slowing down, might need reassignment.",
                "The project is at risk if we don't pick up velocity.",
                "Agent_1 is outperforming expectations, consider for lead role.",
                "Team morale seems low, might need to adjust expectations."
            ]),
            "task_assignments": {},
            "priority_changes": []
        })
    if "environment" in last.lower() or "judge" in last.lower() or "evaluate" in last.lower():
        return json.dumps({
            "work_quality": round(random.uniform(0.5, 1.0), 2),
            "suggestion": random.choice([
                "Consider pair programming for complex tasks",
                "Sprint velocity is on track",
                "Risk of scope creep detected, recommend backlog grooming",
                "Team could benefit from a short break"
            ]),
            "hidden_assessment": random.choice([
                "Interpersonal tension detected between Agent_1 and Agent_3",
                "Overall project health is good but burnout risk is medium",
                "Manager's feedback style may be demotivating Agent_2",
                "Technical debt is accumulating faster than expected"
            ]),
            "risk_level": random.choice(["low", "medium", "high"])
        })
    return json.dumps({"response": "Acknowledged", "status": "ok"})
