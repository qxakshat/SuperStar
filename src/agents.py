"""LLM-powered Engineer and Manager agents."""

import json
import random
from dataclasses import dataclass, field
from typing import Optional
from llm import chat, chat_json
from messages import MessageBus

@dataclass
class AgentProfile:
    agent_id: str
    name: str
    role: str  # "engineer" or "manager"
    skill_level: float = 0.7  # 0-1
    personality: str = "balanced"  # balanced, aggressive, cautious, creative
    reliability: float = 0.8
    wfh_tendency: float = 0.2
    specialization: str = "fullstack"
    is_human: bool = False

ENGINEER_SYSTEM = """You are {name}, a software engineer on a 10-day sprint team.
Your personality: {personality}. Skill level: {skill_level}/1.0. Specialization: {specialization}.
Current morale: {morale}/100, Energy: {energy}/100.

You receive visible messages from your team and manager. Respond with a JSON object:
{{
    "standup": "Your daily standup update (1-2 sentences)",
    "task_progress": <integer 0-100, how much progress you made today>,
    "blockers": "any blockers or 'none'",
    "mood": "one word describing your mood"
}}

Be realistic - if your energy is low, make less progress. If morale is low, your standup might be terse.
Consider events affecting you. Don't always be positive - reflect your actual state."""

MANAGER_SYSTEM = """You are {name}, the engineering manager of a 10-day sprint team.
Your team: {team_list}. Day {day}/10 of the sprint.
Project health: {health}. Completion: {completion_pct}%.

You see all public standups plus hidden performance data. Respond with JSON:
{{
    "visible_feedback": "Public feedback to the team (1-2 sentences)",
    "hidden_feedback": "Private assessment only you can see (honest evaluation)",
    "task_assignments": {{}},
    "priority_changes": [],
    "suggestions": ["suggestion1", "suggestion2"]
}}

Be strategic. Consider team dynamics, individual performance, and project deadlines.
If someone is struggling, suggest help. If the project is at risk, escalate."""

ENV_JUDGE_SYSTEM = """You are the sprint environment judge. Evaluate the day's work objectively.
Team size: {team_size}. Day: {day}/10. Current health: {health}.

Given today's standups and events, produce JSON:
{{
    "work_quality": <float 0-1>,
    "suggestion": "A constructive suggestion for the team",
    "hidden_assessment": "Honest hidden assessment of team dynamics",
    "risk_level": "low|medium|high",
    "agent_adjustments": {{
        "agent_id": {{"morale_delta": 0, "energy_delta": 0, "collaboration": 0.0}}
    }}
}}"""


class EngineerAgent:
    def __init__(self, profile: AgentProfile, model: str = "gpt-4o-mini"):
        self.profile = profile
        self.model = model
        self.history: list[dict] = []

    def act(self, day: int, visible_messages: list, morale: int, energy: int,
            current_task: Optional[str] = None, events_today: list = None) -> dict:
        if self.profile.is_human:
            return None  # handled by HumanAdapter

        events_str = "; ".join(e.description for e in (events_today or []))
        msgs_str = "\n".join(f"[{m.sender}]: {m.content}" for m in visible_messages[-10:])

        system = ENGINEER_SYSTEM.format(
            name=self.profile.name, personality=self.profile.personality,
            skill_level=self.profile.skill_level, specialization=self.profile.specialization,
            morale=morale, energy=energy
        )
        user_msg = f"""Day {day}/10.
Current task: {current_task or 'None assigned'}
Events today: {events_str or 'None'}
Recent messages:
{msgs_str}

Provide your daily standup and progress update as JSON."""

        self.history.append({"role": "user", "content": user_msg})
        messages = [{"role": "system", "content": system}] + self.history[-6:]

        result = chat_json(messages, model=self.model)
        self.history.append({"role": "assistant", "content": json.dumps(result)})

        # Adjust progress by capacity — keep it meaningful (don't over-penalize)
        base_progress = result.get("task_progress", random.randint(20, 40))
        # Blended multiplier: 50% base + 50% skill-weighted so even low-skill agents contribute
        skill_factor = 0.5 + 0.5 * self.profile.skill_level * self.profile.reliability
        energy_factor = 0.6 + 0.4 * min(1.0, energy / 80)
        result["task_progress"] = max(10, int(base_progress * skill_factor * energy_factor))
        return result


class ManagerAgent:
    def __init__(self, profile: AgentProfile, model: str = "gpt-4o-mini"):
        self.profile = profile
        self.model = model
        self.history: list[dict] = []

    def act(self, day: int, visible_messages: list, hidden_messages: list,
            agent_scores: dict, project_health: dict, events_today: list = None) -> dict:

        team_list = ", ".join(f"{aid} (score:{s.total_score:.0f}, morale:{s.morale})"
                              for aid, s in agent_scores.items() if aid != self.profile.agent_id)
        events_str = "; ".join(e.description for e in (events_today or []))
        msgs_str = "\n".join(f"[{m.sender}]: {m.content}" for m in visible_messages[-15:])
        hidden_str = "\n".join(f"[HIDDEN {m.sender}]: {m.content}" for m in hidden_messages[-10:])

        system = MANAGER_SYSTEM.format(
            name=self.profile.name, team_list=team_list, day=day,
            health=project_health.get("health", "unknown"),
            completion_pct=project_health.get("completion_pct", 0)
        )
        user_msg = f"""Day {day}/10.
Events: {events_str or 'None'}
Public messages:
{msgs_str}
Hidden intel:
{hidden_str}

Provide your management decisions as JSON."""

        self.history.append({"role": "user", "content": user_msg})
        messages = [{"role": "system", "content": system}] + self.history[-6:]

        result = chat_json(messages, model=self.model)
        self.history.append({"role": "assistant", "content": json.dumps(result)})
        return result


def env_judge(day: int, standups: dict, events: list, project_health: dict,
              team_size: int, model: str = "gpt-4o-mini") -> dict:
    """Environment LLM that judges work quality and provides hidden/visible feedback."""
    standups_str = "\n".join(f"{aid}: {json.dumps(s)}" for aid, s in standups.items())
    events_str = "; ".join(e.description for e in events)

    system = ENV_JUDGE_SYSTEM.format(
        team_size=team_size, day=day, health=project_health.get("health", "unknown")
    )
    user_msg = f"""Day {day} standups:
{standups_str}
Events: {events_str or 'None'}
Project health: {json.dumps(project_health)}

Evaluate today's progress and provide your assessment as JSON."""

    result = chat_json(
        [{"role": "system", "content": system}, {"role": "user", "content": user_msg}],
        model=model
    )
    return result
