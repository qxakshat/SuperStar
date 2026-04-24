"""Random life & project events that affect the simulation."""

import random
from dataclasses import dataclass

@dataclass
class Event:
    type: str
    target_agent: str  # "all" or specific agent_id
    severity: float  # 0.0 to 1.0
    duration_days: int
    description: str
    effects: dict  # {"energy": -20, "morale": -10, "capacity": 0.5}

EVENT_TEMPLATES = [
    {"type": "sick_leave", "description": "{agent} is out sick today",
     "effects": {"capacity": 0.0, "energy": -30}, "severity": 0.8, "duration": 1, "prob": 0.08},
    {"type": "personal_issue", "description": "{agent} is dealing with a personal matter",
     "effects": {"capacity": 0.5, "morale": -20, "energy": -15}, "severity": 0.5, "duration": 2, "prob": 0.05},
    {"type": "scope_creep", "description": "New requirement added by stakeholder",
     "effects": {"new_tasks": 2}, "severity": 0.6, "duration": 0, "prob": 0.1, "target": "all"},
    {"type": "production_outage", "description": "Production incident! All hands on deck",
     "effects": {"capacity": 0.3, "energy": -25}, "severity": 0.9, "duration": 1, "prob": 0.04, "target": "all"},
    {"type": "morale_boost", "description": "Team lunch and positive feedback from leadership",
     "effects": {"morale": 20, "energy": 10}, "severity": 0.0, "duration": 1, "prob": 0.1, "target": "all"},
    {"type": "wfh_day", "description": "{agent} is working from home",
     "effects": {"capacity": 0.85, "morale": 5}, "severity": 0.1, "duration": 1, "prob": 0.15},
    {"type": "family_emergency", "description": "{agent} has a family emergency",
     "effects": {"capacity": 0.0, "morale": -30, "energy": -40}, "severity": 1.0, "duration": 2, "prob": 0.02},
    {"type": "tech_debt_explosion", "description": "Legacy code broke, needs urgent fix",
     "effects": {"new_tasks": 1, "capacity": 0.7}, "severity": 0.7, "duration": 1, "prob": 0.06, "target": "all"},
    {"type": "conference_day", "description": "{agent} is attending a tech conference",
     "effects": {"capacity": 0.0, "skill_boost": 5}, "severity": 0.2, "duration": 1, "prob": 0.03},
    {"type": "burnout_warning", "description": "{agent} showing signs of burnout",
     "effects": {"capacity": 0.6, "morale": -15, "energy": -20}, "severity": 0.7, "duration": 3, "prob": 0.04},
]

class EventGenerator:
    def __init__(self, frequency: float = 1.0, seed: int = None):
        self.frequency = frequency  # multiplier for event probabilities
        self.rng = random.Random(seed)
        self.active_events: list[tuple[Event, int]] = []  # (event, end_day)

    def generate(self, day: int, agent_ids: list[str]) -> list[Event]:
        """Generate events for a given day."""
        events = []
        # Check active multi-day events
        still_active = []
        for evt, end_day in self.active_events:
            if day <= end_day:
                events.append(evt)
                still_active.append((evt, end_day))
        self.active_events = still_active

        # Generate new events
        for template in EVENT_TEMPLATES:
            if self.rng.random() < template["prob"] * self.frequency:
                target = template.get("target", None)
                if target == "all":
                    agent = "all"
                else:
                    agent = self.rng.choice(agent_ids)
                desc = template["description"].format(agent=agent)
                evt = Event(
                    type=template["type"], target_agent=agent,
                    severity=template["severity"], duration_days=template.get("duration", 1),
                    description=desc, effects=dict(template["effects"])
                )
                events.append(evt)
                if evt.duration_days > 1:
                    self.active_events.append((evt, day + evt.duration_days - 1))
        return events

    def get_agent_capacity(self, agent_id: str, events: list[Event]) -> float:
        """Calculate effective capacity for an agent given active events."""
        capacity = 1.0
        for evt in events:
            if evt.target_agent in (agent_id, "all"):
                cap = evt.effects.get("capacity", 1.0)
                capacity = min(capacity, cap)
        return capacity

    def get_morale_impact(self, agent_id: str, events: list[Event]) -> int:
        total = 0
        for evt in events:
            if evt.target_agent in (agent_id, "all"):
                total += evt.effects.get("morale", 0)
        return total

    def get_energy_impact(self, agent_id: str, events: list[Event]) -> int:
        total = 0
        for evt in events:
            if evt.target_agent in (agent_id, "all"):
                total += evt.effects.get("energy", 0)
        return total
