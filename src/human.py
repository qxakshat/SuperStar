"""Human-in-the-loop adapter — lets a person play as an engineer via the dashboard."""

from dataclasses import dataclass
from typing import Optional
from agents import EngineerAgent, AgentProfile


class HumanEngineerAdapter:
    """
    Drop-in replacement for EngineerAgent that reads input from the dashboard
    instead of calling an LLM. The SprintEnv treats it identically.
    """

    def __init__(self, profile: AgentProfile):
        self.profile = profile
        self.profile.is_human = True
        self._pending_input: Optional[dict] = None
        self.history: list[dict] = []

    def set_input(self, standup: str, task_progress: int = 30,
                  blockers: str = "none", mood: str = "focused"):
        """Called by the dashboard when human submits their standup."""
        self._pending_input = {
            "standup": standup,
            "task_progress": task_progress,
            "blockers": blockers,
            "mood": mood,
        }

    def act(self, day: int, visible_messages: list, morale: int, energy: int,
            current_task: Optional[str] = None, events_today: list = None) -> Optional[dict]:
        """Returns the human's input if available, None otherwise."""
        if self._pending_input is None:
            return None
        result = self._pending_input.copy()
        self.history.append({"day": day, **result})
        self._pending_input = None
        return result

    @property
    def model(self):
        return "human"


def make_human_scenario(base_scenario, human_name: str = "You"):
    """Modify a scenario to include a human player slot."""
    import copy
    scenario = copy.deepcopy(base_scenario)

    # Replace one engineer with human
    for agent in scenario.agents:
        if agent.role == "engineer":
            agent.is_human = True
            agent.name = human_name
            break

    return scenario
