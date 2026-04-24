"""SprintEnv — OpenEnv-native environment for 10-day sprint simulation.

Uses openenv-core's Environment, Observation, Action, State base classes
instead of Gymnasium. This is the single source of truth for the environment.
"""

import copy
import json
import os
import sys
import random
from typing import Optional, Any, Dict, List

# Ensure src/ is importable for bare module imports
_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src")
_src_dir = os.path.normpath(_src_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from pydantic import Field
from openenv.core.env_server import Environment, Observation, Action, State
from openenv.core.env_server.interfaces import EnvironmentMetadata

from agents import EngineerAgent, ManagerAgent, AgentProfile, env_judge
from messages import MessageBus
from events import EventGenerator, Event
from scoring import (
    TaskItem, AgentScore, update_badges, compute_burndown,
    compute_project_health, compute_sprint_reward, compute_agent_reward, compute_velocity
)


# ─── OpenEnv Pydantic Models ────────────────────────────────────────────────

class SprintAction(Action):
    """Action the LLM agent can take in the sprint environment."""
    action_type: str = Field(default="noop", description="standup | manager_feedback | env_suggestion | noop")
    agent_id: str = Field(default="", description="Which agent is acting")
    content: str = Field(default="", description="Standup text, feedback text, etc.")
    task_progress: int = Field(default=0, ge=0, le=100, description="Task progress percentage")


class SprintObservation(Observation):
    """What the agent observes after each step — rich text prompt + structured data."""
    # Core fields (done + reward inherited from Observation)
    day: int = Field(default=0, description="Current sprint day (0-10)")
    sprint_days: int = Field(default=10, description="Total sprint days")
    project_name: str = Field(default="", description="Sprint/project name")

    # Jira-like board
    backlog_summary: str = Field(default="", description="Text summary of the backlog")
    completion_pct: float = Field(default=0.0, description="Backlog completion percentage")
    velocity: float = Field(default=0.0, description="Story points completed per day")
    health: str = Field(default="unknown", description="green | yellow | red")

    # Messages visible to the acting agent
    visible_messages: List[Dict[str, Any]] = Field(default_factory=list)
    current_task: str = Field(default="", description="Currently assigned task")

    # Agent stats
    morale: int = Field(default=70, ge=0, le=100)
    energy: int = Field(default=80, ge=0, le=100)
    score: float = Field(default=0.0, ge=0)

    # Events and team context
    events_today: List[str] = Field(default_factory=list)
    team_standups: List[Dict[str, Any]] = Field(default_factory=list)
    manager_feedback: str = Field(default="")
    env_suggestions: List[str] = Field(default_factory=list)

    def to_prompt(self) -> str:
        """Convert observation to a text prompt for the LLM."""
        lines = [
            f"# Sprint: {self.project_name} — Day {self.day}/{self.sprint_days}",
            f"## Project Status: {self.health.upper()} | {self.completion_pct:.1f}% complete | Velocity: {self.velocity:.1f} pts/day",
            f"\n## Your Status: Morale {self.morale}/100 | Energy {self.energy}/100 | Score {self.score:.0f}",
        ]
        if self.current_task:
            lines.append(f"**Current Task:** {self.current_task}")
        if self.events_today:
            lines.append("\n## Today's Events:")
            for e in self.events_today:
                lines.append(f"- ⚡ {e}")
        if self.team_standups:
            lines.append("\n## Team Standups:")
            for s in self.team_standups:
                lines.append(f"- **{s.get('agent', '?')}**: {s.get('standup', 'No update')}")
        if self.manager_feedback:
            lines.append(f"\n## Manager Feedback: {self.manager_feedback}")
        if self.env_suggestions:
            lines.append("\n## Suggestions:")
            for s in self.env_suggestions:
                lines.append(f"- 💡 {s}")
        lines.append(f"\n## Backlog:\n{self.backlog_summary}")
        lines.append("\n---\nProvide your standup update as JSON:")
        lines.append('{"standup": "...", "task_progress": <0-100>, "blockers": "...", "mood": "..."}')
        return "\n".join(lines)


class SprintState(State):
    """Full environment state (visible + hidden) for training/evaluation."""
    # step_count + episode_id inherited from State
    day: int = 0
    is_done: bool = False
    scenario_name: str = ""
    project_health: Dict[str, Any] = Field(default_factory=dict)
    agent_scores: Dict[str, Any] = Field(default_factory=dict)
    backlog: List[Dict[str, Any]] = Field(default_factory=list)
    visible_messages: List[Dict[str, Any]] = Field(default_factory=list)
    hidden_messages: List[Dict[str, Any]] = Field(default_factory=list)
    events: List[List[Dict[str, Any]]] = Field(default_factory=list)
    burndown: List[Dict[str, Any]] = Field(default_factory=list)
    rewards: List[float] = Field(default_factory=list)
    total_reward: float = 0.0


# ─── The Environment ─────────────────────────────────────────────────────────

class SprintEnv(Environment[SprintAction, SprintObservation, SprintState]):
    """
    OpenEnv-native sprint simulation environment.

    Models a 10-day software sprint with:
    - 4 LLM-powered engineers with unique personalities
    - 1 LLM-powered manager with imperfect information
    - Dual-channel messaging (visible + hidden)
    - Stochastic life events (sick leave, burnout, scope creep, etc.)
    - Composite reward balancing delivery, velocity, morale, quality
    """
    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self, scenario=None, model: str = "gpt-4o-mini",
                 env_model: str = "gpt-4o-mini", seed: int = 42, **kwargs):
        super().__init__(**kwargs)
        from scenarios import get_default_scenario
        self.scenario = scenario or get_default_scenario()
        self.model = model
        self.env_model = env_model
        self._seed = seed
        # Internal state — initialized in reset()
        self.day = 0
        self._done = False
        self.backlog: list[TaskItem] = []
        self.message_bus = MessageBus()
        self.event_gen = EventGenerator(seed=seed)
        self.engineers: dict[str, EngineerAgent] = {}
        self.manager: Optional[ManagerAgent] = None
        self.agent_scores: dict[str, AgentScore] = {}
        self.rng = random.Random(seed)
        self._total_reward = 0.0
        self.history = {"daily_health": [], "daily_events": [], "daily_standups": [],
                        "burndown": [], "env_judgments": [], "rewards": []}

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None,
              **kwargs) -> SprintObservation:
        """Reset environment to start of sprint. Returns initial observation."""
        self._reset_rubric()
        self.rng = random.Random(seed or self._seed)
        self.day = 0
        self._done = False
        self._total_reward = 0.0

        # Deep copy backlog
        self.backlog = [copy.deepcopy(t) for t in self.scenario.backlog]
        self.message_bus = MessageBus()
        self.event_gen = EventGenerator(frequency=self.scenario.event_frequency, seed=seed or self._seed)

        # Initialize agents
        self.engineers = {}
        self.manager = None
        self.agent_scores = {}

        for profile in self.scenario.agents:
            self.agent_scores[profile.agent_id] = AgentScore(
                agent_id=profile.agent_id, morale=70, energy=80
            )
            if profile.role == "engineer":
                self.engineers[profile.agent_id] = EngineerAgent(profile, model=self.model)
            elif profile.role == "manager":
                self.manager = ManagerAgent(profile, model=self.model)

        # Assign initial tasks
        self._assign_initial_tasks()

        # Reset history
        self.history = {"daily_health": [], "daily_events": [], "daily_standups": [],
                        "burndown": [], "env_judgments": [], "rewards": []}

        # Post sprint kickoff message
        self.message_bus.post(0, "system",
            f"Sprint '{self.scenario.name}' started! "
            f"{len(self.backlog)} tasks, {sum(t.story_points for t in self.backlog)} story points. "
            f"Team: {', '.join(p.name for p in self.scenario.agents)}", "system")

        return self._make_observation()

    def step(self, action: SprintAction, timeout_s: Optional[float] = None,
             **kwargs) -> SprintObservation:
        """Advance one day in the sprint. Returns observation with done + reward."""
        if self._done:
            obs = self._make_observation()
            obs.done = True
            obs.reward = 0.0
            return obs

        self.day += 1

        # Handle incoming human/external action
        if action.action_type == "standup" and action.agent_id:
            self.add_human_standup(action.agent_id, action.content, action.task_progress)

        # 1. Generate events
        agent_ids = list(self.engineers.keys())
        events = self.event_gen.generate(self.day, agent_ids)
        self.history["daily_events"].append(events)

        for evt in events:
            self.message_bus.post(self.day, "system", evt.description, "system",
                                  hidden=(evt.severity > 0.5))
            if evt.target_agent == "all":
                for aid in agent_ids:
                    self._apply_event_effects(aid, evt)
            elif evt.target_agent in self.agent_scores:
                self._apply_event_effects(evt.target_agent, evt)

        # 2. Engineers produce standups
        standups = {}
        for aid, eng in self.engineers.items():
            if eng.profile.is_human:
                continue
            score = self.agent_scores[aid]
            capacity = self.event_gen.get_agent_capacity(aid, events)
            visible_msgs = self.message_bus.get_visible(aid, self.day - 1)
            current_task = self._get_assigned_task(aid)
            agent_events = [e for e in events if e.target_agent in (aid, "all")]

            result = eng.act(
                day=self.day, visible_messages=visible_msgs,
                morale=score.morale, energy=score.energy,
                current_task=current_task, events_today=agent_events
            )
            if result:
                standups[aid] = result
                standup_text = result.get("standup", "No update")
                self.message_bus.post(self.day, aid, standup_text, "standup")
                progress = result.get("task_progress", 0) * capacity
                self._update_task_progress(aid, progress)
                score.energy = max(10, min(100, score.energy - self.rng.randint(5, 15) + (5 if capacity < 0.5 else 0)))

        self.history["daily_standups"].append(standups)

        # 3. Environment judges the day
        project_health = compute_project_health(self.backlog, self.day, self.scenario.sprint_days)
        judgment = env_judge(
            self.day, standups, events, project_health,
            len(self.engineers), model=self.env_model
        )
        self.history["env_judgments"].append(judgment)

        # Apply env adjustments
        for aid, adj in judgment.get("agent_adjustments", {}).items():
            if aid in self.agent_scores:
                s = self.agent_scores[aid]
                s.morale = max(0, min(100, s.morale + adj.get("morale_delta", 0)))
                s.energy = max(0, min(100, s.energy + adj.get("energy_delta", 0)))
                s.collaboration_score = max(0, min(1, s.collaboration_score + adj.get("collaboration", 0)))

        suggestion = judgment.get("suggestion", "")
        if suggestion:
            self.message_bus.post(self.day, "environment", suggestion, "env_suggestion")
        hidden = judgment.get("hidden_assessment", "")
        if hidden:
            self.message_bus.post(self.day, "environment", hidden, "env_suggestion", hidden=True)

        # 4. Manager responds
        if self.manager:
            hidden_msgs = self.message_bus.get_hidden(self.day)
            visible_msgs = self.message_bus.get_visible(self.manager.profile.agent_id)
            mgr_result = self.manager.act(
                self.day, visible_msgs, hidden_msgs,
                self.agent_scores, project_health, events
            )
            if mgr_result:
                vis = mgr_result.get("visible_feedback", "")
                hid = mgr_result.get("hidden_feedback", "")
                if vis:
                    self.message_bus.post(self.day, self.manager.profile.agent_id, vis, "review")
                if hid:
                    self.message_bus.post(self.day, self.manager.profile.agent_id, hid, "dm", hidden=True)

        # 5. Compute rewards
        project_health = compute_project_health(self.backlog, self.day, self.scenario.sprint_days)
        self.history["daily_health"].append(project_health)
        self.history["burndown"] = compute_burndown(self.backlog, self.scenario.sprint_days)

        sprint_reward = compute_sprint_reward(project_health, self.agent_scores)
        self.history["rewards"].append(sprint_reward)
        self._total_reward += sprint_reward

        # Update agent scores
        for aid, score in self.agent_scores.items():
            quality = judgment.get("work_quality", 0.7)
            score.quality_avg = (score.quality_avg * max(1, self.day - 1) + quality) / self.day
            update_badges(score)
            score.daily_points.append(int(sprint_reward * 100))

        # 6. Check termination
        self._done = self.day >= self.scenario.sprint_days

        # Build observation
        obs = self._make_observation()
        obs.done = self._done
        obs.reward = sprint_reward
        return obs

    @property
    def state(self) -> SprintState:
        """Full state including hidden information."""
        return SprintState(
            step_count=self.day,
            day=self.day,
            is_done=self._done,
            scenario_name=self.scenario.name,
            project_health=self.history["daily_health"][-1] if self.history["daily_health"] else {},
            agent_scores={
                aid: {"morale": s.morale, "energy": s.energy, "score": s.total_score,
                      "tasks_done": s.tasks_completed, "points": s.story_points_done,
                      "quality": round(s.quality_avg, 2), "badges": s.badges,
                      "streak": s.streak, "velocity": round(s.velocity, 2)}
                for aid, s in self.agent_scores.items()
            },
            backlog=[
                {"id": t.id, "title": t.title, "points": t.story_points,
                 "status": t.status, "assigned_to": t.assigned_to,
                 "quality": t.quality_score}
                for t in self.backlog
            ],
            visible_messages=[
                {"day": m.day, "sender": m.sender, "content": m.content, "channel": m.channel}
                for m in self.message_bus.messages if not m.hidden
            ],
            hidden_messages=[
                {"day": m.day, "sender": m.sender, "content": m.content, "channel": m.channel}
                for m in self.message_bus.messages if m.hidden
            ],
            burndown=self.history.get("burndown", []),
            rewards=self.history.get("rewards", []),
            total_reward=self._total_reward,
            events=[
                [{"type": e.type, "target": e.target_agent, "desc": e.description}
                 for e in day_events]
                for day_events in self.history.get("daily_events", [])
            ],
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="superstar-sprint",
            description="Multi-agent sprint simulation with hidden dynamics, dual-channel messaging, and GRPO-trainable environment LLM",
            version="1.0.0",
        )

    # ─── Public helpers ──────────────────────────────────────────────────

    def add_human_standup(self, agent_id: str, standup_text: str, task_progress: int = 20) -> dict:
        """Add a human player's standup (called from dashboard)."""
        if agent_id not in self.engineers:
            return {"error": "Agent not found"}
        result = {"standup": standup_text, "task_progress": task_progress, "blockers": "none", "mood": "engaged"}
        self.message_bus.post(self.day, agent_id, standup_text, "standup")
        self._update_task_progress(agent_id, task_progress)
        if self.day > 0 and self.day <= len(self.history["daily_standups"]):
            self.history["daily_standups"][-1][agent_id] = result
        return result

    def get_full_state(self) -> dict:
        """Get state as a plain dict (for dashboard/training compatibility)."""
        s = self.state
        return s.model_dump()

    # ─── Internal mechanics ──────────────────────────────────────────────

    def _make_observation(self) -> SprintObservation:
        """Build observation from current state."""
        health = self.history["daily_health"][-1] if self.history["daily_health"] else {}

        # Backlog summary
        backlog_lines = []
        for t in self.backlog:
            icon = {"todo": "📋", "in_progress": "🔄", "done": "✅", "blocked": "🚫"}.get(t.status, "❓")
            backlog_lines.append(f"{icon} {t.id}: {t.title} ({t.story_points}pts) [{t.status}]")

        # Events today
        events_today = []
        if self.history["daily_events"]:
            events_today = [e.description for e in self.history["daily_events"][-1]]

        # Visible messages (latest day)
        visible = [{"day": m.day, "sender": m.sender, "content": m.content, "channel": m.channel}
                    for m in self.message_bus.messages if not m.hidden and m.day == self.day]

        # Standups
        standups = []
        if self.history["daily_standups"]:
            last = self.history["daily_standups"][-1]
            for aid, s in last.items():
                standups.append({"agent": aid, "standup": s.get("standup", "No update")})

        # Manager feedback
        mgr_feedback = ""
        mgr_msgs = [m for m in visible if m.get("channel") == "review"]
        if mgr_msgs:
            mgr_feedback = mgr_msgs[-1]["content"]

        suggestions = [m["content"] for m in visible if m.get("channel") == "env_suggestion"]

        return SprintObservation(
            day=self.day,
            sprint_days=self.scenario.sprint_days,
            project_name=self.scenario.name,
            backlog_summary="\n".join(backlog_lines),
            completion_pct=health.get("completion_pct", 0),
            velocity=health.get("velocity", 0),
            health=health.get("health", "unknown"),
            visible_messages=visible,
            events_today=events_today,
            team_standups=standups,
            manager_feedback=mgr_feedback,
            env_suggestions=suggestions,
            done=self._done,
            reward=self.history["rewards"][-1] if self.history["rewards"] else None,
        )

    def _assign_initial_tasks(self):
        available = [t for t in self.backlog if not t.dependencies]
        engineers = list(self.engineers.keys())
        for i, task in enumerate(available[:len(engineers)]):
            aid = engineers[i % len(engineers)]
            task.status = "in_progress"
            task.assigned_to = aid
            task.day_started = 0

    def _get_assigned_task(self, agent_id: str) -> Optional[str]:
        for t in self.backlog:
            if t.assigned_to == agent_id and t.status == "in_progress":
                return f"{t.id}: {t.title} ({t.story_points}pts)"
        return None

    def _update_task_progress(self, agent_id: str, progress: float):
        for task in self.backlog:
            if task.assigned_to == agent_id and task.status == "in_progress":
                daily_capacity = max(progress, 18) / 100
                effort_needed = task.story_points / 7.0
                increment = daily_capacity / max(effort_needed, 0.3)
                task.quality_score = min(1.0, task.quality_score + increment)

                threshold = 0.55
                if task.quality_score >= threshold:
                    task.status = "done"
                    task.day_completed = self.day
                    score = self.agent_scores[agent_id]
                    score.tasks_completed += 1
                    score.story_points_done += task.story_points
                    score.streak += 1
                    score.velocity = score.story_points_done / max(self.day, 1)
                    self._assign_next_task(agent_id)
                break

    def _assign_next_task(self, agent_id: str):
        done_ids = {t.id for t in self.backlog if t.status == "done"}
        for task in self.backlog:
            if task.status == "todo":
                deps_met = all(d in done_ids for d in task.dependencies)
                if deps_met:
                    task.status = "in_progress"
                    task.assigned_to = agent_id
                    task.day_started = self.day
                    self.message_bus.post(self.day, "system",
                                          f"{agent_id} started {task.id}: {task.title}", "system")
                    return

    def _apply_event_effects(self, agent_id: str, event: Event):
        if agent_id not in self.agent_scores:
            return
        score = self.agent_scores[agent_id]
        score.morale = max(0, min(100, score.morale + event.effects.get("morale", 0)))
        score.energy = max(0, min(100, score.energy + event.effects.get("energy", 0)))
        if "skill_boost" in event.effects:
            for p in self.scenario.agents:
                if p.agent_id == agent_id:
                    p.skill_level = min(1.0, p.skill_level + event.effects["skill_boost"] / 100)


# ─── Convenience runner ──────────────────────────────────────────────────────

def run_simulation(scenario=None, model="gpt-4o-mini", verbose=True) -> dict:
    """Run a complete 10-day sprint simulation."""
    env = SprintEnv(scenario=scenario, model=model)
    obs = env.reset()

    if verbose:
        print(f"\n{'='*60}")
        print(f"🚀 Sprint: {env.scenario.name}")
        print(f"📋 {len(env.backlog)} tasks | {sum(t.story_points for t in env.backlog)} story points")
        print(f"👥 Team: {', '.join(p.name for p in env.scenario.agents)}")
        print(f"{'='*60}\n")

    total_reward = 0
    for day in range(env.scenario.sprint_days):
        obs = env.step(SprintAction())
        reward = obs.reward or 0
        total_reward += reward

        if verbose:
            print(f"📅 Day {env.day}: completion={obs.completion_pct:.1f}% | "
                  f"health={obs.health} | reward={reward:.3f}")
            if obs.events_today:
                for e in obs.events_today:
                    print(f"   ⚡ Event: {e}")

        if obs.done:
            break

    state = env.get_full_state()
    if verbose:
        print(f"\n{'='*60}")
        print(f"🏁 Sprint Complete! Total reward: {total_reward:.3f}")
        for aid, data in state.get("agent_scores", {}).items():
            print(f"   {aid}: score={data['score']:.0f} | tasks={data['tasks_done']} | "
                  f"morale={data['morale']} | badges={data['badges']}")
        print(f"{'='*60}\n")

    return {"env": env, "total_reward": total_reward, "final_state": state}


if __name__ == "__main__":
    result = run_simulation()
