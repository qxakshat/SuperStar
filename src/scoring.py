"""Scoring, gamification, burndown charts, and reward computation."""

from dataclasses import dataclass, field

@dataclass
class TaskItem:
    id: str
    title: str
    story_points: int
    status: str = "todo"  # todo, in_progress, done, blocked
    assigned_to: str = ""
    day_started: int = -1
    day_completed: int = -1
    quality_score: float = 0.0
    dependencies: list[str] = field(default_factory=list)

@dataclass
class AgentScore:
    agent_id: str
    tasks_completed: int = 0
    story_points_done: int = 0
    quality_avg: float = 0.0
    velocity: float = 0.0
    collaboration_score: float = 0.0
    morale: int = 70
    energy: int = 80
    badges: list[str] = field(default_factory=list)
    daily_points: list[int] = field(default_factory=list)
    streak: int = 0  # consecutive productive days

    @property
    def total_score(self) -> float:
        return (self.story_points_done * 10 + self.quality_avg * 20 +
                self.collaboration_score * 15 + len(self.badges) * 5 +
                self.streak * 3)

BADGES = {
    "on_fire": {"name": "On Fire 🔥", "desc": "3+ consecutive productive days", "check": lambda s: s.streak >= 3},
    "quality_king": {"name": "Quality King 👑", "desc": "Average quality > 0.9", "check": lambda s: s.quality_avg > 0.9},
    "speed_demon": {"name": "Speed Demon ⚡", "desc": "Completed 3+ tasks", "check": lambda s: s.tasks_completed >= 3},
    "team_player": {"name": "Team Player 🤝", "desc": "High collaboration score", "check": lambda s: s.collaboration_score > 0.7},
    "resilient": {"name": "Resilient 💪", "desc": "Productive despite events", "check": lambda s: s.morale < 50 and s.velocity > 0},
    "marathon": {"name": "Marathon Runner 🏃", "desc": "Completed work every day", "check": lambda s: len(s.daily_points) >= 10 and all(p > 0 for p in s.daily_points)},
}

def update_badges(score: AgentScore) -> list[str]:
    new_badges = []
    for badge_id, badge in BADGES.items():
        if badge["name"] not in score.badges and badge["check"](score):
            score.badges.append(badge["name"])
            new_badges.append(badge["name"])
    return new_badges

def compute_burndown(backlog: list[TaskItem], total_days: int = 10) -> list[dict]:
    """Compute ideal vs actual burndown data."""
    total_points = sum(t.story_points for t in backlog)
    ideal_per_day = total_points / total_days
    burndown = []
    for day in range(total_days + 1):
        done_points = sum(t.story_points for t in backlog if t.status == "done" and t.day_completed <= day)
        remaining = total_points - done_points
        burndown.append({
            "day": day,
            "ideal": max(0, total_points - ideal_per_day * day),
            "actual": remaining
        })
    return burndown

def compute_velocity(backlog: list[TaskItem], day: int) -> float:
    """Story points completed per day so far."""
    if day == 0:
        return 0.0
    done = sum(t.story_points for t in backlog if t.status == "done" and t.day_completed <= day)
    return done / day

def compute_project_health(backlog: list[TaskItem], day: int, total_days: int = 10) -> dict:
    """Overall project health metrics."""
    total_points = sum(t.story_points for t in backlog)
    done_points = sum(t.story_points for t in backlog if t.status == "done")
    in_progress = sum(1 for t in backlog if t.status == "in_progress")
    blocked = sum(1 for t in backlog if t.status == "blocked")
    expected_progress = (day / total_days) * total_points if total_days > 0 else 0
    actual_progress = done_points
    health = "green" if actual_progress >= expected_progress * 0.9 else "yellow" if actual_progress >= expected_progress * 0.7 else "red"
    return {
        "total_points": total_points, "done_points": done_points,
        "completion_pct": round(done_points / total_points * 100, 1) if total_points > 0 else 0,
        "in_progress_count": in_progress, "blocked_count": blocked,
        "velocity": compute_velocity(backlog, day),
        "health": health, "day": day, "total_days": total_days,
        "on_track": actual_progress >= expected_progress * 0.8,
    }

def compute_sprint_reward(project_health: dict, agent_scores: dict[str, AgentScore]) -> float:
    """Compute overall sprint reward for GRPO training."""
    completion = project_health["completion_pct"] / 100.0
    velocity_bonus = min(project_health["velocity"] / 5.0, 1.0)
    avg_morale = sum(s.morale for s in agent_scores.values()) / max(len(agent_scores), 1) / 100.0
    avg_quality = sum(s.quality_avg for s in agent_scores.values()) / max(len(agent_scores), 1)
    blocked_penalty = project_health["blocked_count"] * 0.05
    reward = (
        0.35 * completion +
        0.20 * velocity_bonus +
        0.15 * avg_morale +
        0.20 * avg_quality -
        blocked_penalty
    )
    return round(max(0.0, min(1.0, reward)), 4)

def compute_agent_reward(score: AgentScore, project_health: dict) -> float:
    """Individual agent reward."""
    individual = (score.story_points_done * 0.3 + score.quality_avg * 0.25 +
                  score.collaboration_score * 0.15 + (score.morale / 100) * 0.15 +
                  (score.energy / 100) * 0.15)
    project_bonus = 0.2 if project_health["on_track"] else -0.1
    return round(max(0.0, min(1.0, individual + project_bonus)), 4)
