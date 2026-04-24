"""Scenario loader and configuration."""

import yaml
import json
import os
from dataclasses import dataclass, field
from typing import Optional
from agents import AgentProfile
from scoring import TaskItem

@dataclass
class ScenarioConfig:
    name: str
    description: str
    sprint_days: int = 10
    agents: list[AgentProfile] = field(default_factory=list)
    backlog: list[TaskItem] = field(default_factory=list)
    event_frequency: float = 1.0
    model: str = "gpt-4o-mini"
    seed: int = 42

def load_scenario(path: str) -> ScenarioConfig:
    """Load scenario from YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    agents = []
    for a in data.get("agents", []):
        agents.append(AgentProfile(**a))

    backlog = []
    for t in data.get("backlog", []):
        backlog.append(TaskItem(**t))

    return ScenarioConfig(
        name=data.get("name", "Default Sprint"),
        description=data.get("description", ""),
        sprint_days=data.get("sprint_days", 10),
        agents=agents, backlog=backlog,
        event_frequency=data.get("event_frequency", 1.0),
        model=data.get("model", "gpt-4o-mini"),
        seed=data.get("seed", 42),
    )

def get_default_scenario() -> ScenarioConfig:
    """Built-in default scenario for quick testing."""
    agents = [
        AgentProfile("manager_1", "Sarah Chen", "manager", skill_level=0.85,
                      personality="strategic", reliability=0.9, specialization="leadership"),
        AgentProfile("eng_1", "Alex Rivera", "engineer", skill_level=0.8,
                      personality="aggressive", reliability=0.85, specialization="backend",
                      wfh_tendency=0.3),
        AgentProfile("eng_2", "Jamie Park", "engineer", skill_level=0.7,
                      personality="cautious", reliability=0.9, specialization="frontend",
                      wfh_tendency=0.1),
        AgentProfile("eng_3", "Morgan Taylor", "engineer", skill_level=0.6,
                      personality="creative", reliability=0.7, specialization="fullstack",
                      wfh_tendency=0.4),
        AgentProfile("eng_4", "Casey Nguyen", "engineer", skill_level=0.9,
                      personality="balanced", reliability=0.95, specialization="devops",
                      wfh_tendency=0.2),
    ]
    backlog = [
        TaskItem("T1", "Set up CI/CD pipeline", 5, dependencies=[]),
        TaskItem("T2", "Design database schema", 8, dependencies=[]),
        TaskItem("T3", "Implement user auth API", 8, dependencies=["T2"]),
        TaskItem("T4", "Build login/signup UI", 5, dependencies=["T3"]),
        TaskItem("T5", "Create product listing API", 5, dependencies=["T2"]),
        TaskItem("T6", "Build product catalog page", 8, dependencies=["T5"]),
        TaskItem("T7", "Implement search functionality", 5, dependencies=["T5"]),
        TaskItem("T8", "Shopping cart backend", 8, dependencies=["T3", "T5"]),
        TaskItem("T9", "Shopping cart UI", 5, dependencies=["T8", "T4"]),
        TaskItem("T10", "Payment integration", 13, dependencies=["T8"]),
        TaskItem("T11", "Write unit tests", 5, dependencies=[]),
        TaskItem("T12", "Performance optimization", 3, dependencies=["T6"]),
        TaskItem("T13", "Deploy to staging", 3, dependencies=["T1"]),
        TaskItem("T14", "Integration testing", 5, dependencies=["T13"]),
        TaskItem("T15", "Documentation", 3, dependencies=[]),
    ]
    return ScenarioConfig(
        name="E-Commerce Platform Sprint",
        description="Build core features of an e-commerce platform in a 10-day sprint",
        agents=agents, backlog=backlog, event_frequency=1.0, seed=42
    )

def get_crunch_scenario() -> ScenarioConfig:
    """High-pressure crunch scenario."""
    agents = [
        AgentProfile("manager_1", "Director Kim", "manager", skill_level=0.75,
                      personality="aggressive", reliability=0.8, specialization="leadership"),
        AgentProfile("eng_1", "Dev Alpha", "engineer", skill_level=0.85,
                      personality="aggressive", reliability=0.8, specialization="backend"),
        AgentProfile("eng_2", "Dev Beta", "engineer", skill_level=0.65,
                      personality="cautious", reliability=0.6, specialization="frontend",
                      wfh_tendency=0.5),
        AgentProfile("eng_3", "Dev Gamma", "engineer", skill_level=0.5,
                      personality="creative", reliability=0.7, specialization="fullstack"),
    ]
    backlog = [
        TaskItem("CT1", "Emergency hotfix deploy", 3),
        TaskItem("CT2", "Rewrite legacy payment module", 13),
        TaskItem("CT3", "Migrate database to new schema", 8, dependencies=["CT1"]),
        TaskItem("CT4", "Build admin dashboard", 8),
        TaskItem("CT5", "Security audit and fixes", 8, dependencies=["CT2"]),
        TaskItem("CT6", "Load testing", 5, dependencies=["CT3"]),
        TaskItem("CT7", "Client demo preparation", 5, dependencies=["CT4"]),
        TaskItem("CT8", "Bug bash and stabilization", 8, dependencies=["CT5", "CT6"]),
    ]
    return ScenarioConfig(
        name="Crunch Mode Sprint",
        description="High-pressure sprint with tight deadlines and technical debt",
        agents=agents, backlog=backlog, event_frequency=1.5, seed=123
    )

BUILT_IN_SCENARIOS = {
    "default": get_default_scenario,
    "crunch": get_crunch_scenario,
}
