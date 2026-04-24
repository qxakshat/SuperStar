<<<<<<< HEAD
# 🚀 Superstar — Sprint Simulation Open Environment

**A multi-agent LLM-powered environment that simulates the hidden human dynamics of software project sprints, with GRPO-based self-improvement.**

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 What Is This?

**Superstar** is an **OpenEnv-native** environment (built on `openenv-core`) that models a **10-day software sprint** with:
- **4 LLM-powered engineers** with unique personalities, skills, and life circumstances
- **1 LLM-powered manager** making strategic decisions with imperfect information
- **Hidden dynamics**: private assessments, interpersonal tension, burnout risks — invisible to the agents but influencing outcomes
- **Random life events**: sick leave, family emergencies, scope creep, production outages
- **A trainable environment LLM** that learns to give better suggestions via GRPO

The key insight: **what people say in standups ≠ what's actually happening**. Superstar models this gap.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│                 SprintEnv (env.py)               │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Engineers │  │ Manager  │  │ Env LLM      │  │
│  │ (frozen)  │  │ (frozen) │  │ (trainable)  │  │
│  └─────┬────┘  └─────┬────┘  └──────┬───────┘  │
│        │             │              │            │
│   ┌────▼─────────────▼──────────────▼────┐      │
│   │         Message Bus                   │      │
│   │   visible channel │ hidden channel    │      │
│   └──────────────────────────────────────┘      │
│        │                                         │
│   ┌────▼────────┐  ┌──────────┐  ┌─────────┐   │
│   │ Scoring     │  │ Events   │  │ Backlog  │   │
│   │ & Rewards   │  │ Generator│  │ Manager  │   │
│   └─────────────┘  └──────────┘  └─────────┘   │
└─────────────────────────────────────────────────┘
         │                              │
    ┌────▼────┐                   ┌─────▼─────┐
    │ GRPO    │                   │  Gradio   │
    │ Trainer │                   │ Dashboard │
    └─────────┘                   └───────────┘
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a simulation (works without API key — uses smart mock)
python main.py --run

# Launch the interactive dashboard
python main.py --dashboard

# Run GRPO training
python main.py --train --episodes 5

# Crunch mode scenario
python main.py --run --scenario crunch
```

### With OpenAI API:
```bash
export OPENAI_API_KEY="sk-..."
python main.py --dashboard --model gpt-4o
```

## 📊 Dashboard

The Gradio dashboard has 5 tabs:

| Tab | Content |
|-----|---------|
| **📊 Project Board** | Jira-like backlog, burndown chart, velocity, team messages |
| **🔒 Hidden Stats** | Agent radar, health heatmap, hidden messages, event timeline |
| **🧑‍💻 Play as Engineer** | Human-in-the-loop: join the sprint, submit daily standups |
| **🧠 GRPO Training** | Run training experiments, compare pre/post metrics |
| **📄 Submission** | Full problem statement, environment design, and evaluation details |

## 🧠 GRPO Training

The environment LLM (judge/narrator) is fine-tuned using TRL's GRPO:

1. **Collect**: Run N sprint simulations, capture env LLM decisions + outcomes
2. **Prepare**: Format as GRPO dataset with group-relative rewards
3. **Train**: Fine-tune Qwen2.5-0.5B-Instruct (or any causal LM)
4. **Evaluate**: Compare pre/post sprint outcomes

```bash
python main.py --train --episodes 5 --scenario default
```

## 📁 Project Structure

```
superstar/
├── env.py              # OpenEnv-native Environment subclass (the heart)
├── app.py              # OpenEnv create_app() + HF Spaces entry
├── openenv_client.py   # OpenEnv client + TRL rollout_func
├── agents.py           # Engineer + Manager + Env Judge LLM agents
├── llm.py              # Thin OpenAI-compatible LLM client
├── messages.py         # Dual-channel message bus (visible/hidden)
├── scoring.py          # Points, badges, burndown, rewards
├── events.py           # Random life & project events
├── scenarios.py        # Scenario configs and loaders
├── training.py         # TRL GRPO training pipeline
├── train_colab.py      # Colab-compatible training script
├── dashboard.py        # Gradio dashboard (5 tabs)
├── human.py            # Human-in-the-loop adapter
├── main.py             # CLI entry point
├── openenv.yaml        # OpenEnv manifest
├── Dockerfile          # HF Spaces container
├── configs/            # YAML scenario files
└── requirements.txt
```

## 🏆 Reward Design

**Sprint Reward** (trains the env LLM):
```
R = 0.35×completion + 0.20×velocity + 0.15×morale + 0.20×quality − blocked_penalty
```

**Agent Reward** (individual scoring):
```
R_i = 0.30×points + 0.25×quality + 0.15×collab + 0.15×morale + 0.15×energy + project_bonus
```

## 🎲 Events System

| Event | Probability | Effect |
|-------|-------------|--------|
| Sick Leave | 8% | Capacity → 0% |
| WFH Day | 15% | Capacity → 85% |
| Personal Issue | 5% | Morale -20, Capacity 50% |
| Scope Creep | 10% | +2 new tasks |
| Production Outage | 4% | All hands, Capacity 30% |
| Morale Boost | 10% | Morale +20, Energy +10 |
| Family Emergency | 2% | Capacity 0%, 2-day |
| Burnout Warning | 4% | Capacity 60%, 3-day |

---

*Built for the OpenEnv Hackathon Apr '26*
=======
---
title: SuperStar
emoji: 🏃
colorFrom: red
colorTo: red
sdk: gradio
sdk_version: 6.13.0
app_file: app.py
pinned: false
short_description: Tech team gamification in a multi agent setting
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 3c834e9 (initial commit)
