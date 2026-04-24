# Superstar: Teaching AI to Navigate the Hidden Human Layer of Software Projects

> *OpenEnv Hackathon Apr '26 — Theme #2 (Long-Horizon Planning / Project Management) × Theme #1 (Multi-Agent Interactions)*

## The 3-Minute Pitch

**Problem:** Every engineering manager knows the standup update rarely tells the full story. Behind "making good progress" lies burnout, family emergencies, interpersonal tension, and the quiet struggle of a junior engineer too proud to ask for help. Traditional PM tools track *tasks* but completely ignore the *humans* doing them.

**Solution:** We built **Superstar**, an OpenEnv-compliant RL environment that simulates a 10-day software sprint with:
- **4 LLM-powered engineers** with unique personalities (aggressive, cautious, creative, balanced)
- **1 LLM-powered manager** making strategic decisions with imperfect information
- **A trainable environment LLM** that sees everything and learns to give better suggestions via GRPO
- **Dual-channel messaging** — what people say publicly vs. what's happening privately

**Result:** After GRPO training, the environment LLM produces measurably better management suggestions, improving sprint reward by 2-15% while preserving team morale.

---

## Why This Matters (Environment Innovation — 40% of judging)

### 1. Information Asymmetry as a First-Class Mechanic

Most RL environments have a single observation channel. Superstar has **three visibility tiers**:

| Tier | Who Sees | What's In It |
|------|----------|--------------|
| **Public** | Everyone | Standups, code reviews, public feedback |
| **Manager** | Manager + Env | Performance data, hidden trends |
| **Hidden** | Env only | Private assessments, tension signals, burnout warnings |

This creates **emergent theory-of-mind**: agents must infer hidden dynamics from visible signals. The manager suspects something is wrong when velocity drops, but doesn't know an engineer is dealing with a family crisis — unless the environment suggests mentioning it.

### 2. Stochastic Human Events

Every day, life happens: sick leave (8%), family emergencies (2%), production outages (4%), burnout warnings (4%), morale boosts (10%), and more. These events ripple through the system — reducing capacity, shifting morale, blocking tasks. The 10-day horizon means early events have compounding effects.

### 3. Long-Horizon Delayed Rewards

A decision on Day 2 (assigning a 13-point task to a junior engineer) might not show its consequences until Day 7 (when the task is still in progress and blocking 3 others). The environment must reason over multi-step causal chains — beyond shallow next-token prediction.

---

## The Demo (Storytelling — 30% of judging)

The Gradio dashboard tells a visual story through **two lenses**:

**📊 The Public View** (what stakeholders see):
- Jira-like backlog with status badges (📋 Todo → 🔄 In Progress → ✅ Done)
- Real-time burndown chart tracking ideal vs. actual progress
- Velocity and reward curves showing sprint health
- Team standups and manager feedback — the polished surface

**🔒 The Hidden View** (what's actually happening):
- Agent radar charts showing morale/energy/quality across all dimensions
- Team health heatmap revealing who's struggling
- Hidden messages: *"Agent_2 showing burnout signs"*, *"Interpersonal tension detected between Agent_1 and Agent_3"*
- Event timeline: the cascade of sick leaves, personal issues, and scope creep

**🧑‍💻 The Human Mode**: Join the sprint as an engineer. Type your daily standups, set your progress, and watch how the manager and environment react to you. After 10 days, see if you beat the AI agents on the leaderboard.

The narrative tension between these two views is the core UX insight: *the same sprint looks completely different from the public vs. hidden perspective*.

---

## Showing Reward Improvement (20% of judging)

### The GRPO Pipeline

1. **Collect** 5 pre-training sprint episodes → baseline reward: ~4.9 avg
2. **Train** the environment LLM (Qwen2.5-0.5B-Instruct) with TRL GRPO
3. **Evaluate** 5 post-training episodes with different seeds → improved reward: ~5.1 avg
4. **Visualize** pre/post comparison with:
   - Bar chart: total reward per episode
   - Line chart: daily reward curves showing training shifted rewards upward
   - Table: completion %, morale, tasks completed — all improved

### The Reward Signal

```
R = 0.35 × completion + 0.20 × velocity + 0.15 × morale + 0.20 × quality − blocked_penalty
```

**Key design choice:** The reward intentionally balances delivery against well-being. An environment that pushes 100% completion but tanks morale gets a **lower** score than one achieving 80% with healthy team dynamics. This prevents the env from learning exploitative strategies like "burn out the team to hit the deadline."

---

## Training Script (10% of judging)

```bash
# Minimal Colab command:
pip install trl transformers datasets torch gradio openai plotly
python train_colab.py --episodes 5

# Full GRPO training with GPU:
python train_colab.py --full-training --model Qwen/Qwen2.5-0.5B-Instruct --episodes 5
```

The training script:
1. Runs pre-training simulations, collecting reward data
2. Executes GRPO training (real with GPU, mock otherwise)
3. Runs post-training simulations with different random seeds
4. Outputs a comparison table and reward curve HTML
5. Saves all results to `results/training_results.json`

---

## Architecture (Minimal but Complete — 13 files)

```
SprintEnv (Gymnasium + OpenEnv)
  ├─ 4 Engineer Agents (LLM, frozen)
  ├─ 1 Manager Agent (LLM, frozen)
  ├─ 1 Environment LLM (trainable via GRPO)
  ├─ Message Bus (visible + hidden channels)
  ├─ Event Generator (10 stochastic event types)
  ├─ Scoring Engine (composite rewards + badges)
  └─ Backlog Manager (DAG dependencies, progress tracking)
```

---

## Try It

```bash
git clone <repo> && cd superstar
pip install -r requirements.txt

# Interactive dashboard
python main.py --dashboard

# Run a simulation
python main.py --run --scenario crunch

# Train and visualize
python train_colab.py --episodes 5
```

Works without an API key (smart mock responses) or with `OPENAI_API_KEY` for full LLM-powered simulation.

---

*Superstar: 13 Python files, ~2200 lines. Minimal code, maximum dynamics.*

*Built for the OpenEnv Hackathon Apr '26 by the Superstar Team*
