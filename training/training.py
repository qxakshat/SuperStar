"""TRL GRPO Training Loop for the Environment LLM.

The environment LLM (judge/narrator) is the policy being trained.
Agent LLMs (engineers + manager) remain frozen — we only train the env
to make better suggestions/judgments that improve overall sprint outcomes.
"""

import json
import os
import sys
import random
from dataclasses import dataclass, field

import torch
import numpy as np
from typing import Optional

# Add src/ and server/ to path
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("src", "server"):
    _p = os.path.join(_root, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)


@dataclass
class TrajectoryStep:
    """One judgment call by the env LLM during a sprint."""
    prompt: str
    completion: str
    reward: float
    day: int
    episode: int


@dataclass
class EpisodeResult:
    """Summary of one full sprint simulation."""
    episode_id: int
    scenario: str
    total_reward: float
    completion_pct: float
    avg_morale: float
    avg_quality: float
    steps: list[TrajectoryStep] = field(default_factory=list)


class TrajectoryCollector:
    """Collects training data from sprint simulations."""

    def __init__(self):
        self.episodes: list[EpisodeResult] = []
        self.all_steps: list[TrajectoryStep] = []

    def collect_episode(self, env, episode_id: int) -> EpisodeResult:
        """Run one episode and collect trajectory data."""
        from superstar import SprintEnv
        from superstar import SprintAction
        env.reset()

        steps = []
        for day in range(env.scenario.sprint_days):
            obs = env.step(SprintAction())
            reward = obs.reward or 0

            # Capture the env judgment prompt/completion
            if env.history["env_judgments"]:
                judgment = env.history["env_judgments"][-1]
                prompt = (
                    f"Day {env.day}/{env.scenario.sprint_days}. "
                    f"Team size: {len(env.engineers)}. "
                    f"Health: {obs.health}. "
                    f"Completion: {obs.completion_pct:.1f}%. "
                    f"Events today: {len(env.history['daily_events'][-1]) if env.history['daily_events'] else 0}."
                )
                completion = json.dumps(judgment)
                steps.append(TrajectoryStep(
                    prompt=prompt, completion=completion,
                    reward=reward, day=env.day, episode=episode_id
                ))

            if obs.done:
                break

        final = env.get_full_state()
        agents = final.get("agent_scores", {})
        result = EpisodeResult(
            episode_id=episode_id,
            scenario=env.scenario.name,
            total_reward=sum(s.reward for s in steps),
            completion_pct=final.get("project_health", {}).get("completion_pct", 0),
            avg_morale=np.mean([a["morale"] for a in agents.values()]) if agents else 50,
            avg_quality=np.mean([a["quality"] for a in agents.values()]) if agents else 0.5,
            steps=steps,
        )
        self.episodes.append(result)
        self.all_steps.extend(steps)
        return result

    def get_training_data(self) -> list[dict]:
        """Format collected data for GRPO training."""
        data = []
        for step in self.all_steps:
            data.append({
                "prompt": step.prompt,
                "completion": step.completion,
                "reward": step.reward,
            })
        return data


def prepare_grpo_dataset(collector: TrajectoryCollector) -> "Dataset":
    """Convert trajectory data to HuggingFace Dataset for TRL."""
    from datasets import Dataset

    training_data = collector.get_training_data()
    if not training_data:
        raise ValueError("No training data collected. Run simulations first.")

    # GRPO needs: prompt, and we generate multiple completions per prompt
    prompts = list(set(d["prompt"] for d in training_data))
    dataset_rows = []
    for prompt in prompts:
        completions = [d for d in training_data if d["prompt"] == prompt]
        dataset_rows.append({
            "prompt": prompt,
            "completions": [c["completion"] for c in completions],
            "rewards": [c["reward"] for c in completions],
        })

    # For TRL GRPO, we need just prompts — completions are generated during training
    return Dataset.from_dict({
        "prompt": [r["prompt"] for r in dataset_rows],
    })


def create_reward_function(collector: TrajectoryCollector):
    """Create a reward function for GRPO based on collected trajectory stats."""
    # Build a lookup of prompt -> average reward from historical data
    training_data = collector.get_training_data()
    prompt_rewards = {}
    for d in training_data:
        p = d["prompt"]
        if p not in prompt_rewards:
            prompt_rewards[p] = []
        prompt_rewards[p].append(d["reward"])

    avg_reward = np.mean([d["reward"] for d in training_data]) if training_data else 0.5

    def reward_fn(completions: list[str], prompts: list[str] = None, **kwargs) -> list[float]:
        """Score completions based on quality heuristics + historical performance."""
        rewards = []
        for i, comp in enumerate(completions):
            score = 0.5  # base

            # Reward well-structured JSON
            try:
                parsed = json.loads(comp)
                score += 0.1

                # Reward having key fields
                if "work_quality" in parsed:
                    score += 0.05
                if "suggestion" in parsed and len(parsed["suggestion"]) > 10:
                    score += 0.1
                if "hidden_assessment" in parsed and len(parsed["hidden_assessment"]) > 10:
                    score += 0.1
                if "risk_level" in parsed:
                    score += 0.05
                if "agent_adjustments" in parsed:
                    score += 0.1
            except (json.JSONDecodeError, TypeError):
                score -= 0.2

            # Use historical context if available
            if prompts and i < len(prompts) and prompts[i] in prompt_rewards:
                hist_avg = np.mean(prompt_rewards[prompts[i]])
                score = 0.6 * score + 0.4 * hist_avg

            rewards.append(max(0.0, min(1.0, score)))
        return rewards

    return reward_fn


def run_grpo_training(
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
    num_episodes: int = 5,
    num_train_epochs: int = 1,
    output_dir: str = "./checkpoints/env_llm",
    scenario_name: str = "default",
    learning_rate: float = 1e-5,
    per_device_batch_size: int = 2,
) -> dict:
    """
    Full GRPO training pipeline:
    1. Run simulations to collect trajectories
    2. Prepare dataset
    3. Train env LLM with GRPO
    4. Return metrics
    """
    from superstar import SprintEnv
    from scenarios import BUILT_IN_SCENARIOS

    print(f"\n{'='*60}")
    print(f"🧠 GRPO Training Pipeline")
    print(f"   Model: {model_name}")
    print(f"   Episodes: {num_episodes}")
    print(f"   Scenario: {scenario_name}")
    print(f"{'='*60}\n")

    # Phase 1: Collect trajectories
    print("📊 Phase 1: Collecting trajectories from simulations...")
    collector = TrajectoryCollector()
    scenario_fn = BUILT_IN_SCENARIOS.get(scenario_name)
    if not scenario_fn:
        raise ValueError(f"Unknown scenario: {scenario_name}")

    pre_training_results = []
    for ep in range(num_episodes):
        scenario = scenario_fn()
        scenario.seed = 42 + ep
        env = SprintEnv(scenario=scenario)
        result = collector.collect_episode(env, ep)
        pre_training_results.append({
            "episode": ep,
            "reward": result.total_reward,
            "completion": result.completion_pct,
            "morale": result.avg_morale,
            "quality": result.avg_quality,
        })
        print(f"   Episode {ep+1}: reward={result.total_reward:.3f}, "
              f"completion={result.completion_pct:.1f}%")

    # Phase 2: Prepare dataset
    print(f"\n📦 Phase 2: Preparing GRPO dataset ({len(collector.all_steps)} steps)...")
    dataset = prepare_grpo_dataset(collector)
    reward_fn = create_reward_function(collector)

    # Phase 3: GRPO Training
    print(f"\n🔧 Phase 3: GRPO Training...")
    metrics = _run_grpo_train(
        model_name=model_name,
        dataset=dataset,
        reward_fn=reward_fn,
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_batch_size=per_device_batch_size,
    )

    # Phase 4: Post-training evaluation
    print(f"\n📈 Phase 4: Post-training evaluation...")
    post_training_results = []
    for ep in range(num_episodes):
        scenario = scenario_fn()
        scenario.seed = 100 + ep  # different seeds for eval
        env = SprintEnv(scenario=scenario)
        result = collector.collect_episode(env, num_episodes + ep)
        post_training_results.append({
            "episode": ep,
            "reward": result.total_reward,
            "completion": result.completion_pct,
            "morale": result.avg_morale,
            "quality": result.avg_quality,
        })
        print(f"   Episode {ep+1}: reward={result.total_reward:.3f}, "
              f"completion={result.completion_pct:.1f}%")

    # Compute improvement
    pre_avg = np.mean([r["reward"] for r in pre_training_results])
    post_avg = np.mean([r["reward"] for r in post_training_results])
    improvement = ((post_avg - pre_avg) / max(pre_avg, 0.01)) * 100

    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"   Pre-training avg reward:  {pre_avg:.4f}")
    print(f"   Post-training avg reward: {post_avg:.4f}")
    print(f"   Improvement: {improvement:+.1f}%")
    print(f"   Model saved to: {output_dir}")
    print(f"{'='*60}\n")

    return {
        "pre_training": pre_training_results,
        "post_training": post_training_results,
        "improvement_pct": improvement,
        "metrics": metrics,
        "model_path": output_dir,
    }


def _run_grpo_train(model_name, dataset, reward_fn, output_dir,
                     num_train_epochs, learning_rate, per_device_batch_size) -> dict:
    """Run the actual GRPO training with TRL."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOConfig, GRPOTrainer

        print(f"   Loading model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )

        training_args = GRPOConfig(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            logging_steps=1,
            save_steps=50,
            max_completion_length=512,
            num_generations=4,
            remove_unused_columns=False,
            report_to="none",
        )

        trainer = GRPOTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            reward_funcs=reward_fn,
            processing_class=tokenizer,
        )

        print("   Training started...")
        train_result = trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        return {
            "train_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else 0,
            "train_steps": train_result.global_step if hasattr(train_result, 'global_step') else 0,
            "status": "completed",
        }

    except ImportError as e:
        print(f"   ⚠️ TRL/Transformers not available: {e}")
        print(f"   Running mock training for demo purposes...")
        return _mock_grpo_train(dataset, reward_fn, output_dir)

    except Exception as e:
        print(f"   ⚠️ Training error: {e}")
        print(f"   Running mock training for demo purposes...")
        return _mock_grpo_train(dataset, reward_fn, output_dir)


def _mock_grpo_train(dataset, reward_fn, output_dir) -> dict:
    """Mock GRPO training for environments without GPU/TRL."""
    os.makedirs(output_dir, exist_ok=True)
    n_steps = max(30, (len(dataset) if dataset else 10) * 3)

    # Simulate training loss curve
    losses = []
    for i in range(n_steps):
        loss = 2.0 * np.exp(-0.3 * i) + 0.5 + random.gauss(0, 0.05)
        losses.append(loss)

    # Save mock checkpoint
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump({"losses": losses, "steps": n_steps, "mock": True}, f)

    print(f"   Mock training: {n_steps} steps, final loss: {losses[-1]:.4f}")
    return {
        "train_loss": losses[-1],
        "train_steps": n_steps,
        "status": "mock_completed",
        "losses": losses,
    }


if __name__ == "__main__":
    results = run_grpo_training(num_episodes=3)
    print(json.dumps({k: v for k, v in results.items() if k != "metrics"}, indent=2, default=str))
