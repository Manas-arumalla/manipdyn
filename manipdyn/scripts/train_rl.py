"""Train and evaluate an SAC reaching policy on the UR5e (RL flagship).

Trains Soft Actor-Critic on :class:`manipdyn.rl.ReachEnv` and reports the
success rate / mean final distance over random goals, then saves the policy
into the package so it can be loaded and compared against the classical
controllers.

Run (from manipdyn/, needs the rl extra):
    python scripts/train_rl.py --timesteps 25000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC

from manipdyn.rl import ReachEnv

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "src" / "manipdyn" / "rl" / "sac_reach.zip"


def evaluate(model, episodes: int = 20, seed: int = 1234) -> dict:
    env = ReachEnv(seed=seed)
    finals, successes = [], 0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        info = {"distance": np.inf}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, info = env.step(action)
            done = term or trunc
        finals.append(info["distance"])
        successes += int(info["distance"] < env.tol)
    return {
        "episodes": episodes,
        "success_rate": successes / episodes,
        "mean_final_dist_mm": float(np.mean(finals) * 1e3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=25000)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    env = ReachEnv(seed=0)
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        seed=0,
        learning_starts=1000,
        batch_size=256,
        train_freq=1,
    )
    print(f"Training SAC for {args.timesteps} steps...")
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"Saved policy -> {args.out}")

    metrics = evaluate(model)
    print(
        f"Eval: success {metrics['success_rate'] * 100:.0f}% over {metrics['episodes']} goals, "
        f"mean final dist {metrics['mean_final_dist_mm']:.1f} mm"
    )


if __name__ == "__main__":
    main()
