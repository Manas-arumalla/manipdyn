# Reinforcement learning

A learned baseline alongside the classical controllers: the same MuJoCo physics
is exposed as a Gymnasium environment and solved with Soft Actor-Critic (SAC).

```bash
pip install -e "./manipdyn[rl]"
python scripts/train_rl.py --timesteps 40000
```

## Environment — `ReachEnv`

* **Observation** (18-d): arm position (6), velocity (6), end-effector position
  (3), and the goal-relative vector `goal − ee` (3).
* **Action** (6-d): joint-position targets about the home pose, tracked by an
  internal PD + gravity-compensation loop. This position-style action space is
  standard in manipulation RL and far more sample-efficient than commanding raw
  torques (which, in our experiments, SAC failed to learn within a small
  budget).
* **Reward**: potential-based progress `10·(d_prev − d)` − `0.1·d` − control
  penalty, plus a success bonus; the episode terminates on reaching the goal
  (3 cm tolerance).
* Goals are the forward kinematics of configurations near a forward-reaching
  nominal, so they are always reachable and in a consistent region.

## Result

SAC, 40k environment steps:

| metric | value |
|--------|-------|
| success rate (3 cm) | **80%** (20 random goals) |
| mean final distance | **34 mm** |

The trained policy ships with the package (`manipdyn/rl/sac_reach.zip`) and can
be loaded with `stable_baselines3.SAC.load(...)`. This is a deliberately small
budget to keep the demo fast; longer training and reward shaping push the
success rate higher.

## Why it belongs here

It closes the loop on the project's thesis — *compare methods on identical
physics*. The RL policy is evaluated with the same reach scenarios and the same
end-effector-error metric as the eight classical controllers, so learned and
model-based control sit on one bench.
