# so101-rl

External Isaac Lab package with the SO-101 visual-manipulation task
environments consumed by [`pi05-self-improve/`](../../../pi05-self-improve/)
for autonomous VLA rollouts and self-improvement data collection. It contains
no policy training code: the task's termination terms double as the scripted
success oracle for recorded rollouts.

A shared platform config owns the robot, workcell, cameras, observations, and
actions; the scenario config owns its assets, reset layout, rewards, and
terminations.

## Scenario

| Scenario | Fixed task | Randomized task |
|---|---|---|
| One box into one cup | `SO101-Object-In-Cup-Vision-Fixed-v0` | `SO101-Object-In-Cup-Vision-v0` |

Both IDs are registered against `isaaclab.envs:ManagerBasedRLEnv` and export
the same three-camera + six-joint interface that mirrors the real
`monomanual_dual_overhead` rig.

Install this package into the existing Isaac Lab Python environment. Do not
install a second copy of Isaac Lab as a package dependency.

See the repository-level [`isaaclab/README.md`](../../README.md) for asset
preparation, live vectorized simulation, and verification commands.
