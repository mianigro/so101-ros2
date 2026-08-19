# so101-rl

External Isaac Lab package for SO-101 three-camera manipulation tasks. A shared
platform config owns the robot, workcell, cameras, actor observations/actions,
randomization, visual actor, and PPO defaults. Explicit scenario configs own
their assets, reset layout, rewards, terminations, and exact critic state.

The package currently registers single-box placement with a 34-value critic and
permutation-invariant three-box placement with an 84-value critic. Both export
the same camera-and-joints actor interface.

## Scenarios

| Scenario | Fixed task | Randomized task |
|---|---|---|
| One box into one cup | `SO101-Object-In-Cup-Vision-Fixed-v0` | `SO101-Object-In-Cup-Vision-v0` |
| Three boxes into three distinct cups | `SO101-Three-Boxes-In-Cups-Vision-Fixed-v0` | `SO101-Three-Boxes-In-Cups-Vision-v0` |

Select an ID with `--task`. Train the fixed variant first, then resume its
checkpoint into the randomized variant from the same row. Checkpoints must not
cross between rows because the critic state widths differ.

Install this package into the existing Isaac Lab Python environment. Do not
install a second copy of Isaac Lab as a package dependency.

See the repository-level `isaaclab/README.md` for asset preparation, live
vectorized simulation, training, playback, and verification commands. See
[`isaaclab/METHODOLOGY.md`](../../METHODOLOGY.md) for the environment design,
exact rewards, PPO and network configuration, supported algorithms, and
extension guide.
