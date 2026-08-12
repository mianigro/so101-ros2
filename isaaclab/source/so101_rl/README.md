# so101-rl

External Isaac Lab package for the SO-101 three-camera object-in-cup PPO tasks.
The deployable actor consumes cameras and measured joints; a training-only MLP
critic consumes one exact 34-value simulator task-state group.

Install this package into the existing Isaac Lab Python environment. Do not
install a second copy of Isaac Lab as a package dependency.

See the repository-level `isaaclab/README.md` for asset preparation, live
vectorized simulation, training, playback, and verification commands. See
[`isaaclab/METHODOLOGY.md`](../../METHODOLOGY.md) for the environment design,
exact rewards, PPO and network configuration, supported algorithms, and
extension guide.
