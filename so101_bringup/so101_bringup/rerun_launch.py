"""Shared Rerun bridge launch wiring with fail-fast environment validation.

The bridges run through Pixi, so their working directory must be the repository
root that owns ``pixi.toml``. Top-level launches expose this through the
``rerun_env_dir`` argument (default: ``$SO101_RERUN_ENV_DIR``). Requesting a
bridge without a usable directory fails the launch immediately instead of
spawning ``pixi`` with an empty working directory.
"""

from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    TimerAction,
)
from launch.substitutions import EnvironmentVariable, LaunchConfiguration

_TRUTHY = {"1", "true", "yes", "on"}


def declare_rerun_arguments(rerun_delay_default="2.0"):
    """Declare the four cross-cutting Rerun arguments used by top-level launches."""
    return [
        DeclareLaunchArgument("use_rerun", default_value="false"),
        DeclareLaunchArgument(
            "use_rerun_3d",
            default_value="false",
            description="Launch the 3D Rerun bridge (animated URDF + TF + cameras + plots)",
        ),
        DeclareLaunchArgument(
            "rerun_env_dir",
            # Best: set env var once, no need to pass each run:
            # export SO101_RERUN_ENV_DIR=/abs/path/to/so101-ros-physical-ai
            default_value=EnvironmentVariable(
                "SO101_RERUN_ENV_DIR", default_value=""
            ),
        ),
        DeclareLaunchArgument("rerun_delay_s", default_value=rerun_delay_default),
    ]


def _bridge_process(pixi_task, camera_profile, rerun_env_dir):
    return ExecuteProcess(
        cmd=["pixi", "run", pixi_task, "--", "--camera-profile", camera_profile],
        cwd=rerun_env_dir,
        additional_env={"PYTHONUNBUFFERED": "1"},
        output="screen",
    )


def _spawn_bridges(
    context,
    *,
    camera_profile,
    use_rerun,
    use_rerun_3d,
    rerun_env_dir,
    rerun_delay_s,
    also_requires_dir=None,
):
    want_2d = use_rerun.perform(context).strip().lower() in _TRUTHY
    want_3d = use_rerun_3d.perform(context).strip().lower() in _TRUTHY
    also_wants_dir = (
        also_requires_dir.perform(context).strip().lower() in _TRUTHY
        if also_requires_dir is not None
        else False
    )
    env_dir = rerun_env_dir.perform(context).strip()
    if (want_2d or want_3d or also_wants_dir) and not env_dir:
        raise RuntimeError(
            "A Pixi-managed process was requested but rerun_env_dir is empty. "
            "Pixi tasks need the repository root that owns pixi.toml as their "
            "working directory. Export "
            "SO101_RERUN_ENV_DIR=/abs/path/to/so101-ros-physical-ai or pass "
            "rerun_env_dir:=/abs/path/to/so101-ros-physical-ai."
        )
    profile = camera_profile.perform(context)
    delay = float(rerun_delay_s.perform(context))
    actions = []
    if want_2d:
        actions.append(
            TimerAction(
                period=delay,
                actions=[_bridge_process("bridge", profile, env_dir)],
            )
        )
    if want_3d:
        actions.append(
            TimerAction(
                period=delay,
                actions=[_bridge_process("bridge-3d", profile, env_dir)],
            )
        )
    return actions


def rerun_bridge_actions(camera_profile, also_requires_dir=None):
    """Return the validated 2D/3D Rerun bridge actions for a top-level launch.

    ``also_requires_dir`` optionally names another ``LaunchConfiguration``
    (e.g. ``use_inference``) for a Pixi-managed process sharing the same
    working directory, so it is covered by the same fail-fast check.
    """
    return OpaqueFunction(
        function=_spawn_bridges,
        kwargs={
            "camera_profile": camera_profile,
            "use_rerun": LaunchConfiguration("use_rerun"),
            "use_rerun_3d": LaunchConfiguration("use_rerun_3d"),
            "rerun_env_dir": LaunchConfiguration("rerun_env_dir"),
            "rerun_delay_s": LaunchConfiguration("rerun_delay_s"),
            "also_requires_dir": also_requires_dir,
        },
    )
