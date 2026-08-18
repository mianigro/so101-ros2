"""Setup-driven arm bringup.

Top-level launches read the active setup YAML and spawn one leader/follower
bringup pair (plus one teleop relay per pair) for every arm the setup
defines. The setup YAML is the single source of truth for namespaces, USB
ports, and world placement — edit the file (or point ``setup_config_file``
at an edited copy) to change them.
"""

from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

from so101_bringup.setup_config import (
    Setup,
    SetupConfigError,
    default_setups_dir,
    load_setup,
)

_TRUTHY = {"1", "true", "yes", "on"}


def declare_use_follower_argument(*, default_value: str = "true") -> list:
    return [
        DeclareLaunchArgument(
            "use_follower",
            default_value=default_value,
            description=(
                "Start the physical/mock follower controller managers. Set false "
                "when Isaac Sim is the follower (single-pair setups only)."
            ),
        ),
        DeclareLaunchArgument("leader_rviz", default_value="false"),
        DeclareLaunchArgument("follower_rviz", default_value="false"),
    ]


def active_setup(context) -> Setup:
    """Resolve the setup selected by the setup / setup_config_file arguments."""
    name = LaunchConfiguration("setup").perform(context).strip()
    override = LaunchConfiguration("setup_config_file").perform(context).strip()
    return load_setup(name, default_setups_dir(), path=override or None)


_CONTROLLER_CONFIGS = {
    "leader.launch.py": "leader_controllers.yaml",
    "follower.launch.py": "follower_controllers.yaml",
}


def _bringup_include(launch_file: str, arm, hardware_type, use_rviz):
    # controller_config_file must be passed explicitly: launch deduplicates
    # identically-named arguments across includes, so the per-file defaults in
    # leader/follower.launch.py would collide.
    controller_config = PathJoinSubstitution([
        FindPackageShare("so101_bringup"),
        "config",
        "ros2_control",
        _CONTROLLER_CONFIGS[launch_file],
    ])
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("so101_bringup"), "launch", launch_file
            ])
        ),
        launch_arguments={
            "namespace": arm.namespace,
            "hardware_type": hardware_type,
            "usb_port": arm.usb_port,
            "frame_prefix": arm.frame_prefix,
            "controller_config_file": controller_config,
            "use_rviz": use_rviz,
        }.items(),
    )


def _teleop_relay_include(leader_namespace: str, follower_namespace: str, params_file):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("so101_teleop"), "launch", "teleop.launch.py"
            ])
        ),
        launch_arguments={
            "leader_namespace": leader_namespace,
            "follower_namespace": follower_namespace,
            "node_namespace": follower_namespace,
            "params_file": params_file,
        }.items(),
    )


def spawn_teleop_arms(context):
    """Leader + follower bringup and one teleop relay per arm pair."""
    setup = active_setup(context)
    hardware_type = LaunchConfiguration("hardware_type").perform(context)
    use_follower = (
        LaunchConfiguration("use_follower").perform(context).strip().lower() in _TRUTHY
    )
    if not use_follower and setup.pairs > 1:
        raise SetupConfigError(
            "use_follower:=false (Isaac Sim follower) is only supported for "
            "single-pair setups; Isaac Sim has no bimanual support"
        )
    leader_rviz = LaunchConfiguration("leader_rviz").perform(context)
    follower_rviz = LaunchConfiguration("follower_rviz").perform(context)
    params_file = LaunchConfiguration("teleop_params_file")
    delay = float(LaunchConfiguration("teleop_delay_s").perform(context))

    actions = []
    for arm in setup.leaders:
        actions.append(_bringup_include("leader.launch.py", arm, hardware_type, leader_rviz))
    if use_follower:
        for arm in setup.followers:
            actions.append(
                _bringup_include("follower.launch.py", arm, hardware_type, follower_rviz)
            )
    for leader, follower in zip(setup.leaders, setup.followers):
        actions.append(
            TimerAction(
                period=delay,
                actions=[_teleop_relay_include(leader.namespace, follower.namespace, params_file)],
            )
        )
    return actions


def spawn_follower_arms(context):
    """Follower bringup only (vision / recording / inference stacks)."""
    setup = active_setup(context)
    hardware_type = LaunchConfiguration("hardware_type").perform(context)
    use_rviz = LaunchConfiguration("use_rviz")
    return [
        _bringup_include("follower.launch.py", arm, hardware_type, use_rviz)
        for arm in setup.followers
    ]


def teleop_arms_action() -> OpaqueFunction:
    return OpaqueFunction(function=spawn_teleop_arms)


def follower_arms_action() -> OpaqueFunction:
    return OpaqueFunction(function=spawn_follower_arms)


def include_layout_tf():
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("so101_bringup"), "launch", "layout_tf.launch.py"
            ])
        ),
        launch_arguments={
            "setup": LaunchConfiguration("setup"),
            "setup_config_file": LaunchConfiguration("setup_config_file"),
        }.items(),
    )
