from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    leader_ns = LaunchConfiguration('leader_namespace')
    follower_ns = LaunchConfiguration('follower_namespace')
    node_ns = LaunchConfiguration('node_namespace')
    params_file = LaunchConfiguration('params_file')

    default_params = PathJoinSubstitution(
        [FindPackageShare('so101_teleop'), 'config', 'teleop.yaml']
    )

    # Derived topics from namespaces
    leader_topic = PathJoinSubstitution(['/', leader_ns, 'joint_states'])
    fwd_topic = PathJoinSubstitution(
        ['/', follower_ns, 'forward_controller', 'commands']
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument('leader_namespace', default_value='leader'),
            DeclareLaunchArgument('follower_namespace', default_value='follower'),
            DeclareLaunchArgument(
                'node_namespace',
                default_value=LaunchConfiguration('follower_namespace'),
                description=(
                    'Namespace for the relay node; defaults to the follower '
                    'namespace so multiple relays can coexist'
                ),
            ),
            DeclareLaunchArgument('params_file', default_value=default_params),
            Node(
                package='so101_teleop',
                executable='teleop',
                name='follower_command_relay',
                namespace=node_ns,
                output='screen',
                parameters=[
                    params_file,
                    {
                        'leader_topic': leader_topic,
                        'fwd_topic': fwd_topic,
                    },
                ],
            ),
        ]
    )
