from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_pelorus(context):
    node_name = LaunchConfiguration("node_name")
    namespace = LaunchConfiguration("namespace")
    arguments = [
        "online",
        "--node-name",
        node_name,
        "--namespace",
        namespace,
    ]

    config_path = LaunchConfiguration("config_path").perform(context)
    if config_path:
        arguments.extend(["--config-path", config_path])

    cpu_only = LaunchConfiguration("cpu_only").perform(context).lower()
    if cpu_only in ("1", "true", "yes", "on"):
        arguments.append("--cpu-only")

    return [
        Node(
            package="pelorus",
            executable="pelorus",
            name=node_name,
            namespace=namespace,
            output="screen",
            arguments=arguments,
            ros_arguments=["--log-level", LaunchConfiguration("log_level")],
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_path",
                default_value="",
                description="Optional path to a Pelorus .rl configuration overlay.",
            ),
            DeclareLaunchArgument(
                "namespace",
                default_value="pelorus",
                description="ROS namespace for the Pelorus node.",
            ),
            DeclareLaunchArgument(
                "node_name",
                default_value="pelorus",
                description="ROS node name.",
            ),
            DeclareLaunchArgument(
                "log_level",
                default_value="info",
                description="ROS log level.",
            ),
            DeclareLaunchArgument(
                "cpu_only",
                default_value="false",
                description="Force CPU-only processing.",
            ),
            OpaqueFunction(function=_launch_pelorus),
        ]
    )
