from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _launch_pelorus(context):
    node_name = LaunchConfiguration("node_name")
    namespace = LaunchConfiguration("namespace")
    nodes = []
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

    export_trajectory = LaunchConfiguration("export_trajectory").perform(context)
    if export_trajectory:
        arguments.extend(["--export-trajectory", export_trajectory])

    export_processing_time = LaunchConfiguration("export_processing_time").perform(context)
    if export_processing_time:
        arguments.extend(["--export-processing-time", export_processing_time])

    cpu_only = LaunchConfiguration("cpu_only").perform(context).lower()
    if cpu_only in ("1", "true", "yes", "on"):
        arguments.extend(["-o", "parallelism.cpu_only=true"])

    nodes.append(
        Node(
            package="pelorus",
            executable="pelorus",
            name=node_name,
            namespace=namespace,
            output="screen",
            arguments=arguments,
            ros_arguments=["--log-level", LaunchConfiguration("log_level")],
        )
    )

    rviz = LaunchConfiguration("rviz").perform(context).lower()
    if rviz in ("1", "true", "yes", "on"):
        nodes.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", LaunchConfiguration("rviz_config")],
            )
        )

    return nodes


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
            DeclareLaunchArgument(
                "export_trajectory",
                default_value="",
                description="Optional path to write a TUM trajectory file.",
            ),
            DeclareLaunchArgument(
                "export_processing_time",
                default_value="",
                description="Optional path to write processing time samples.",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Start RViz with the configured RViz display file.",
            ),
            DeclareLaunchArgument(
                "rviz_config",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("pelorus"), "rviz", "lio.rviz"]
                ),
                description="Path to the RViz configuration file.",
            ),
            OpaqueFunction(function=_launch_pelorus),
        ]
    )
