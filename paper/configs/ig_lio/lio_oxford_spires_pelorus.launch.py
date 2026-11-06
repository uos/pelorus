import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory("ig_lio")
    config_path = os.path.join(package_dir, "config", "oxford_spires_pelorus.yaml")
    rviz_path = os.path.join(package_dir, "rviz", "lio_show.rviz")

    with open(config_path, "r") as config_file:
        params = yaml.safe_load(config_file)

    return LaunchDescription([
        DeclareLaunchArgument(
            "rviz",
            default_value="false",
            description="Start RViz2",
        ),
        Node(
            package="ig_lio",
            executable="ig_lio_node",
            name="ig_lio_node",
            output="screen",
            parameters=[params],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            arguments=["-d", rviz_path],
            output="screen",
            condition=IfCondition(LaunchConfiguration("rviz")),
        ),
    ])
