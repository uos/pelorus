from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lio',
            executable='lio',
            parameters=[
                { 'verbose': True},
                { 'global_frame': ''},
                { 'global_zero_frame': ''},
                { 'min_dist': 2.0 },
                #{ 'min_intensity': 30.0 },
                #{ 'max_intensity': 255.0 },
                { 'lidar_vendor': 'ouster' },
                { 'voxel_size': 0.10 },
            ],
            remappings=[
                ('cloud', 'ouster/points'),
                ('imu', 'imu'),
                ('registered', 'ouster/registered'),
            ]
        ),
    ])
