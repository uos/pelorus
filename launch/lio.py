from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lio',
            executable='lio',
            parameters=[
                { 'verbose': True },
                { 'map_frame': 'map' },
                { 'imu_lidar_tf_override': False }, # Use TF to get IMU to sensor transformation.
                { 'filter_global_zero_frame': '' }, # Global Ground Frame. When set, points of polygons are transformed to this frame first before filtering. - TODO create global ground frame gravity vector from fusion with lio pose. then this can default to "-auto-"
                { 'min_dist': 2.0 }, # filter out points closer than this distance
                #{ 'min_intensity': 30.0 },
                #{ 'max_intensity': 255.0 },
                { 'lidar_vendor': 'ouster' },
                { 'voxel_size': 0.10 },
            ],
            remappings=[
                ('cloud', 'ouster/points'),
                ('imu', 'ouster/imu'),
                ('registered', 'ouster/registered'),
            ]
        ),
    ])
