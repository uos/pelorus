from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cloudfilter',
            namespace='cloudfilter',
            executable='cloudfilter',
            name='cloudfilter',
            parameters=[
                {'verbose': True},
                {'global_frame': 'slam'},
                {'global_zero_frame': 'base_footprint'},
                {'filter_type': 'aabb'},
                {'invert_aabb': False},
                {'aabb_min_x': -1.0},
                {'aabb_max_x': 4.0},
                {'aabb_min_y': -1.5},
                {'aabb_max_y': 1.5},
                {'aabb_min_z': -0.1},
                {'aabb_max_z': 1.5},
                {'min_dist': 0.85},
                {'max_dist': 4.0},
                {'invert_distance': False},
                {'fov_left': -90.0},
                {'fov_right': 90.0},
                {'enable_fov': True},
                {'invert_polygon': False},
                {'disable_with_first_polygon': False},
                {'min_intensity': 30},
                {'max_intensity': 255},
                {'invert_polygon': False},
            ],
            remappings=[
                ('cloud', 'ouster/points'),
                ('filtered', 'ouster/points_filtered'),
                ('polygon', 'bounding_box'),
            ]
        ),
    ])