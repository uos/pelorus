from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pelorus',
            executable='pelorus',
            parameters=[
                { 'verbose': True },
                { 'feedback_fps': 0 }, # terminal fps of performance metrics. 0 to disable.
                { 'map_frame': 'map' },
                { 'imu_lidar_tf_override': False }, # Use TF to get IMU to sensor transformation.
                { 'filter_global_zero_frame': '' }, # Global Ground Frame. When set, points of polygons are transformed to this frame first before filtering. - TODO create global ground frame gravity vector from fusion with lio pose. then this can default to "-auto-"
                { 'min_dist': 3.0 }, # filter out points closer than this distance

                #{ 'max_dist': 20.0 },
                #{ 'min_intensity': 30.0 },
                #{ 'max_intensity': 255.0 },

                { 'lidar_vendor': 'robosense' },
                { 'imu_qos': 'system_default' }, # for ouster
                { 'lidar_qos': 'system_default' }, # for ouster

                { 'lidar_to_imu_pose_x': 0.0 },
                { 'lidar_to_imu_pose_y': 0.0 },
                { 'lidar_to_imu_pose_z': -0.1 },

                #{ 'lidar_to_imu_pose_roll': 0.0 }, # TODO change to rotation matrix?
                #{ 'lidar_to_imu_pose_pitch': 0.0 },
                #{ 'lidar_to_imu_pose_yaw': 0.0 },

                { 'enable_live_extrinsic_calibration': True },

                { 'imu_cov_gyr': 0.1 },
                { 'imu_cov_gyr_bias': 0.0001 },
                { 'imu_cov_acc': 0.1 },
                { 'imu_cov_acc_bias': 0.0001 },
                { 'imu_init_length': 100 },

                # TODO: point_cov constant is param but not yet used as params but still const
                { 'map_resolution': 0.25 }, # local map resolution
                { 'voxel_size': 0.5 }, # for downsampling
            ],
            remappings=[
                #('cloud', 'ouster/points'),
                #('imu', 'ouster/imu'),
                #('registered', 'ouster/registered'),
                ('cloud', '/rslidar_points'),
                ('imu', 'imu/data'),
                ('registered', '/registered'),
            ]
        ),
    ])
