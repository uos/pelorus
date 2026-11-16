# Demo configuration for a custom robot setup with a Velodyne lidar
# Adjust the parameters as needed for your specific setup of the bagfile.

lidar.vendor = Velodyne

imu.from_lidar_transform {
    translation = [0.0m, 0.0m, 0.0m] # Translation Vector Lidar_to_IMU

    # Rotation Matrix Lidar_to_IMU
    rotation = [ [1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0],
                 [0.0, 0.0, -1.0] ]
}

preprocessing.downsample.resolution = 50cm
registration.map.resolution = 50cm
