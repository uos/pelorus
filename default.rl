preprocessing {

  downsample.resolution = 25cm # "downsample" feature flag needs to be enabled in crates/pelorus/Cargo.toml.

  initial_scan_accumulate = 1 # Assume the first n scans are recorded while standing still. This can help with non-repetitive patterns because this method heavily depends on spatial resolution.

  filter {

    bounding_box {
      # The Bounding Box is built from MIN_{x, y, z} and MAX_{x, y, z}. Empty ranges mean maximum or minimum limits, so ".." is an infinitely large Bounding Box.
      x = .. # <MIN_x>..<MAX_x>, Example 1m..3m
      y = .. # <MIN_y>..<MAX_y>, Example 2m..
      z = .. # <MIN_z>..<MAX_z>, Example 4cm..100m
      invert = false # true = only accept points inside this bounding_box
    }

    distance {
      range = .. # Example ..2m (up to 2m from origin) or 1cm..4m (everything from 1cm to 4m distance)
      invert = false # true = only accept points outside the range
    }

    horizontal_fov {
      degrees = .. # Degrees, Example 10..42
      forward = [1.0, 0.0] # Axis to determine where field of view ranges start and end.
      enable = false
      invert = false # true = only accept points outside the range
    }

    polygon_2d {
      coords = [] # 2D coordinates of each vertex. Example [ [1.0m, -1.5m], [-1.0m, 1.5m], [0m, 0m] ]
      global_zero_frame = "" # If set to a TF frame, the polygon is transformed to that frame first. This enables global scan boundaries in any global coordinate system given by TF.
      invert = false # true = only accept points outside the polygon
    }

    intensity {
      values = .. # Floating range normalized to 0..1.0
      invert = false # true = only accept points outside the range
    }
    
  }

}

registration {

  map {
    resolution = 50cm # Voxel side length. NOTE: This will heavily affect your registration accuracy.
    capacity = 120000 # How many Voxels of the previous defined resolution to save at any time. With Cuda enabled, the capacity will be round up to the next power of 2.
   
    distribution_update_sensitivity = 10.0 # Magic weighting value for handling the sensivity to new scans. The higher this value, the sensitive is it to brand new scans.
  }

  k_neighborhood = 12 # How many points to get from the neighborhood. Higher values yield higher accuracy at the cost of runtime.

  max_iter = 2 # Maximum IEKF correction iterations per scan.

  neighbors_max_range = 5m # Maximum distance to any neighbor. Default is mostly ok if you do not drastically reduce the map resolution.
}

lidar {
  topic = "/ouster/points"
  vendor = Ouster # Velodyne | Ouster | Livox | Robosense | Skip (Use any 3D sensor when just Lidar Odometry is selected)

  qos.reliability = Reliable # Reliable | BestEffort | SystemDefault
  covariance = 0.001 # Weight for ICP correction in the IEKF

  as_sync = false # When true, use incoming Lidar messages as a forced sync trigger. Note: This looses accuracy but some outer libraries work better with this mode.
}


imu {
  topic = "/ouster/imu"
  qos.reliability = Reliable # Reliable | BestEffort | SystemDefault

  initial_setup = 500ms

  from_lidar_transform {
    use_tf_tree = false # Ignore the next manual translation and use the TF tree instead. The lookup will be once with receiving the first IMU message.

    # lidar -> imu
    translation = [-6.253mm, 11.775mm, 28.535mm]
    rotation = [ [-1.0, 0.0, 0.0],
                 [0.0, -1.0, 0.0],
                 [0.0, 0.0, 1.0] ]
  }

  ahrs_filtering.method = FilterAccelerationAndCreateOrientation # CreateOrientationOnly | FilterAccelerationAndCreateOrientation | FilterAccelerationOnly | Disable (recommended when using on-device filtering in the driver)

  gravity = 9.81 # Initial guess in m/s^2 for initialization.

  covariance_gyroscope = 0.1
  covariance_gyroscope_bias = 0.0001

  covariance_acceleration = 0.1
  covariance_acceleration_bias = 0.0001

  calibration {
    gyroscope = true # Take a few measurements first without rotating in the beginning to smooth out rotations.
    samples = 100 # Use this many measurements for the gyroscope
    parameter_file = "" # Filter out static bias. Relative path "./<path>" from this config file or absolute path "/<path>" to a calibration file created by https://github.com/stelzo/imu-calib-ros.
  }

}

wheel {
  topic = "/odom"
  qos.reliability = Reliable # Reliable | BestEffort | SystemDefault

  to_imu_transform {
    use_tf_tree = false # Ignore the next manual translation and use the TF tree instead. The lookup will be once with receiving the first odom message.

    # Rotation Matrix Wheel_to_IMU
    rotation = [ [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0] ]
  }

  # covariance = 0.001 # Weight for Odometry correction in the IEKF. If not given, the covariance provided in the Odometry message will be used.

  kinematic_constraint_std = 0.1 # Standard deviation for kinematic constraints in undistortion (lower = stronger constraints)

  kinematic_model = Full3D # Full3D | Differential | Holonomic2D | Ackermann. Applies kinematic constraints on the IEKF.
  
  # Ackermann steering configuration (only used when kinematic_model = Ackermann)
  ackermann {
    imu_to_rear_axle = [0.0m, 0.0m, 0.0m] # Offset from IMU to rear axle center in imu frame
  }
}

output {

  terminal {
    ui = true # Show terminal outputs like timings or other metrics.
    fps = 1 # Refresh rate for rendering to the terminal in frames per second
    warning_debounce = 1s # Debounce time for specific warnings
     
    # Log level (accepted: trace, debug, info, warn, error).
    log_level = "error"
  }

  global_frame = "lio_init" # Which frame to publish the TF messages to

  lidar_as_map_tf_base = true # The cloud is processed in the IMU coordinate system. With this parameter, each point that wants to be published is first transformed back to the Lidar coordinate system again.

  mapping {
    resolution = 20cm # When "mapping" flag is enabled, this defines the resolution of the global map.
  }

  rerun {
    undistorted_static_world = false # Logs the registered clouds static in world coordinates so they do not vanish
  }

  lio_odometry_msg_qos = SystemDefault # Reliable | BestEffort | SystemDefault
  lio_path_msg_qos = SystemDefault # Reliable | BestEffort | SystemDefault

  imu_odometry_msg_qos = SystemDefault # Reliable | BestEffort | SystemDefault
  imu_path_msg_qos = SystemDefault # Reliable | BestEffort | SystemDefault

  imu_ahrs_filtered_msg_qos = Reliable # Reliable | BestEffort | SystemDefault

  imu_ahrs_filtered_topic = "imu/filtered" # AHRS filtered output

  logger_prefix = "pelorus"
}



