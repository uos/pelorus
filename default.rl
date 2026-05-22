registration {
  map {

    # @recommended
    # Feature size used for registration. NOTE: This will heavily affect your registration accuracy and runtime.
    resolution = 50cm

    # @recommended
    # How many Voxels of the previous defined resolution to save at any time. With Cuda enabled, the capacity will be round up to the next power of 2.
    capacity = 131072
   
    # Magic weighting value for handling the sensivity to new scans. The higher this value, the sensitive is it to brand new scans.
    distribution_update_sensitivity = 10.0
  }

  # Maximum IEKF correction iterations per scan.
  max_iter = 4

  # Maximum distance to any neighbor. Default is mostly ok if you do not drastically reduce the map resolution.
  neighbors_max_range = 5m
}

preprocessing {
  downsample {
    # @recommended
    # Output the full undistorted cloud on the cloud output topic.
    bypass_output_downsample = true

    # @recommended
    # Downsample the incoming scan by keeping only every n-th point. 1 means no points are skipped.
    point_skip = 1

    # Where to run preprocessing downsampling. Backend uses the active acceleration backend when available; Cpu always uses the CPU path.
    # CPU | Backend
    backend = CPU

    # Estimated sizes of features in the incoming scan, expressed as a factor of registration.map.resolution. Expected values are between 0.0 and 1.0.
    query_resolution_factor = 0.5
  }

  # Assume the first n scans are recorded while standing still. This fixes non-repetitive patterns because the method heavily depends on spatial resolution.
  initial_scan_accumulate = 3

  filter {
    bounding_box {
      # The Bounding Box is built from MIN_{x, y, z} and MAX_{x, y, z}. Empty ranges mean maximum or minimum limits, so ".." is an infinitely large Bounding Box.
      
      # <MIN_x>..<MAX_x>, Example 1m..3m
      x = ..

      # <MIN_y>..<MAX_y>, Example 2m..
      y = ..
      
      # <MIN_z>..<MAX_z>, Example 4cm..100m
      z = ..

      # if true, we only accept points inside this bounding_box
      invert = false
    }

    distance {
      # Example ..2m (up to 2m from origin) or 1cm..4m (everything from 1cm to 4m distance)
      range = 4m..

      # if true we only accept points outside the range
      invert = false
    }

    horizontal_fov {
      enable = false

      # Degrees, Example 10..42
      degrees = ..

      # Axis to determine where field of view ranges start and end.
      forward = [1.0, 0.0]

      # if true we only accept points outside the range
      invert = false
    }

    polygon_2d {
      # 2D coordinates of each vertex. Example [ [1.0m, -1.5m], [-1.0m, 1.5m], [0m, 0m] ]
      coords = []

      # If set to a TF frame, the polygon is transformed to that frame first. This enables global scan boundaries in any global coordinate system given by TF.
      global_zero_frame = ""

      # if true we only accept points outside the polygon
      invert = false
    }

    intensity {
      # Floating range normalized to 0..1.0
      values = ..

      # if true we only accept points outside the range
      invert = false
    }
    
  }

}

lidar {
  # @recommended
  topic = "/ouster/points"

  # @recommended
  # Use the vendor specific cloud format timestamps for undeskewing
  # Possible values: Velodyne | Ouster | Livox | Robosense | LivoxMid360 | Hesai | Blickfeld | SickMultiScan | NoTimestamp
  vendor = NoTimestamp

  # @recommended
  # Reliable | BestEffort | SystemDefault. Any other setting than Reliable is undefined behavior.
  qos.reliability = Reliable 

  # Weight for ICP correction in the IEKF
  covariance = 0.001

 # If true, use incoming Lidar messages as a forced sync trigger. Note: This looses accuracy but some outer libraries work better with this mode.
  as_sync = false
}


imu {

  # @recommended
  topic = "/ouster/imu"

  # @recommended
  # Reliable | BestEffort | SystemDefault. Any other setting than Reliable is undefined behavior.
  qos.reliability = Reliable

  initial_setup = 100ms

  # @recommended
  from_lidar_transform {

    # Ignore the next manual translation and use the TF tree instead. The lookup will be once with receiving the first IMU message.
    use_tf_tree = false

    # lidar -> imu
    translation = [-6.253mm, 11.775mm, 28.535mm]
    rotation = [  [1.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0],
                  [0.0, 0.0, 1.0] ]
  }

  # CreateOrientationOnly | FilterAccelerationAndCreateOrientation | FilterAccelerationOnly | Disable (recommended when using on-device filtering in the driver)
  ahrs_filtering.method = FilterAccelerationAndCreateOrientation

  # Initial guess in m/s^2 for initialization.
  gravity = 9.81

  covariance_gyroscope = 0.1
  covariance_gyroscope_bias = 0.0001

  covariance_acceleration = 0.1
  covariance_acceleration_bias = 0.0001

  calibration {
    # Take a few measurements first without rotating in the beginning to smooth out rotations.
    gyroscope = true
    
    # Use this many measurements for the gyroscope
    samples = 100

    # Filter out static bias. Relative path "./<path>" from this config file or absolute path "/<path>" to a calibration file created by https://github.com/stelzo/imu-calib-ros.
    parameter_file = ""
  }

}

wheel {
  # Processing odometry messages is not tested enough yet, thus experimental.
  topic = "/odom_experimental"

  # Reliable | BestEffort | SystemDefault
  qos.reliability = Reliable 

  to_imu_transform {
    # Ignore the next manual translation and use the TF tree instead. The lookup will be once with receiving the first odom message.
    use_tf_tree = false

    # Rotation Matrix Wheel_to_IMU
    rotation = [ [1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0] ]
  }

  # Weight for Odometry correction in the IEKF. If not given, the covariance provided in the Odometry message will be used.
  # covariance = 0.001

  # Standard deviation for kinematic constraints in undistortion (lower = stronger constraints)
  kinematic_constraint_std = 0.001

  # @recommended
  # Full3D | Differential | Holonomic2D | Ackermann. Applies kinematic constraints on the IEKF.
  kinematic_model = Full3D
  
  # @recommended
  # Ackermann steering configuration (only used when kinematic_model = Ackermann)
  ackermann.imu_to_rear_axle = [0.0m, 0.0m, 0.0m] # Offset from IMU to rear axle center in imu frame
}

output {

  terminal {
    # @recommended
    # Show the interactive TUI with position, velocity, and performance metrics.
    # Set to false for non-interactive sessions.
    ui = true

    # Show timing metrics and debug output in the terminal.
    debug = false

    # Refresh rate for debug terminal output in frames per second
    debug_fps = 1

    # Debounce time for specific warnings
    warning_debounce = 1s
     
    # @recommended
    # Log level (Trace, Debug, Info, Warn, Error)
    log_level = Warn
  }

  # @recommended
  # Which frame to publish the TF messages to
  global_frame = "lio_init"

  # @recommended
  base_link {
    # If not empty, the odometry and TF will be published for this frame instead of the LiDAR frame.
    frame_id = ""

    from_lidar_transform {
      # Ignore the next manual translation and use the TF tree instead. The lookup will be once with receiving the first sensor message.
      use_tf_tree = false

      # Pose of the LiDAR in the base_link frame.
      translation = [0.0mm, 0.0mm, 0.0mm]
      rotation = [ [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0] ]
    }
  }

  rerun {
    # Logs the registered clouds static in world coordinates so they do not vanish.
    # This will kill your rerun instance very quickly but you'll get some nice visualizations for a few seconds :)
    undistorted_static_world = false
  }

  # Reliable | BestEffort | SystemDefault
  lio_odometry_msg_qos = Reliable

  # Reliable | BestEffort | SystemDefault
  lio_path_msg_qos = SystemDefault

  # Reliable | BestEffort | SystemDefault
  imu_odometry_msg_qos = SystemDefault

  # Reliable | BestEffort | SystemDefault
  imu_path_msg_qos = SystemDefault

  # Reliable | BestEffort | SystemDefault
  imu_ahrs_filtered_msg_qos = Reliable

  # AHRS filtered output relative to the pelorus topic namespace.
  imu_ahrs_filtered_topic = "imu/filtered"

  logger_prefix = "pelorus"
}

parallelism {
  # @recommended
  # Maximum number of CPU threads used for parallel point-cloud processing.
  # Set to `all` to use all available cores or specify a number like `4` for a fixed number.
  # High numbers will result in more substantial CPU load.
  num_threads = 3

  # Force CPU-only processing regardless of available GPU backends.
  cpu_only = false
}
