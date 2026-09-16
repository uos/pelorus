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

sensor_drivers {
  ouster {
    # Sensor hostname or address used only to read metadata over HTTP. The
    # sensor must already be configured to send UDP data to this host.
    hostname = "os.local"

    # Optional Ouster metadata JSON file. Empty fetches live metadata from
    # hostname; a file is useful for deterministic replay and offline testing.
    metadata_path = ""

    # Local receive address and the sensor's configured UDP destination ports.
    bind_address = "0.0.0.0"
    lidar_port = 7502
    imu_port = 7503

    # Optional sender filter. Empty accepts Ouster packets from any address.
    device_ip = ""

    # Ouster points are transformed into the forward-left-up Sensor frame.
    lidar_frame_id = "os_sensor"
    imu_frame_id = "os_imu"

    timestamp {
      # Ptp | Host. Ptp requires TIME_FROM_PTP_1588 sensor configuration and
      # validates sensor timestamps against host UTC. Host preserves relative
      # sensor timing while anchoring each scan or IMU packet at receive time.
      mode = Ptp
      max_host_skew = 20ms
      startup_timeout = 15s
    }
  }
}

lidar {
  driver {
    # Disabled | Velodyne | Ouster
    # Disabled keeps the existing external point-cloud input configured by lidar.topic.
    # Velodyne and Ouster enable in-process UDP drivers and require Pelorus to be
    # built with their matching feature (or the aggregate `sensor-drivers` feature).
    type = Disabled

    velodyne {
      # Vlp16 | Vlp32c | PuckLite | PuckHiRes
      # The configured model must match the product identifier in received packets.
      model = Vlp32c

      # Local IPv4 or IPv6 address on which to receive UDP packets. "0.0.0.0"
      # listens on every IPv4 interface.
      bind_address = "0.0.0.0"

      # UDP data-packet port. Must be in 1..=65535 and differ from position_port.
      data_port = 2368

      # UDP position/NMEA packet port. Must be in 1..=65535 and differ from data_port.
      position_port = 8308

      # Optional sender-address filter. Empty accepts packets from any device;
      # otherwise this must be an IPv4 or IPv6 address, for example "192.168.1.201".
      device_ip = ""

      # Frame assigned to native scans. Coordinates follow ROS REP-103:
      # +X forward, +Y left, +Z up.
      frame_id = "velodyne"

      # Revolution boundary in radians, measured in the sensor azimuth convention.
      # A scan is emitted whenever azimuth crosses this angle. Expected range is
      # 0.0 (inclusive) to 2*pi (exclusive). Ratslang currently has no angle quantity type.
      cut_angle = 0.0

      timestamp {
        # Sensor | Host
        # Sensor prefers UTC reconstructed from valid position-packet NMEA. If no
        # valid NMEA anchor is available and allow_host_hour_anchor is true, the
        # packet's microseconds-since-hour are combined with the nearest host UTC hour.
        # Host timestamps packets when they are received and stamps a scan from its
        # first packet. Host is useful when sensor UTC is unavailable, but includes
        # network and scheduling latency. Both modes keep per-point firing offsets.
        mode = Sensor

        # In Sensor mode, permit nearest-host-hour reconstruction when position/NMEA
        # UTC is unavailable. The host clock must be synchronized to the sensor's
        # time reference. This option has no effect in Host mode.
        allow_host_hour_anchor = true

        # Maximum accepted absolute difference between reconstructed sensor UTC and
        # host UTC. Must be positive. Used by Sensor mode; Host mode already uses host time.
        max_host_skew = 20ms

        # Maximum positive duration to wait for usable packets/timestamps during
        # startup. Pelorus fails startup instead of silently changing clock modes.
        startup_timeout = 15s
      }
    }
  }
  # @recommended
  topic = "/ouster/points"

  # PointCloud2 | Compressed. Selects the input message type when using mt_pubsub.
  # Compressed requires the default-enabled `compressed-pointcloud` Cargo feature.
  # Other communication backends always subscribe to PointCloud2.
  subscriber_mode = PointCloud2

  # @recommended
  # Use the vendor specific cloud format timestamps for deskewing
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
  driver {
    # Disabled | Xsens | Ouster
    # Disabled keeps the existing external IMU input configured by imu.topic.
    # Xsens and Ouster enable in-process drivers and require Pelorus to be built with
    # their matching feature (or the aggregate `sensor-drivers` feature).
    type = Disabled

    xsens {
      # Serial device path. A stable /dev/serial/by-id/... path is recommended over
      # /dev/ttyUSB0 because USB enumeration order can change between boots.
      port = "/dev/ttyUSB0"

      # Serial baud rate in bits per second. Must be positive and must match the device.
      baudrate = 115200

      # Frame assigned to native IMU measurements.
      frame_id = "imu"

      # Requested MTData2 output frequency in Hz when configure_device is true.
      # Must be in 1..=2000 and supported by the connected Xsens model and baud rate.
      output_rate = 400

      # Enu | Ned | Nwu
      # World-coordinate convention requested for orientation output:
      # Enu = East-North-Up, Ned = North-East-Down, Nwu = North-West-Up.
      # Acceleration and rate-of-turn remain expressed in the IMU body frame.
      coordinate_system = Enu

      # If true, put the device in configuration mode, request UTC, calibrated
      # acceleration, rate-of-turn, and quaternion MTData2 fields at output_rate,
      # then return it to measurement mode. If false, the device must already emit
      # all required fields in compatible precision and coordinates.
      configure_device = true

      timestamp {
        # Utc | Host
        # Utc requires a present and valid MTData2 UTC field and rejects measurements
        # whose clock differs from the host beyond max_host_skew. Host timestamps a
        # complete decoded frame at receipt time and is useful before GNSS/UTC becomes
        # valid, but includes serial transport and scheduling latency.
        mode = Utc

        # Maximum accepted absolute difference between Xsens UTC and host UTC.
        # Must be positive. Used by Utc mode; Host mode already uses host time.
        max_host_skew = 20ms

        # Maximum positive duration to wait for usable MTData2 input during startup.
        # Pelorus fails startup instead of silently changing from Utc to Host.
        startup_timeout = 15s
      }
    }
  }

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

    # Keep refining the extrinsic in the filter instead of holding it at the
    # values above, which then serve as the starting point.
    #
    # This is refinement, not calibration: the filter needs a starting guess
    # close enough to converge, and the extrinsic is only observable while the
    # platform is actually being excited. On a robot that mostly drives
    # straight it will drift on the unobservable directions rather than
    # improve. Leave it off unless you know the motion excites all axes, and
    # check the result against what you started with.
    estimate_online = false
  }

  # Ignore the IMU entirely and run the odometry on the Lidar alone, with the
  # filter propagating at a constant twist between scans.
  #
  # This exists to produce a trajectory for calibrating the IMU against, which
  # is circular if the IMU helped produce it. It is not a navigation mode: with
  # no IMU the gravity direction is unobservable, so roll and pitch are
  # relative to wherever the sensor started rather than to level, and there is
  # nothing to carry the estimate through a stretch where the geometry is
  # degenerate. The covariances below stop describing sensor noise and start
  # describing how much the constant-twist model can be trusted, so raise them.
  lidar_only = false

  # CreateOrientationOnly | FilterAccelerationAndCreateOrientation | FilterAccelerationOnly | Disable (recommended when using on-device filtering in the driver)
  ahrs_filtering.method = Disable

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

    # Kalibr-format IMU calibration to apply to every sample: scale factors,
    # axis misalignment and biases. Relative path "./<path>" from this config
    # file, or absolute "/<path>".
    #
    # Produced by https://github.com/stelzo/dedrunk, or by Kalibr itself with
    # --imu-models scale-misalignment. Record the bag for it with
    # `pelorus online raw imu intrinsics --output-rosbag <dir>`.
    parameter_file = ""

    # Take the four covariances above from parameter_file instead of the
    # constants, deriving them from its measured noise densities and bias
    # random walks. Needs a file that carries the stochastic half too, i.e.
    # one from `dedrunk merge`; `dedrunk intrinsics` alone does not.
    #
    # Off by default because it moves numbers the odometry is tuned around.
    # Both the derived and the configured values are logged at startup, so the
    # ratio for this sensor is visible before you commit to it.
    noise_from_file = false

    # Seconds the IMU clock runs ahead of the Lidar clock, as reported by
    # `dedrunk extrinsics`. Pelorus subtracts it from every IMU timestamp, so
    # paste the number exactly as dedrunk prints it, sign included.
    #
    # This is a constant skew between the two sensors' clocks, not jitter. It
    # matters most while turning: at 1 rad/s, 10 ms of skew smears the scan by
    # about half a degree, which the registration then has to absorb.
    time_offset = 0s

    # Multiplier on the measured densities before they become covariances.
    #
    # Allan variance is measured on a completely still sensor at constant
    # temperature. A moving robot has vibration, thermal drift and timing
    # jitter that dominate it, so the honest noise level in operation is
    # higher than the one on the bench, and no amount of measuring will tell
    # you by how much. The Kalibr wiki suggests 10x or more for MEMS parts.
    noise_inflation = 10.0
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

  # @recommended
  # Lidar | ImuAndLidar. ImuAndLidar adds low-latency IMU predictions between Lidar corrections.
  odometry_rate = Lidar

  # PointCloud2 | Compressed. Selects the registered-cloud message type when using mt_pubsub.
  # Compressed requires the default-enabled `compressed-pointcloud` Cargo feature.
  # The topic name stays "registered" in either mode.
  publisher_mode = PointCloud2

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
    # If not empty, the odometry and TF will be published for this frame instead of the Lidar frame.
    frame_id = ""

    from_lidar_transform {
      # Ignore the next manual translation and use the TF tree instead. The lookup will be once with receiving the first sensor message.
      use_tf_tree = false

      # Pose of the Lidar in the base_link frame.
      translation = [0.0mm, 0.0mm, 0.0mm]
      rotation = [ [1.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0],
                   [0.0, 0.0, 1.0] ]
    }
  }

  rerun {
    # Address of an already running Rerun server. Visualization is activated with --rerun.
    server_addr = "127.0.0.1:9876"

    # Periodic snapshot of the whole registration map, colored by relevance score.
    # Off by default. Large maps can be expensive to render.
    map {
      enabled = false
      # Minimum time between snapshots.
      publish_interval = 1s
      # Ellipsoid size, scaled by each voxel's covariance.
      visual_scale = 3.0
    }
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
  # Maximum number of CPU threads used for parallel processing.
  # Set to `all` to use all available cores or specify a number like `4` for a fixed number.
  # High numbers will result in more substantial CPU load.
  num_threads = 3

  # Force CPU-only processing regardless of available GPU backends.
  cpu_only = false

  # @recommended
  # Auto benchmarks both modes at runtime. Other options overwrite automatic decisions.
  # This is recommended when measuring computational footprint or runtime for scientific papers.
  # Setting num_threads to 1 will still use rayon implementations. To truly run single-core,
  # disable the rayon feature or set various pipeline steps to SingleCore with this setting.
  # Auto | SingleCore | Rayon
  execution_mode = Auto

  # Number of standard errors by which one CPU execution mode must beat the other
  # before Pelorus decides the execution mode for the next phase.
  decision_threshold_stddevs = 2.0

  # Number of measurements collected for each mode before a decision is allowed.
  decision_samples_per_mode = 20
}
