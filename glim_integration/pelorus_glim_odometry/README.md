# pelorus_glim_odometry

GLIM odometry backend plugin that runs Pelorus as the odometry estimation module.

```sh
cd pelorus/glim_integration/pelorus_glim_odometry
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make
sudo -E make install
```

## Enable Pelorus odometry in GLIM

> Note: Pelorus needs GLIM's IMU subscription in `config_ros` to use reliable
> QoS (`glim_ros.imu_qos.reliability = "reliable"`).

In GLIM's `config.json`, set:
 - `"config_odometry": "config_odometry_pelorus.json"`

Create the config file in your GLIM config root:

```json
{
  "odometry_estimation": {
    "so_name": "libodometry_estimation_pelorus.so",
    "use_glim_config": true,

    // optional, overrides all glim configs
    //"pelorus_rl_path": "path/to/your/pelorus.rl",

    "query_resolution_factor": 0.5,
    "map_resolution": 0.5
  }
}
```

`query_resolution_factor` is forwarded unchanged to Pelorus. The resulting
query voxel size is `query_resolution_factor * map_resolution`.

The generated Pelorus overlay selects backend preprocessing. CUDA, Vulkan, or
Metal is used when the corresponding backend is built and available. Otherwise
Pelorus falls back to CPU preprocessing.

Pelorus accelerometer and gyroscope bias estimates are forwarded to GLIM's
`EstimationFrame::imu_bias` in `[accelerometer, gyroscope]` order.

Now run GLIM as usual.
