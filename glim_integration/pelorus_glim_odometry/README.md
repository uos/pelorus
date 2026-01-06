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

> ½Note: Pelorus needs the ROS communication in `config_sensors` to be set to `reliable` in order to work correctly.

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

    "input_resolution": 0.2,
    "map_resolution": 0.5
  }
}
```

Now run GLIM as usual.
