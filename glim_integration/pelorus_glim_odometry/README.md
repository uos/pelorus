# pelorus_glim_odometry

GLIM odometry backend plugin that runs Pelorus as the odometry estimation module.

This is an alternative to GLIM's built-in `odometry_estimation_cpu/gpu/ct` modules.

## Quick start

You need:

- gtsam
- gtsam_points
- glim
- Pelorus

```sh
cd pelorus/glim_integration/pelorus_glim_odometry
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make
sudo -E make install
```

## Enable Pelorus odometry in GLIM

> ½Note: Pelorus needs the ROS communication in `config_sensors` to be set to `reliable` in order to work correctly.

1. Create a config file (e.g., `config_odometry_pelorus.json`) in your GLIM config root.
2. Set `so_name` to:
   - Linux: `libodometry_estimation_pelorus.so`
   - macOS: `libodometry_estimation_pelorus.dylib`
3. In GLIM's `config.json`, set:
   - `"config_odometry": "config_odometry_pelorus.json"`
4. Run GLIM as usual

## Configuration


Example with all options:
```json
{
  "odometry_estimation": {
    "so_name": "libodometry_estimation_pelorus.so",
    "use_glim_config": true,
  }
}
```

- `so_name` (required): Path or name of the plugin library
- `pelorus_rl_path` (optional): Path to Pelorus config file (empty = defaults)
- `use_glim_config` (optional): If `true` and `pelorus_rl_path` is empty, the plugin will generate a Pelorus config based on GLIM's settings.
