# pelorus_glim_frontend

> [!WARNING]
> This module does not work correctly at the moment because of some internal differences with transforms. The problem is related to the untested behavior when the IMU extrinsic rotation matrix is not orthogonal.

GLIM extension module that feeds Pelorus and injects **additional odometry constraints** into GLIM's odometry factor graph (similar to `orb_slam_frontend` in `glim_ext`).

This **does not replace** GLIM odometry. It only adds extra `BetweenFactor<Pose3>` constraints.

## Quick start


### Build and install

```sh
cd pelorus/glim_integration/pelorus_glim_frontend
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
make
sudo -E make install
```

### Enable the extension in GLIM

Add the plugin library name to `glim_ros.extension_modules` (in your GLIM config): `libpelorus_glim_frontend.so`

## Configuration

The extension reads parameters from a config file named `config_pelorus_frontend.json` in your GLIM config root:

```json
{
  "pelorus_frontend": {
    // optional
    "pelorus_rl_path": "/abs/path/to/pelorus.rl",
    // if config not set
    "use_glim_config": true,

    "query_resolution_factor": 0.5,
    "map_resolution": 0.5,

    // untested, see above warning
    "stamp_match_tolerance_sec": 0.02,
    "between_sigma_trans": 0.2,
    "between_sigma_rot_rad": 0.15,
    "max_frame_buffer": 400
  }
}
```
