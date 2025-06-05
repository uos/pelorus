# Pelorus

Lidar Inertial Odometry from scratch.

## Usage


Install [Rust](https://www.rust-lang.org/tools/install) first if not done already.

```sh 
# with sourced ROS2
cargo install \
  --git ssh://git@gitlabti.informatik.uni-osnabrueck.de/chsieh/pelorus.git pelorus \
  --locked \
  --no-default-features \
  --features gicp,r2r,pelorus-reg/nn_kdt

# then you can create a default config
pelorus config init
# creates ./lio.toml. change the parameters as you like

pelorus run
```

This just runs the current most stable version: a *not-so-basic* Lidar Odometry. IMU or Odom is ignored at the moment.

## Dev

We use mask to save some time typing out usual commands.
```sh
cargo install mask
```

In the project directory, where the `./maskfile.md` lives:
```sh
mask --help
```

Dev uses rerun for proper visualisations. Run it in background.
```sh
cargo install rerun

tmux new -s rerun
rerun # then detach
```


