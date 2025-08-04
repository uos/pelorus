# Pelorus

Lidar Inertial Odometry.

## Usage

Install [Rust](https://www.rust-lang.org/tools/install) first if not done already.


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
For remote rerun instances, set the `RERUN_SERVER_ADDR` environment variable. The default is `127.0.0.1:9876`.

