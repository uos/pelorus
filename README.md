# ½[lio] - LiDAR Inertial Odometry

## Usage
See
```sh
cargo run -- --help
```

## Dev
[lio] is developed with the nightly toolchain. On debug builds, the following settings are optimized for compilation speed.

```sh
rustup default nightly
rustup update
rustup component add rustc-codegen-cranelift-preview --toolchain nightly
RUSTFLAGS="-Z threads=8" cargo build
```

or export `RUSTFLAGS` to your environment.

Now you can set a general config for cargo to use the cranelift backend for debug builds.

`~/.cargo/config`
```sh
[unstable]
codegen-backend = true

[profile.dev]
codegen-backend = "cranelift"
```
