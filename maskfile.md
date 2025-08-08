# Pelorus Tasks

Quick commands usually needed for developing or to setup Pelorus.

## setup-rust

> Install Rust via rustup.

~~~sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
rustup default stable
rustup update
~~~

## build-release-ros2

> Build for your current system with ROS2 as sensor abstraction layer with recommended settings.

~~~sh
RUSTFLAGS="-C target-cpu=native -C llvm-args=-cost-kind=latency" cargo build --release --features mimalloc,downsample,parallel,ros2c
~~~

## demo-bag

> Downloads test bag file to current directory.

~~~sh
curl -sLo dlg_cut.zip "https://myshare.uni-osnabrueck.de/f/5faf4154af384854ab94?dl=1" && unzip dlg_cut.zip && rm dlg_cut.zip
~~~

## opt

> Maximum speed using default flags.

~~~sh
RUSTFLAGS="-C target-cpu=native -C llvm-args=-cost-kind=latency" cargo run --release -- run -c pelorus.rl
~~~

## opt-ratsync

> Run within Minot toolchain as fast as possible.

~~~sh
RUSTFLAGS="-C target-cpu=native -C llvm-args=-cost-kind=latency" cargo run --release --features offline,parallel -- run -c pelorus.rl
~~~

## opt-ros2

> Maximum speed using default flags.

~~~sh
RUSTFLAGS="-C target-cpu=native -C llvm-args=-cost-kind=latency" cargo run --release --features offline -- run -c pelorus.rl
~~~

## dev

> Start with development flags and logging.

~~~sh
RUST_BACKTRACE=1 RUST_LOG=debug,lighthouse=off,tracing=off,sea=off,h2=off,hyper_util=off,re_sdk=off,re_chunk=off,tower=off,ratpub=off,tonic=off,tonic_web=off cargo run --release --no-default-features --features dev -- run -c pelorus.rl
~~~

