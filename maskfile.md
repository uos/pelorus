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
RUSTFLAGS="-C target-cpu=native -C llvm-args=-cost-kind=latency" cargo build --no-default-features --release --features mimalloc,downsample,parallel,ros2c
~~~

## demo-bag

> Downloads test bag file to current directory.

~~~sh
curl -sLo dlg_cut.zip "https://myshare.uni-osnabrueck.de/f/5faf4154af384854ab94?dl=1" && unzip dlg_cut.zip && rm dlg_cut.zip
~~~

## opt

> Maximum speed using default flags.

~~~sh
RUSTFLAGS="-C target-cpu=native -C llvm-args=-cost-kind=latency" cargo run --release -- online -c pelorus.rl
~~~

## opt-ratsync

> Run within Minot toolchain as fast as possible.

~~~sh
RUSTFLAGS="-C target-cpu=native -C llvm-args=-cost-kind=latency" cargo run --release --no-default-features --features mimalloc,offline,parallel -- online -rc pelorus.rl
~~~

## opt-ros2

> Maximum speed using default flags.

~~~sh
RUSTFLAGS="-C target-cpu=native -C llvm-args=-cost-kind=latency" cargo run --release --no-default-features --features mimalloc,parallel,downsample,ros2 -- online -c pelorus.rl
~~~

## dev

> Start with development flags and logging.

~~~sh
RUST_BACKTRACE=1 RUST_LOG=debug,minot=off,rat=off,tracing=off,mt_sea=off,h2=off,hyper_util=off,re_sdk=off,re_chunk=off,tower=off,ratpub=off,tonic=off,tonic_web=off cargo mommy run --profile dbg-release --no-default-features --features dev -- online -rc pelorus.rl
~~~


## docs

> Generate the configuration pages from pelorus_core/default.rl, then build the site into docs/site.

~~~sh
cargo run -p pelorus_docs
cd docs && zensical build
~~~

## docs-serve

> Same as `docs`, but serves the site locally with live reload.

~~~sh
cargo run -p pelorus_docs
cd docs && zensical serve
~~~

## nostd-check

> Check that the no_std library computes what the std library computes, on the demo bags in web/demo (create them with `web/make-demo.sh`).
>
> Three builds of `pelorus-nostd-check` replay each bag on one thread. The std build and the no_std build differ in the float maths only, because no_std takes libm and faer's portable kernels. The third build runs the no_std code with the maths of the std build and has to match the std build bit for bit.

~~~sh
set -eu
out=target/nostd-check
rustup target add thumbv7em-none-eabihf
cargo check -p pelorus_sdk --no-default-features --target thumbv7em-none-eabihf
cargo build --release -p pelorus_nostd_check --features convert --target-dir "$out/std"
cargo build --release -p pelorus_nostd_check --target-dir "$out/no-std"
cargo build --release -p pelorus_nostd_check --features system-maths --target-dir "$out/system-maths"
if cargo tree -p pelorus_nostd_check -e features --prefix none | grep -F 'feature "std"'; then
  echo "a dependency of the no_std build turns on std" >&2
  exit 1
fi

check="release/pelorus-nostd-check"
mkdir -p "$out/runs"
while read -r name config lidar imu; do
  stream="$out/runs/$name.stream"
  "$out/std/$check" convert "web/configs/$config" "web/demo/$name.mcap" "$lidar" "$imu" "$stream"
  for build in std no-std system-maths; do
    "$out/$build/$check" run "$stream" "$out/runs/$name.$build.scans"
  done
  "$out/std/$check" compare "$out/runs/$name.std.scans" "$out/runs/$name.system-maths.scans"
  "$out/std/$check" compare "$out/runs/$name.std.scans" "$out/runs/$name.no-std.scans" || true
done <<RUNS
helipr_bridge02_20s pelorus-helipr-ouster.rl /ouster/points /imu/data_raw
keble_college_02_30s pelorus-spires.rl /hesai/pandar /alphasense_driver_ros/imu
grand_tour_eth1_30s pelorus-grandtour.rl /boxi/hesai/points /boxi/stim320/imu
RUNS
~~~
