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
> Three builds of `pelorus-nostd-check` replay each bag on one thread. The std build and the no_std build differ in the float maths only, because no_std takes libm and faer's portable kernels. The third build runs the no_std code with the maths of the std build and has to match the std build bit for bit. A fourth runs the no_std code in `f32`.
>
> The `f32` build may not compute a single `f64`. `pelorus_nostd_probe` calls every entry point of the library for the Cortex-M4F, whose FPU is single precision, so each `f64` operation becomes a call into compiler-builtins. Linked with only what the entry points reach, the `f32` probe must contain none of them and the `f64` probe must. The `f32` probe also builds for an RV32IMF core without the A and C extensions, from the target file in `pelorus_leszy_qemu`, and may contain no `f64` routine and no atomic or compressed instruction there. With `qemu-system-riscv32` installed, `pelorus_leszy_qemu` then replays one bag on such an emulated core and compares it with the no_std build on the host.

~~~sh
set -eu
out=target/nostd-check
rustup target add thumbv7em-none-eabihf
cargo check -p pelorus_sdk --no-default-features --target thumbv7em-none-eabihf
cargo build --release -p pelorus_nostd_check --features convert --target-dir "$out/std"
cargo build --release -p pelorus_nostd_check --target-dir "$out/no-std"
cargo build --release -p pelorus_nostd_check --features system-maths --target-dir "$out/system-maths"
cargo build --release -p pelorus_nostd_check --features f32 --target-dir "$out/no-std-f32"
cargo build --release -p pelorus_nostd_check --features convert,f32 --target-dir "$out/std-f32"
if cargo tree -p pelorus_nostd_check -e features --prefix none | grep -F 'feature "std"'; then
  echo "a dependency of the no_std build turns on std" >&2
  exit 1
fi

# Soft-float routines of compiler-builtins that take or return an f64.
doubles='^(__aeabi_(d[a-z0-9]*|f2d|[iu]2d|[iu]l2d)|__(add|sub|mul|div|neg)df3|__(extend|trunc)[a-z]*df2|__float[a-z]*df|__fix[a-z]*df[a-z]*|__(eq|ne|lt|le|gt|ge|un)df2|__powidf2)$'
lld="$(rustc --print sysroot)/lib/rustlib/$(rustc -vV | sed -n 's/^host: //p')/bin/rust-lld"
riscv=pelorus_leszy_qemu/riscv32imf-unknown-none-elf.json
# probe <"" or 32> [linker arguments]: the f64 routines in the linked probe.
# PROBE_TARGET picks the target, the Cortex-M4F unless set.
probe() {
  target=${PROBE_TARGET:-thumbv7em-none-eabihf}
  name=$(basename "$target" .json)
  dir="$out/probe-$name$1"
  features=${1:+--features=f32}
  shift
  case $target in
    *.json) cargo="cargo +nightly -Zbuild-std=core,alloc -Zjson-target-spec" ;;
    *) cargo=cargo ;;
  esac
  CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1 \
    CARGO_PROFILE_RELEASE_PANIC=abort CARGO_PROFILE_RELEASE_STRIP=none \
    $cargo build --release -p pelorus_nostd_probe --target "$target" \
    $features --target-dir "$dir"
  # The atomics of the RISC-V target are calls into the firmware.
  "$lld" -flavor gnu --gc-sections --entry probe_new \
    -u probe_imu -u probe_odom -u probe_xyzit -u probe_xyzt -u probe_raw -u probe_drain -u probe_free \
    --defsym host_alloc=0 --defsym host_free=0 --defsym host_publish=0 \
    --unresolved-symbols=ignore-all \
    "$@" "$dir/$name/release/libpelorus_nostd_probe.a" -o "$dir/probe.elf"
  readelf -sW "$dir/probe.elf" | awk '{ print $8 }' | grep -E "$doubles" || true
}
if [ -z "$(probe "")" ]; then
  echo "the f64 probe contains no f64 routine, so the check cannot see any" >&2
  exit 1
fi
found=$(probe 32)
if [ -n "$found" ]; then
  echo "the f32 build computes in f64:" >&2
  probe 32 $(printf -- '--why-live=%s ' $found) | grep -v '(no section)' >&2
  exit 1
fi
echo "f32: no f64 operation"

# RV32IMF without the A and C extensions, built with nightly from the target file.
rustup component add rust-src --toolchain nightly
found=$(PROBE_TARGET=$riscv probe 32)
if [ -n "$found" ]; then
  echo "the f32 build for RV32IMF computes in f64:" >&2
  PROBE_TARGET=$riscv probe 32 $(printf -- '--why-live=%s ' $found) | grep -v '(no section)' >&2
  exit 1
fi
elf="$out/probe-riscv32imf-unknown-none-elf32/probe.elf"
# Compressed instructions are the ones encoded in 16 bits.
if llvm-objdump -d --mattr=+a,+c "$elf" | grep -qE '\s(lr|sc|amo[a-z]+)\.w|^ *[0-9a-f]+:\s[0-9a-f]{4}\s'; then
  echo "the RV32IMF probe contains atomic or compressed instructions" >&2
  exit 1
fi
echo "RV32IMF f32: no f64 operation, no atomic or compressed instruction"

check="release/pelorus-nostd-check"
mkdir -p "$out/runs"
while read -r name config lidar imu; do
  stream="$out/runs/$name.stream"
  "$out/std/$check" convert "web/configs/$config" "web/demo/$name.mcap" "$lidar" "$imu" "$stream"
  for build in std no-std system-maths; do
    "$out/$build/$check" run "$stream" "$out/runs/$name.$build.scans"
  done
  "$out/std-f32/$check" config "web/configs/$config" "$out/runs/$name.f32.config"
  "$out/no-std-f32/$check" run "$stream" "$out/runs/$name.no-std-f32.scans" "$out/runs/$name.f32.config"
  "$out/std/$check" compare "$out/runs/$name.std.scans" "$out/runs/$name.system-maths.scans"
  "$out/std/$check" compare "$out/runs/$name.std.scans" "$out/runs/$name.no-std.scans" || true
  "$out/std/$check" compare "$out/runs/$name.no-std.scans" "$out/runs/$name.no-std-f32.scans" || true
done <<RUNS
helipr_bridge02_20s pelorus-helipr-ouster.rl /ouster/points /imu/data_raw
keble_college_02_30s pelorus-spires.rl /hesai/pandar /alphasense_driver_ros/imu
grand_tour_eth1_30s pelorus-grandtour.rl /boxi/hesai/points /boxi/stim320/imu
RUNS

# The f32 build on an emulated RV32IMF core without the A and C extensions.
if ! command -v qemu-system-riscv32 >/dev/null; then
  echo "qemu-system-riscv32 is missing, so the RV32IMF run is skipped" >&2
  exit 0
fi
name=keble_college_02_30s
cargo +nightly build --release -Zbuild-std=core,alloc -Zbuild-std-features=compiler-builtins-mem \
  -Zjson-target-spec -p pelorus_leszy_qemu --features f32 --target "$riscv" --target-dir "$out/qemu"
log="$out/runs/$name.qemu.log"
if ! qemu-system-riscv32 -M virt -m 1536M -bios none -display none -serial "file:$log" \
  -cpu rv32,a=false,c=false,d=false,zaamo=false,zalrsc=false,zawrs=false,zca=false,zcf=false,zcd=false,zcb=false,zcmp=false,zcmt=false \
  -kernel "$out/qemu/riscv32imf-unknown-none-elf/release/pelorus-leszy-qemu" \
  -device "loader,file=$out/runs/$name.f32.config,addr=0x9ff00000" \
  -device "loader,file=$out/runs/$name.stream,addr=0xa0000000"; then
  grep '^#' "$log" >&2
  echo "the RV32IMF run failed" >&2
  exit 1
fi
grep '^#' "$log"
grep -v '^#' "$log" > "$out/runs/$name.qemu-f32.scans"
"$out/std/$check" compare "$out/runs/$name.no-std.scans" "$out/runs/$name.qemu-f32.scans" || true
~~~
