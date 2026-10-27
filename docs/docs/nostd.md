# Embedded (no_std)

The Rust library builds without the standard library. It needs an allocator and nothing else, so Pelorus runs on microcontrollers and in kernels as well as on a desktop. Only the library does: the command line tool, the ROS 2 node and the C and Python bindings use std.

## Adding the crate

The crate is `pelorus_sdk`, imported as `pelorus`. Turn its default features off to drop std:

```sh
cargo add pelorus_sdk --no-default-features
```

The target needs atomic compare-and-swap. Every Cortex-M3 or newer, every RISC-V core with the A extension such as `riscv32imac`, and every 32-bit or 64-bit application processor has it. Cortex-M0 and RISC-V cores without the A extension, such as `riscv32i` and `riscv32imc`, do not. The binary needs an `#[global_allocator]`.

## Running the odometry

Without std, Pelorus cannot read Ratslang configuration files. The `Odometer` takes a resolved `LioConfig` instead. `LioConfig::defaults()` is the configuration the embedded `default.rl` resolves to. Change its fields for your sensors before you create the odometer:

Create the odometer once and feed it every message the sensor driver receives:

```rust
use pelorus::{ImuMeasurement, LioConfig, Odometer};

/// One message from the sensor driver.
pub enum Input {
    Imu(ImuMeasurement),
    Points {
        sec: i32,
        nanosec: u32,
        points: Vec<[f32; 5]>,
    },
}

pub fn run(
    mut receive: impl FnMut() -> Input,
    mut publish: impl FnMut((i32, u32), [f64; 3], [f64; 4]),
) -> Result<(), pelorus::Error> {
    let mut config = LioConfig::defaults();
    config.map_resolution = 0.5;
    let mut odometer = Odometer::from_config(config, true)?;

    loop {
        match receive() {
            Input::Imu(imu) => odometer.add_imu(imu),
            Input::Points { sec, nanosec, points } => {
                // Skip a scan while an older one still waits, so the queue cannot grow.
                if odometer.pending_scans() < 2 {
                    odometer.add_points_xyzit(sec, nanosec, &points);
                }
            }
        }
        // An IMU message can complete a waiting scan, so drain after every input.
        while let Some(scan) = odometer.next_scan() {
            publish(scan.timestamp(), scan.position(), scan.orientation());
        }
    }
}
```

Everything runs on the calling thread:

- `add_imu` and `add_points_xyzit` only queue their input. A cloud is filtered and downsampled as it is added.
- `next_scan` takes the oldest scan whose IMU interval is complete, deskews it and registers it against the map. It returns once that is done.
- A scan is complete once IMU messages past its end have arrived, so it usually comes out a few IMU messages after its cloud went in.
- At the start, clouds that arrive during the IMU initialization are dropped and the first scans only build the map. `next_scan` returns nothing for about the first second.
- While `next_scan` works, the loop receives nothing. Buffer the network packets in the driver, for example in DMA rings, so none are lost during a registration.

## Bridging a configuration to the device

A configuration written in Ratslang can still reach the device. Resolve it on a host with std, where `pelorus::load_config` reads the file, and serialize the `LioConfig` with serde, for example with [postcard](https://docs.rs/postcard):

```rust
// On the host, with the std feature.
let config = pelorus::load_config(Some("pelorus.rl"))?;
std::fs::write("pelorus.bin", postcard::to_allocvec(&config)?)?;
```

```rust
// On the device.
let config: pelorus::LioConfig = postcard::from_bytes(include_bytes!("pelorus.bin"))?;
```
