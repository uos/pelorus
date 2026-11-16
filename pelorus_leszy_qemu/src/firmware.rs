extern crate alloc;

use alloc::string::String;
use core::arch::{asm, global_asm};
use core::fmt::Write;

use pelorus::{ImuMeasurement, LioConfig, Odometer, PointCloud2Msg, Real, Scan};
use ros_pointcloud2::ros::{HeaderMsg, PointFieldMsg, TimeMsg};
use ros_pointcloud2::{CloudDimensions, Denseness, Endian};

const CONFIG: usize = 0x9FF0_0000;
const STREAM: usize = 0xA000_0000;
const UART: *mut u8 = 0x1000_0000 as *mut u8;
const TEST_DEVICE: *mut u32 = 0x10_0000 as *mut u32;

// Set the stack, enable the FPU, clear .bss and enter Rust. Every trap ends
// the run.
global_asm!(
    r#"
    .section .text.start
    .global _start
_start:
    la sp, _stack_top
    la t0, trap_entry
    csrw mtvec, t0
    li t0, 0x2000
    csrs mstatus, t0
    csrwi fcsr, 0
    la t0, _bss_start
    la t1, _bss_end
1:  bgeu t0, t1, 2f
    sw zero, 0(t0)
    addi t0, t0, 4
    j 1b
2:  call rust_main
3:  j 3b

    .align 2
trap_entry:
    csrr a0, mcause
    csrr a1, mepc
    csrr a2, mtval
    call rust_trap
4:  j 4b
"#
);

unsafe extern "C" {
    static _heap_start: u8;
}

#[global_allocator]
static ALLOCATOR: talc::Talck<talc::locking::AssumeUnlockable, talc::ErrOnOom> =
    talc::Talc::new(talc::ErrOnOom).lock();

/// Interrupts are the only concurrency on a single core, so a critical section
/// clears the machine interrupt enable bit.
struct SingleCore;
critical_section::set_impl!(SingleCore);

unsafe impl critical_section::Impl for SingleCore {
    unsafe fn acquire() -> bool {
        let mstatus: usize;
        unsafe { asm!("csrrci {}, mstatus, 8", out(reg) mstatus) };
        mstatus & 8 != 0
    }

    unsafe fn release(enabled: bool) {
        if enabled {
            unsafe { asm!("csrsi mstatus, 8") };
        }
    }
}

// The `__sync` functions of the embedded docs, which the target turns atomic
// read-modify-write operations into.
macro_rules! sync_functions {
    ($t:ty, $cas:ident, $add:ident, $sub:ident, $and:ident, $or:ident) => {
        #[unsafe(no_mangle)]
        extern "C" fn $cas(ptr: *mut $t, expected: $t, new: $t) -> $t {
            critical_section::with(|_| unsafe {
                let old = ptr.read_volatile();
                if old == expected {
                    ptr.write_volatile(new);
                }
                old
            })
        }
        sync_functions!(@op $t, $add, |a: $t, b: $t| a.wrapping_add(b));
        sync_functions!(@op $t, $sub, |a: $t, b: $t| a.wrapping_sub(b));
        sync_functions!(@op $t, $and, |a: $t, b: $t| a & b);
        sync_functions!(@op $t, $or, |a: $t, b: $t| a | b);
    };
    (@op $t:ty, $name:ident, $f:expr) => {
        #[unsafe(no_mangle)]
        extern "C" fn $name(ptr: *mut $t, value: $t) -> $t {
            critical_section::with(|_| unsafe {
                let old = ptr.read_volatile();
                ptr.write_volatile($f(old, value));
                old
            })
        }
    };
}

sync_functions!(
    u8,
    __sync_val_compare_and_swap_1,
    __sync_fetch_and_add_1,
    __sync_fetch_and_sub_1,
    __sync_fetch_and_and_1,
    __sync_fetch_and_or_1
);
sync_functions!(
    u32,
    __sync_val_compare_and_swap_4,
    __sync_fetch_and_add_4,
    __sync_fetch_and_sub_4,
    __sync_fetch_and_and_4,
    __sync_fetch_and_or_4
);

struct Uart;

impl Write for Uart {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        for byte in s.bytes() {
            unsafe { UART.write_volatile(byte) };
        }
        Ok(())
    }
}

fn exit(code: u32) -> ! {
    let value = if code == 0 {
        0x5555
    } else {
        (code << 16) | 0x3333
    };
    unsafe { TEST_DEVICE.write_volatile(value) };
    loop {
        unsafe { asm!("wfi") };
    }
}

#[unsafe(no_mangle)]
extern "C" fn rust_trap(cause: usize, pc: usize, value: usize) -> ! {
    let _ = writeln!(
        Uart,
        "# trap: mcause {cause:#x} mepc {pc:#x} mtval {value:#x}"
    );
    exit(2)
}

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    let _ = writeln!(Uart, "# panic: {info}");
    exit(3)
}

/// Reads the stream in RAM. QEMU leaves the RAM after the stream zeroed, so a
/// record tag of 0 is its end.
struct Cursor {
    at: usize,
}

impl Cursor {
    fn take(&mut self, n: usize) -> &'static [u8] {
        let bytes = unsafe { core::slice::from_raw_parts(self.at as *const u8, n) };
        self.at += n;
        bytes
    }

    fn array<const N: usize>(&mut self) -> [u8; N] {
        self.take(N).try_into().unwrap()
    }

    fn u8(&mut self) -> u8 {
        self.array::<1>()[0]
    }

    fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.array())
    }

    fn i32(&mut self) -> i32 {
        i32::from_le_bytes(self.array())
    }

    fn f64(&mut self) -> f64 {
        f64::from_le_bytes(self.array())
    }

    fn bytes(&mut self) -> &'static [u8] {
        let n = self.u32() as usize;
        self.take(n)
    }

    fn string(&mut self) -> String {
        String::from(core::str::from_utf8(self.bytes()).unwrap())
    }

    fn imu(&mut self) -> ImuMeasurement {
        let mut imu = ImuMeasurement {
            timestamp_sec: self.i32(),
            timestamp_nanosec: self.u32(),
            orientation: [0.0; 4],
            orientation_covariance: [0.0; 9],
            angular_velocity: [0.0; 3],
            angular_velocity_covariance: [0.0; 9],
            linear_acceleration: [0.0; 3],
            linear_acceleration_covariance: [0.0; 9],
        };
        for value in imu
            .orientation
            .iter_mut()
            .chain(&mut imu.orientation_covariance)
            .chain(&mut imu.angular_velocity)
            .chain(&mut imu.angular_velocity_covariance)
            .chain(&mut imu.linear_acceleration)
            .chain(&mut imu.linear_acceleration_covariance)
        {
            *value = pelorus_core::real(self.f64());
        }
        imu
    }

    fn cloud(&mut self) -> PointCloud2Msg {
        let stamp = TimeMsg {
            sec: self.i32(),
            nanosec: self.u32(),
        };
        let frame_id = self.string();
        let dimensions = CloudDimensions {
            width: self.u32(),
            height: self.u32(),
        };
        let field_count = self.u32();
        let fields = (0..field_count)
            .map(|_| PointFieldMsg {
                name: self.string().into(),
                offset: self.u32(),
                datatype: self.u8(),
                count: self.u32(),
            })
            .collect();
        let endian = if self.u8() != 0 {
            Endian::Big
        } else {
            Endian::Little
        };
        let point_step = self.u32();
        let row_step = self.u32();
        let data = self.bytes().to_vec();
        let dense = if self.u8() != 0 {
            Denseness::Dense
        } else {
            Denseness::Sparse
        };
        PointCloud2Msg {
            header: HeaderMsg {
                seq: 0,
                stamp,
                frame_id,
            },
            dimensions,
            fields,
            endian,
            point_step,
            row_step,
            data,
            dense,
        }
    }
}

#[unsafe(no_mangle)]
extern "C" fn rust_main() -> ! {
    unsafe {
        let base = &raw const _heap_start as *mut u8;
        let span = talc::Span::from_base_size(base, CONFIG - base as usize);
        ALLOCATOR.lock().claim(span).unwrap();
    }
    let config = unsafe { core::slice::from_raw_parts(CONFIG as *const u8, STREAM - CONFIG) };
    let (config, _): (LioConfig, _) = postcard::take_from_bytes(config)
        .unwrap_or_else(|e| panic!("the configuration does not decode in this precision: {e}"));
    let mut odometer = Odometer::from_config(config, true)
        .unwrap_or_else(|_| panic!("the odometer does not start"));

    let mut stream = Cursor { at: STREAM };
    assert!(
        stream.take(16) == b"PELORUS-STREAM-1",
        "no input stream at {STREAM:#x}"
    );
    // The stream's own configuration is replaced by the one at CONFIG.
    stream.bytes();

    let mut out = Uart;
    let mut scans = 0_usize;
    loop {
        match stream.u8() {
            0 => break,
            1 => odometer.add_imu(stream.imu()),
            2 => {
                odometer.add_pointcloud2(stream.cloud());
            }
            tag => panic!("unknown record tag {tag}"),
        }
        while let Some(scan) = odometer.next_scan() {
            write_scan(&mut out, &scan);
            scans += 1;
        }
    }
    let precision = if pelorus_core::REAL_IS_F32 {
        "f32"
    } else {
        "f64"
    };
    let _ = writeln!(
        out,
        "# riscv32imf no_std {precision}: {scans} scans registered"
    );
    exit(0)
}

/// The line of `pelorus-nostd-check run` for a scan.
fn write_scan(out: &mut Uart, scan: &Scan<'_>) {
    let (sec, nanosec) = scan.timestamp();
    let [x, y, z] = scan.position();
    let [qw, qx, qy, qz] = scan.orientation();
    let (linear, angular) = scan.twist();
    let state = fnv1a(
        scan.pose_covariance()
            .iter()
            .chain(scan.twist_covariance())
            .chain(&linear)
            .chain(&angular)
            .chain(&scan.accel_bias())
            .chain(&scan.gyro_bias())
            .map(|value| bits(*value)),
    );
    let points = fnv1a(scan.points().iter().flat_map(|p| {
        [p.x, p.y, p.z, p.intensity, p.timestamp].map(|value| u64::from(value.to_bits()))
    }));
    let _ = writeln!(
        out,
        "{} {sec}.{nanosec:09} {x:?} {y:?} {z:?} {qw:?} {qx:?} {qy:?} {qz:?} {state:016x} {points:016x} {}",
        scan.scan_id(),
        scan.points().len(),
    );
}

fn bits(value: Real) -> u64 {
    if pelorus_core::REAL_IS_F32 {
        u64::from(pelorus_core::single(value).to_bits())
    } else {
        pelorus_core::double(value).to_bits()
    }
}

fn fnv1a(values: impl Iterator<Item = u64>) -> u64 {
    values.fold(0xcbf2_9ce4_8422_2325, |hash, value| {
        value.to_le_bytes().iter().fold(hash, |hash, byte| {
            (hash ^ u64::from(*byte)).wrapping_mul(0x0100_0000_01b3)
        })
    })
}
