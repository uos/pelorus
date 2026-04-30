#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;

use core::panic::PanicInfo;

#[cfg(feature = "alloc")]
use core::{
    alloc::{GlobalAlloc, Layout},
    ptr::null_mut,
    sync::atomic::{AtomicUsize, Ordering},
};

#[panic_handler]
fn panic(_info: &PanicInfo<'_>) -> ! {
    loop {}
}

#[unsafe(no_mangle)]
pub extern "C" fn rust_eh_personality() {}

#[cfg(feature = "alloc")]
const NO_STD_ALLOC_HEAP_SIZE: usize = 1024 * 1024;

#[cfg(feature = "alloc")]
#[repr(align(16))]
struct NoStdHeap([u8; NO_STD_ALLOC_HEAP_SIZE]);

#[cfg(feature = "alloc")]
static mut NO_STD_HEAP: NoStdHeap = NoStdHeap([0; NO_STD_ALLOC_HEAP_SIZE]);

#[cfg(feature = "alloc")]
struct NoStdBumpAllocator {
    cursor: AtomicUsize,
}

#[cfg(feature = "alloc")]
impl NoStdBumpAllocator {
    fn reset(&self) {
        self.cursor.store(0, Ordering::SeqCst);
    }
}

#[cfg(feature = "alloc")]
unsafe impl GlobalAlloc for NoStdBumpAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let align = layout.align();
        let size = layout.size();
        let mut current = self.cursor.load(Ordering::Relaxed);

        loop {
            let aligned = (current + align - 1) & !(align - 1);
            let Some(next) = aligned.checked_add(size) else {
                return null_mut();
            };
            if next > NO_STD_ALLOC_HEAP_SIZE {
                return null_mut();
            }

            if self
                .cursor
                .compare_exchange_weak(current, next, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                // SAFETY: aligned index is bounded by NO_STD_ALLOC_HEAP_SIZE and within static heap.
                let base = unsafe { core::ptr::addr_of_mut!(NO_STD_HEAP.0).cast::<u8>() };
                // SAFETY: pointer arithmetic stays within static heap.
                return unsafe { base.add(aligned) };
            }
            current = self.cursor.load(Ordering::Relaxed);
        }
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {}
}

#[cfg(feature = "alloc")]
#[global_allocator]
static NO_STD_ALLOCATOR: NoStdBumpAllocator = NoStdBumpAllocator {
    cursor: AtomicUsize::new(0),
};

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct pelorus_pico_point_t {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub intensity: f32,
    pub timestamp: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct pelorus_odometry_t {
    pub timestamp: f64,
    pub position: [f64; 3],
    pub orientation: [f64; 4], // [w, x, y, z]
    pub linear_velocity: [f64; 3],
    pub valid: i32,
}

unsafe extern "C" {
    fn pelorus_pico_c_init();
    fn pelorus_pico_c_add_imu(
        timestamp: f64,
        gyro_x: f32,
        gyro_y: f32,
        gyro_z: f32,
        accel_x: f32,
        accel_y: f32,
        accel_z: f32,
    ) -> pelorus_odometry_t;
    fn pelorus_pico_c_add_cloud(
        pts: *const pelorus_pico_point_t,
        n: i32,
        cloud_start_time: f64,
        cloud_end_time: f64,
    ) -> pelorus_odometry_t;
}

#[unsafe(no_mangle)]
pub extern "C" fn pelorus_pico_init() {
    #[cfg(feature = "alloc")]
    NO_STD_ALLOCATOR.reset();

    // SAFETY: forwards to linked pico C implementation with matching ABI.
    unsafe { pelorus_pico_c_init() };
}

#[unsafe(no_mangle)]
pub extern "C" fn pelorus_pico_add_imu(
    timestamp: f64,
    gyro_x: f32,
    gyro_y: f32,
    gyro_z: f32,
    accel_x: f32,
    accel_y: f32,
    accel_z: f32,
) -> pelorus_odometry_t {
    // SAFETY: forwards to linked pico C implementation with matching ABI.
    unsafe { pelorus_pico_c_add_imu(timestamp, gyro_x, gyro_y, gyro_z, accel_x, accel_y, accel_z) }
}

#[unsafe(no_mangle)]
pub extern "C" fn pelorus_pico_add_cloud(
    pts: *const pelorus_pico_point_t,
    n: i32,
    cloud_start_time: f64,
    cloud_end_time: f64,
) -> pelorus_odometry_t {
    // SAFETY: forwards to linked pico C implementation with matching ABI.
    unsafe { pelorus_pico_c_add_cloud(pts, n, cloud_start_time, cloud_end_time) }
}
