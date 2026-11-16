use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;
use core::alloc::{GlobalAlloc, Layout};
use core::ptr::read_unaligned;
use core::slice::from_raw_parts;

use pelorus::{ImuMeasurement, LioConfig, Odometer, Pcloud2Field, RawPointCloud};

unsafe extern "C" {
    fn host_alloc(size: usize, align: usize) -> *mut u8;
    fn host_free(ptr: *mut u8, size: usize, align: usize);
    fn host_publish(data: *const u8, len: usize);
}

struct HostAlloc;
unsafe impl GlobalAlloc for HostAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { host_alloc(layout.size(), layout.align()) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { host_free(ptr, layout.size(), layout.align()) }
    }
}

#[global_allocator]
static ALLOC: HostAlloc = HostAlloc;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

fn publish<T: ?Sized>(value: &T) {
    unsafe {
        host_publish(
            value as *const T as *const u8,
            core::mem::size_of_val(value),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn probe_new(config: *const u8, len: usize, vendor: bool) -> *mut Odometer {
    let bytes = unsafe { from_raw_parts(config, len) };
    let Ok(config) = postcard::from_bytes::<LioConfig>(bytes) else {
        return core::ptr::null_mut();
    };
    match Odometer::from_config(config, vendor) {
        Ok(odometer) => Box::into_raw(Box::new(odometer)),
        Err(_) => core::ptr::null_mut(),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn probe_imu(odometer: *mut Odometer, imu: *const u8) {
    let odometer = unsafe { &mut *odometer };
    odometer.add_imu(unsafe { read_unaligned(imu.cast::<ImuMeasurement>()) });
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn probe_odom(
    odometer: *mut Odometer,
    sec: i32,
    nanosec: u32,
    linear: *const u8,
    angular: *const u8,
    covariance: *const u8,
) {
    let odometer = unsafe { &mut *odometer };
    unsafe {
        odometer.add_odom(
            sec,
            nanosec,
            read_unaligned(linear.cast()),
            read_unaligned(angular.cast()),
            read_unaligned(covariance.cast()),
        )
    };
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn probe_xyzit(
    o: *mut Odometer,
    sec: i32,
    nanosec: u32,
    p: *const [f32; 5],
    n: usize,
) -> u64 {
    unsafe { (*o).add_points_xyzit(sec, nanosec, from_raw_parts(p, n)) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn probe_xyzt(
    o: *mut Odometer,
    sec: i32,
    nanosec: u32,
    p: *const [f32; 4],
    n: usize,
) -> u64 {
    unsafe { (*o).add_points_xyzt(sec, nanosec, from_raw_parts(p, n)) }
}

/// Field `i` has the name `names[name_starts[i]..name_starts[i + 1]]`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn probe_raw(
    o: *mut Odometer,
    sec: i32,
    nanosec: u32,
    width: u32,
    names: *const u8,
    name_starts: *const usize,
    offsets: *const u32,
    datatypes: *const u8,
    field_count: usize,
    point_step: u32,
    data: *const u8,
    len: usize,
    dense: bool,
) -> u64 {
    let starts = unsafe { from_raw_parts(name_starts, field_count + 1) };
    let fields: Vec<Pcloud2Field> = (0..field_count)
        .map(|i| {
            let name = unsafe { from_raw_parts(names.add(starts[i]), starts[i + 1] - starts[i]) };
            Pcloud2Field {
                name: String::from(core::str::from_utf8(name).unwrap_or("")),
                offset: unsafe { *offsets.add(i) },
                datatype: unsafe { *datatypes.add(i) },
            }
        })
        .collect();
    let cloud = RawPointCloud {
        header_sec: sec,
        header_nanosec: nanosec,
        width,
        fields: &fields,
        point_step,
        row_step: point_step * width,
        data: unsafe { from_raw_parts(data, len) },
        is_dense: dense,
    };
    unsafe { (*o).add_pointcloud_raw(cloud) }
}

/// Publishes every output of every finished scan.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn probe_drain(o: *mut Odometer) -> usize {
    let odometer = unsafe { &mut *o };
    let mut scans = 0;
    publish(&odometer.pending_scans());
    publish(&odometer.imu_initialized());
    while let Some(scan) = odometer.next_scan() {
        publish(&scan.scan_id());
        publish(&scan.timestamp());
        publish(&scan.position());
        publish(&scan.orientation());
        publish(scan.pose_covariance());
        publish(&scan.twist());
        publish(scan.twist_covariance());
        publish(&scan.accel_bias());
        publish(&scan.gyro_bias());
        publish(scan.frame_id().as_bytes());
        publish(scan.points());
        publish(&scan.points_xyzi().collect::<Vec<_>>()[..]);
        publish(&scan.world_points().collect::<Vec<_>>()[..]);
        publish(&scan.local_position());
        publish(&scan.origin_cells());
        publish(&scan.cell_size());
        publish(&scan.local_points().collect::<Vec<_>>()[..]);
        scans += 1;
    }
    scans
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn probe_free(o: *mut Odometer) {
    drop(unsafe { Box::from_raw(o) });
}
