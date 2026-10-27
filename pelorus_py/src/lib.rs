//! Python bindings for Pelorus.
//!
//! [`PelorusHandle`] wraps [`pelorus::Handle`], so Python runs exactly the odometry of the
//! Rust and C libraries and of `pelorus offline`.

use numpy::{IntoPyArray, PyReadonlyArray2};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySequence};

use log::error;

use ::pelorus::{Error, ImuMeasurement, Odometry, PointCloud2Msg};
use ros_pointcloud2::prelude::*;
use ros2_interfaces_jazzy_serde::sensor_msgs::msg::{PointCloud2, PointField};

#[cfg(any(
    all(feature = "cuda", feature = "metal"),
    all(feature = "cuda", feature = "vulkan"),
    all(feature = "metal", feature = "vulkan"),
))]
compile_error!("Select at most one Pelorus acceleration backend: Metal, CUDA, or Vulkan");

impl_pointcloud2_for_ros2_interfaces_jazzy_serde!();

/// Helper to convert sec/nanosec to f64 seconds
fn sec_nanosec_to_timestamp(sec: i32, nanosec: u32) -> f64 {
    sec as f64 + (nanosec as f64 / 1e9)
}

fn to_py_error(error: Error) -> PyErr {
    match error {
        Error::CallbackActive => PyRuntimeError::new_err(
            "Callback is set. try_recv() and recv() are unavailable. Use the callback or call clear_callback() first.",
        ),
        Error::Disconnected => PyRuntimeError::new_err("Channel disconnected"),
        other => PyRuntimeError::new_err(format!("{other:#}")),
    }
}

trait NumpyPointScalar: Copy {
    fn to_f64(self) -> f64;
}

impl NumpyPointScalar for f32 {
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl NumpyPointScalar for f64 {
    fn to_f64(self) -> f64 {
        self
    }
}

fn point_component_to_f32(value: f64, point_index: usize, field: &str) -> PyResult<f32> {
    if !value.is_finite() || value < f32::MIN as f64 || value > f32::MAX as f64 {
        return Err(PyValueError::new_err(format!(
            "point {point_index} field '{field}' must be a finite float32 value"
        )));
    }
    Ok(value as f32)
}

fn point_timestamp_to_u32(value: f64, point_index: usize) -> PyResult<u32> {
    if !value.is_finite() || value < 0.0 || value > u32::MAX as f64 {
        return Err(PyValueError::new_err(format!(
            "point {point_index} field 't' must be a finite nanosecond offset in 0..={}",
            u32::MAX
        )));
    }
    let rounded = value.round();
    if rounded > u32::MAX as f64 {
        return Err(PyValueError::new_err(format!(
            "point {point_index} field 't' must be a finite nanosecond offset in 0..={}",
            u32::MAX
        )));
    }
    Ok(rounded as u32)
}

fn validate_pointcloud_buffer(
    height: u32,
    width: u32,
    point_step: u32,
    row_step: u32,
    data_len: usize,
) -> PyResult<()> {
    if height == 0 || width == 0 {
        return Err(PyValueError::new_err(
            "PointCloud2 height and width must be greater than zero",
        ));
    }
    if point_step == 0 {
        return Err(PyValueError::new_err(
            "PointCloud2 point_step must be greater than zero",
        ));
    }

    let minimum_row_step = width
        .checked_mul(point_step)
        .ok_or_else(|| PyValueError::new_err("PointCloud2 row size overflows u32"))?;
    if row_step < minimum_row_step {
        return Err(PyValueError::new_err(format!(
            "PointCloud2 row_step ({row_step}) is smaller than width * point_step ({minimum_row_step})"
        )));
    }

    let required_data_len = (height as usize)
        .checked_mul(row_step as usize)
        .ok_or_else(|| PyValueError::new_err("PointCloud2 data size overflows usize"))?;
    if data_len < required_data_len {
        return Err(PyValueError::new_err(format!(
            "PointCloud2 data contains {data_len} bytes but at least {required_data_len} are required"
        )));
    }

    Ok(())
}

/// The fields of a PointCloud2 message, before validation.
struct CloudParts {
    sec: i32,
    nanosec: u32,
    frame_id: String,
    height: u32,
    width: u32,
    fields: Vec<PointField>,
    is_bigendian: bool,
    point_step: u32,
    row_step: u32,
    data: Vec<u8>,
    is_dense: bool,
}

/// Validate the parts of a PointCloud2 message and build it.
fn checked_cloud(parts: CloudParts) -> PyResult<PointCloud2Msg> {
    if parts.fields.is_empty() {
        return Err(PyValueError::new_err("PointCloud2 fields cannot be empty"));
    }
    validate_pointcloud_buffer(
        parts.height,
        parts.width,
        parts.point_step,
        parts.row_step,
        parts.data.len(),
    )?;
    Ok(impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(
        PointCloud2 {
            header: ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
                stamp: ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time {
                    sec: parts.sec,
                    nanosec: parts.nanosec,
                },
                frame_id: parts.frame_id,
            },
            height: parts.height,
            width: parts.width,
            fields: parts.fields,
            is_bigendian: parts.is_bigendian,
            point_step: parts.point_step,
            row_step: parts.row_step,
            data: parts.data,
            is_dense: parts.is_dense,
        },
    ))
}

/// Point stride of the clouds built from NumPy arrays and sequences.
const XYZIT_POINT_STEP: usize = 20;

/// A cloud with Ouster-like fields x, y, z, intensity (f32) and t (u32), from
/// `data` laid out with [`XYZIT_POINT_STEP`].
fn xyzit_cloud(sec: i32, nanosec: u32, num_points: usize, data: Vec<u8>) -> PointCloud2Msg {
    let field = |name: &str, offset: u32, datatype: u8| PointField {
        name: name.to_string(),
        offset,
        datatype,
        count: 1,
    };
    impl_ros2_interfaces_jazzy_serde::to_pointcloud2_msg(PointCloud2 {
        header: ros2_interfaces_jazzy_serde::std_msgs::msg::Header {
            stamp: ros2_interfaces_jazzy_serde::builtin_interfaces::msg::Time { sec, nanosec },
            frame_id: String::new(),
        },
        height: 1,
        width: num_points as u32,
        fields: vec![
            field("x", 0, 7),
            field("y", 4, 7),
            field("z", 8, 7),
            field("intensity", 12, 7),
            field("t", 16, 6),
        ],
        is_bigendian: false,
        point_step: XYZIT_POINT_STEP as u32,
        row_step: (num_points * XYZIT_POINT_STEP) as u32,
        data,
        is_dense: true,
    })
}

/// Fill a Python dict with an odometry result.
fn fill_odom_dict(dict: &Bound<'_, PyDict>, odom: &Odometry) -> PyResult<()> {
    dict.set_item("timestamp_sec", odom.timestamp_sec)?;
    dict.set_item("timestamp_nanosec", odom.timestamp_nanosec)?;
    // Keep a float timestamp for convenience.
    dict.set_item(
        "timestamp",
        sec_nanosec_to_timestamp(odom.timestamp_sec, odom.timestamp_nanosec),
    )?;
    dict.set_item("frame_id", &odom.frame_id)?;
    dict.set_item("position", odom.position.to_vec())?;
    dict.set_item("orientation", odom.orientation.to_vec())?;
    dict.set_item("pose_covariance", odom.pose_covariance.to_vec())?;
    dict.set_item("linear_velocity", odom.linear_velocity.to_vec())?;
    dict.set_item("angular_velocity", odom.angular_velocity.to_vec())?;
    dict.set_item("accel_bias", odom.accel_bias.to_vec())?;
    dict.set_item("gyro_bias", odom.gyro_bias.to_vec())?;
    dict.set_item("twist_covariance", odom.twist_covariance.to_vec())?;

    let flat: Vec<f64> = odom
        .points
        .iter()
        .flat_map(|p| [p.x as f64, p.y as f64, p.z as f64, p.intensity as f64])
        .collect();
    let array = ndarray::Array2::from_shape_vec((odom.points.len(), 4), flat)
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to create array: {}", e)))?;
    dict.set_item("points", array.into_pyarray(dict.py()))?;
    dict.set_item("scan_id", odom.scan_id)?;
    #[cfg(feature = "processing-time-export")]
    dict.set_item("processing_time_ns", odom.processing_time_ns)?;
    Ok(())
}

/// Python wrapper for Pelorus handle
#[pyclass(name = "Handle", module = "pelorus")]
struct PelorusHandle {
    inner: ::pelorus::Handle,
}

#[pymethods]
impl PelorusHandle {
    #[new]
    #[pyo3(signature = (config_path=None, device_specific_pointcloud2=true))]
    /// Create a new Pelorus instance.
    ///
    /// - `config_path`: optional path or HTTP(S) URL to a `.rl` config file. If omitted, embedded
    ///   defaults are used.
    /// - `device_specific_pointcloud2`: when true (default) Pelorus will use device-specific parsing
    ///   for PointCloud2 inputs (field meanings and timestamps are vendor/device-dependent). When
    ///   false, Pelorus uses the generic parsing path where per-point `t` is interpreted as a
    ///   header-relative nanoseconds offset (see README for the recommended layout).
    ///
    /// Inputs are processed in call order on a worker thread with the odometry of
    /// `pelorus offline`, so the same input sequence produces the same results.
    fn new(config_path: Option<&str>, device_specific_pointcloud2: bool) -> PyResult<Self> {
        // Ensure Rust logging works when used from Python (RUST_LOG env var)
        let _ = env_logger::try_init();
        let inner =
            ::pelorus::Handle::new_with_options(config_path, device_specific_pointcloud2)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Pelorus: {e:#}")))?;
        Ok(PelorusHandle { inner })
    }

    /// Set an odometry callback (async) like `pelorus_set_callback` in the C bindings.
    ///
    /// Semantics (mirrors C):
    /// - If a callback is set (non-None), odometry results are delivered to the callback from a
    ///   background thread and are NOT queued for `try_recv()` / `recv()`.
    /// - If callback is None, results are queued and can be consumed via `try_recv()` / `recv()`.
    fn set_callback(&mut self, py: Python<'_>, callback: Option<Py<PyAny>>) -> PyResult<()> {
        let Some(callback) = callback else {
            self.inner.clear_callback();
            return Ok(());
        };
        if !callback.bind(py).is_callable() {
            return Err(PyValueError::new_err("callback must be callable or None"));
        }
        self.inner.set_callback(move |odom| {
            Python::attach(|py| {
                let dict = PyDict::new(py);
                if let Err(e) = fill_odom_dict(&dict, &odom) {
                    error!(
                        "failed to build odometry dict of scan #{}: {e}",
                        odom.scan_id
                    );
                    return;
                }
                if let Err(e) = callback.bind(py).call1((dict,)) {
                    error!("odometry callback raised for scan #{}: {e}", odom.scan_id);
                    e.print(py);
                }
            });
        });
        Ok(())
    }

    /// Disable odometry callback (equivalent to `set_callback(None)`).
    fn clear_callback(&mut self) -> PyResult<()> {
        self.inner.clear_callback();
        Ok(())
    }

    /// Add an IMU message
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (timestamp_sec, timestamp_nanosec, orientation, orientation_covariance, angular_velocity, angular_velocity_covariance, linear_acceleration, linear_acceleration_covariance))]
    fn add_imu(
        &self,
        timestamp_sec: i32,
        timestamp_nanosec: u32,
        orientation: [f64; 4],
        orientation_covariance: [f64; 9],
        angular_velocity: [f64; 3],
        angular_velocity_covariance: [f64; 9],
        linear_acceleration: [f64; 3],
        linear_acceleration_covariance: [f64; 9],
    ) -> PyResult<()> {
        self.inner
            .add_imu(ImuMeasurement {
                timestamp_sec,
                timestamp_nanosec,
                orientation,
                orientation_covariance,
                angular_velocity,
                angular_velocity_covariance,
                linear_acceleration,
                linear_acceleration_covariance,
            })
            .map_err(to_py_error)
    }

    /// Add a ROS2 `sensor_msgs/msg/Imu` Python object directly.
    ///
    /// Works with messages coming from `mcap_ros2`, `rosbags` or `rclpy`.
    fn add_imu_ros(&self, imu: &Bound<'_, PyAny>) -> PyResult<()> {
        let (sec, nanosec, _frame_id) = Self::extract_header_stamp_frame_id(imu)?;

        let orientation_any = imu.getattr("orientation")?;
        let orientation = [
            orientation_any.getattr("w")?.extract::<f64>()?,
            orientation_any.getattr("x")?.extract::<f64>()?,
            orientation_any.getattr("y")?.extract::<f64>()?,
            orientation_any.getattr("z")?.extract::<f64>()?,
        ];

        let angular_any = imu.getattr("angular_velocity")?;
        let angular_velocity = [
            angular_any.getattr("x")?.extract::<f64>()?,
            angular_any.getattr("y")?.extract::<f64>()?,
            angular_any.getattr("z")?.extract::<f64>()?,
        ];

        let linear_any = imu.getattr("linear_acceleration")?;
        let linear_acceleration = [
            linear_any.getattr("x")?.extract::<f64>()?,
            linear_any.getattr("y")?.extract::<f64>()?,
            linear_any.getattr("z")?.extract::<f64>()?,
        ];

        let covariance = |name: &str| -> PyResult<[f64; 9]> {
            let values: Vec<f64> = imu.getattr(name)?.extract()?;
            values
                .try_into()
                .map_err(|_| PyValueError::new_err(format!("{name} must have 9 elements")))
        };

        self.add_imu(
            sec,
            nanosec,
            orientation,
            covariance("orientation_covariance")?,
            angular_velocity,
            covariance("angular_velocity_covariance")?,
            linear_acceleration,
            covariance("linear_acceleration_covariance")?,
        )
    }

    /// Add an odometry message
    #[pyo3(signature = (timestamp_sec, timestamp_nanosec, twist_linear, twist_angular, twist_covariance))]
    fn add_odom(
        &self,
        timestamp_sec: i32,
        timestamp_nanosec: u32,
        twist_linear: [f64; 3],
        twist_angular: [f64; 3],
        twist_covariance: [f64; 36],
    ) -> PyResult<()> {
        self.inner
            .add_odom(
                timestamp_sec,
                timestamp_nanosec,
                twist_linear,
                twist_angular,
                twist_covariance,
            )
            .map_err(to_py_error)
    }

    /// Add a ROS2 `nav_msgs/msg/Odometry` Python object directly.
    ///
    /// Only uses `header.stamp` and `twist` fields.
    fn add_odom_ros(&self, odom: &Bound<'_, PyAny>) -> PyResult<()> {
        let (sec, nanosec, _frame_id) = Self::extract_header_stamp_frame_id(odom)?;
        let twist = odom.getattr("twist")?;
        let twist_twist = twist.getattr("twist")?;
        let lin = twist_twist.getattr("linear")?;
        let ang = twist_twist.getattr("angular")?;

        let twist_linear = [
            lin.getattr("x")?.extract::<f64>()?,
            lin.getattr("y")?.extract::<f64>()?,
            lin.getattr("z")?.extract::<f64>()?,
        ];
        let twist_angular = [
            ang.getattr("x")?.extract::<f64>()?,
            ang.getattr("y")?.extract::<f64>()?,
            ang.getattr("z")?.extract::<f64>()?,
        ];

        let cov_vec: Vec<f64> = twist.getattr("covariance")?.extract()?;
        let twist_covariance: [f64; 36] = cov_vec
            .try_into()
            .map_err(|_| PyValueError::new_err("twist.covariance must have 36 elements"))?;

        self.add_odom(sec, nanosec, twist_linear, twist_angular, twist_covariance)
    }

    /// Add a point cloud
    ///
    /// Accepts multiple input formats:
    /// 1. PointCloud2 dict (ROS2 format)
    /// 2. NumPy float32/float64 array (Nx3/Nx4/Nx5)
    ///    - Nx3: [x, y, z]
    ///    - Nx4: [x, y, z, t] where `t` is `uint32` nanoseconds offset (relative to header timestamp)
    ///    - Nx5: [x, y, z, intensity, t]
    /// 3. Python sequence of point sequences with the same shapes as above
    ///
    /// Timestamp semantics:
    /// - For dict/ROS2 PointCloud2: uses `header.stamp` as the scan reference time.
    /// - For numpy/list inputs: you must provide `timestamp_sec`/`timestamp_nanosec` which become the
    ///   scan header timestamp (absolute), and optional per-point `t` values are interpreted as
    ///   nanoseconds offset relative to that header time.
    ///
    /// Returns the scan id, or 0 for a cloud the odometry drops because it is older than the
    /// previous one.
    #[pyo3(signature = (cloud, timestamp_sec=None, timestamp_nanosec=None))]
    fn add_pointcloud(
        &mut self,
        cloud: &Bound<'_, PyAny>,
        timestamp_sec: Option<i32>,
        timestamp_nanosec: Option<u32>,
    ) -> PyResult<u64> {
        let stamp = |kind: &str| -> PyResult<(i32, u32)> {
            let sec = timestamp_sec.ok_or_else(|| {
                PyValueError::new_err(format!("timestamp_sec required for {kind} input"))
            })?;
            let nanosec = timestamp_nanosec.ok_or_else(|| {
                PyValueError::new_err(format!("timestamp_nanosec required for {kind} input"))
            })?;
            Ok((sec, nanosec))
        };
        let cloud_msg = if let Ok(dict) = cloud.cast::<PyDict>() {
            Self::cloud_from_dict(dict)?
        } else if let Ok(array) = cloud.extract::<PyReadonlyArray2<f32>>() {
            let (sec, nanosec) = stamp("numpy array")?;
            Self::cloud_from_numpy(array.as_array(), sec, nanosec)?
        } else if let Ok(array) = cloud.extract::<PyReadonlyArray2<f64>>() {
            let (sec, nanosec) = stamp("numpy array")?;
            Self::cloud_from_numpy(array.as_array(), sec, nanosec)?
        } else if let Ok(sequence) = cloud.cast::<PySequence>() {
            let (sec, nanosec) = stamp("sequence")?;
            Self::cloud_from_sequence(sequence, sec, nanosec)?
        } else {
            return Err(PyValueError::new_err(
                "cloud must be a PointCloud2 dict, float32/float64 NumPy array (Nx3/Nx4/Nx5), or sequence of points",
            ));
        };
        self.inner.add_pointcloud2(cloud_msg).map_err(to_py_error)
    }

    /// Add a ROS2 `sensor_msgs/msg/PointCloud2` Python object directly.
    ///
    /// Returns the scan id, or 0 for a cloud the odometry drops because it is older than the
    /// previous one.
    fn add_pointcloud2_ros(&mut self, cloud: &Bound<'_, PyAny>) -> PyResult<u64> {
        let cloud_msg = Self::cloud_from_ros_msg(cloud)?;
        self.inner.add_pointcloud2(cloud_msg).map_err(to_py_error)
    }

    /// Try to receive odometry (non-blocking)
    fn try_recv(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let Some(odom) = self.inner.try_recv().map_err(to_py_error)? else {
            return Ok(None);
        };
        let dict = PyDict::new(py);
        fill_odom_dict(&dict, &odom)?;
        Ok(Some(dict.into()))
    }

    /// Blocking receive odometry
    fn recv(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let inner = &self.inner;
        // Release the GIL while waiting, so a callback thread can run.
        let odom = py.detach(|| inner.recv()).map_err(to_py_error)?;
        let dict = PyDict::new(py);
        fill_odom_dict(&dict, &odom)?;
        Ok(dict.into())
    }

    /// Get version
    #[staticmethod]
    fn version() -> &'static str {
        env!("CARGO_PKG_VERSION")
    }
}

impl PelorusHandle {
    fn extract_stamp_sec_nanosec(stamp: &Bound<'_, PyAny>) -> PyResult<(i32, u32)> {
        let sec: i32 = stamp.getattr("sec")?.extract()?;
        let nanosec: u32 = stamp.getattr("nanosec")?.extract()?;
        Ok((sec, nanosec))
    }

    fn extract_header_stamp_frame_id(msg: &Bound<'_, PyAny>) -> PyResult<(i32, u32, String)> {
        let header = msg.getattr("header")?;
        let stamp = header.getattr("stamp")?;
        let (sec, nanosec) = Self::extract_stamp_sec_nanosec(&stamp)?;
        let frame_id: String = header.getattr("frame_id")?.extract()?;
        Ok((sec, nanosec, frame_id))
    }

    /// Point cloud from a PointCloud2 dict
    fn cloud_from_dict(cloud_dict: &Bound<'_, PyDict>) -> PyResult<PointCloud2Msg> {
        let item = |name: &str| -> PyResult<Bound<'_, PyAny>> {
            cloud_dict
                .get_item(name)?
                .ok_or_else(|| PyValueError::new_err(format!("Missing '{name}' field")))
        };
        let header = item("header")?;
        // header is a Bound<PyAny>, whose get_item() raises KeyError if not found
        let stamp = header.get_item("stamp")?;

        let fields_bound = item("fields")?;
        let fields_list = fields_bound.cast::<PyList>()?;
        let mut fields = Vec::with_capacity(fields_list.len());
        for field in fields_list.iter() {
            fields.push(PointField {
                name: field.get_item("name")?.extract()?,
                offset: field.get_item("offset")?.extract()?,
                datatype: field.get_item("datatype")?.extract()?,
                count: field.get_item("count")?.extract()?,
            });
        }

        checked_cloud(CloudParts {
            sec: stamp.get_item("sec")?.extract()?,
            nanosec: stamp.get_item("nanosec")?.extract()?,
            frame_id: header.get_item("frame_id")?.extract()?,
            height: item("height")?.extract()?,
            width: item("width")?.extract()?,
            fields,
            is_bigendian: item("is_bigendian")?.extract()?,
            point_step: item("point_step")?.extract()?,
            row_step: item("row_step")?.extract()?,
            data: item("data")?.extract()?,
            is_dense: item("is_dense")?.extract()?,
        })
    }

    /// Point cloud from a ROS2 `sensor_msgs/msg/PointCloud2` Python object
    fn cloud_from_ros_msg(cloud_msg: &Bound<'_, PyAny>) -> PyResult<PointCloud2Msg> {
        let (sec, nanosec, frame_id) = Self::extract_header_stamp_frame_id(cloud_msg)?;

        let fields_any = cloud_msg.getattr("fields")?;
        let fields_list = fields_any.cast::<PyList>()?;
        let mut fields = Vec::with_capacity(fields_list.len());
        for field_any in fields_list.iter() {
            fields.push(PointField {
                name: field_any.getattr("name")?.extract()?,
                offset: field_any.getattr("offset")?.extract()?,
                datatype: field_any.getattr("datatype")?.extract()?,
                count: field_any.getattr("count")?.extract()?,
            });
        }

        checked_cloud(CloudParts {
            sec,
            nanosec,
            frame_id,
            height: cloud_msg.getattr("height")?.extract()?,
            width: cloud_msg.getattr("width")?.extract()?,
            fields,
            is_bigendian: cloud_msg.getattr("is_bigendian")?.extract()?,
            point_step: cloud_msg.getattr("point_step")?.extract()?,
            row_step: cloud_msg.getattr("row_step")?.extract()?,
            data: cloud_msg.getattr("data")?.extract()?,
            is_dense: cloud_msg.getattr("is_dense")?.extract()?,
        })
    }

    /// Point cloud from a NumPy array.
    fn cloud_from_numpy<T: NumpyPointScalar>(
        array: ndarray::ArrayView2<'_, T>,
        timestamp_sec: i32,
        timestamp_nanosec: u32,
    ) -> PyResult<PointCloud2Msg> {
        let shape = array.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err("Array must be 2D (Nx3/Nx4/Nx5)"));
        }
        let num_points = shape[0];
        let num_fields = shape[1];
        if num_points == 0 {
            return Err(PyValueError::new_err("Point cloud cannot be empty"));
        }
        if num_fields != 3 && num_fields != 4 && num_fields != 5 {
            return Err(PyValueError::new_err(
                "Array must have 3, 4, or 5 columns: [x, y, z], [x, y, z, t], or [x, y, z, intensity, t]",
            ));
        }

        // For Nx4, the 4th column is t and intensity is 0.
        let has_t = num_fields >= 4;
        let has_intensity = num_fields == 5;
        let mut data = Vec::with_capacity(num_points * XYZIT_POINT_STEP);
        for i in 0..num_points {
            for (j, field) in ["x", "y", "z"].into_iter().enumerate() {
                let value = point_component_to_f32(array[[i, j]].to_f64(), i, field)?;
                data.extend_from_slice(&value.to_le_bytes());
            }
            let intensity = if has_intensity {
                point_component_to_f32(array[[i, 3]].to_f64(), i, "intensity")?
            } else {
                0.0
            };
            data.extend_from_slice(&intensity.to_le_bytes());
            let t: u32 = if has_t {
                let idx = if has_intensity { 4 } else { 3 };
                point_timestamp_to_u32(array[[i, idx]].to_f64(), i)?
            } else {
                0
            };
            data.extend_from_slice(&t.to_le_bytes());
        }
        Ok(xyzit_cloud(
            timestamp_sec,
            timestamp_nanosec,
            num_points,
            data,
        ))
    }

    /// Point cloud from a sequence of point sequences.
    fn cloud_from_sequence(
        points: &Bound<'_, PySequence>,
        timestamp_sec: i32,
        timestamp_nanosec: u32,
    ) -> PyResult<PointCloud2Msg> {
        let num_points = points.len()?;
        if num_points == 0 {
            return Err(PyValueError::new_err("Point cloud cannot be empty"));
        }

        // The first point determines the format.
        let first_point = points.get_item(0)?;
        let first_point_sequence = first_point.cast::<PySequence>().map_err(|_| {
            PyValueError::new_err("Each point must be a list, tuple, or other sequence")
        })?;
        let num_fields = first_point_sequence.len()?;
        if num_fields != 3 && num_fields != 4 && num_fields != 5 {
            return Err(PyValueError::new_err(
                "Each point must have 3, 4, or 5 values: [x, y, z], [x, y, z, t], or [x, y, z, intensity, t]",
            ));
        }

        let has_t = num_fields >= 4;
        let has_intensity = num_fields == 5;
        let mut data = Vec::with_capacity(num_points * XYZIT_POINT_STEP);
        for point_index in 0..num_points {
            let point = points.get_item(point_index)?;
            let point_sequence = point.cast::<PySequence>().map_err(|_| {
                PyValueError::new_err("Each point must be a list, tuple, or other sequence")
            })?;
            if point_sequence.len()? != num_fields {
                return Err(PyValueError::new_err(
                    "All points must have the same number of fields",
                ));
            }

            for (i, field) in ["x", "y", "z"].into_iter().enumerate() {
                let value: f64 = point_sequence.get_item(i)?.extract()?;
                let value = point_component_to_f32(value, point_index, field)?;
                data.extend_from_slice(&value.to_le_bytes());
            }
            let intensity: f32 = if has_intensity {
                let value: f64 = point_sequence.get_item(3)?.extract()?;
                point_component_to_f32(value, point_index, "intensity")?
            } else {
                0.0
            };
            data.extend_from_slice(&intensity.to_le_bytes());
            let t: u32 = if has_t {
                let idx = if has_intensity { 4 } else { 3 };
                let value: f64 = point_sequence.get_item(idx)?.extract()?;
                point_timestamp_to_u32(value, point_index)?
            } else {
                0
            };
            data.extend_from_slice(&t.to_le_bytes());
        }
        Ok(xyzit_cloud(
            timestamp_sec,
            timestamp_nanosec,
            num_points,
            data,
        ))
    }
}

#[pymodule]
fn pelorus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PelorusHandle>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
