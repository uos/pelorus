//! The input stream both builds replay: the resolved configuration followed by
//! the IMU messages and clouds in bag order. Every value is little-endian.
//!
//! ```text
//! magic         b"PELORUS-STREAM-1"
//! config        u32 length, postcard-encoded LioConfig
//! records until the end of the file:
//!   1 IMU       i32 sec, u32 nanosec, 37 f64 (orientation [w x y z] and covariance,
//!               angular velocity and covariance, linear acceleration and covariance)
//!   2 cloud     i32 sec, u32 nanosec, string frame_id, u32 width, u32 height,
//!               u32 field count, per field (string name, u32 offset, u8 datatype,
//!               u32 count), u8 big endian, u32 point_step, u32 row_step,
//!               u32 length and the point data, u8 dense
//! string        u32 length, UTF-8 bytes
//! ```

use std::io::{self, Read};

use pelorus::{ImuMeasurement, LioConfig, PointCloud2Msg};
use ros_pointcloud2::ros::{HeaderMsg, PointFieldMsg, TimeMsg};
use ros_pointcloud2::{CloudDimensions, Denseness, Endian};

pub const MAGIC: &[u8; 16] = b"PELORUS-STREAM-1";
pub const IMU: u8 = 1;
pub const CLOUD: u8 = 2;

pub enum Record {
    Imu(ImuMeasurement),
    Cloud(PointCloud2Msg),
}

pub struct Reader<R> {
    inner: R,
}

impl<R: Read> Reader<R> {
    /// Check the magic and read the configuration.
    pub fn open(mut inner: R) -> io::Result<(LioConfig, Self)> {
        let mut magic = [0; 16];
        inner.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(invalid("not a Pelorus input stream"));
        }
        let mut reader = Self { inner };
        let config = reader.bytes()?;
        let config = postcard::from_bytes(&config)
            .map_err(|e| invalid(&format!("invalid configuration: {e}")))?;
        Ok((config, reader))
    }

    /// The next record, or `None` at the end of the stream.
    pub fn next_record(&mut self) -> io::Result<Option<Record>> {
        let mut tag = [0];
        if self.inner.read(&mut tag)? == 0 {
            return Ok(None);
        }
        match tag[0] {
            IMU => {
                let timestamp_sec = self.i32()?;
                let timestamp_nanosec = self.u32()?;
                let mut imu = ImuMeasurement {
                    timestamp_sec,
                    timestamp_nanosec,
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
                    *value = self.f64()?;
                }
                Ok(Some(Record::Imu(imu)))
            }
            CLOUD => {
                let stamp = TimeMsg {
                    sec: self.i32()?,
                    nanosec: self.u32()?,
                };
                let frame_id = self.string()?;
                let dimensions = CloudDimensions {
                    width: self.u32()?,
                    height: self.u32()?,
                };
                let field_count = self.u32()?;
                let fields = (0..field_count)
                    .map(|_| {
                        Ok(PointFieldMsg {
                            name: self.string()?.into(),
                            offset: self.u32()?,
                            datatype: self.u8()?,
                            count: self.u32()?,
                        })
                    })
                    .collect::<io::Result<_>>()?;
                let endian = if self.u8()? != 0 {
                    Endian::Big
                } else {
                    Endian::Little
                };
                let point_step = self.u32()?;
                let row_step = self.u32()?;
                let data = self.bytes()?;
                let dense = if self.u8()? != 0 {
                    Denseness::Dense
                } else {
                    Denseness::Sparse
                };
                Ok(Some(Record::Cloud(PointCloud2Msg {
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
                })))
            }
            other => Err(invalid(&format!("unknown record tag {other}"))),
        }
    }

    fn array<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let mut bytes = [0; N];
        self.inner.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.array::<1>()?[0])
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn i32(&mut self) -> io::Result<i32> {
        Ok(i32::from_le_bytes(self.array()?))
    }

    fn f64(&mut self) -> io::Result<f64> {
        Ok(f64::from_le_bytes(self.array()?))
    }

    fn bytes(&mut self) -> io::Result<Vec<u8>> {
        let mut bytes = vec![0; self.u32()? as usize];
        self.inner.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn string(&mut self) -> io::Result<String> {
        String::from_utf8(self.bytes()?).map_err(|_| invalid("a string is not UTF-8"))
    }
}

pub fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
