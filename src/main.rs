#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

use futures::executor::LocalPool;
use futures::task::LocalSpawnExt;
use futures::{future, StreamExt};
use r2r::geometry_msgs::msg::PolygonStamped;
use r2r::sensor_msgs::msg::PointCloud2;
use r2r::{log_error, log_info, log_warn, Publisher, QosProfile};
use std::sync::{Arc, Mutex};
use tf_r2r::TfListener;

use ros_pointcloud2::prelude::*;

mod config;

struct State {
    pub global_polygon: Option<Vec<nalgebra::Point2<f32>>>,
    pub point_buffer: Vec<PointXYZI>,
    pub sensor_tf: Option<nalgebra::geometry::Isometry3<f32>>,
    pub config: Arc<Mutex<config::ConfigParse>>,
}

impl State {
    pub fn new(config: Arc<Mutex<config::ConfigParse>>) -> Self {
        State {
            global_polygon: None,
            point_buffer: Vec::new(),
            sensor_tf: None,
            config: config,
        }
    }
}
/*
fn polygon_callback(
    logger: &str,
    state: Arc<Mutex<State>>,
    tf_listener: Arc<TfListener>,
    msg: PolygonStamped,
) {
    let target_frame = state
        .lock()
        .unwrap()
        .config
        .lock()
        .unwrap()
        .global_frame
        .clone();

    let poly: Vec<nalgebra::Point2<f32>> = if target_frame.is_empty() {
        let tf = tf_listener.clone().lookup_transform(
            msg.header.frame_id.as_str(),
            state
                .lock()
                .unwrap()
                .config
                .lock()
                .unwrap()
                .global_frame
                .as_str(),
            msg.header.stamp,
        );
        let tf = match tf {
            Ok(t) => t,
            Err(e) => {
                log_warn!(logger, "TF: {:?}", e);
                return;
            }
        };
        let tf = nalgebra::geometry::Isometry3::from_parts(
            nalgebra::geometry::Translation3::new(
                tf.transform.translation.x as f32,
                tf.transform.translation.y as f32,
                tf.transform.translation.z as f32,
            ),
            nalgebra::geometry::UnitQuaternion::from_quaternion(
                nalgebra::geometry::Quaternion::new(
                    tf.transform.rotation.w as f32,
                    tf.transform.rotation.x as f32,
                    tf.transform.rotation.y as f32,
                    tf.transform.rotation.z as f32,
                ),
            ),
        )
        .inverse();

        msg.polygon
            .points
            .iter()
            .map(|p| {
                let p = tf * nalgebra::geometry::Point3::new(p.x as f32, p.y as f32, p.z as f32);
                nalgebra::Point2::new(p.x, p.y)
            })
            .collect()
    } else {
        msg.polygon
            .points
            .iter()
            .map(|p| nalgebra::Point2::new(p.x as f32, p.y as f32))
            .collect()
    };

    match state.lock() {
        Ok(mut state) => {
            state.global_polygon = Some(poly);
            match state.config.lock() {
                Ok(mut config) => {
                    if config.disable_with_first_polygon {
                        config.fov_enabled = false;
                        config.aabb_max_x = f32::MAX;
                        config.aabb_min_x = f32::MIN;
                        config.aabb_max_y = f32::MAX;
                        config.aabb_min_y = f32::MIN;
                        config.max_dist = 9999.999;
                    }
                }
                Err(e) => {
                    log_error!(
                        logger,
                        "Failed to lock config in filter for polygon service: {:?}",
                        e
                    );
                }
            }
        }
        Err(e) => {
            log_error!(
                logger,
                "Failed to lock state in filter for polygon service: {:?}",
                e
            );
        }
    }
}

fn cloud_callback(
    logger: &str,
    state: Arc<Mutex<State>>,
    tf_listener: Arc<TfListener>,
    filtered_pub: Arc<Publisher<PointCloud2>>,
    msg: PointCloud2,
) {
    let start_callback = std::time::Instant::now();
    #[cfg(feature = "cuda")]
    let stream = cupcl::CudaStream::new();
    let orig_header = msg.header.clone();

    let in_msg: PointCloud2Msg = msg.into();

    let in_stamp = orig_header.stamp;
    let incoming_header = in_msg.header.clone();
    let points_n = (in_msg.dimensions.width * in_msg.dimensions.height) as usize;
    if points_n == 0 {
        return;
    }

    let (sensor_tf, global_zero_frame, verbose) = {
        match state.lock() {
            Ok(mut state) => {
                state.point_buffer.clear();

                let (sensor_tf, global_zero_frame, verbose) = match state.config.lock() {
                    Ok(config) => {
                        let sensor_tf = match state.sensor_tf {
                            Some(tf) => tf,
                            None => {
                                if config.global_zero_frame.is_empty() {
                                    nalgebra::geometry::Isometry3::identity()
                                } else {
                                    let tf = tf_listener.lookup_transform(
                                        orig_header.frame_id.clone().as_str(),
                                        config.global_zero_frame.clone().as_str(),
                                        in_stamp.clone(),
                                    );
                                    let tf = match tf {
                                        Ok(t) => t,
                                        Err(e) => {
                                            log_warn!(logger, "TF: {:?}", e);
                                            return;
                                        }
                                    };
                                    nalgebra::geometry::Isometry3::from_parts(
                                        nalgebra::geometry::Translation3::new(
                                            tf.transform.translation.x as f32,
                                            tf.transform.translation.y as f32,
                                            tf.transform.translation.z as f32,
                                        ),
                                        nalgebra::geometry::UnitQuaternion::from_quaternion(
                                            nalgebra::geometry::Quaternion::new(
                                                tf.transform.rotation.w as f32,
                                                tf.transform.rotation.x as f32,
                                                tf.transform.rotation.y as f32,
                                                tf.transform.rotation.z as f32,
                                            ),
                                        ),
                                    )
                                    .inverse()
                                }
                            }
                        };

                        let out_frame = if config.global_zero_frame.is_empty() {
                            orig_header.frame_id.clone()
                        } else {
                            config.global_zero_frame.clone()
                        };

                        (sensor_tf, out_frame, config.verbose)
                    }
                    Err(e) => {
                        log_error!(
                            logger,
                            "Failed to lock config in filter for polygon service: {:?}",
                            e
                        );
                        return;
                    }
                };

                state.sensor_tf = Some(sensor_tf);

                if state.point_buffer.capacity() < points_n {
                    state.point_buffer = Vec::with_capacity(points_n);
                }

                (sensor_tf, global_zero_frame, verbose)
            }

            Err(e) => {
                log_error!(
                    logger,
                    "Failed to lock state in filter for polygon service: {:?}",
                    e
                );
                return;
            }
        }
    };

    let filter_params = {
        match state.lock() {
            Ok(state) => {
                match state.config.lock() {
                    Ok(config) => {
                        let mut params = cupcl::PassthroughFilterParameters::default();
                        params.min = (
                            config.aabb_min_x.into(),
                            config.aabb_min_y.into(),
                            config.aabb_min_z.into(),
                        );
                        params.max = (
                            config.aabb_max_x.into(),
                            config.aabb_max_y.into(),
                            config.aabb_max_z.into(),
                        );
                        params.invert_bounding_box = config.invert_aabb;
                        params.min_dist = config.min_dist.into();
                        params.max_dist = config.max_dist.into();
                        params.invert_distance = config.invert_distance;
                        params.rotation = (
                            sensor_tf.rotation.i.into(),
                            sensor_tf.rotation.j.into(),
                            sensor_tf.rotation.k.into(),
                            sensor_tf.rotation.w.into(),
                        );
                        params.translation = (
                            sensor_tf.translation.x.into(),
                            sensor_tf.translation.y.into(),
                            sensor_tf.translation.z.into(),
                        );
                        params.fov_left = config.fov_left.into();
                        params.fov_right = config.fov_right.into();
                        params.enable_horizontal_fov = config.fov_enabled;
                        params.invert_fov = config.invert_fov;
                        params.min_intensity = config.min_intensity.into();
                        params.max_intensity = config.max_intensity.into();
                        params.invert_intensity = config.invert_intensity;

                        match state.global_polygon.as_ref() {
                            None => {}
                            Some(poly) => {
                                let tf = if config.global_zero_frame.is_empty() {
                                    nalgebra::geometry::Isometry3::identity()
                                } else {
                                    let tf = tf_listener.lookup_transform(
                                        // TODO might be dangerous because of the lock
                                        config.global_frame.as_str(),
                                        config.global_zero_frame.as_str(),
                                        in_stamp,
                                    );
                                    let tf = match tf {
                                        Ok(t) => t,
                                        Err(e) => {
                                            log_warn!(logger, "TF: {:?}", e);
                                            return;
                                        }
                                    };
                                    nalgebra::geometry::Isometry3::from_parts(
                                        nalgebra::geometry::Translation3::new(
                                            tf.transform.translation.x as f32,
                                            tf.transform.translation.y as f32,
                                            tf.transform.translation.z as f32,
                                        ),
                                        nalgebra::geometry::UnitQuaternion::from_quaternion(
                                            nalgebra::geometry::Quaternion::new(
                                                tf.transform.rotation.w as f32,
                                                tf.transform.rotation.x as f32,
                                                tf.transform.rotation.y as f32,
                                                tf.transform.rotation.z as f32,
                                            ),
                                        ),
                                    )
                                    .inverse()
                                };

                                let mut cupcl_poly: Vec<(f64, f64)> = Vec::new();
                                for p in poly {
                                    let tp = tf * nalgebra::Point3::new(p.x, p.y, 0.0);
                                    cupcl_poly.push((tp.x.into(), tp.y.into()));
                                }
                                #[cfg(feature = "cpu")]
                                {
                                    params.polygon = Some(cupcl_poly);
                                }
                                #[cfg(feature = "cuda")]
                                {
                                    let poly = cupcl::CudaBuffer::from_vec(
                                        &stream,
                                        cupcl_poly,
                                        std::mem::size_of::<cupcl::Point2>(),
                                    );
                                    params.polygon = Some(poly);
                                }
                                params.invert_polygon = config.invert_polygon;
                            }
                        }

                        /*let cloud: cupcl::PointCloud;

                        // TODO reuse buffer
                        #[cfg(feature = "cuda")]
                        {
                            cloud = cupcl::PointCloud::from_full_cloud(
                                &stream,
                                state.point_buffer.clone(),
                            );
                        }

                        #[cfg(feature = "cpu")]
                        {
                            cloud = cupcl::PointCloud::from_full_cloud(state.point_buffer.clone());
                        }*/

                        params
                    }
                    Err(e) => {
                        log_error!(
                            logger,
                            "Failed to lock config in filter for polygon service: {:?}",
                            e
                        );
                        return;
                    }
                }
            }
            Err(e) => {
                log_error!(
                    logger,
                    "Failed to lock state in filter for polygon service: {:?}",
                    e
                );
                return;
            }
        }
    };

    let start_filter = std::time::Instant::now();
    let mut output = cupcl::passthrough_filter(in_msg, filter_params);
    let end_filter = std::time::Instant::now();

    output.header = incoming_header;
    output.header.frame_id = global_zero_frame.clone();

    let out_msg: PointCloud2 = output.into();
    if let Err(e) = filtered_pub.publish(&out_msg) {
        log_error!(logger, "Could not send filtered pointcloud: {:?}", e);
    }
    let end_callback = std::time::Instant::now();
    if verbose {
        log_info!(logger, "Filtering: {:?}", end_filter - start_filter);
        log_info!(logger, "Callback: {:?}", end_callback - start_callback);
    }
}*/

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /*
    let config = Arc::new(Mutex::new(config::ConfigParse::default()));
    let ctx = r2r::Context::create()?;
    let mut node = r2r::Node::create(ctx, "cloudfilter", "")?;
    let nl = "cloudfilter";
    let (_, _) = node.make_derived_parameter_handler(config.clone())?;

    let conf_out_copy = config.clone();

    let tf_listener = Arc::new(TfListener::new(&mut node));
    let filtered_pub = Arc::new(
        node.create_publisher::<PointCloud2>("filtered_cloud", QosProfile::default())
            .expect("Failed to create publisher"),
    );

    let mut pool = LocalPool::new();
    let spawner = pool.spawner();

    let state = Arc::new(Mutex::new(State::new(config.clone())));

    let cloud_subscriber = node.subscribe::<PointCloud2>("cloud", QosProfile::default())?;
    let state_clone = state.clone();
    let tf_listener_clone = tf_listener.clone();
    let cloud_pub_clone = filtered_pub.clone();
    spawner
        .spawn_local(async move {
            cloud_subscriber
                .for_each(|msg| {
                    cloud_callback(
                        nl,
                        state_clone.clone(),
                        tf_listener_clone.clone(),
                        cloud_pub_clone.clone(),
                        msg,
                    );
                    future::ready(())
                })
                .await;
        })
        .expect("Failed to spawn cloud subscriber");

    let state_clone = state.clone();
    let tf_listener_clone = tf_listener.clone();
    let _poly_subber = node.subscribe::<PolygonStamped>("polygon", QosProfile::default())?;
    spawner
        .spawn_local(async move {
            _poly_subber
                .for_each(|msg| {
                    polygon_callback(nl, state_clone.clone(), tf_listener_clone.clone(), msg);
                    future::ready(())
                })
                .await;
        })
        .expect("Failed to spawn polygon subscriber");

    let state_clone = Arc::clone(&state);
    let _service_set_fov =
        node.create_service::<r2r::cloudfilter_msgs::srv::SetFov::Service>("set_fov");
    spawner
        .spawn_local(async move {
            _service_set_fov
                .expect("could not create service")
                .for_each(|req| {
                    let left: f32 = req.message.left;
                    let right: f32 = req.message.right;
                    let enable: bool = req.message.enable;

                    let resp = r2r::cloudfilter_msgs::srv::SetFov::Response {
                        success: match state_clone
                            .lock()
                            .expect("could not lock state")
                            .config
                            .lock()
                        {
                            Ok(mut state) => {
                                state.fov_enabled = enable;
                                state.fov_left = left;
                                state.fov_right = right;
                                true
                            }
                            Err(e) => {
                                log_error!(
                                    nl,
                                    "Failed to lock state in filter for polygon service: {:?}",
                                    e
                                );
                                false
                            }
                        },
                    };
                    req.respond(resp).expect("could not send response");
                    future::ready(())
                })
                .await;
        })
        .expect("Failed to spawn set_fov service");

    let state_clone = Arc::clone(&state);
    let _service_set_distances =
        node.create_service::<r2r::cloudfilter_msgs::srv::SetDistances::Service>("set_distances");
    spawner
        .spawn_local(async move {
            _service_set_distances
                .expect("could not create service")
                .for_each(|req| {
                    let min_distance: f32 = req.message.min;
                    let max_distance: f32 = req.message.max;
                    let invert: bool = req.message.invert;

                    let resp = r2r::cloudfilter_msgs::srv::SetDistances::Response {
                        success: match state_clone
                            .lock()
                            .expect("could not lock state")
                            .config
                            .lock()
                        {
                            Ok(mut state) => {
                                state.min_dist = min_distance;
                                state.max_dist = max_distance;
                                state.invert_distance = invert;
                                true
                            }
                            Err(e) => {
                                log_error!(
                                    nl,
                                    "Failed to lock state in filter for polygon service: {:?}",
                                    e
                                );
                                false
                            }
                        },
                    };
                    req.respond(resp).expect("could not send response");
                    future::ready(())
                })
                .await;
        })
        .expect("Failed to spawn set_distances service");

    log_info!(nl, "Config: {:?}", conf_out_copy.lock().unwrap());

    log_info!(nl, "-- spawned listeners --");

    #[cfg(debug_assertions)]
    log_info!(nl, "I am in debug mode :(");

    loop {
        node.spin_once(std::time::Duration::from_millis(100)); // 100ms timeout
        pool.run_until_stalled();
    }
    */
    Ok(())
}
