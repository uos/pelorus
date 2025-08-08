#![cfg(feature = "trajectory")]

use kiss3d::camera::ArcBall;
use kiss3d::light::Light;
use kiss3d::window::Window;
use nalgebra::{Quaternion, Translation3, UnitQuaternion};
use plotters::prelude::*;
use sophus_lie::traits::IsTranslationProductGroup;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

type SE3 = sophus_lie::Isometry3<f64, 1>;
type SO3 = sophus_lie::Rotation3<f64, 1>;

fn draw_world_frame(window: &mut Window, size: f32) {
    let origin = kiss3d::nalgebra::Point3::origin();
    let x_axis = kiss3d::nalgebra::Point3::new(size, 0.0, 0.0);
    let y_axis = kiss3d::nalgebra::Point3::new(0.0, 0.0, -size);
    let z_axis = kiss3d::nalgebra::Point3::new(0.0, size, 0.0);

    // X: Red, Y: Green, Z: Blue (a common convention)
    window.draw_line(
        &origin,
        &x_axis,
        &kiss3d::nalgebra::Point3::new(1.0, 0.0, 0.0),
    );
    window.draw_line(
        &origin,
        &y_axis,
        &kiss3d::nalgebra::Point3::new(0.0, 1.0, 0.0),
    );
    window.draw_line(
        &origin,
        &z_axis,
        &kiss3d::nalgebra::Point3::new(0.0, 0.0, 1.0),
    );
}

fn draw_camera_frustum(
    window: &mut Window,
    pose: &Pose,
    color: &kiss3d::nalgebra::Point3<f32>,
    scale: f32,
) {
    // Define the 5 points of the camera frustum in its own coordinate system
    // (origin + 4 corners of the near plane)
    let pyramid_points: [kiss3d::nalgebra::Point3<f32>; 5] = [
        kiss3d::nalgebra::Point3::origin(),
        kiss3d::nalgebra::Point3::new(1.0, -0.5, -0.5) * scale, // Forward, Right, Down
        kiss3d::nalgebra::Point3::new(1.0, 0.5, -0.5) * scale,  // Forward, Left,  Down
        kiss3d::nalgebra::Point3::new(1.0, 0.5, 0.5) * scale,   // Forward, Left,  Up
        kiss3d::nalgebra::Point3::new(1.0, -0.5, 0.5) * scale,  // Forward, Right, Up
    ];

    // Create the transformation from the pose
    let transform = kiss3d::nalgebra::Isometry3::from_parts(
        kiss3d::nalgebra::Translation3::new(
            pose.translation.x,
            pose.translation.y,
            pose.translation.z,
        )
        .cast::<f32>(),
        kiss3d::nalgebra::UnitQuaternion::from_quaternion(kiss3d::nalgebra::Quaternion::new(
            pose.rotation.quaternion().w,
            pose.rotation.quaternion().i,
            pose.rotation.quaternion().j,
            pose.rotation.quaternion().k,
        ))
        .cast::<f32>(),
    );

    // Transform the points from camera coordinates to world coordinates
    let transformed_points: Vec<kiss3d::nalgebra::Point3<f32>> =
        pyramid_points.iter().map(|p| transform * p).collect();

    let o = &transformed_points[0]; // Origin
    let p1 = &transformed_points[1];
    let p2 = &transformed_points[2];
    let p3 = &transformed_points[3];
    let p4 = &transformed_points[4];

    // Draw the pyramid lines
    window.draw_line(o, p1, color);
    window.draw_line(o, p2, color);
    window.draw_line(o, p3, color);
    window.draw_line(o, p4, color);

    // Draw the near plane rectangle
    window.draw_line(p1, p2, color);
    window.draw_line(p2, p3, color);
    window.draw_line(p3, p4, color);
    window.draw_line(p4, p1, color);
}

#[derive(Debug, Clone, Copy)]
struct Pose {
    timestamp: f64,
    translation: Translation3<f64>,
    rotation: UnitQuaternion<f64>,
}

fn parse_pose(line: &str, gt_coord_switch: bool) -> Result<Pose, Box<dyn Error>> {
    let parts: Vec<f64> = line
        .split_whitespace()
        .map(|s| s.parse::<f64>())
        .collect::<Result<Vec<f64>, _>>()?;

    if parts.len() != 8 {
        return Err("Each line must have 8 values".into());
    }

    let translation = if gt_coord_switch {
        // 3MGRS oder so, GroundFusion++
        // Translation3::new(-parts[1], parts[3], parts[2])
        Translation3::new(parts[1], parts[3], -parts[2])
    } else {
        // Translation3::new(parts[1], parts[3], -parts[2])
        Translation3::new(parts[2], parts[3], parts[1])
    };
    Ok(Pose {
        timestamp: parts[0],
        translation,
        rotation: UnitQuaternion::from_quaternion(Quaternion::new(
            parts[7], parts[4], parts[5], parts[6], // w, i, j, k
        )),
    })
}

fn read_trajectory(path: &Path, gt_coord_switch: bool) -> Result<Vec<Pose>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    reader
        .lines()
        .map(|line| parse_pose(&line?, gt_coord_switch))
        .collect()
}

fn interpolate_pose(p1: &Pose, p2: &Pose, timestamp: f64) -> Pose {
    let t = (timestamp - p1.timestamp) / (p2.timestamp - p1.timestamp);
    let p1 = SE3::from_translation_and_rotation(
        &p1.translation.vector,
        &SO3::try_from_mat(&p1.rotation.to_rotation_matrix().matrix()).unwrap(),
    );
    let p2 = SE3::from_translation_and_rotation(
        &p2.translation.vector,
        &SO3::try_from_mat(&p2.rotation.to_rotation_matrix().matrix()).unwrap(),
    );
    let diff = p1.interpolate(&p2, t);
    Pose {
        timestamp,
        translation: Translation3::from(diff.translation()),
        rotation: UnitQuaternion::from_matrix(&diff.rotation().matrix()),
    }
}

fn create_2d_plots(gt_traj: &[Pose], est_traj: &[Pose]) -> Result<(), Box<dyn std::error::Error>> {
    let root_xy = BitMapBackend::new("side_view_XZ.png", (1024, 768)).into_drawing_area();
    root_xy.fill(&WHITE)?;

    // Find data range for XY plot
    let (x_min, x_max, y_min, y_max) = gt_traj.iter().chain(est_traj.iter()).fold(
        (f64::MAX, f64::MIN, f64::MAX, f64::MIN),
        |(x_min, x_max, y_min, y_max), pose| {
            (
                x_min.min(pose.translation.x),
                x_max.max(pose.translation.x),
                y_min.min(pose.translation.y),
                y_max.max(pose.translation.y),
            )
        },
    );

    let mut chart_xy = ChartBuilder::on(&root_xy)
        .caption("Side View", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, y_min..y_max)?;

    chart_xy.configure_mesh().draw()?;

    // Draw Ground Truth (Blue)
    chart_xy
        .draw_series(LineSeries::new(
            gt_traj.iter().map(|p| (p.translation.x, p.translation.y)),
            &BLUE,
        ))?
        .label("Ground Truth")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    // Draw Estimated (Red)
    chart_xy
        .draw_series(LineSeries::new(
            est_traj.iter().map(|p| (p.translation.x, p.translation.y)),
            &RED,
        ))?
        .label("Estimated")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart_xy
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    // --- Top View Plot ---
    let root_xz = BitMapBackend::new("top_view_XY.png", (1024, 768)).into_drawing_area();
    root_xz.fill(&WHITE)?;

    let (x_min, x_max, z_min, z_max) = gt_traj.iter().chain(est_traj.iter()).fold(
        (f64::MAX, f64::MIN, f64::MAX, f64::MIN),
        |(x_min, x_max, z_min, z_max), pose| {
            (
                x_min.min(pose.translation.x),
                x_max.max(pose.translation.x),
                z_min.min(pose.translation.z),
                z_max.max(pose.translation.z),
            )
        },
    );

    let mut chart_xz = ChartBuilder::on(&root_xz)
        .caption("Top View", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(x_min..x_max, z_min..z_max)?;

    chart_xz.configure_mesh().draw()?;

    // Draw Ground Truth (Blue)
    chart_xz.draw_series(LineSeries::new(
        gt_traj.iter().map(|p| (p.translation.x, p.translation.z)),
        &BLUE,
    ))?;

    // Draw Estimated (Red)
    chart_xz.draw_series(LineSeries::new(
        est_traj.iter().map(|p| (p.translation.x, p.translation.z)),
        &RED,
    ))?;

    root_xy.present()?;
    root_xz.present()?;
    println!("Generated 2D plots: top_view_XY.png and side_view_XZ.png");

    Ok(())
}

pub fn visualize_results(
    estimated_path: &Path,
    ground_truth_path: &Path,
    rotation_error: bool,
) -> anyhow::Result<()> {
    let estimated_trajectory = read_trajectory(estimated_path, false).unwrap();
    let ground_truth_trajectory = read_trajectory(ground_truth_path, true).unwrap();

    let mut translation_errors = Vec::new();
    let mut rotation_errors = Vec::new();
    let mut interpolated_estimated_trajectory = Vec::new();
    let mut corresponding_gt_trajectory = Vec::new();

    for gt_pose in &ground_truth_trajectory {
        // Find the two closest estimated poses in time
        let mut p1_idx = 0;
        while p1_idx < estimated_trajectory.len() - 2
            && estimated_trajectory[p1_idx + 1].timestamp < gt_pose.timestamp
        {
            p1_idx += 1;
        }
        let p2_idx = p1_idx + 1;

        if p2_idx >= estimated_trajectory.len() {
            // Not enough data to interpolate
            continue;
        }

        let p1 = &estimated_trajectory[p1_idx];
        let p2 = &estimated_trajectory[p2_idx];

        let interpolated_pose = interpolate_pose(p1, p2, gt_pose.timestamp);
        interpolated_estimated_trajectory.push(interpolated_pose);
        corresponding_gt_trajectory.push(*gt_pose);

        let trans_error =
            (gt_pose.translation.vector - interpolated_pose.translation.vector).norm();
        translation_errors.push(trans_error.powi(2));

        if rotation_error {
            let rot_error = gt_pose.rotation.angle_to(&interpolated_pose.rotation);
            rotation_errors.push(rot_error.powi(2));
        }
    }

    // --- Calculate RMSE ---
    let trans_rmse =
        (translation_errors.iter().sum::<f64>() / translation_errors.len() as f64).sqrt();
    println!("Translational RMSE: {}", trans_rmse);

    if rotation_error {
        let rot_rmse = (rotation_errors.iter().sum::<f64>() / rotation_errors.len() as f64).sqrt();
        println!("Rotational RMSE (radians): {}", rot_rmse);
        println!("Rotational RMSE (degrees): {}", rot_rmse.to_degrees());
    }

    // --- Visualization ---
    create_2d_plots(
        &corresponding_gt_trajectory,
        &interpolated_estimated_trajectory,
    )
    .unwrap();

    let mut window = Window::new("Trajectory Viewer");
    window.set_light(Light::StickToCamera);

    // Set up camera
    let eye = kiss3d::nalgebra::Point3::new(0.5, 0.5, 2.0);
    let at = kiss3d::nalgebra::Point3::origin();
    let mut camera = ArcBall::new(eye, at);

    while window.render_with_camera(&mut camera) {
        draw_world_frame(&mut window, 1.0); // Draw a frame of size 1.0

        // Draw ground truth trajectory (blue)
        for poses in ground_truth_trajectory.windows(2) {
            let start = kiss3d::nalgebra::Point3::new(
                poses[0].translation.vector.x,
                poses[0].translation.vector.y,
                poses[0].translation.vector.z,
            )
            .cast();
            let end = kiss3d::nalgebra::Point3::new(
                poses[1].translation.vector.x,
                poses[1].translation.vector.y,
                poses[1].translation.vector.z,
            )
            .cast();
            window.draw_line(&start, &end, &kiss3d::nalgebra::Point3::new(0.0, 0.0, 1.0));
        }

        // Draw interpolated estimated trajectory (red)
        for poses in interpolated_estimated_trajectory.windows(2) {
            let start = kiss3d::nalgebra::Point3::new(
                poses[0].translation.vector.x,
                poses[0].translation.vector.y,
                poses[0].translation.vector.z,
            )
            .cast();
            let end = kiss3d::nalgebra::Point3::new(
                poses[1].translation.vector.x,
                poses[1].translation.vector.y,
                poses[1].translation.vector.z,
            )
            .cast();
            window.draw_line(&start, &end, &kiss3d::nalgebra::Point3::new(1.0, 0.0, 0.0));
        }

        for (gt_pose, est_pose) in corresponding_gt_trajectory
            .iter()
            .zip(interpolated_estimated_trajectory.iter())
        {
            let start = kiss3d::nalgebra::Point3::new(
                gt_pose.translation.vector.x,
                gt_pose.translation.vector.y,
                gt_pose.translation.vector.z,
            )
            .cast();
            let end = kiss3d::nalgebra::Point3::new(
                est_pose.translation.vector.x,
                est_pose.translation.vector.y,
                est_pose.translation.vector.z,
            )
            .cast();

            window.draw_line(&start, &end, &kiss3d::nalgebra::Point3::new(0.5, 0.5, 0.5));
        }

        // for pose in ground_truth_trajectory.iter().step_by(20) {
        //     draw_camera_frustum(
        //         &mut window,
        //         pose,
        //         &kiss3d::nalgebra::Point3::new(0.0, 0.0, 1.0),
        //         0.1,
        //     );
        // }

        // Draw estimated camera frustums (red)
        for pose in interpolated_estimated_trajectory.iter().step_by(20) {
            draw_camera_frustum(
                &mut window,
                pose,
                &kiss3d::nalgebra::Point3::new(1.0, 0.0, 0.0),
                0.1,
            );
        }
    }

    Ok(())
}
