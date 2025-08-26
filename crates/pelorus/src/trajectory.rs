#![cfg(feature = "trajectory")]

use nalgebra::{Matrix3, Quaternion, Translation3, UnitQuaternion, SVD};
use plotters::prelude::*;
use plotters::style::full_palette::GREY;
use rerun as rr;
use rerun::external::glam;
use sophus_lie::traits::IsTranslationProductGroup;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

type SE3 = sophus_lie::Isometry3<f64, 1>;
type SO3 = sophus_lie::Rotation3<f64, 1>;

const PLOT_WIDTH: u32 = 800;
const PLOT_HEIGHT: u32 = 600;

fn pose_to_rerun_transform(pose: &Pose) -> rr::Transform3D {
    let correction = nalgebra::UnitQuaternion::from_axis_angle(
        &nalgebra::Vector3::x_axis(),
        -std::f32::consts::FRAC_PI_2,
    );

    let corrected_rot = pose.rotation.cast::<f32>() * correction;

    rr::Transform3D::from_translation_rotation(
        glam::Vec3::new(
            pose.translation.x as f32,
            pose.translation.y as f32,
            pose.translation.z as f32,
        ),
        rr::datatypes::Quaternion::from_xyzw([
            corrected_rot.i,
            corrected_rot.j,
            corrected_rot.k,
            corrected_rot.w,
        ]),
    )
}

fn get_rerun_color_for_error(error: f64, max_error: f64) -> rr::Color {
    let normalized_error = (error / max_error).min(1.0).max(0.0) as f32;
    rr::Color::from_rgb(
        (normalized_error * 255.0) as u8,
        0,
        ((1.0 - normalized_error) * 255.0) as u8,
    )
}

fn visualize_with_rerun(
    corresponding_gt_trajectory: &[Pose],
    aligned_est_trajectory: &[Pose],
    ape_data_for_plot: &[(f64, f64)],
    rpe_data_for_plot: &[(f64, f64)],
) -> anyhow::Result<()> {
    let rec = rr::RecordingStreamBuilder::new("Trajectory Analysis").spawn()?;

    rec.log("world", &rr::ViewCoordinates::RDF())?;
    rec.log_static(
        "world/axes",
        &rr::Arrows3D::from_vectors([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            .with_colors([[255, 0, 0], [0, 255, 0], [0, 0, 255]])
            .with_labels(["X", "Y", "Z"]),
    )?;

    let gt_points: Vec<glam::Vec3> = corresponding_gt_trajectory
        .iter()
        .map(|p| {
            glam::vec3(
                p.translation.x as f32,
                p.translation.y as f32,
                p.translation.z as f32,
            )
        })
        .collect();

    rec.log_static(
        "world/ground_truth_path",
        &rr::LineStrips3D::new([gt_points])
            .with_colors([rr::Color::from_rgb(128, 128, 128)])
            .with_labels(["Ground Truth"])
            .with_radii([rr::Radius::new_ui_points(0.85)]),
    )?;

    let max_ape_for_color = ape_data_for_plot
        .iter()
        .map(|(_, e)| *e)
        .fold(0.0, f64::max);

    let est_points: Vec<glam::Vec3> = aligned_est_trajectory
        .iter()
        .map(|p| {
            glam::vec3(
                p.translation.x as f32,
                p.translation.y as f32,
                p.translation.z as f32,
            )
        })
        .collect();
    let ape_colors: Vec<rr::Color> = ape_data_for_plot
        .iter()
        .map(|(_, err)| get_rerun_color_for_error(*err, max_ape_for_color))
        .collect();

    log_colored_trajectory(
        &rec,
        "world/estimated_path_ape",
        &est_points,
        &ape_colors,
        "Estimated (APE)",
        0.5,
    )?;

    let max_rpe_for_color = rpe_data_for_plot
        .iter()
        .map(|(_, e)| *e)
        .fold(0.0, f64::max);
    let rpe_points: Vec<glam::Vec3> = est_points[0..rpe_data_for_plot.len()].to_vec();
    let rpe_colors: Vec<rr::Color> = rpe_data_for_plot
        .iter()
        .map(|(_, err)| get_rerun_color_for_error(*err, max_rpe_for_color))
        .collect();

    log_colored_trajectory(
        &rec,
        "world/estimated_path_rpe",
        &rpe_points,
        &rpe_colors,
        "Estimated (RPE)",
        0.5,
    )?;

    for i in 0..corresponding_gt_trajectory.len() {
        let gt_pose = &corresponding_gt_trajectory[i];
        let est_pose = &aligned_est_trajectory[i];
        let (timestamp, ape) = ape_data_for_plot[i];

        rec.set_timestamp_secs_since_epoch("timestamp", timestamp);

        rec.log("world/ground_truth_pose", &pose_to_rerun_transform(gt_pose))?;
        rec.log("world/estimated_pose", &pose_to_rerun_transform(est_pose))?;

        rec.log(
            "world/estimated_pose/camera",
            &rr::Pinhole::from_focal_length_and_resolution(
                [300.0, 300.0], // Example focal length
                [640.0, 480.0], // Example resolution
            ),
        )?;

        // Log the line connecting the two poses to visualize the error
        let gt_point = glam::vec3(
            gt_pose.translation.x as f32,
            gt_pose.translation.y as f32,
            gt_pose.translation.z as f32,
        );
        let est_point = glam::vec3(
            est_pose.translation.x as f32,
            est_pose.translation.y as f32,
            est_pose.translation.z as f32,
        );
        rec.log(
            "world/error_line",
            &rr::LineStrips3D::new([[gt_point, est_point]])
                .with_colors([rr::Color::from_rgb(255, 128, 128)]),
        )?;

        rec.log("errors/ape_translation", &rr::Scalars::single(ape))?;
    }

    for (timestamp, rpe) in rpe_data_for_plot {
        rec.set_timestamp_secs_since_epoch("timestamp", *timestamp);
        rec.log("errors/rpe_translation", &rr::Scalars::single(*rpe))?;
    }

    Ok(())
}

fn log_colored_trajectory(
    rec: &rr::RecordingStream,
    entity_path: &str,
    points: &[glam::Vec3],
    colors: &[rr::Color],
    label: &str,
    radius: f32,
) -> anyhow::Result<()> {
    if points.len() < 2 {
        return Ok(());
    }

    let strips = points.windows(2).map(|segment| segment.to_vec());

    let segment_colors: Vec<rr::Color> = colors[0..points.len() - 1].to_vec();

    rec.log_static(
        entity_path,
        &rr::LineStrips3D::new(strips)
            .with_colors(segment_colors)
            .with_labels([label])
            .with_radii([rr::Radius::new_ui_points(radius)]),
    )?;

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct Pose {
    timestamp: f64,
    translation: Translation3<f64>,
    rotation: UnitQuaternion<f64>,
}

fn umeyama_alignment(
    gt_traj: &[Pose],
    est_traj: &[Pose],
) -> (f64, Matrix3<f64>, nalgebra::Vector3<f64>) {
    let gt_points: Vec<_> = gt_traj.iter().map(|p| p.translation.vector).collect();
    let est_points: Vec<_> = est_traj.iter().map(|p| p.translation.vector).collect();

    let gt_centroid: nalgebra::Vector3<f64> =
        gt_points.iter().sum::<nalgebra::Vector3<f64>>() / gt_points.len() as f64;
    let est_centroid: nalgebra::Vector3<f64> =
        est_points.iter().sum::<nalgebra::Vector3<f64>>() / est_points.len() as f64;

    let gt_centered: Vec<_> = gt_points.iter().map(|p| p - gt_centroid).collect();
    let est_centered: Vec<_> = est_points.iter().map(|p| p - est_centroid).collect();

    let mut h = Matrix3::<f64>::zeros();
    for i in 0..gt_points.len() {
        h += gt_centered[i] * est_centered[i].transpose();
    }
    h /= gt_points.len() as f64;

    let svd = SVD::new(h, true, true);
    let u = svd.u.unwrap();
    let vt = svd.v_t.unwrap();
    let s_diag = svd.singular_values;

    let mut s_identity = Matrix3::<f64>::identity();
    if (u * vt).determinant() < 0.0 {
        // Handle reflection case
        s_identity[(2, 2)] = -1.0;
    }
    let r = u * s_identity * vt;

    let var_est =
        est_centered.iter().map(|p| p.norm_squared()).sum::<f64>() / est_points.len() as f64;
    let s = (s_diag.component_mul(&s_identity.diagonal())).sum() / var_est;

    let t = gt_centroid - s * r * est_centroid;

    (s, r, t)
}

fn create_error_plot(
    filename: &str,
    title: &str,
    data: &[(f64, f64)],
) -> Result<(), Box<dyn Error>> {
    if data.is_empty() {
        println!("No data to plot for {}", title);
        return Ok(());
    }
    let root = SVGBackend::new(filename, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root.fill(&WHITE)?;

    let start_t = data[0].0;

    let max_t = data.iter().last().map_or(0.0, |(t, _)| *t);
    let max_err = data.iter().fold(0.0, |max: f64, (_, e)| max.max(*e));

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 40).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0.0..(max_t - start_t), 0.0..max_err * 1.1)?;

    chart
        .configure_mesh()
        .x_desc("Time (s)")
        .y_desc("Error")
        .draw()?;

    chart.draw_series(LineSeries::new(
        data.iter().map(|(t, e)| (*t - start_t, *e)),
        &RED,
    ))?;

    root.present()?;
    Ok(())
}

fn parse_pose(line: &str) -> Result<Pose, Box<dyn Error>> {
    let parts: Vec<f64> = line
        .split_whitespace()
        .map(|s| s.parse::<f64>())
        .collect::<Result<Vec<f64>, _>>()?;

    if parts.len() != 8 {
        return Err("Each line must have 8 values".into());
    }

    let translation = Translation3::new(parts[1], parts[2], parts[3]);

    Ok(Pose {
        timestamp: parts[0],
        translation,
        rotation: UnitQuaternion::from_quaternion(Quaternion::new(
            parts[7], parts[4], parts[5], parts[6], // w, i, j, k
        )),
    })
}

fn read_trajectory(path: &Path) -> Result<Vec<Pose>, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    reader.lines().map(|line| parse_pose(&line?)).collect()
}

fn pose_to_se3(pose: &Pose) -> SE3 {
    SE3::from_translation_and_rotation(
        &&pose.translation.vector,
        &SO3::try_from_mat(&&pose.rotation.to_rotation_matrix().matrix()).unwrap(),
    )
}

fn get_plotters_color_for_error(error: f64, max_error: f64) -> RGBColor {
    let normalized_error = (error / max_error).min(1.0).max(0.0);
    let r = (normalized_error * 255.0) as u8;
    let b = ((1.0 - normalized_error) * 255.0) as u8;
    RGBColor(r, 0, b)
}

fn interpolate_pose(p1: &Pose, p2: &Pose, timestamp: f64) -> Pose {
    let t = (timestamp - p1.timestamp) / (p2.timestamp - p1.timestamp);

    let p1 = SE3::from_translation_and_rotation(
        &&p1.translation.vector,
        &SO3::try_from_mat(&&p1.rotation.to_rotation_matrix().matrix()).unwrap(),
    );

    let p2 = SE3::from_translation_and_rotation(
        &p2.translation.vector,
        &SO3::try_from_mat(&&p2.rotation.to_rotation_matrix().matrix()).unwrap(),
    );

    let diff = p1.interpolate(&p2, t);

    Pose {
        timestamp,
        translation: Translation3::from(diff.translation()),
        rotation: UnitQuaternion::from_matrix(&diff.rotation().matrix()),
    }
}

fn create_2d_plots(
    gt_traj: &[Pose],
    est_traj: &[Pose],
    ape_data: &[(f64, f64)],
    rpe_data: &[(f64, f64)],
) -> Result<(), Box<dyn std::error::Error>> {
    let max_ape = ape_data.iter().map(|(_, e)| *e).fold(0.0, f64::max);

    let root_ape_side =
        SVGBackend::new("side_view_APE.svg", (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root_ape_side.fill(&WHITE)?;
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
    let mut chart_ape_side = ChartBuilder::on(&root_ape_side)
        .caption(
            "Side View Global Error (APE)",
            ("sans-serif", 50).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x_min..x_max, z_min..z_max)?;
    chart_ape_side.configure_mesh().draw()?;
    for i in 0..est_traj.len() - 1 {
        let p1 = &est_traj[i];
        let p2 = &est_traj[i + 1];
        let error = ape_data[i].1;
        let color = get_plotters_color_for_error(error, max_ape);
        chart_ape_side.draw_series(LineSeries::new(
            vec![
                (p1.translation.x, p1.translation.z),
                (p2.translation.x, p2.translation.z),
            ],
            color.stroke_width(2),
        ))?;
    }
    chart_ape_side
        .draw_series(LineSeries::new(vec![], RED.stroke_width(3)))?
        .label("High Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));
    chart_ape_side
        .draw_series(LineSeries::new(vec![], BLUE.stroke_width(3)))?
        .label("Low Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));
    chart_ape_side
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    let root_ape_top =
        SVGBackend::new("top_view_APE.svg", (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root_ape_top.fill(&WHITE)?;
    let (x_min_top, x_max_top, y_min_top, y_max_top) = gt_traj.iter().chain(est_traj.iter()).fold(
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
    let mut chart_ape_top = ChartBuilder::on(&root_ape_top)
        .caption(
            "Top View Global Error (APE)",
            ("sans-serif", 50).into_font(),
        )
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x_min_top..x_max_top, y_min_top..y_max_top)?;
    chart_ape_top.configure_mesh().draw()?;
    for i in 0..est_traj.len() - 1 {
        let p1 = &est_traj[i];
        let p2 = &est_traj[i + 1];
        let error = ape_data[i].1;
        let color = get_plotters_color_for_error(error, max_ape);
        chart_ape_top.draw_series(LineSeries::new(
            vec![
                (p1.translation.x, p1.translation.y),
                (p2.translation.x, p2.translation.y),
            ],
            color.stroke_width(2),
        ))?;
    }
    chart_ape_top
        .draw_series(LineSeries::new(vec![], RED.stroke_width(3)))?
        .label("High Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));
    chart_ape_top
        .draw_series(LineSeries::new(vec![], BLUE.stroke_width(3)))?
        .label("Low Error")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));
    chart_ape_top
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    let max_rpe = rpe_data.iter().map(|(_, e)| *e).fold(0.0, f64::max);

    let root_rpe_side =
        SVGBackend::new("side_view_drift_RPE.svg", (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root_rpe_side.fill(&WHITE)?;
    let mut chart_rpe_side = ChartBuilder::on(&root_rpe_side)
        .caption("Side View Drift (RPE)", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x_min..x_max, z_min..z_max)?;
    chart_rpe_side.configure_mesh().draw()?;
    for i in 0..rpe_data.len() {
        let p1 = &est_traj[i];
        let p2 = &est_traj[i + 1];
        let error = rpe_data[i].1;
        let color = get_plotters_color_for_error(error, max_rpe);
        chart_rpe_side.draw_series(LineSeries::new(
            vec![
                (p1.translation.x, p1.translation.z),
                (p2.translation.x, p2.translation.z),
            ],
            color.stroke_width(2),
        ))?;
    }
    for i in rpe_data.len()..est_traj.len() - 1 {
        let p1 = &est_traj[i];
        let p2 = &est_traj[i + 1];
        chart_rpe_side.draw_series(LineSeries::new(
            vec![
                (p1.translation.x, p1.translation.z),
                (p2.translation.x, p2.translation.z),
            ],
            GREY.stroke_width(2),
        ))?;
    }
    chart_rpe_side
        .draw_series(LineSeries::new(vec![], RED.stroke_width(3)))?
        .label("High Drift")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));
    chart_rpe_side
        .draw_series(LineSeries::new(vec![], BLUE.stroke_width(3)))?
        .label("Low Drift")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));
    chart_rpe_side
        .draw_series(LineSeries::new(vec![], GREY.stroke_width(3)))?
        .label("No RPE Data")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREY.filled()));
    chart_rpe_side
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    let root_rpe_top =
        SVGBackend::new("top_view_drift_RPE.svg", (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
    root_rpe_top.fill(&WHITE)?;
    let mut chart_rpe_top = ChartBuilder::on(&root_rpe_top)
        .caption("Top View Drift (RPE)", ("sans-serif", 50).into_font())
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(50)
        .build_cartesian_2d(x_min_top..x_max_top, y_min_top..y_max_top)?;
    chart_rpe_top.configure_mesh().draw()?;
    for i in 0..rpe_data.len() {
        let p1 = &est_traj[i];
        let p2 = &est_traj[i + 1];
        let error = rpe_data[i].1;
        let color = get_plotters_color_for_error(error, max_rpe);
        chart_rpe_top.draw_series(LineSeries::new(
            vec![
                (p1.translation.x, p1.translation.y),
                (p2.translation.x, p2.translation.y),
            ],
            color.stroke_width(2),
        ))?;
    }
    for i in rpe_data.len()..est_traj.len() - 1 {
        let p1 = &est_traj[i];
        let p2 = &est_traj[i + 1];
        chart_rpe_top.draw_series(LineSeries::new(
            vec![
                (p1.translation.x, p1.translation.y),
                (p2.translation.x, p2.translation.y),
            ],
            GREY.stroke_width(2),
        ))?;
    }
    chart_rpe_top
        .draw_series(LineSeries::new(vec![], RED.stroke_width(3)))?
        .label("High Drift")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED.filled()));
    chart_rpe_top
        .draw_series(LineSeries::new(vec![], BLUE.stroke_width(3)))?
        .label("Low Drift")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE.filled()));
    chart_rpe_top
        .draw_series(LineSeries::new(vec![], GREY.stroke_width(3)))?
        .label("No RPE Data")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], GREY.filled()));
    chart_rpe_top
        .configure_series_labels()
        .border_style(&BLACK)
        .draw()?;

    root_ape_side.present()?;
    root_ape_top.present()?;
    root_rpe_side.present()?;
    root_rpe_top.present()?;

    Ok(())
}

pub fn visualize_results(estimated_path: &Path, ground_truth_path: &Path) -> anyhow::Result<()> {
    let estimated_trajectory_raw = read_trajectory(estimated_path).unwrap();
    let ground_truth_trajectory = read_trajectory(ground_truth_path).unwrap();

    let mut interpolated_estimated_trajectory = Vec::new();
    let mut synced_gt_poses = Vec::new();

    for gt_pose in &ground_truth_trajectory {
        let mut p1_idx = 0;
        while p1_idx < estimated_trajectory_raw.len() - 2
            && estimated_trajectory_raw[p1_idx + 1].timestamp < gt_pose.timestamp
        {
            p1_idx += 1;
        }
        let p2_idx = p1_idx + 1;
        if p2_idx >= estimated_trajectory_raw.len() {
            continue;
        }

        let p1 = &estimated_trajectory_raw[p1_idx];
        let p2 = &estimated_trajectory_raw[p2_idx];

        let interpolated_pose = interpolate_pose(p1, p2, gt_pose.timestamp);
        interpolated_estimated_trajectory.push(interpolated_pose);
        synced_gt_poses.push(*gt_pose);
    }

    if synced_gt_poses.is_empty() {
        println!("No overlapping timestamps found. Cannot perform analysis.");
        return Ok(());
    }

    let first_gt_pose_inv = pose_to_se3(&synced_gt_poses[0]).inverse();
    let corresponding_gt_trajectory: Vec<Pose> = synced_gt_poses
        .iter()
        .map(|pose| {
            let se3_pose = pose_to_se3(pose);
            let normalized_se3 = first_gt_pose_inv.group_mul(&se3_pose);
            Pose {
                timestamp: pose.timestamp,
                translation: Translation3::from(normalized_se3.translation()),
                rotation: UnitQuaternion::from_matrix(&normalized_se3.rotation().matrix()),
            }
        })
        .collect();

    let (s, r, t) = umeyama_alignment(
        &corresponding_gt_trajectory,
        &interpolated_estimated_trajectory,
    );
    println!(
        "Umeyama Alignment Found: s={:.4}, t=[{:.4}, {:.4}, {:.4}]",
        s, t.x, t.y, t.z,
    );

    let aligned_est_trajectory: Vec<Pose> = interpolated_estimated_trajectory
        .iter()
        .map(|pose| {
            let aligned_trans = s * r * pose.translation.vector + t;
            let aligned_rot =
                UnitQuaternion::from_matrix(&(r * pose.rotation.to_rotation_matrix().matrix()));
            Pose {
                timestamp: pose.timestamp,
                translation: Translation3::from(aligned_trans),
                rotation: aligned_rot,
            }
        })
        .collect();

    let mut ape_trans_errors = Vec::new();
    let mut ape_rot_errors = Vec::new();
    let mut ape_data_for_plot = Vec::new();

    for (gt, est) in corresponding_gt_trajectory
        .iter()
        .zip(aligned_est_trajectory.iter())
    {
        let trans_error = (gt.translation.vector - est.translation.vector).norm();
        let rot_error = gt.rotation.angle_to(&est.rotation);
        ape_trans_errors.push(trans_error.powi(2));
        ape_rot_errors.push(rot_error.powi(2));
        ape_data_for_plot.push((gt.timestamp, trans_error));
    }

    let ape_trans_rmse =
        (ape_trans_errors.iter().sum::<f64>() / ape_trans_errors.len() as f64).sqrt();
    let ape_rot_rmse = (ape_rot_errors.iter().sum::<f64>() / ape_rot_errors.len() as f64).sqrt();
    println!("--- Absolute Pose Error (APE) ---");
    println!("Translational RMSE: {:.4} meters", ape_trans_rmse);
    println!("Rotational RMSE: {:.4} degrees", ape_rot_rmse.to_degrees());
    create_error_plot(
        "ape_plot.svg",
        "Absolute Pose Error (Translation)",
        &ape_data_for_plot,
    )
    .unwrap();

    let mut rpe_trans_errors = Vec::new();
    let mut rpe_rot_errors = Vec::new();
    let mut rpe_data_for_plot = Vec::new();
    let delta = 10;

    for i in 0..(corresponding_gt_trajectory.len() - delta) {
        let gt_pose1 = pose_to_se3(&corresponding_gt_trajectory[i]);
        let gt_pose2 = pose_to_se3(&corresponding_gt_trajectory[i + delta]);
        let est_pose1 = pose_to_se3(&aligned_est_trajectory[i]);
        let est_pose2 = pose_to_se3(&aligned_est_trajectory[i + delta]);
        let rel_gt = gt_pose1.inverse().group_mul(&gt_pose2);
        let rel_est = est_pose1.inverse().group_mul(&est_pose2);

        let error_se3 = rel_gt.inverse().group_mul(&rel_est);
        let trans_error = error_se3.translation().norm();
        let rot_error = error_se3.rotation().log().norm(); // Angle in radians

        rpe_trans_errors.push(trans_error.powi(2));
        rpe_rot_errors.push(rot_error.powi(2));
        rpe_data_for_plot.push((corresponding_gt_trajectory[i].timestamp, trans_error));
    }

    let rpe_trans_rmse =
        (rpe_trans_errors.iter().sum::<f64>() / rpe_trans_errors.len() as f64).sqrt();
    let rpe_rot_rmse = (rpe_rot_errors.iter().sum::<f64>() / rpe_rot_errors.len() as f64).sqrt();
    println!("--- Relative Pose Error (RPE, delta={}) ---", delta);
    println!("Translational RMSE: {:.4} meters", rpe_trans_rmse);
    println!("Rotational RMSE: {:.4} degrees", rpe_rot_rmse.to_degrees());
    create_error_plot(
        "rpe_plot.svg",
        "Relative Pose Error (Translation)",
        &rpe_data_for_plot,
    )
    .unwrap();

    create_2d_plots(
        &corresponding_gt_trajectory,
        &aligned_est_trajectory,
        &ape_data_for_plot,
        &rpe_data_for_plot,
    )
    .unwrap();

    visualize_with_rerun(
        &corresponding_gt_trajectory,
        &aligned_est_trajectory,
        &ape_data_for_plot,
        &rpe_data_for_plot,
    )?;

    Ok(())
}
