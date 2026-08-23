//! Shared handling of on-disk recording targets: path derivation, topic
//! normalization, and the overwrite prompt.
//!
//! Both the odometry outputs of `pelorus online`/`offline` and the sensor
//! outputs of `pelorus online raw` land in the same kind of MCAP file or ROS 2
//! bag directory, so they answer the same questions the same way.

use std::io::{IsTerminal, Write};
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use log::info;

/// ROS topics are absolute; a bare name from the command line is still meant as
/// one.
pub fn normalize_mcap_topic(topic: impl Into<String>) -> String {
    let topic = topic.into();
    if topic.starts_with('/') {
        topic
    } else {
        format!("/{topic}")
    }
}

/// A ROS 2 bag directory holds its MCAP file under the directory's own name.
pub fn rosbag_mcap_path(rosbag_path: &Path) -> PathBuf {
    let bag_name = rosbag_path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or("output");
    rosbag_path.join(format!("{bag_name}.mcap"))
}

pub fn confirm_output_overwrite(label: &str, path: &Path, assume_yes: bool) -> Result<bool> {
    if !path.exists() {
        return Ok(true);
    }
    if assume_yes {
        return Ok(true);
    }

    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return Err(anyhow!(
            "{} '{}' already exists. This command is non-interactive, so refusing to overwrite it.",
            label,
            path.display()
        ));
    }

    print!(
        "{} '{}' already exists. Overwrite it? [y/N]: ",
        label,
        path.display()
    );
    std::io::stdout()
        .flush()
        .map_err(|e| anyhow!("Failed to flush prompt output: {}", e))?;

    let mut answer = String::new();
    std::io::stdin()
        .read_line(&mut answer)
        .map_err(|e| anyhow!("Failed to read overwrite confirmation: {}", e))?;

    match answer.trim().to_ascii_lowercase().as_str() {
        "y" | "yes" => Ok(true),
        "" | "n" | "no" => Ok(false),
        _ => Err(anyhow!(
            "Invalid response '{}'. Please answer with 'y' or 'n'.",
            answer.trim()
        )),
    }
}

/// Creates the bag directory and confirms every file that would be replaced.
///
/// Returns `false` when the user declined, which callers treat as a clean exit
/// rather than an error: nothing has been written yet at this point.
pub fn prepare_output_targets(
    rosbag_dir: Option<&Path>,
    mcap_path: Option<&Path>,
    rosbag_metadata_path: Option<&Path>,
    assume_yes: bool,
) -> Result<bool> {
    let mut rosbag_overwrite_confirmed = false;
    if let Some(rosbag_path) = rosbag_dir {
        if !confirm_output_overwrite("ROS bag output directory", rosbag_path, assume_yes)? {
            info!(
                "ROS bag output directory '{}' was not overwritten; exiting.",
                rosbag_path.display()
            );
            return Ok(false);
        }
        // Confirming the directory covers the files inside it, which are about
        // to be listed again individually.
        rosbag_overwrite_confirmed = rosbag_path.exists();
        std::fs::create_dir_all(rosbag_path).map_err(|e| {
            anyhow!(
                "Failed to create ROS bag output directory '{}': {}",
                rosbag_path.display(),
                e
            )
        })?;
        info!("ROS bag output directory: {}", rosbag_path.display());
    }
    if let Some(mcap_path) = mcap_path {
        if !confirm_output_overwrite(
            "MCAP output",
            mcap_path,
            assume_yes || rosbag_overwrite_confirmed,
        )? {
            info!(
                "MCAP output '{}' was not overwritten; exiting.",
                mcap_path.display()
            );
            return Ok(false);
        }
        info!("MCAP output target: {}", mcap_path.display());
        if let Some(metadata_path) = rosbag_metadata_path {
            if !confirm_output_overwrite(
                "ROS bag metadata",
                metadata_path,
                assume_yes || rosbag_overwrite_confirmed,
            )? {
                info!(
                    "ROS bag metadata '{}' was not overwritten; exiting.",
                    metadata_path.display()
                );
                return Ok(false);
            }
            info!("ROS bag metadata target: {}", metadata_path.display());
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_bare_topic_names() {
        assert_eq!(normalize_mcap_topic("imu"), "/imu");
        assert_eq!(normalize_mcap_topic("/imu"), "/imu");
    }

    #[test]
    fn bag_directory_holds_mcap_named_after_itself() {
        assert_eq!(
            rosbag_mcap_path(Path::new("/tmp/session_01")),
            PathBuf::from("/tmp/session_01/session_01.mcap")
        );
    }
}
