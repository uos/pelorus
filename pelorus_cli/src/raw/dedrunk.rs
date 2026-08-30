//! Optional `dedrunk` calibration after a guided recording.

use std::io::{IsTerminal, Write};
use std::path::Path;
use std::process::Command;

use anyhow::Result;
use log::{info, warn};

/// The binary this looks for on `PATH`.
const DEDRUNK: &str = "dedrunk";

/// Whether `dedrunk` can be run from here.
fn is_installed() -> bool {
    Command::new(DEDRUNK)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Offers to fit calibration parameters from a recorded bag.
///
/// Missing bags, unavailable `dedrunk`, and non-interactive sessions are skipped.
pub fn offer_fit(bag: Option<&Path>, assume_yes: bool) -> Result<()> {
    let Some(bag) = bag else {
        info!(
            "No bag was written, so there is nothing to calibrate from. \
             Re-run with --output-rosbag <DIR> to keep the recording."
        );
        return Ok(());
    };

    if !is_installed() {
        info!(
            "Recording finished. Install dedrunk (https://github.com/stelzo/dedrunk) \
             and run:\n    dedrunk intrinsics {}",
            bag.display()
        );
        return Ok(());
    }

    if !assume_yes && !prompt(bag)? {
        info!(
            "Fit it later with:\n    dedrunk intrinsics {}",
            bag.display()
        );
        return Ok(());
    }

    info!("Running dedrunk intrinsics {}", bag.display());
    let status = Command::new(DEDRUNK)
        .arg("intrinsics")
        .arg(bag)
        .status()
        .map_err(|error| anyhow::anyhow!("Failed to run {DEDRUNK}: {error}"))?;

    if !status.success() {
        // Calibration failure does not invalidate the saved recording.
        warn!(
            "dedrunk exited with {status}. The recording is unharmed; re-run:\n    \
             dedrunk intrinsics {}",
            bag.display()
        );
    }
    Ok(())
}

fn prompt(bag: &Path) -> Result<bool> {
    if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
        return Ok(false);
    }
    print!(
        "\nCompute the calibration now with dedrunk from '{}'? [Y/n]: ",
        bag.display()
    );
    std::io::stdout().flush()?;
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    let answer = answer.trim();
    Ok(answer.is_empty() || answer.eq_ignore_ascii_case("y"))
}
