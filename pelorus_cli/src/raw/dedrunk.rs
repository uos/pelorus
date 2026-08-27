//! Optional hand-off to `dedrunk` once a guided recording is finished.
//!
//! Pelorus records the bag, `dedrunk intrinsics` fits the parameters.
//!
//! The hand-off is best-effort. `dedrunk` is a separate tool that
//! may not be installed, so its absence is a hint rather than a failure.

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

/// Offer to fit the calibration from the bag that was just recorded.
///
/// Does nothing when `dedrunk` is missing, when there is no bag to hand it, or
/// when the session is not interactive — an unattended recording should not
/// block on a prompt nobody will answer.
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
        // The recording itself is fine and is still on disk, so a failed fit
        // must not look like a failed run.
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
