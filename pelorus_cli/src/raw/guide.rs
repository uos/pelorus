//! Terminal guidance for a `pelorus online raw imu intrinsics` recording.

use std::io::Write;
use std::time::{Duration, Instant};

use imu_calib::estimate::{GuidedCalibration, GuidedConfig, Status};

/// How often the live progress line is redrawn.
const PROGRESS_INTERVAL: Duration = Duration::from_millis(100);

/// Drives the prompts for one guided recording.
pub struct Guide {
    session: GuidedCalibration,
    last_phase: Option<&'static str>,
    last_progress: Option<Instant>,
    progress_shown: bool,
}

impl Guide {
    /// Start a guided run over `poses` orientations.
    pub fn new(poses: usize) -> Self {
        let session = GuidedCalibration::new(GuidedConfig {
            num_poses: poses,
            // Only the prompting is wanted here. The fit runs later from the
            // bag, so there is no point spending the session on it.
            estimate_gyro_intrinsics: false,
            ..GuidedConfig::default()
        });

        println!(
            "\nGuided IMU intrinsic recording: {poses} orientations.\n\
             Spread them over all octants and rotate about a different axis each time —\n\
             the gyroscope scale is recovered from those rotations."
        );

        Self {
            session,
            last_phase: None,
            last_progress: None,
            progress_shown: false,
        }
    }

    /// Feed one sample. Returns true once every orientation has been captured.
    pub fn push(&mut self, t: f64, accel: [f64; 3], gyro: [f64; 3]) -> bool {
        let status = self.session.push(t, accel, gyro);
        self.show(status);
        matches!(status, Status::Done)
    }

    /// Orientations captured so far, for the closing summary.
    pub fn captured(&self) -> usize {
        self.session.captured_poses()
    }

    fn show(&mut self, status: Status) {
        let phase = match status {
            Status::Initialising { .. } => "init",
            Status::WaitingForMotion { .. } => "move",
            Status::WaitingForStillness { .. } => "settle",
            Status::Collecting { .. } => "collect",
            Status::PoseCaptured { .. } => "captured",
            Status::Done => "done",
        };

        if self.last_phase != Some(phase) {
            self.last_phase = Some(phase);
            self.clear_progress();
            println!("{}", Self::instruction(status));
            let _ = std::io::stdout().flush();
        }

        if let Some(text) = Self::progress(status) {
            self.print_progress(&text);
        }
    }

    fn instruction(status: Status) -> String {
        match status {
            Status::Initialising { .. } => "\nPut the IMU on a flat, solid surface and do not \
                 touch it.\nMeasuring the noise floor to set the motion thresholds..."
                .to_string(),
            Status::WaitingForMotion { next_pose, total } => format!(
                "\nPick the IMU up and set it down in a NEW orientation ({next_pose} of {total}).\n\
                 Tilt it about a different axis than last time."
            ),
            Status::WaitingForStillness { .. } => {
                "  Movement detected. Set it down and hold it still.".to_string()
            }
            Status::Collecting { pose, .. } => {
                format!("  Holding still, capturing orientation {pose}...")
            }
            Status::PoseCaptured { pose, total } => {
                format!("  Orientation {pose} of {total} captured.")
            }
            Status::Done => "\nAll orientations captured. Finalizing the recording...".to_string(),
        }
    }

    fn progress(status: Status) -> Option<String> {
        match status {
            Status::Initialising { collected, total } => {
                Some(format!("  measuring {collected}/{total} samples"))
            }
            Status::Collecting {
                collected, total, ..
            } => Some(format!("  capturing {collected}/{total} samples")),
            _ => None,
        }
    }

    /// Rate-limited, so a 200 Hz stream does not flood the terminal.
    fn print_progress(&mut self, text: &str) {
        let now = Instant::now();
        if let Some(last) = self.last_progress
            && now.duration_since(last) < PROGRESS_INTERVAL
        {
            return;
        }
        self.last_progress = Some(now);
        self.progress_shown = true;
        print!("\r{text}    ");
        let _ = std::io::stdout().flush();
    }

    fn clear_progress(&mut self) {
        if self.progress_shown {
            print!("\r{: <60}\r", "");
            self.progress_shown = false;
            self.last_progress = None;
        }
    }
}
