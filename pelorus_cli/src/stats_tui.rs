use alloc::sync::Arc;
use core::sync::atomic::{AtomicBool, Ordering};
use core::time::Duration;
use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{Terminal, backend::CrosstermBackend, layout::Rect};
use std::io::{self, IsTerminal};
use std::sync::Mutex;
use std::time::Instant;

use mt_log::{LogBuffer, LogLevel, Logger};
use pelorus_core::LidarVendor;
pub use pelorus_tui::TuiStats;
use pelorus_tui::{Key, KeyAction, KeyPress, StatsView};

const MAX_TRAJECTORY_POINTS: usize = 10000;

/// The log buffer the pane draws, filled by the logger installed below.
static LOG_BUFFER: std::sync::OnceLock<Arc<LogBuffer>> = std::sync::OnceLock::new();
/// Routes log records to the pane while the TUI owns the terminal.
static TUI_ACTIVE: std::sync::OnceLock<Arc<AtomicBool>> = std::sync::OnceLock::new();
/// Kept so the filter can be replaced once the config has been read.
static LOGGER: std::sync::OnceLock<&'static Logger> = std::sync::OnceLock::new();

/// Get or initialize the global log buffer.
pub fn get_log_buffer() -> Arc<LogBuffer> {
    LOG_BUFFER
        .get_or_init(|| Arc::new(LogBuffer::new("pelorus")))
        .clone()
}

fn tui_active() -> Arc<AtomicBool> {
    TUI_ACTIVE
        .get_or_init(|| Arc::new(AtomicBool::new(false)))
        .clone()
}

/// Chatter from the Hiroz backend that no reader has ever wanted.
fn is_noisy(level: LogLevel, _target: &str, message: &str) -> bool {
    level == LogLevel::Info
        && (message.starts_with("pub_build;")
            || message.starts_with("[SRV] Declaring queryable on key expression:"))
}

/// Install the TUI-aware logger. Call this early in main().
///
/// Starts at the level `RUST_LOG` says, or info. [`update_log_level`] applies
/// whatever the config asks for once it has been read.
pub fn init_logger() {
    let spec = log_spec();
    if let Ok(logger) = Logger::install_with(
        "Pelorus",
        &spec,
        get_log_buffer(),
        tui_active(),
        Some(is_noisy),
    ) {
        let _ = LOGGER.set(logger);
    }

    // Suppress console output immediately when running in a terminal. Any log
    // messages before the TUI starts would otherwise end up in the main buffer
    // and reappear after LeaveAlternateScreen. If we later determine TUI won't
    // be used, call set_tui_active(false) to restore stderr output.
    if std::io::stdout().is_terminal() {
        set_tui_active(true);
    }
}

/// Update the log filter after configuration is loaded.
pub fn update_log_level() {
    if let Some(logger) = LOGGER.get() {
        logger.set_spec(&log_spec());
    }
}

/// What to filter by, with the transport held out of it.
///
/// The configured level arrives here through `RUST_LOG` as a bare level, which
/// on its own lets Zenoh narrate every expired query at warn -- and a
/// best-effort peer walking away makes every publisher a source of that at
/// once. Whatever the spec says otherwise is left alone.
fn log_spec() -> String {
    mt_log::quieted(&std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_owned()))
}

pub fn set_tui_active(active: bool) {
    tui_active().store(active, Ordering::Relaxed);
}

/// Re-emit one line of a child's output as a record of ours.
///
/// Parsed child records keep their original level, source, and target. Records
/// pass through the `log` facade so configured filtering applies.
pub fn relay_child_line(line: &str, source: &str) {
    let record = mt_log::parse(line).or_source(source);
    log::log!(
        target: "pelorus_cli::minot",
        record.level.into(),
        "{}",
        record.pane_message()
    );
}

/// Thread-safe wrapper for TuiStats
pub struct SharedStats {
    inner: Arc<Mutex<TuiStats>>,
    start_time: Instant,
    last_position: Arc<Mutex<Option<[f64; 3]>>>,
    comp_times: Arc<Mutex<Vec<f64>>>,
    cpu_percents: Arc<Mutex<Vec<f64>>>,
    last_comp_time: Arc<Mutex<Option<Instant>>>,
    scan_timestamps: Arc<Mutex<Vec<Instant>>>,
    scan_data_timestamps: Arc<Mutex<Vec<f64>>>,
    trajectory: Arc<Mutex<Vec<[f64; 3]>>>,

    #[cfg(target_os = "linux")]
    last_cpu: Arc<Mutex<Option<(u64, u64, u64)>>>,

    #[cfg(target_os = "macos")]
    last_cpu_macos: Arc<Mutex<Option<(u64, f64)>>>,
}

impl Default for SharedStats {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedStats {
    pub fn new() -> Self {
        let num_cores = num_cpus::get();
        let inner = Arc::new(Mutex::new(TuiStats {
            num_cores,
            system_stats: true,
            backend: "CPU".to_string(),
            accelerator: None,
            target_lidar_rate_hz: 10.0,
            ..Default::default()
        }));

        Self {
            inner,
            start_time: Instant::now(),
            last_position: Arc::new(Mutex::new(None)),
            comp_times: Arc::new(Mutex::new(Vec::new())),
            cpu_percents: Arc::new(Mutex::new(Vec::new())),
            last_comp_time: Arc::new(Mutex::new(None)),
            scan_timestamps: Arc::new(Mutex::new(Vec::new())),
            scan_data_timestamps: Arc::new(Mutex::new(Vec::new())),
            trajectory: Arc::new(Mutex::new(Vec::with_capacity(MAX_TRAJECTORY_POINTS))),
            #[cfg(target_os = "linux")]
            last_cpu: Arc::new(Mutex::new(None)),
            #[cfg(target_os = "macos")]
            last_cpu_macos: Arc::new(Mutex::new(None)),
        }
    }

    pub fn set_mode(&self, mode: impl Into<String>) {
        let mut stats = self.inner.lock().unwrap();
        stats.mode = mode.into();
    }

    pub fn set_bagfile(&self, bagfile: impl Into<String>) {
        let mut stats = self.inner.lock().unwrap();
        stats.bagfile = Some(bagfile.into());
    }

    pub fn set_backend_info(
        &self,
        backend: String,
        accelerator: Option<String>,
        accelerator_details: Option<String>,
    ) {
        let mut stats = self.inner.lock().unwrap();
        stats.backend = backend;
        stats.accelerator = accelerator;
        stats.accelerator_details = accelerator_details;
    }

    pub fn get(&self) -> TuiStats {
        let mut stats = self.inner.lock().unwrap().clone();
        stats.elapsed_secs = self.start_time.elapsed().as_secs_f64();
        stats.trajectory = self.trajectory.lock().unwrap().clone();
        stats
    }

    pub fn set_target_lidar_rate(&self, rate_hz: f64) {
        let mut stats = self.inner.lock().unwrap();
        stats.target_lidar_rate_hz = rate_hz;
    }

    pub fn record_scan(&self) {
        let now = Instant::now();
        let mut timestamps = self.scan_timestamps.lock().unwrap();
        timestamps.push(now);

        let len = timestamps.len();
        if len > 100 {
            timestamps.drain(0..len - 100);
        }

        let measured_rate = if timestamps.len() >= 2 {
            let oldest = timestamps.first().unwrap();
            let newest = timestamps.last().unwrap();
            let duration = newest.duration_since(*oldest).as_secs_f64();
            if duration > 0.0 {
                (timestamps.len() - 1) as f64 / duration
            } else {
                0.0
            }
        } else {
            0.0
        };

        let mut stats = self.inner.lock().unwrap();
        stats.measured_sensor_rate_hz = measured_rate;
    }

    pub fn record_scan_timestamp(&self, stamp_sec: f64) {
        if stamp_sec <= 0.0 {
            self.record_scan();
            return;
        }

        let mut timestamps = self.scan_data_timestamps.lock().unwrap();
        if let Some(&last) = timestamps.last()
            && stamp_sec <= last
        {
            return;
        }
        timestamps.push(stamp_sec);

        let len = timestamps.len();
        if len > 100 {
            timestamps.drain(0..len - 100);
        }

        let measured_rate = if timestamps.len() >= 2 {
            let oldest = timestamps.first().unwrap();
            let newest = timestamps.last().unwrap();
            let duration = newest - oldest;
            if duration > 0.0 {
                (timestamps.len() - 1) as f64 / duration
            } else {
                0.0
            }
        } else {
            0.0
        };

        let mut stats = self.inner.lock().unwrap();
        stats.measured_sensor_rate_hz = measured_rate;
    }

    pub fn update_pose(
        &self,
        position: [f64; 3],
        orientation: [f64; 4],
        linear_velocity: [f64; 3],
        angular_velocity: [f64; 3],
    ) {
        let mut stats = self.inner.lock().unwrap();
        let mut last_pos = self.last_position.lock().unwrap();
        let mut trajectory = self.trajectory.lock().unwrap();

        if let Some(prev) = *last_pos {
            let dx = position[0] - prev[0];
            let dy = position[1] - prev[1];
            let dz = position[2] - prev[2];
            let dist = (dx * dx + dy * dy + dz * dz).sqrt();
            if dist >= 0.01 {
                stats.distance_traveled += dist;
                *last_pos = Some(position);

                // Add to trajectory
                trajectory.push(position);
                if trajectory.len() > MAX_TRAJECTORY_POINTS {
                    trajectory.remove(0);
                }
            }
        } else {
            *last_pos = Some(position);
            trajectory.push(position);
        }

        stats.distance_to_origin =
            (position[0].powi(2) + position[1].powi(2) + position[2].powi(2)).sqrt();

        stats.position = position;
        stats.orientation = orientation;
        stats.linear_velocity = linear_velocity;
        stats.angular_velocity = angular_velocity;
    }

    pub fn update_bias(&self, accel_bias: [f64; 3], gyro_bias: [f64; 3]) {
        let mut stats = self.inner.lock().unwrap();
        stats.accel_bias = accel_bias;
        stats.gyro_bias = gyro_bias;
    }

    pub fn set_lidar_vendor(&self, vendor: LidarVendor) {
        let mut stats = self.inner.lock().unwrap();
        stats.lidar_vendor = Some(vendor);
    }

    pub fn record_comp_time(&self, time_ms: f64) {
        let mut times = self.comp_times.lock().unwrap();
        times.push(time_ms);

        let len = times.len();
        if len > 1000 {
            times.drain(0..len - 1000);
        }

        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let max = times.iter().cloned().fold(0.0_f64, f64::max);

        let mut last_time = self.last_comp_time.lock().unwrap();
        let freq_hz = if let Some(prev) = *last_time {
            let elapsed = prev.elapsed().as_secs_f64();
            if elapsed > 0.0 { 1.0 / elapsed } else { 0.0 }
        } else {
            0.0
        };
        *last_time = Some(Instant::now());

        let mut stats = self.inner.lock().unwrap();
        stats.last_comp_time_ms = time_ms;
        stats.avg_comp_time_ms = avg;
        stats.max_comp_time_ms = max;
        stats.processing_freq_hz = freq_hz;

        let sensor_rate = if stats.measured_sensor_rate_hz > 0.0 {
            stats.measured_sensor_rate_hz
        } else {
            stats.target_lidar_rate_hz
        };

        if sensor_rate > 0.0 && avg > 0.0 {
            let sensor_period_ms = 1000.0 / sensor_rate;
            stats.real_time_factor = sensor_period_ms / avg;
        }
    }

    pub fn update_system_stats(&self) {
        let ram_mb = get_ram_usage_mb();

        #[cfg(target_os = "linux")]
        let cpu_percent = {
            let mut last = self.last_cpu.lock().unwrap();
            let (percent, new_state) = get_cpu_usage_linux(last.take());
            *last = Some(new_state);
            percent
        };

        #[cfg(target_os = "macos")]
        let cpu_percent = {
            let mut last = self.last_cpu_macos.lock().unwrap();
            let (percent, new_state) = get_cpu_usage_macos(last.take());
            *last = Some(new_state);
            percent
        };

        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        let cpu_percent = 0.0;

        let mut percents = self.cpu_percents.lock().unwrap();
        if cpu_percent >= 0.0 {
            percents.push(cpu_percent);
            let len = percents.len();
            if len > 100 {
                percents.drain(0..len - 100);
            }
        }

        let avg_cpu = if percents.is_empty() {
            0.0
        } else {
            percents.iter().sum::<f64>() / percents.len() as f64
        };
        let max_cpu = percents.iter().cloned().fold(0.0_f64, f64::max);

        let mut stats = self.inner.lock().unwrap();
        stats.ram_mb = ram_mb;
        stats.cpu_percent = cpu_percent.max(0.0);
        stats.avg_cpu_percent = avg_cpu;
        stats.max_cpu_percent = max_cpu;

        #[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
        {
            let (gpu_used, gpu_total, gpu_util) = get_gpu_stats();
            stats.gpu_memory_used_mb = gpu_used;
            stats.gpu_memory_total_mb = gpu_total;
            stats.gpu_utilization = gpu_util;
        }
    }
}

// ============================================================================
// System stats helpers
// ============================================================================

fn get_ram_usage_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
            for line in content.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:")
                    && let Some(kb) = rest
                        .split_whitespace()
                        .next()
                        .and_then(|v| v.parse::<f64>().ok())
                {
                    return kb / 1024.0;
                }
            }
        }
        0.0
    }

    #[cfg(target_os = "macos")]
    {
        unsafe {
            let mut rusage: libc::rusage = std::mem::zeroed();
            if libc::getrusage(libc::RUSAGE_SELF, &mut rusage) == 0 {
                // On macOS ru_maxrss is in bytes
                return rusage.ru_maxrss as f64 / (1024.0 * 1024.0);
            }
        }
        0.0
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    0.0
}

#[cfg(target_os = "linux")]
fn get_cpu_usage_linux(last: Option<(u64, u64, u64)>) -> (f64, (u64, u64, u64)) {
    use std::time::SystemTime;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let (utime, stime) = if let Ok(content) = std::fs::read_to_string("/proc/self/stat") {
        let parts: Vec<&str> = content.split_whitespace().collect();
        if parts.len() > 14 {
            (
                parts[13].parse::<u64>().unwrap_or(0),
                parts[14].parse::<u64>().unwrap_or(0),
            )
        } else {
            (0, 0)
        }
    } else {
        (0, 0)
    };

    let current = (now, stime, utime);

    if let Some((last_clock, last_sys, last_user)) = last {
        let clock_diff = now.saturating_sub(last_clock) as f64;
        if clock_diff > 0.0 {
            let cpu_time = ((stime - last_sys) + (utime - last_user)) as f64;
            let cpu_ms = cpu_time * 10.0;
            let percent = (cpu_ms / clock_diff) * 100.0;
            return (percent, current);
        }
    }

    (-1.0, current)
}

#[cfg(target_os = "macos")]
fn get_cpu_usage_macos(last: Option<(u64, f64)>) -> (f64, (u64, f64)) {
    use std::time::SystemTime;

    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);

    let cpu_time_secs = unsafe {
        let mut rusage: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut rusage) == 0 {
            let user = rusage.ru_utime.tv_sec as f64 + rusage.ru_utime.tv_usec as f64 / 1_000_000.0;
            let sys = rusage.ru_stime.tv_sec as f64 + rusage.ru_stime.tv_usec as f64 / 1_000_000.0;
            user + sys
        } else {
            0.0
        }
    };

    let current = (now, cpu_time_secs);

    if let Some((last_time, last_cpu_time)) = last {
        let time_diff_ms = now.saturating_sub(last_time) as f64;
        if time_diff_ms > 0.0 {
            let cpu_diff = cpu_time_secs - last_cpu_time;
            let percent = (cpu_diff * 1000.0 / time_diff_ms) * 100.0;
            return (percent, current);
        }
    }

    (-1.0, current)
}

#[cfg(any(feature = "cuda", feature = "metal", feature = "vulkan"))]
fn get_gpu_stats() -> (f64, f64, f64) {
    #[cfg(feature = "metal")]
    {
        #[cfg(target_os = "macos")]
        {
            let used = pelorus_metal::get_used_memory() as f64 / (1024.0 * 1024.0);
            if used > 0.0 {
                return (used, 0.0, 0.0);
            }
        }
    }

    #[cfg(feature = "cuda")]
    {
        let is_jetson = std::path::Path::new("/sys/devices/platform/host1x").exists()
            || std::fs::read_to_string("/proc/device-tree/model")
                .map(|s| s.to_lowercase().contains("jetson"))
                .unwrap_or(false);

        let stats = if is_jetson {
            get_jetson_gpu_stats()
        } else {
            get_nvidia_gpu_stats()
        };
        if stats != (0.0, 0.0, 0.0) {
            return stats;
        }
    }

    #[cfg(feature = "vulkan")]
    {
        let used = pelorus_vulkan::get_used_memory() as f64 / (1024.0 * 1024.0);
        if used > 0.0 {
            return (used, 0.0, 0.0);
        }
    }

    (0.0, 0.0, 0.0)
}

#[cfg(feature = "cuda")]
fn get_jetson_gpu_stats() -> (f64, f64, f64) {
    let gpu_util = std::fs::read_to_string("/sys/devices/platform/host1x/gpu.0/load")
        .or_else(|_| std::fs::read_to_string("/sys/devices/gpu.0/load"))
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|v| v / 10.0)
        .unwrap_or(0.0);
    (0.0, 0.0, gpu_util)
}

#[cfg(feature = "cuda")]
fn get_nvidia_gpu_stats() -> (f64, f64, f64) {
    use std::process::Command;

    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();

    if let Ok(output) = output
        && output.status.success()
        && let Ok(stdout) = std::str::from_utf8(&output.stdout)
    {
        let parts: Vec<&str> = stdout.trim().split(',').map(|s| s.trim()).collect();
        if parts.len() >= 3 {
            return (
                parts[0].parse().unwrap_or(0.0),
                parts[1].parse().unwrap_or(0.0),
                parts[2].parse().unwrap_or(0.0),
            );
        }
    }
    (0.0, 0.0, 0.0)
}

/// The status view in the terminal, redrawn from [`SharedStats`] until the
/// user quits or a stop is requested.
pub struct StatsTui {
    stats: Arc<SharedStats>,
    stop_rx: Option<std::sync::mpsc::Receiver<()>>,
    shutdown_tx: Option<std::sync::mpsc::Sender<()>>,
    view: StatsView,
}

impl StatsTui {
    pub fn new(stats: Arc<SharedStats>) -> Self {
        Self {
            stats,
            stop_rx: None,
            shutdown_tx: None,
            view: StatsView::new(get_log_buffer()),
        }
    }

    pub fn with_stop_receiver(mut self, rx: std::sync::mpsc::Receiver<()>) -> Self {
        self.stop_rx = Some(rx);
        self
    }

    pub fn with_shutdown_sender(mut self, tx: std::sync::mpsc::Sender<()>) -> Self {
        self.shutdown_tx = Some(tx);
        self
    }

    pub fn run(&mut self) -> io::Result<()> {
        set_tui_active(true);
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let mut last_system_update = Instant::now();

        loop {
            if last_system_update.elapsed().as_millis() > 500 {
                self.stats.update_system_stats();
                last_system_update = Instant::now();
            }

            let stats = self.stats.get();
            self.view.refresh_dynamic_layout_thresholds(&stats);

            terminal.draw(|f| {
                self.view.render(f, &stats);
            })?;

            if event::poll(Duration::from_millis(50))?
                && let Event::Key(key) = event::read()?
                && let Some(key) = key_press(key)
            {
                let size = terminal.size()?;
                let area = Rect::new(0, 0, size.width, size.height);
                if self.view.handle_key(key, area) == KeyAction::Exit {
                    disable_raw_mode()?;
                    execute!(terminal.backend_mut(), LeaveAlternateScreen, cursor::Show)?;
                    self.shutdown_tx.as_ref().map(|sender| sender.send(()));
                    return Ok(());
                }
            }

            if let Some(rx) = &self.stop_rx
                && rx.try_recv().is_ok()
            {
                // External shutdown requested. Forward to main via shutdown_tx and exit.
                if let Some(tx) = &self.shutdown_tx {
                    let _ = tx.send(());
                }
                break;
            }
        }

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen, cursor::Show)?;
        Ok(())
    }
}

impl Drop for StatsTui {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen);
    }
}

/// The view's key for a terminal key event, or `None` for keys it ignores.
fn key_press(event: event::KeyEvent) -> Option<KeyPress> {
    let key = match event.code {
        KeyCode::Char(c) => Key::Char(c),
        KeyCode::Tab => Key::Tab,
        KeyCode::Esc => Key::Esc,
        KeyCode::Up => Key::Up,
        KeyCode::Down => Key::Down,
        KeyCode::PageUp => Key::PageUp,
        KeyCode::PageDown => Key::PageDown,
        KeyCode::Home => Key::Home,
        KeyCode::End => Key::End,
        _ => return None,
    };
    Some(KeyPress {
        key,
        ctrl: event.modifiers.contains(KeyModifiers::CONTROL),
    })
}
