use crossterm::{
    cursor,
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols::Marker,
    text::{Line, Span},
    widgets::{
        Block, Borders, List, ListItem, Paragraph, Tabs,
        canvas::{Canvas, Line as CanvasLine, Points},
    },
};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use pelorus_core::LidarVendor;

const MAX_TRAJECTORY_POINTS: usize = 10000;
const MAX_LOG_ENTRIES: usize = 1000;

/// Log level for TUI display
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

impl LogLevel {
    fn color(&self) -> Color {
        match self {
            LogLevel::Trace => Color::DarkGray,
            LogLevel::Debug => Color::Gray,
            LogLevel::Info => Color::Cyan,
            LogLevel::Warn => Color::Yellow,
            LogLevel::Error => Color::Red,
        }
    }

    fn prefix(&self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO ",
            LogLevel::Warn => "WARN ",
            LogLevel::Error => "ERROR",
        }
    }
}

impl From<log::Level> for LogLevel {
    fn from(level: log::Level) -> Self {
        match level {
            log::Level::Trace => LogLevel::Trace,
            log::Level::Debug => LogLevel::Debug,
            log::Level::Info => LogLevel::Info,
            log::Level::Warn => LogLevel::Warn,
            log::Level::Error => LogLevel::Error,
        }
    }
}

/// A single log entry
#[derive(Debug, Clone)]
pub struct LogEntry {
    pub timestamp: f64,
    pub level: LogLevel,
    pub message: String,
}

/// Global log buffer accessible from anywhere
static LOG_BUFFER: std::sync::OnceLock<Arc<Mutex<LogBuffer>>> = std::sync::OnceLock::new();
static TUI_ACTIVE: AtomicBool = AtomicBool::new(false);
static LOG_START_TIME: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();

pub struct LogBuffer {
    entries: Vec<LogEntry>,
}

impl LogBuffer {
    fn new() -> Self {
        Self {
            entries: Vec::with_capacity(MAX_LOG_ENTRIES),
        }
    }

    pub fn push(&mut self, level: LogLevel, message: String) {
        let start = LOG_START_TIME.get_or_init(Instant::now);
        let timestamp = start.elapsed().as_secs_f64();
        self.entries.push(LogEntry {
            timestamp,
            level,
            message,
        });
        if self.entries.len() > MAX_LOG_ENTRIES {
            self.entries.remove(0);
        }
    }

    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Get or initialize the global log buffer
pub fn get_log_buffer() -> Arc<Mutex<LogBuffer>> {
    LOG_BUFFER
        .get_or_init(|| Arc::new(Mutex::new(LogBuffer::new())))
        .clone()
}

/// Custom logger that captures to TUI buffer and also prints to console
pub struct TuiLogger;

impl log::Log for TuiLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        // Use the max level set by init_logger
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &log::Record) {
        if !self.enabled(record.metadata()) {
            return;
        }

        let message = format!("{}", record.args());
        let level = LogLevel::from(record.level());

        // Always capture to buffer for TUI
        if let Some(buffer) = LOG_BUFFER.get() {
            if let Ok(mut buf) = buffer.lock() {
                buf.push(level, message.clone());
            }
        }

        // Only print to console if TUI is NOT active
        if !TUI_ACTIVE.load(Ordering::Relaxed) {
            let prefix = match record.level() {
                log::Level::Error => "\x1b[31mERROR\x1b[0m",
                log::Level::Warn => "\x1b[33mWARN\x1b[0m",
                log::Level::Info => "\x1b[36mINFO\x1b[0m",
                log::Level::Debug => "\x1b[90mDEBUG\x1b[0m",
                log::Level::Trace => "\x1b[90mTRACE\x1b[0m",
            };
            eprintln!("[{}] {}", prefix, message);
        }
    }

    fn flush(&self) {}
}

static LOGGER: TuiLogger = TuiLogger;

/// Initialize the TUI-aware logger. Call this early in main().
/// Starts with Info level, can be reconfigured later with update_log_level().
pub fn init_logger() {
    // Initialize the log buffer first
    let _ = get_log_buffer();
    let _ = LOG_START_TIME.get_or_init(Instant::now);

    if log::set_logger(&LOGGER).is_ok() {
        // Start with Info, will be updated after config is loaded
        log::set_max_level(log::LevelFilter::Info);
    }
}

/// Update the log level after configuration is loaded.
/// Call this after config has set RUST_LOG environment variable.
pub fn update_log_level() {
    let level = std::env::var("RUST_LOG")
        .ok()
        .and_then(|s| match s.to_lowercase().as_str() {
            "trace" => Some(log::LevelFilter::Trace),
            "debug" => Some(log::LevelFilter::Debug),
            "info" => Some(log::LevelFilter::Info),
            "warn" | "warning" => Some(log::LevelFilter::Warn),
            "error" => Some(log::LevelFilter::Error),
            "off" => Some(log::LevelFilter::Off),
            _ => None,
        })
        .unwrap_or(log::LevelFilter::Info);

    log::set_max_level(level);
}

/// Mark TUI as active (suppresses console output)
pub fn set_tui_active(active: bool) {
    TUI_ACTIVE.store(active, Ordering::Relaxed);
}

/// Shared statistics for the TUI
#[derive(Debug, Clone, Default)]
pub struct TuiStats {
    // Mode
    pub mode: String,
    pub bagfile: Option<String>,

    // Sensor info
    pub lidar_vendor: Option<LidarVendor>,
    pub target_lidar_rate_hz: f64,

    // State
    pub position: [f64; 3],
    pub orientation: [f64; 4], // [w, x, y, z]
    pub linear_velocity: [f64; 3],
    pub angular_velocity: [f64; 3],
    pub accel_bias: [f64; 3],
    pub gyro_bias: [f64; 3],

    // Distance tracking
    pub distance_traveled: f64,
    pub distance_to_origin: f64,

    // Timing / Performance
    pub last_comp_time_ms: f64,
    pub avg_comp_time_ms: f64,
    pub max_comp_time_ms: f64,
    pub processing_freq_hz: f64,
    pub measured_sensor_rate_hz: f64,
    pub real_time_factor: f64,

    // System
    pub cpu_percent: f64,
    pub avg_cpu_percent: f64,
    pub max_cpu_percent: f64,
    pub ram_mb: f64,
    pub num_cores: usize,

    // GPU (CUDA/Jetson)
    #[cfg(feature = "cuda")]
    pub gpu_memory_used_mb: f64,
    #[cfg(feature = "cuda")]
    pub gpu_memory_total_mb: f64,
    #[cfg(feature = "cuda")]
    pub gpu_utilization: f64,

    // Elapsed
    pub elapsed_secs: f64,

    // Trajectory history
    pub trajectory: Vec<[f64; 3]>,
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
    trajectory: Arc<Mutex<Vec<[f64; 3]>>>,

    #[cfg(target_os = "linux")]
    last_cpu: Arc<Mutex<Option<(u64, u64, u64)>>>,

    #[cfg(target_os = "macos")]
    last_cpu_macos: Arc<Mutex<Option<(u64, f64)>>>,
}

impl SharedStats {
    pub fn new() -> Self {
        let num_cores = num_cpus::get();
        let inner = Arc::new(Mutex::new(TuiStats {
            num_cores,
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

        if sensor_rate > 0.0 && freq_hz > 0.0 {
            stats.real_time_factor = freq_hz / sensor_rate;
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

        #[cfg(feature = "cuda")]
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
        if let Ok(content) = std::fs::read_to_string("/proc/self/stat") {
            let parts: Vec<&str> = content.split_whitespace().collect();
            if parts.len() > 23 {
                if let Ok(rss) = parts[23].parse::<i64>() {
                    let page_size = unsafe { libc::sysconf(libc::_SC_PAGE_SIZE) } as f64;
                    return (rss as f64 * page_size) / (1024.0 * 1024.0);
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

#[cfg(feature = "cuda")]
fn get_gpu_stats() -> (f64, f64, f64) {
    let is_jetson = std::path::Path::new("/sys/devices/platform/host1x").exists()
        || std::fs::read_to_string("/proc/device-tree/model")
            .map(|s| s.to_lowercase().contains("jetson"))
            .unwrap_or(false);

    if is_jetson {
        get_jetson_gpu_stats()
    } else {
        get_nvidia_gpu_stats()
    }
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

    if let Ok(output) = output {
        if output.status.success() {
            if let Ok(stdout) = std::str::from_utf8(&output.stdout) {
                let parts: Vec<&str> = stdout.trim().split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    return (
                        parts[0].parse().unwrap_or(0.0),
                        parts[1].parse().unwrap_or(0.0),
                        parts[2].parse().unwrap_or(0.0),
                    );
                }
            }
        }
    }
    (0.0, 0.0, 0.0)
}

// ============================================================================
// TUI
// ============================================================================

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Stats,
    Trajectory,
    Logs,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewMode {
    FollowCurrent,
    FitAll,
}

pub struct StatsTui {
    stats: Arc<SharedStats>,
    stop_rx: Option<std::sync::mpsc::Receiver<()>>,
    shutdown_tx: Option<std::sync::mpsc::Sender<()>>,
    current_tab: Tab,
    view_mode: ViewMode,
    log_scroll: Option<usize>, // None = auto-follow (show latest), Some(n) = manual scroll position
    log_buffer: Arc<Mutex<LogBuffer>>,
}

impl StatsTui {
    pub fn new(stats: Arc<SharedStats>) -> Self {
        Self {
            stats,
            stop_rx: None,
            shutdown_tx: None,
            current_tab: Tab::Stats,
            view_mode: ViewMode::FollowCurrent,
            log_scroll: None, // Auto-follow by default
            log_buffer: get_log_buffer(),
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

    fn scroll_logs_up(&mut self) {
        match self.log_scroll {
            None => {
                // Switch from auto-follow to manual, start near bottom
                let total = self.log_buffer.lock().map(|b| b.len()).unwrap_or(0);
                if total > 1 {
                    self.log_scroll = Some(total.saturating_sub(2));
                }
            }
            Some(scroll) if scroll > 0 => {
                self.log_scroll = Some(scroll - 1);
            }
            _ => {}
        }
    }

    fn scroll_logs_down(&mut self) {
        if let Some(scroll) = self.log_scroll {
            let total = self.log_buffer.lock().map(|b| b.len()).unwrap_or(0);
            if scroll + 1 >= total.saturating_sub(1) {
                // At or near bottom, switch back to auto-follow
                self.log_scroll = None;
            } else {
                self.log_scroll = Some(scroll + 1);
            }
        }
        // If already auto-follow (None), do nothing
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

            terminal.draw(|f| {
                self.render(f, &stats);
            })?;

            if event::poll(std::time::Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            set_tui_active(false);
                            disable_raw_mode()?;
                            execute!(terminal.backend_mut(), LeaveAlternateScreen, cursor::Show)?;
                            self.shutdown_tx.as_ref().map(|sender| sender.send(()));
                            return Ok(());
                        }
                        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                            self.shutdown_tx.as_ref().map(|sender| sender.send(()));
                            set_tui_active(false);
                            disable_raw_mode()?;
                            execute!(terminal.backend_mut(), LeaveAlternateScreen, cursor::Show)?;
                            return Ok(());
                        }
                        KeyCode::Char('1') => self.current_tab = Tab::Stats,
                        KeyCode::Char('2') => self.current_tab = Tab::Trajectory,
                        KeyCode::Char('3') => self.current_tab = Tab::Logs,
                        KeyCode::Tab => {
                            self.current_tab = match self.current_tab {
                                Tab::Stats => Tab::Trajectory,
                                Tab::Trajectory => Tab::Logs,
                                Tab::Logs => Tab::Stats,
                            };
                        }
                        KeyCode::Char('f') if self.current_tab == Tab::Trajectory => {
                            self.view_mode = match self.view_mode {
                                ViewMode::FollowCurrent => ViewMode::FitAll,
                                ViewMode::FitAll => ViewMode::FollowCurrent,
                            };
                        }
                        KeyCode::Char('k') if self.current_tab == Tab::Logs => {
                            self.scroll_logs_up();
                        }
                        KeyCode::Up if self.current_tab == Tab::Logs => {
                            self.scroll_logs_up();
                        }
                        KeyCode::Char('j') if self.current_tab == Tab::Logs => {
                            self.scroll_logs_down();
                        }
                        KeyCode::Down if self.current_tab == Tab::Logs => {
                            self.scroll_logs_down();
                        }
                        KeyCode::PageUp if self.current_tab == Tab::Logs => {
                            for _ in 0..20 {
                                self.scroll_logs_up();
                            }
                        }
                        KeyCode::PageDown if self.current_tab == Tab::Logs => {
                            for _ in 0..20 {
                                self.scroll_logs_down();
                            }
                        }
                        KeyCode::Home if self.current_tab == Tab::Logs => {
                            self.log_scroll = Some(0);
                        }
                        KeyCode::End if self.current_tab == Tab::Logs => {
                            self.log_scroll = None; // Auto-follow = show latest
                        }
                        _ => {}
                    }
                }
            }

            if let Some(rx) = &self.stop_rx {
                if rx.try_recv().is_ok() {
                    // External shutdown requested; forward to main via shutdown_tx and exit
                    if let Some(tx) = &self.shutdown_tx {
                        let _ = tx.send(());
                    }
                    break;
                }
            }
        }

        set_tui_active(false);
        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        Ok(())
    }

    fn render(&self, f: &mut ratatui::Frame, stats: &TuiStats) {
        let size = f.area();

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // Header
                Constraint::Length(3), // Tabs
                Constraint::Min(10),   // Content
                Constraint::Length(1), // Help bar
            ])
            .split(size);

        self.render_title_bar(f, chunks[0], stats);
        self.render_tabs(f, chunks[1]);

        match self.current_tab {
            Tab::Stats => self.render_stats_tab(f, chunks[2], stats),
            Tab::Trajectory => self.render_trajectory_tab(f, chunks[2], stats),
            Tab::Logs => self.render_logs_tab(f, chunks[2]),
        }

        self.render_help_bar(f, chunks[3]);
    }

    fn render_title_bar(&self, f: &mut ratatui::Frame, area: Rect, stats: &TuiStats) {
        let mode_text = if stats.mode.is_empty() {
            format!("Pelorus v{}", VERSION)
        } else {
            format!("Pelorus v{} [{}]", VERSION, stats.mode)
        };

        let mut spans = vec![Span::styled(
            mode_text,
            Style::default()
                .add_modifier(Modifier::BOLD)
                .fg(Color::Cyan),
        )];

        if let Some(bag) = &stats.bagfile {
            spans.push(Span::raw(" | Bag: "));
            spans.push(Span::styled(bag, Style::default().fg(Color::Yellow)));
        }

        let paragraph = Paragraph::new(Line::from(spans));
        f.render_widget(paragraph, area);
    }

    fn render_tabs(&self, f: &mut ratatui::Frame, area: Rect) {
        let titles = vec![
            Line::from(" 1 Stats "),
            Line::from(" 2 Trajectory "),
            Line::from(" 3 Logs "),
        ];

        let selected = match self.current_tab {
            Tab::Stats => 0,
            Tab::Trajectory => 1,
            Tab::Logs => 2,
        };

        let tabs = Tabs::new(titles)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray)),
            )
            .select(selected)
            .style(Style::default().fg(Color::DarkGray))
            .highlight_style(
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            );

        f.render_widget(tabs, area);
    }

    fn render_stats_tab(&self, f: &mut ratatui::Frame, area: Rect, stats: &TuiStats) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4), // Header info
                Constraint::Length(9), // State
                Constraint::Length(4), // Navigation
                Constraint::Min(6),    // Performance
            ])
            .split(area);

        self.render_header(f, chunks[0], stats);
        self.render_state(f, chunks[1], stats);
        self.render_navigation(f, chunks[2], stats);
        self.render_performance(f, chunks[3], stats);
    }

    fn render_trajectory_tab(&self, f: &mut ratatui::Frame, area: Rect, stats: &TuiStats) {
        let chunks = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);

        // Left: Top-down view (X-Y)
        self.render_trajectory_view(f, chunks[0], stats, "Top (X-Y)", 0, 1);

        // Right: Side view (X-Z)
        self.render_trajectory_view(f, chunks[1], stats, "Side (X-Z)", 0, 2);
    }

    fn render_trajectory_view(
        &self,
        f: &mut ratatui::Frame,
        area: Rect,
        stats: &TuiStats,
        title: &str,
        axis_h: usize,
        axis_v: usize,
    ) {
        if stats.trajectory.is_empty() {
            let block = Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(Span::styled(
                    format!(" {} ", title),
                    Style::default()
                        .fg(Color::Blue)
                        .add_modifier(Modifier::BOLD),
                ));
            let paragraph = Paragraph::new("Waiting for data...")
                .style(Style::default().fg(Color::DarkGray))
                .block(block);
            f.render_widget(paragraph, area);
            return;
        }

        // Calculate bounds
        let (min_h, max_h, min_v, max_v) = if self.view_mode == ViewMode::FitAll {
            let mut min_h = f64::MAX;
            let mut max_h = f64::MIN;
            let mut min_v = f64::MAX;
            let mut max_v = f64::MIN;

            for p in &stats.trajectory {
                min_h = min_h.min(p[axis_h]);
                max_h = max_h.max(p[axis_h]);
                min_v = min_v.min(p[axis_v]);
                max_v = max_v.max(p[axis_v]);
            }

            // Add padding
            let padding_h = (max_h - min_h).abs() * 0.1 + 1.0;
            let padding_v = (max_v - min_v).abs() * 0.1 + 1.0;
            (
                min_h - padding_h,
                max_h + padding_h,
                min_v - padding_v,
                max_v + padding_v,
            )
        } else {
            // Follow current position
            let current_h = stats.position[axis_h];
            let current_v = stats.position[axis_v];
            let range = 20.0; // View range around current position
            (
                current_h - range,
                current_h + range,
                current_v - range,
                current_v + range,
            )
        };

        // Collect points for trajectory
        let trajectory_points: Vec<(f64, f64)> = stats
            .trajectory
            .iter()
            .map(|p| (p[axis_h], p[axis_v]))
            .collect();

        let current_pos = (stats.position[axis_h], stats.position[axis_v]);

        // Calculate orientation axes from quaternion
        // Quaternion: [w, x, y, z]
        let q = stats.orientation;
        let (qw, qx, qy, qz) = (q[0], q[1], q[2], q[3]);

        // Rotation matrix columns (forward=X, left=Y, up=Z in body frame)
        // Forward (X-axis, red)
        let forward = [
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy + qw * qz),
            2.0 * (qx * qz - qw * qy),
        ];
        // Left (Y-axis, green)
        let left = [
            2.0 * (qx * qy - qw * qz),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz + qw * qx),
        ];
        // Up (Z-axis, blue)
        let up = [
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ];

        // Project axes onto 2D view plane and scale
        let axis_len = (max_h - min_h).min(max_v - min_v) * 0.04;
        let forward_2d = (forward[axis_h] * axis_len, forward[axis_v] * axis_len);
        let left_2d = (left[axis_h] * axis_len, left[axis_v] * axis_len);
        let up_2d = (up[axis_h] * axis_len, up[axis_v] * axis_len);

        // View mode indicator
        let mode_str = match self.view_mode {
            ViewMode::FollowCurrent => "Follow",
            ViewMode::FitAll => "Fit All",
        };

        let (rt_color, _) = get_realtime_color_and_status(stats.real_time_factor);
        let vel_mag = (stats.linear_velocity[0].powi(2)
            + stats.linear_velocity[1].powi(2)
            + stats.linear_velocity[2].powi(2))
        .sqrt();

        // Calculate text positions relative to view bounds
        let range_h = max_h - min_h;
        let range_v = max_v - min_v;
        // Align text exactly to the left edge to avoid jitter from floating point padding
        let text_x = min_h;
        let text_y_1 = max_v - range_v * 0.03;
        let text_y_2 = max_v - range_v * 0.06;
        let text_y_3 = max_v - range_v * 0.09;

        // Axis labels
        let axis_h_label = ["X", "Y", "Z"][axis_h];
        let axis_v_label = ["X", "Y", "Z"][axis_v];

        let canvas = Canvas::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Color::DarkGray))
                    .title(Span::styled(
                        format!(" {} [{}] ", title, mode_str),
                        Style::default()
                            .fg(Color::Blue)
                            .add_modifier(Modifier::BOLD),
                    )),
            )
            .x_bounds([min_h, max_h])
            .y_bounds([min_v, max_v])
            .marker(Marker::Braille)
            .paint(move |ctx| {
                // Draw trajectory
                if !trajectory_points.is_empty() {
                    ctx.draw(&Points {
                        coords: &trajectory_points,
                        color: Color::Green,
                    });
                }

                // Draw current position (larger dot)
                ctx.draw(&Points {
                    coords: &[current_pos],
                    color: Color::Yellow,
                });

                // Draw orientation axes at current position
                // Forward (X) - Red
                ctx.draw(&CanvasLine {
                    x1: current_pos.0,
                    y1: current_pos.1,
                    x2: current_pos.0 + forward_2d.0,
                    y2: current_pos.1 + forward_2d.1,
                    color: Color::Red,
                });
                // Left (Y) - Green (for top view) or Up (Z) - Blue (contextual)
                ctx.draw(&CanvasLine {
                    x1: current_pos.0,
                    y1: current_pos.1,
                    x2: current_pos.0 + left_2d.0,
                    y2: current_pos.1 + left_2d.1,
                    color: Color::LightGreen,
                });
                // Up (Z) - Cyan
                ctx.draw(&CanvasLine {
                    x1: current_pos.0,
                    y1: current_pos.1,
                    x2: current_pos.0 + up_2d.0,
                    y2: current_pos.1 + up_2d.1,
                    color: Color::Cyan,
                });

                // Draw origin
                ctx.draw(&Points {
                    coords: &[(0.0, 0.0)],
                    color: Color::Red,
                });

                // Corner info - position
                ctx.print(
                    text_x,
                    text_y_1,
                    Line::from(vec![
                        Span::styled("X", Style::default().fg(Color::Red)),
                        Span::styled(
                            format!("{:.1} ", stats.position[0]),
                            Style::default().fg(Color::White),
                        ),
                        Span::styled("Y", Style::default().fg(Color::LightGreen)),
                        Span::styled(
                            format!("{:.1} ", stats.position[1]),
                            Style::default().fg(Color::White),
                        ),
                        Span::styled("Z", Style::default().fg(Color::Cyan)),
                        Span::styled(
                            format!("{:.1}", stats.position[2]),
                            Style::default().fg(Color::White),
                        ),
                    ]),
                );

                // Corner info - velocity
                ctx.print(
                    text_x,
                    text_y_2,
                    Line::from(vec![
                        Span::styled("V: ", Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            format!("{:.2} m/s", vel_mag),
                            Style::default().fg(Color::Green),
                        ),
                    ]),
                );

                // Corner info - RT factor
                ctx.print(
                    text_x,
                    text_y_3,
                    Line::from(vec![
                        Span::styled("RT: ", Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            format!("{:.2}x", stats.real_time_factor),
                            Style::default().fg(rt_color),
                        ),
                    ]),
                );

                // Scale labels on edges
                // Bottom left - min values
                ctx.print(
                    min_h,
                    min_v + range_v * 0.02,
                    Line::from(Span::styled(
                        format!("{}: {:.1}m", axis_h_label, min_h),
                        Style::default().fg(Color::DarkGray),
                    )),
                );

                // Bottom right - max H
                ctx.print(
                    max_h - range_h * 0.25,
                    min_v + range_v * 0.02,
                    Line::from(Span::styled(
                        format!("{}: {:.1}m", axis_h_label, max_h),
                        Style::default().fg(Color::DarkGray),
                    )),
                );

                // Left side - min V (bottom)
                ctx.print(
                    min_h,
                    min_v + range_v * 0.05,
                    Line::from(Span::styled(
                        format!("{}: {:.1}m", axis_v_label, min_v),
                        Style::default().fg(Color::DarkGray),
                    )),
                );

                // Left side - max V (top area, below info)
                ctx.print(
                    min_h,
                    max_v - range_v * 0.13,
                    Line::from(Span::styled(
                        format!("{}: {:.1}m", axis_v_label, max_v),
                        Style::default().fg(Color::DarkGray),
                    )),
                );

                // Show total range in corner
                ctx.print(
                    max_h - range_h * 0.35,
                    max_v - range_v * 0.03,
                    Line::from(Span::styled(
                        format!("Δ{:.1}m × Δ{:.1}m", range_h, range_v),
                        Style::default().fg(Color::DarkGray),
                    )),
                );
            });

        f.render_widget(canvas, area);
    }

    fn render_header(&self, f: &mut ratatui::Frame, area: Rect, stats: &TuiStats) {
        let vendor = match stats.lidar_vendor {
            Some(LidarVendor::Ouster) => "Ouster",
            Some(LidarVendor::Velodyne) => "Velodyne",
            Some(LidarVendor::Livox) => "Livox",
            Some(LidarVendor::LivoxMid360) => "Livox Mid-360",
            Some(LidarVendor::Robosense) => "Robosense",
            None => "Unknown",
        };

        let elapsed = format_duration(stats.elapsed_secs);
        let sensor_rate_str = if stats.measured_sensor_rate_hz > 0.0 {
            format!(" @ {:.1} Hz", stats.measured_sensor_rate_hz)
        } else {
            String::new()
        };

        let lines = vec![
            Line::from(vec![
                Span::raw("⏱️  "),
                Span::styled(elapsed, Style::default().fg(Color::White)),
                Span::raw("  │  📡 "),
                Span::styled(vendor, Style::default().fg(Color::Cyan)),
                Span::styled(sensor_rate_str, Style::default().fg(Color::Cyan)),
            ]),
            Line::from(vec![
                Span::raw("💻 "),
                Span::styled(
                    format!("{} cores", stats.num_cores),
                    Style::default().fg(Color::White),
                ),
            ]),
        ];

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(Span::styled(
                " Info ",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ));
        let paragraph = Paragraph::new(lines).block(block);
        f.render_widget(paragraph, area);
    }

    fn render_state(&self, f: &mut ratatui::Frame, area: Rect, stats: &TuiStats) {
        let p = stats.position;
        let q = stats.orientation;
        let lv = stats.linear_velocity;
        let av = stats.angular_velocity;
        let ab = stats.accel_bias;
        let gb = stats.gyro_bias;

        let lines = vec![
            Line::from(vec![
                Span::styled("Position    ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:>9.3} {:>9.3} {:>9.3}", p[0], p[1], p[2]),
                    Style::default().fg(Color::White),
                ),
                Span::styled(" m", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("Orientation ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:>9.4} {:>9.4} {:>9.4} {:>9.4}", q[0], q[1], q[2], q[3]),
                    Style::default().fg(Color::White),
                ),
            ]),
            Line::from(vec![
                Span::styled("Velocity    ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:>9.3} {:>9.3} {:>9.3}", lv[0], lv[1], lv[2]),
                    Style::default().fg(Color::Green),
                ),
                Span::styled(" m/s", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("Angular Vel ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:>9.4} {:>9.4} {:>9.4}", av[0], av[1], av[2]),
                    Style::default().fg(Color::Yellow),
                ),
                Span::styled(" rad/s", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("Accel Bias  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:>9.6} {:>9.6} {:>9.6}", ab[0], ab[1], ab[2]),
                    Style::default().fg(Color::Magenta),
                ),
            ]),
            Line::from(vec![
                Span::styled("Gyro Bias   ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:>9.6} {:>9.6} {:>9.6}", gb[0], gb[1], gb[2]),
                    Style::default().fg(Color::Magenta),
                ),
            ]),
        ];

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(Span::styled(
                " State ",
                Style::default()
                    .fg(Color::Blue)
                    .add_modifier(Modifier::BOLD),
            ));
        let paragraph = Paragraph::new(lines).block(block);
        f.render_widget(paragraph, area);
    }

    fn render_navigation(&self, f: &mut ratatui::Frame, area: Rect, stats: &TuiStats) {
        let lines = vec![
            Line::from(vec![
                Span::styled("Traveled    ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:>9.2}", stats.distance_traveled),
                    Style::default().fg(Color::Green),
                ),
                Span::styled(" m", Style::default().fg(Color::DarkGray)),
            ]),
            Line::from(vec![
                Span::styled("To Origin   ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:>9.2}", stats.distance_to_origin),
                    Style::default().fg(Color::Yellow),
                ),
                Span::styled(" m", Style::default().fg(Color::DarkGray)),
            ]),
        ];

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(Span::styled(
                " Navigation ",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ));
        let paragraph = Paragraph::new(lines).block(block);
        f.render_widget(paragraph, area);
    }

    fn render_performance(&self, f: &mut ratatui::Frame, area: Rect, stats: &TuiStats) {
        let (rt_color, rt_status) = get_realtime_color_and_status(stats.real_time_factor);

        #[allow(unused_mut)]
        let mut lines = vec![
            Line::from(vec![
                Span::styled("Comp Time   ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:>6.1} ms", stats.last_comp_time_ms),
                    Style::default().fg(Color::White),
                ),
                Span::styled("  avg ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1}", stats.avg_comp_time_ms),
                    Style::default().fg(Color::Gray),
                ),
                Span::styled(" / max ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1}", stats.max_comp_time_ms),
                    Style::default().fg(Color::Gray),
                ),
            ]),
            Line::from(vec![
                Span::styled("Proc Rate   ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:>6.1} Hz", stats.processing_freq_hz),
                    Style::default().fg(rt_color),
                ),
                Span::styled("  sensor ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1} Hz", stats.measured_sensor_rate_hz),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
            Line::from(vec![
                Span::styled("RT Factor   ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:>6.2}x", stats.real_time_factor),
                    Style::default().fg(rt_color).add_modifier(Modifier::BOLD),
                ),
                Span::raw("  "),
                Span::styled(rt_status, Style::default().fg(rt_color)),
            ]),
            Line::from(vec![
                Span::styled("CPU         ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:>6.1} %", stats.cpu_percent),
                    Style::default().fg(Color::White),
                ),
                Span::styled("  avg ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1}", stats.avg_cpu_percent),
                    Style::default().fg(Color::Gray),
                ),
                Span::styled(" / max ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1}", stats.max_cpu_percent),
                    Style::default().fg(Color::Gray),
                ),
            ]),
            Line::from(vec![
                Span::styled("RAM         ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:>6.0} MB", stats.ram_mb),
                    Style::default().fg(Color::White),
                ),
            ]),
        ];

        #[cfg(feature = "cuda")]
        {
            if stats.gpu_memory_total_mb > 0.0 {
                lines.push(Line::from(vec![
                    Span::styled("GPU Mem     ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!(
                            "{:>6.0} / {:.0} MB",
                            stats.gpu_memory_used_mb, stats.gpu_memory_total_mb
                        ),
                        Style::default().fg(Color::Cyan),
                    ),
                ]));
            }
            if stats.gpu_utilization > 0.0 {
                lines.push(Line::from(vec![
                    Span::styled("GPU Load    ", Style::default().fg(Color::DarkGray)),
                    Span::styled(
                        format!("{:>6.1} %", stats.gpu_utilization),
                        Style::default().fg(Color::Cyan),
                    ),
                ]));
            }
        }

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(Span::styled(
                " Performance ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ));
        let paragraph = Paragraph::new(lines).block(block);
        f.render_widget(paragraph, area);
    }

    fn render_logs_tab(&self, f: &mut ratatui::Frame, area: Rect) {
        let entries = self.log_buffer.lock().ok();
        let entries = entries.as_ref().map(|b| b.entries()).unwrap_or(&[]);

        let visible_height = area.height.saturating_sub(2) as usize; // -2 for borders
        let content_width = area.width.saturating_sub(2) as usize; // -2 for borders
        let total_entries = entries.len();

        // Prefix length: "   45.47s WARN  " = 16 chars
        let prefix_len = 16;
        let continuation_indent = "                  "; // 18 spaces for continuation lines
        let msg_width = content_width.saturating_sub(prefix_len);

        // Build wrapped lines with entry indices
        let mut wrapped_lines: Vec<(usize, Line)> = Vec::new();

        for (entry_idx, entry) in entries.iter().enumerate() {
            let time_str = format!("{:>8.2}s", entry.timestamp);
            let msg = &entry.message;
            let msg_color = if entry.level == LogLevel::Error || entry.level == LogLevel::Warn {
                entry.level.color()
            } else {
                Color::White
            };

            if msg.len() <= msg_width || msg_width == 0 {
                // Single line
                wrapped_lines.push((
                    entry_idx,
                    Line::from(vec![
                        Span::styled(time_str, Style::default().fg(Color::DarkGray)),
                        Span::raw(" "),
                        Span::styled(
                            entry.level.prefix(),
                            Style::default()
                                .fg(entry.level.color())
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::raw(" "),
                        Span::styled(msg.as_str(), Style::default().fg(msg_color)),
                    ]),
                ));
            } else {
                // Wrap long messages
                let mut remaining = msg.as_str();
                let mut first_line = true;

                while !remaining.is_empty() {
                    let take_len = remaining.len().min(if first_line {
                        msg_width
                    } else {
                        msg_width.saturating_sub(2)
                    });
                    let (chunk, rest) = remaining.split_at(take_len);
                    remaining = rest;

                    if first_line {
                        wrapped_lines.push((
                            entry_idx,
                            Line::from(vec![
                                Span::styled(
                                    time_str.clone(),
                                    Style::default().fg(Color::DarkGray),
                                ),
                                Span::raw(" "),
                                Span::styled(
                                    entry.level.prefix(),
                                    Style::default()
                                        .fg(entry.level.color())
                                        .add_modifier(Modifier::BOLD),
                                ),
                                Span::raw(" "),
                                Span::styled(chunk.to_string(), Style::default().fg(msg_color)),
                            ]),
                        ));
                        first_line = false;
                    } else {
                        wrapped_lines.push((
                            entry_idx,
                            Line::from(vec![
                                Span::styled(
                                    continuation_indent,
                                    Style::default().fg(Color::DarkGray),
                                ),
                                Span::styled("↳ ", Style::default().fg(Color::DarkGray)),
                                Span::styled(chunk.to_string(), Style::default().fg(msg_color)),
                            ]),
                        ));
                    }
                }
            }
        }

        let total_lines = wrapped_lines.len();

        // Calculate scroll position (in lines, not entries)
        let (start_line, is_auto_follow, selected_entry) = match self.log_scroll {
            None => {
                // Auto-follow: show the latest lines
                (total_lines.saturating_sub(visible_height), true, None)
            }
            Some(scroll) => {
                // Manual scroll - scroll is in entry indices, convert to line index
                // Find first line of the scrolled-to entry
                let line_idx = wrapped_lines
                    .iter()
                    .position(|(idx, _)| *idx >= scroll)
                    .unwrap_or(0);
                let max_scroll = total_lines.saturating_sub(visible_height);
                (line_idx.min(max_scroll), false, Some(scroll))
            }
        };

        let end_line = (start_line + visible_height).min(total_lines);

        let items: Vec<ListItem> = wrapped_lines
            .get(start_line..end_line)
            .unwrap_or(&[])
            .iter()
            .map(|(entry_idx, line)| {
                // Highlight the selected entry with a marker
                let is_selected = selected_entry.map(|s| *entry_idx == s).unwrap_or(false);
                if is_selected {
                    let mut spans = vec![Span::styled(
                        "▶",
                        Style::default()
                            .fg(Color::Cyan)
                            .add_modifier(Modifier::BOLD),
                    )];
                    spans.extend(line.spans.iter().cloned());
                    ListItem::new(Line::from(spans)).style(Style::default().bg(Color::DarkGray))
                } else {
                    let mut spans = vec![Span::raw(" ")];
                    spans.extend(line.spans.iter().cloned());
                    ListItem::new(Line::from(spans))
                }
            })
            .collect();

        let scroll_info = if is_auto_follow {
            format!(" [AUTO {} entries] ", total_entries)
        } else {
            let percent = if total_lines > visible_height {
                let max_scroll = total_lines.saturating_sub(visible_height);
                (start_line as f64 / max_scroll as f64) * 100.0
            } else {
                100.0
            };
            format!(
                " [line {}/{} {:.0}%] ",
                start_line + 1,
                total_lines,
                percent
            )
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray))
            .title(Span::styled(
                " Logs ",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ))
            .title_bottom(Span::styled(
                scroll_info,
                Style::default().fg(if is_auto_follow {
                    Color::Cyan
                } else {
                    Color::Yellow
                }),
            ));

        let list = List::new(items).block(block);
        f.render_widget(list, area);
    }

    fn render_help_bar(&self, f: &mut ratatui::Frame, area: Rect) {
        let help = match self.current_tab {
            Tab::Stats => Line::from(vec![
                Span::styled(
                    " Tab",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("/", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "1-3",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Switch  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "q",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("/", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "Esc",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Quit  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "Ctrl+C",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Stop", Style::default().fg(Color::DarkGray)),
            ]),
            Tab::Trajectory => Line::from(vec![
                Span::styled(
                    " Tab",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("/", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "1-3",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Switch  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "f",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Toggle View  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "q",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("/", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "Esc",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Quit", Style::default().fg(Color::DarkGray)),
            ]),
            Tab::Logs => Line::from(vec![
                Span::styled(
                    " Tab",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("/", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "1-3",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Switch  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "↑↓",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("/", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "jk",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Scroll  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "PgUp/Dn",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Page  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "Home/End",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Jump  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "q",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Quit", Style::default().fg(Color::DarkGray)),
            ]),
        };
        let paragraph = Paragraph::new(help);
        f.render_widget(paragraph, area);
    }
}

fn get_realtime_color_and_status(rt_factor: f64) -> (Color, &'static str) {
    if rt_factor <= 0.0 {
        (Color::Gray, "⏳ Initializing")
    } else if rt_factor < 0.8 {
        (Color::Red, "🔴 CRITICAL")
    } else if rt_factor < 0.95 {
        (Color::Yellow, "🟡 Warning")
    } else if rt_factor < 1.5 {
        (Color::Green, "🟢 OK")
    } else if rt_factor < 3.0 {
        (Color::Green, "🟢 Good")
    } else {
        (Color::Cyan, "⚡ Excellent")
    }
}

impl Drop for StatsTui {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen);
    }
}

fn format_duration(secs: f64) -> String {
    let total_secs = secs as u64;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let seconds = total_secs % 60;

    if hours > 0 {
        format!("{:02}:{:02}:{:02}", hours, minutes, seconds)
    } else {
        format!("{:02}:{:02}", minutes, seconds)
    }
}
