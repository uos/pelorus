//! The Pelorus status view: statistics, trajectory and logs drawn with ratatui.
//!
//! The view only renders and reacts to keys. The application owns the
//! terminal, or any other ratatui backend, and the data. It fills a
//! [`TuiStats`], calls [`StatsView::render`] inside its draw loop and passes key
//! presses to [`StatsView::handle_key`].

use alloc::sync::Arc;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols::Marker,
    text::{Line, Span},
    widgets::{
        Block, Borders, Clear, Padding, Paragraph,
        canvas::{Canvas, Line as CanvasLine, Points},
    },
};

use mt_log::{LogBuffer, LogEntry};
use pelorus_core::LidarVendor;

extern crate alloc;

const FOLLOW_VIEW_RANGE_STEP_M: f64 = 5.0;
const FOLLOW_VIEW_RANGE_MIN_M: f64 = 5.0;
const FOLLOW_VIEW_RANGE_MAX_M: f64 = 200.0;
const STATS_LEFT_PADDING: u16 = 2;

/// What the status view shows. Filled by the application, read by [`StatsView::render`].
#[derive(Debug, Clone, Default)]
pub struct TuiStats {
    // Mode
    pub mode: String,
    pub bagfile: Option<String>,

    // Backend info
    pub backend: String,
    pub accelerator: Option<String>,
    pub accelerator_details: Option<String>,

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
    /// Whether CPU and RAM usage are measured. Without them the view leaves
    /// those rows out.
    pub system_stats: bool,
    pub cpu_percent: f64,
    pub avg_cpu_percent: f64,
    pub max_cpu_percent: f64,
    pub ram_mb: f64,
    pub num_cores: usize,

    // GPU (CUDA/Metal/Jetson) - Process-local buffer usage. Zero without a GPU backend.
    pub gpu_memory_used_mb: f64,
    pub gpu_memory_total_mb: f64,
    pub gpu_utilization: f64,

    // Elapsed
    pub elapsed_secs: f64,

    // Trajectory history
    pub trajectory: Vec<[f64; 3]>,
}

const MERGED_TABS_MIN_WIDTH: u16 = 120;
const MERGED_TABS_MIN_HEIGHT: u16 = 42;
const HYBRID_MERGED_MIN_HEIGHT: u16 = 30;
const STATS_FULL_MIN_HEIGHT: u16 = 24;
const LOGS_MIN_HEIGHT: u16 = 8;
const DOCKED_HELP_PANEL_MIN_WIDTH: u16 = 38;
const DOCKED_HELP_PANEL_MAX_WIDTH: u16 = 72;
const TRAJECTORY_HINT_PANEL_WIDTH: u16 = 34;
const TRAJECTORY_HINT_MIN_TOTAL_WIDTH: u16 = 90;

/// One projection of the trajectory panel and what to draw around it.
#[derive(Clone, Copy)]
struct TrajectoryView<'a> {
    title: &'a str,
    axis_h: usize,
    axis_v: usize,
    show_top_stats: bool,
    show_left_divider: bool,
    show_rt_stat: bool,
}

/// Tab state that decides which hints the help bar shows.
#[derive(Clone, Copy)]
struct HelpBarContext {
    active_tab: Tab,
    logs_integrated: bool,
    show_trajectory_tab: bool,
    show_logs_tab: bool,
    trajectory_visible: bool,
}

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

#[derive(Clone, Copy, PartialEq, Eq)]
enum TrajectoryPaneMode {
    Split,
    LeftOnly,
    RightOnly,
}

#[derive(Clone, Copy)]
struct TabVisibility {
    show_trajectory_tab: bool,
    show_logs_tab: bool,
    logs_integrated: bool,
}

#[derive(Clone, Copy)]
struct TabbedLayout {
    header: Rect,
    tabs: Rect,
    content: Rect,
    help: Rect,
}

/// What the caller does after [`StatsView::handle_key`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KeyAction {
    Continue,
    /// The user asked to quit.
    Exit,
}

/// A key the status view reacts to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Key {
    Char(char),
    Tab,
    Esc,
    Up,
    Down,
    PageUp,
    PageDown,
    Home,
    End,
}

/// A key press, with whether Ctrl was held.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct KeyPress {
    pub key: Key,
    pub ctrl: bool,
}

/// What the status view shows and how: active tab, trajectory zoom and log
/// scroll position. Draw it with [`render`](StatsView::render) and feed it keys
/// with [`handle_key`](StatsView::handle_key).
pub struct StatsView {
    current_tab: Tab,
    view_mode: ViewMode,
    follow_view_range_m: f64,
    trajectory_pane_mode: TrajectoryPaneMode,
    log_view: mt_log::LogView,
    log_buffer: Arc<LogBuffer>,
    show_minot_logs: bool,
    show_help_overlay: bool,
    merged_layout_min_width: u16,
    hybrid_layout_min_width: u16,
    /// The application draws the trajectory itself, so the view leaves it out.
    trajectory_elsewhere: bool,
}

impl StatsView {
    /// A view on the Stats tab that draws the logs in `log_buffer`.
    pub fn new(log_buffer: Arc<LogBuffer>) -> Self {
        Self {
            current_tab: Tab::Stats,
            view_mode: ViewMode::FollowCurrent,
            follow_view_range_m: 20.0,
            trajectory_pane_mode: TrajectoryPaneMode::Split,
            log_view: mt_log::LogView::default(),
            log_buffer,
            show_minot_logs: true,
            show_help_overlay: false,
            merged_layout_min_width: MERGED_TABS_MIN_WIDTH,
            hybrid_layout_min_width: MERGED_TABS_MIN_WIDTH,
            trajectory_elsewhere: false,
        }
    }

    /// Leave the trajectory panes and tab out, for an application that draws
    /// the trajectory elsewhere. The statistics and logs then get the space.
    pub fn set_trajectory_elsewhere(&mut self, elsewhere: bool) {
        self.trajectory_elsewhere = elsewhere;
    }

    fn status_bar_style() -> Style {
        Style::default().bg(Color::Rgb(20, 26, 34))
    }

    fn use_merged_tabs_layout(&self, area: Rect) -> bool {
        !self.trajectory_elsewhere
            && area.width >= self.merged_layout_min_width
            && area.height >= MERGED_TABS_MIN_HEIGHT
    }

    fn use_hybrid_merged_layout(&self, area: Rect) -> bool {
        !self.trajectory_elsewhere
            && area.width >= self.hybrid_layout_min_width
            && area.height >= HYBRID_MERGED_MIN_HEIGHT
            && area.height < MERGED_TABS_MIN_HEIGHT
    }

    fn render_stats_with_optional_logs(
        &self,
        f: &mut ratatui::Frame,
        area: Rect,
        stats: &TuiStats,
    ) {
        if Self::stats_shows_logs(area) {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(STATS_FULL_MIN_HEIGHT),
                    Constraint::Length(1),
                    Constraint::Min(1),
                ])
                .split(area);
            self.render_stats_tab(f, chunks[0], stats);
            self.render_section_divider(f, chunks[1]);
            self.render_logs_tab(f, chunks[2]);
        } else {
            self.render_stats_tab(f, area, stats);
        }
    }

    fn stats_shows_logs(area: Rect) -> bool {
        area.height >= STATS_FULL_MIN_HEIGHT + 1 + LOGS_MIN_HEIGHT
    }

    fn tab_visibility_for_area(&self, area: Rect) -> TabVisibility {
        let logs_integrated = if self.use_merged_tabs_layout(area) {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1),
                    Constraint::Length(1),
                    Constraint::Min(10),
                    Constraint::Length(1),
                ])
                .split(area);
            let content_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
                .split(chunks[2]);
            Self::stats_shows_logs(content_chunks[0])
        } else if self.use_hybrid_merged_layout(area) {
            let pre_content = Self::pre_tabbed_content_area(area);
            let content_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
                .split(pre_content);
            Self::stats_shows_logs(content_chunks[0])
        } else {
            let pre_content = Self::pre_tabbed_content_area(area);
            Self::stats_shows_logs(pre_content)
        };

        TabVisibility {
            show_trajectory_tab: self.show_trajectory_tab_for_size(area),
            show_logs_tab: !self.use_merged_tabs_layout(area) && !logs_integrated,
            logs_integrated,
        }
    }

    fn effective_tab(&self, visibility: TabVisibility) -> Tab {
        match self.current_tab {
            Tab::Trajectory if !visibility.show_trajectory_tab => Tab::Stats,
            Tab::Logs if !visibility.show_logs_tab => Tab::Stats,
            tab => tab,
        }
    }

    fn cycle_tab(&mut self, visibility: TabVisibility) {
        self.current_tab = match (
            visibility.show_trajectory_tab,
            visibility.show_logs_tab,
            self.current_tab,
        ) {
            (true, true, Tab::Stats) => Tab::Trajectory,
            (true, true, Tab::Trajectory) => Tab::Logs,
            (true, true, Tab::Logs) => Tab::Stats,
            (true, false, Tab::Stats) => Tab::Trajectory,
            (true, false, Tab::Trajectory) => Tab::Stats,
            (true, false, Tab::Logs) => Tab::Stats,
            (false, true, Tab::Stats) => Tab::Logs,
            (false, true, Tab::Logs) => Tab::Stats,
            (false, true, Tab::Trajectory) => Tab::Stats,
            (false, false, _) => Tab::Stats,
        };
    }

    fn pre_tabbed_content_area(size: Rect) -> Rect {
        Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Min(10),
                Constraint::Length(1),
            ])
            .split(size)[2]
    }

    fn split_tabbed_layout(size: Rect, show_tabs_row: bool) -> TabbedLayout {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(if show_tabs_row { 1 } else { 0 }),
                Constraint::Min(10),
                Constraint::Length(1),
            ])
            .split(size);

        TabbedLayout {
            header: chunks[0],
            tabs: chunks[1],
            content: chunks[2],
            help: chunks[3],
        }
    }

    fn logs_controls_active(&self, visibility: TabVisibility) -> bool {
        self.effective_tab(visibility) == Tab::Logs || visibility.logs_integrated
    }

    /// React to a key press. `area` is the size the view is drawn at.
    pub fn handle_key(&mut self, key: KeyPress, area: Rect) -> KeyAction {
        let visibility = self.tab_visibility_for_area(area);

        if matches!(key.key, Key::Char('q') | Key::Esc)
            || (matches!(key.key, Key::Char('c')) && key.ctrl)
        {
            return KeyAction::Exit;
        }

        match key.key {
            Key::Char('1') => self.current_tab = Tab::Stats,
            Key::Char('2') => {
                if visibility.show_trajectory_tab {
                    self.current_tab = Tab::Trajectory;
                } else if visibility.show_logs_tab {
                    self.current_tab = Tab::Logs;
                }
            }
            Key::Char('3') => {
                if visibility.show_trajectory_tab && visibility.show_logs_tab {
                    self.current_tab = Tab::Logs;
                }
            }
            Key::Tab => self.cycle_tab(visibility),
            Key::Char('?') => self.show_help_overlay = !self.show_help_overlay,
            Key::Char('f') => {
                if self.trajectory_visible_for_size(area) {
                    self.view_mode = match self.view_mode {
                        ViewMode::FollowCurrent => ViewMode::FitAll,
                        ViewMode::FitAll => ViewMode::FollowCurrent,
                    };
                }
            }
            Key::Char('v') => {
                if self.trajectory_visible_for_size(area) {
                    self.trajectory_pane_mode = match self.trajectory_pane_mode {
                        TrajectoryPaneMode::Split => TrajectoryPaneMode::LeftOnly,
                        TrajectoryPaneMode::LeftOnly => TrajectoryPaneMode::RightOnly,
                        TrajectoryPaneMode::RightOnly => TrajectoryPaneMode::Split,
                    };
                }
            }
            Key::Char('-') | Key::Char('_') => {
                if self.trajectory_visible_for_size(area) {
                    self.follow_view_range_m = (self.follow_view_range_m
                        + FOLLOW_VIEW_RANGE_STEP_M)
                        .clamp(FOLLOW_VIEW_RANGE_MIN_M, FOLLOW_VIEW_RANGE_MAX_M);
                }
            }
            Key::Char('+') | Key::Char('=') => {
                if self.trajectory_visible_for_size(area) {
                    self.follow_view_range_m = (self.follow_view_range_m
                        - FOLLOW_VIEW_RANGE_STEP_M)
                        .clamp(FOLLOW_VIEW_RANGE_MIN_M, FOLLOW_VIEW_RANGE_MAX_M);
                }
            }
            Key::Char('k') | Key::Up => {
                if self.logs_controls_active(visibility) {
                    self.scroll_logs_up();
                }
            }
            Key::Char('j') | Key::Down => {
                if self.logs_controls_active(visibility) {
                    self.scroll_logs_down();
                }
            }
            Key::PageUp => {
                if self.logs_controls_active(visibility) {
                    for _ in 0..20 {
                        self.scroll_logs_up();
                    }
                }
            }
            Key::PageDown => {
                if self.logs_controls_active(visibility) {
                    for _ in 0..20 {
                        self.scroll_logs_down();
                    }
                }
            }
            Key::Home => {
                if self.logs_controls_active(visibility) {
                    self.log_view.to_start();
                }
            }
            Key::End => {
                if self.logs_controls_active(visibility) {
                    self.log_view.to_end();
                }
            }
            Key::Char('m') if self.logs_controls_active(visibility) => {
                self.show_minot_logs = !self.show_minot_logs;
            }
            _ => {}
        }

        KeyAction::Continue
    }

    fn show_trajectory_tab_for_size(&self, size: Rect) -> bool {
        !self.trajectory_elsewhere
            && !(self.use_merged_tabs_layout(size) || self.use_hybrid_merged_layout(size))
    }

    fn trajectory_visible_for_size(&self, size: Rect) -> bool {
        if self.trajectory_elsewhere {
            return false;
        }
        if self.use_merged_tabs_layout(size) {
            return true;
        }

        if self.use_hybrid_merged_layout(size) {
            return true;
        }

        self.current_tab == Tab::Trajectory
    }

    fn visible_tab_count(show_trajectory_tab: bool, show_logs_tab: bool) -> u8 {
        1 + u8::from(show_trajectory_tab) + u8::from(show_logs_tab)
    }

    fn should_show_trajectory_unlock_hint(&self, size: Rect, active_tab: Tab) -> bool {
        if active_tab != Tab::Stats || self.trajectory_elsewhere {
            return false;
        }
        if self.use_merged_tabs_layout(size) || self.use_hybrid_merged_layout(size) {
            return false;
        }
        if size.width < TRAJECTORY_HINT_MIN_TOTAL_WIDTH || size.height < 14 {
            return false;
        }
        let target_width = self.trajectory_unlock_target_width(size);
        let width_short = target_width.saturating_sub(size.width);
        let height_short = HYBRID_MERGED_MIN_HEIGHT.saturating_sub(size.height);
        width_short > 0 || height_short > 0
    }

    fn trajectory_unlock_target_width(&self, size: Rect) -> u16 {
        if size.height >= MERGED_TABS_MIN_HEIGHT {
            self.merged_layout_min_width
        } else {
            self.hybrid_layout_min_width
        }
    }

    fn estimate_stats_min_width(&self, stats: &TuiStats) -> u16 {
        let vendor = match stats.lidar_vendor {
            Some(LidarVendor::Ouster) => "Ouster",
            Some(LidarVendor::Velodyne) => "Velodyne",
            Some(LidarVendor::Livox) => "Livox",
            Some(LidarVendor::LivoxMid360) => "Livox Mid-360",
            Some(LidarVendor::Robosense) => "Robosense",
            Some(LidarVendor::Hesai) => "Hesai",
            Some(LidarVendor::Blickfeld) => "Blickfeld",
            Some(LidarVendor::SickMultiscan) => "SickMultiScan",
            Some(LidarVendor::NoTimestamp) => "No Timestamp",
            None => "Unknown",
        };
        let elapsed = format_duration(stats.elapsed_secs);
        let sensor_rate_str = if stats.measured_sensor_rate_hz > 0.0 {
            format!(" @ {:.1} Hz", stats.measured_sensor_rate_hz)
        } else {
            String::new()
        };
        let header_line_1 = format!("{}  │  {}{}", elapsed, vendor, sensor_rate_str);
        let mut header_line_2 = stats.backend.clone();
        if let Some(accel) = &stats.accelerator {
            header_line_2.push_str(" + ");
            header_line_2.push_str(accel);
            if let Some(details) = &stats.accelerator_details {
                header_line_2.push_str(" (");
                header_line_2.push_str(details);
                header_line_2.push(')');
            }
        }

        let lv = stats.linear_velocity;
        let av = stats.angular_velocity;
        let vel_abs_ms = (lv[0].powi(2) + lv[1].powi(2) + lv[2].powi(2)).sqrt();
        let vel_abs_kmh = vel_abs_ms * 3.6;
        let state_lines = [
            format!(
                "Position    {:>9.3} {:>9.3} {:>9.3} m",
                stats.position[0], stats.position[1], stats.position[2]
            ),
            format!(
                "Orientation {:>9.4} {:>9.4} {:>9.4} {:>9.4}",
                stats.orientation[0],
                stats.orientation[1],
                stats.orientation[2],
                stats.orientation[3]
            ),
            format!(
                "Velocity    {:>9.3} {:>9.3} {:>9.3} m/s",
                lv[0], lv[1], lv[2]
            ),
            format!(
                "Angular Vel {:>9.4} {:>9.4} {:>9.4} rad/s",
                av[0], av[1], av[2]
            ),
            format!(
                "Speed |v|   {:^12} // {:^12}",
                format!("{:.2} m/s", vel_abs_ms),
                format!("{:.1} km/h", vel_abs_kmh)
            ),
        ];
        let nav_lines = [
            format!("Traveled    {:>9.2} m", stats.distance_traveled),
            format!("To Origin   {:>9.2} m", stats.distance_to_origin),
        ];
        let perf_lines = [
            format!(
                "Comp Time   {:>6.1} ms  avg {:.1} / max {:.1}",
                stats.last_comp_time_ms, stats.avg_comp_time_ms, stats.max_comp_time_ms
            ),
            format!(
                "Proc Rate   {:>6.1} Hz  sensor {:.1} Hz",
                stats.processing_freq_hz, stats.measured_sensor_rate_hz
            ),
            format!(
                "CPU         {:>6.1} %  avg {:.1} / max {:.1}",
                stats.cpu_percent, stats.avg_cpu_percent, stats.max_cpu_percent
            ),
            format!("RAM         {:>6.0} MB", stats.ram_mb),
        ];

        let mut max_chars = header_line_1
            .chars()
            .count()
            .max(header_line_2.chars().count());
        for line in state_lines.into_iter().chain(nav_lines).chain(perf_lines) {
            max_chars = max_chars.max(line.chars().count());
        }

        (max_chars as u16).saturating_add(2)
    }

    fn render_section_divider(&self, f: &mut ratatui::Frame, area: Rect) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        let line = "─".repeat(area.width as usize);
        f.render_widget(
            Paragraph::new(Span::styled(line, Style::default().fg(Color::DarkGray))),
            area,
        );
    }

    fn div_ceil_u16(n: u32, d: u32) -> u16 {
        n.div_ceil(d) as u16
    }

    /// Fit the layout breakpoints to how wide `stats` renders. Call it before
    /// each [`render`](Self::render).
    pub fn refresh_dynamic_layout_thresholds(&mut self, stats: &TuiStats) {
        let left_required = self.estimate_stats_min_width(stats).max(40) as u32;
        let right_required = 36u32;

        // merged split: 42% left / 58% right
        let merged_from_left = Self::div_ceil_u16(left_required * 100, 42);
        let merged_from_right = Self::div_ceil_u16(right_required * 100, 58);
        self.merged_layout_min_width =
            MERGED_TABS_MIN_WIDTH.max(merged_from_left.max(merged_from_right));

        // hybrid split: 60% left / 40% right
        let hybrid_from_left = Self::div_ceil_u16(left_required * 100, 60);
        let hybrid_from_right = Self::div_ceil_u16(right_required * 100, 40);
        self.hybrid_layout_min_width =
            MERGED_TABS_MIN_WIDTH.max(hybrid_from_left.max(hybrid_from_right));
    }

    fn render_vertical_divider(&self, f: &mut ratatui::Frame, area: Rect) {
        if area.width == 0 || area.height == 0 {
            return;
        }
        let lines: Vec<Line> = (0..area.height)
            .map(|_| Line::from(Span::styled("│", Style::default().fg(Color::DarkGray))))
            .collect();
        f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), area);
    }

    fn render_trajectory_unlock_hint(&self, f: &mut ratatui::Frame, area: Rect, size: Rect) {
        if area.width < 8 || area.height < 4 {
            return;
        }
        let target_width = self.trajectory_unlock_target_width(size);
        let width_short = target_width.saturating_sub(size.width);
        let height_short = HYBRID_MERGED_MIN_HEIGHT.saturating_sub(size.height);

        let mut lines = vec![
            Line::from(Span::styled(
                "Trajectory Pane",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(Span::styled(
                "not yet available",
                Style::default().fg(Color::DarkGray),
            )),
            Line::default(),
        ];

        if width_short > 0 {
            lines.push(Line::from(vec![
                Span::styled("+", Style::default().fg(Color::Yellow)),
                Span::styled(
                    format!("{width_short} cols"),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" needed", Style::default().fg(Color::DarkGray)),
            ]));
        }
        if height_short > 0 {
            lines.push(Line::from(vec![
                Span::styled("+", Style::default().fg(Color::Yellow)),
                Span::styled(
                    format!("{height_short} rows"),
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" needed", Style::default().fg(Color::DarkGray)),
            ]));
        }

        lines.push(Line::default());
        lines.push(Line::from(Span::styled(
            format!("target: {}w × {}h", target_width, HYBRID_MERGED_MIN_HEIGHT),
            Style::default().fg(Color::DarkGray),
        )));
        lines.push(Line::from(Span::styled(
            "open with Tab or 2",
            Style::default().fg(Color::Yellow),
        )));

        f.render_widget(Paragraph::new(lines).alignment(Alignment::Center), area);
    }

    fn scroll_logs_up(&mut self) {
        self.log_view.scroll_up(self.log_buffer.len());
    }

    fn scroll_logs_down(&mut self) {
        self.log_view.scroll_down(self.log_buffer.len());
    }

    /// Draw the whole view into `f`.
    pub fn render(&self, f: &mut ratatui::Frame, stats: &TuiStats) {
        let size = f.area();

        if self.use_merged_tabs_layout(size) {
            let chunks = Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Length(1), // Header
                    Constraint::Min(10),   // Content
                    Constraint::Length(1), // Help bar
                ])
                .split(size);

            self.render_title_bar(f, chunks[0], stats);

            let content_chunks = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(42), Constraint::Percentage(58)])
                .split(chunks[1]);

            self.render_stats_with_optional_logs(f, content_chunks[0], stats);

            self.render_trajectory_tab(f, content_chunks[1], stats, true, false);
            self.render_help_bar_merged(f, chunks[2]);
            self.render_help_overlay(
                f,
                chunks[1],
                self.help_overlay_items(
                    Tab::Stats,
                    Self::stats_shows_logs(content_chunks[0]),
                    false,
                    false,
                    true,
                    true,
                ),
            );
        } else if self.use_hybrid_merged_layout(size) {
            let pre_logs_integrated = Self::stats_shows_logs(Self::pre_tabbed_content_area(size));
            let show_tabs_row = Self::visible_tab_count(false, !pre_logs_integrated) > 1;
            let layout = Self::split_tabbed_layout(size, show_tabs_row);

            self.render_title_bar(f, layout.header, stats);
            let logs_integrated = Self::stats_shows_logs(layout.content);
            let visibility = TabVisibility {
                show_trajectory_tab: false,
                show_logs_tab: !logs_integrated,
                logs_integrated,
            };
            let effective_tab = self.effective_tab(visibility);
            if Self::visible_tab_count(visibility.show_trajectory_tab, visibility.show_logs_tab) > 1
            {
                self.render_tabs(
                    f,
                    layout.tabs,
                    visibility.show_trajectory_tab,
                    visibility.show_logs_tab,
                    effective_tab,
                );
            }

            if effective_tab == Tab::Logs && visibility.show_logs_tab {
                self.render_logs_tab(f, layout.content);
            } else {
                let content_chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
                    .split(layout.content);

                self.render_stats_with_optional_logs(f, content_chunks[0], stats);
                self.render_trajectory_tab(f, content_chunks[1], stats, true, false);
            }

            self.render_help_bar_context(
                f,
                layout.help,
                HelpBarContext {
                    active_tab: effective_tab,
                    logs_integrated,
                    show_trajectory_tab: visibility.show_trajectory_tab,
                    show_logs_tab: visibility.show_logs_tab,
                    trajectory_visible: true,
                },
            );
            self.render_help_overlay(
                f,
                layout.content,
                self.help_overlay_items(
                    effective_tab,
                    logs_integrated,
                    visibility.show_trajectory_tab,
                    visibility.show_logs_tab,
                    effective_tab != Tab::Logs,
                    false,
                ),
            );
        } else {
            let show_trajectory_tab = self.show_trajectory_tab_for_size(size);
            let pre_logs_integrated = Self::stats_shows_logs(Self::pre_tabbed_content_area(size));
            let show_tabs_row =
                Self::visible_tab_count(show_trajectory_tab, !pre_logs_integrated) > 1;

            let layout = Self::split_tabbed_layout(size, show_tabs_row);

            self.render_title_bar(f, layout.header, stats);
            let logs_integrated = Self::stats_shows_logs(layout.content);
            let visibility = TabVisibility {
                show_trajectory_tab,
                show_logs_tab: !logs_integrated,
                logs_integrated,
            };
            let effective_tab = self.effective_tab(visibility);
            if Self::visible_tab_count(visibility.show_trajectory_tab, visibility.show_logs_tab) > 1
            {
                self.render_tabs(
                    f,
                    layout.tabs,
                    visibility.show_trajectory_tab,
                    visibility.show_logs_tab,
                    effective_tab,
                );
            }

            match effective_tab {
                Tab::Stats => {
                    let required_stats_width = self.estimate_stats_min_width(stats);
                    if self.should_show_trajectory_unlock_hint(size, effective_tab)
                        && layout.content.width
                            > required_stats_width.saturating_add(TRAJECTORY_HINT_PANEL_WIDTH + 1)
                    {
                        let split = Layout::default()
                            .direction(Direction::Horizontal)
                            .constraints([
                                Constraint::Min(required_stats_width),
                                Constraint::Length(1),
                                Constraint::Length(TRAJECTORY_HINT_PANEL_WIDTH),
                            ])
                            .split(layout.content);
                        self.render_stats_with_optional_logs(f, split[0], stats);
                        self.render_vertical_divider(f, split[1]);
                        self.render_trajectory_unlock_hint(f, split[2], size);
                    } else {
                        self.render_stats_with_optional_logs(f, layout.content, stats);
                    }
                }
                Tab::Trajectory => {
                    self.render_trajectory_tab(f, layout.content, stats, false, true)
                }
                Tab::Logs => self.render_logs_tab(f, layout.content),
            }

            self.render_help_bar_context(
                f,
                layout.help,
                HelpBarContext {
                    active_tab: effective_tab,
                    logs_integrated,
                    show_trajectory_tab: visibility.show_trajectory_tab,
                    show_logs_tab: visibility.show_logs_tab,
                    trajectory_visible: effective_tab == Tab::Trajectory,
                },
            );
            self.render_help_overlay(
                f,
                layout.content,
                self.help_overlay_items(
                    effective_tab,
                    logs_integrated,
                    visibility.show_trajectory_tab,
                    visibility.show_logs_tab,
                    effective_tab == Tab::Trajectory,
                    false,
                ),
            );
        }
    }

    fn help_overlay_items(
        &self,
        active_tab: Tab,
        logs_integrated: bool,
        show_trajectory_tab: bool,
        show_logs_tab: bool,
        trajectory_visible: bool,
        merged: bool,
    ) -> Vec<String> {
        let mut items = vec![];
        if merged {
            items.push("f  Toggle Follow/Fit".to_string());
            items.push("+/-  Zoom Follow View (5m)".to_string());
            items.push("v  Cycle Trajectory Pane Layout".to_string());
            items.push("j/k, PgUp/PgDn, Home/End  Scroll Logs".to_string());
            items.push("m  Toggle Minot Logs".to_string());
            items.push("?  Close Help".to_string());
            items.push("q / Esc / Ctrl+C  Quit".to_string());
            return items;
        }

        let tab_hint = match (show_trajectory_tab, show_logs_tab) {
            (true, true) => "Tab or 1-3",
            (true, false) => "Tab or 1-2",
            (false, true) => "Tab or 1-2",
            (false, false) => "1",
        };
        items.push(format!("{}  Switch Tab", tab_hint));

        match active_tab {
            Tab::Stats => {
                if trajectory_visible {
                    items.push("f  Toggle Follow/Fit (Trajectory)".to_string());
                    items.push("+/-  Zoom Follow View (5m)".to_string());
                    items.push("v  Cycle Trajectory Pane Layout".to_string());
                }
                if logs_integrated {
                    items.push("j/k, PgUp/PgDn, Home/End  Scroll Logs".to_string());
                    items.push("m  Toggle Minot Logs".to_string());
                }
            }
            Tab::Trajectory => {
                items.push("f  Toggle Follow/Fit".to_string());
                items.push("+/-  Zoom Follow View (5m)".to_string());
                items.push("v  Cycle Pane Layout".to_string());
            }
            Tab::Logs => {
                items.push("j/k or Up/Down  Scroll".to_string());
                items.push("PgUp/PgDn, Home/End  Jump".to_string());
                items.push("m  Toggle Minot Logs".to_string());
            }
        }

        items.push("?  Close Help".to_string());
        items.push("q / Esc / Ctrl+C  Quit".to_string());
        items
    }

    fn render_help_overlay(&self, f: &mut ratatui::Frame, area: Rect, items: Vec<String>) {
        if !self.show_help_overlay || area.width < 20 || area.height < 6 {
            return;
        }

        let max_item_width = items.iter().map(|s| s.chars().count()).max().unwrap_or(20) as u16;
        let width = (max_item_width + 4)
            .min(DOCKED_HELP_PANEL_MAX_WIDTH)
            .min(area.width.saturating_sub(2))
            .max(DOCKED_HELP_PANEL_MIN_WIDTH);
        let height = ((items.len() as u16) + 2)
            .min(area.height.saturating_sub(2))
            .max(5);
        // Dock-style help panel: fixed to bottom-right of the active content area.
        let x = area.x + area.width.saturating_sub(width + 1);
        let y = area.y + area.height.saturating_sub(height + 1);
        let overlay = Rect {
            x,
            y,
            width,
            height,
        };

        let lines: Vec<Line> = items.into_iter().map(Line::from).collect();
        let panel = Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(Span::styled(
                    " Help (? close) ",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                )),
        );

        f.render_widget(Clear, overlay);
        f.render_widget(panel, overlay);
    }

    fn render_title_bar(&self, f: &mut ratatui::Frame, area: Rect, stats: &TuiStats) {
        let mode_text = if stats.mode.is_empty() {
            "Pelorus".to_owned()
        } else {
            format!("Pelorus {}", stats.mode)
        };

        let mut spans = vec![
            Span::styled(
                format!(" {mode_text} "),
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  backend ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                stats.backend.clone(),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  rate ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1} Hz", stats.processing_freq_hz),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  RT ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.2}x", stats.real_time_factor),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!("  {}", format_duration(stats.elapsed_secs)),
                Style::default().fg(Color::DarkGray),
            ),
        ];

        if let Some(bag) = &stats.bagfile {
            spans.push(Span::styled("  bag ", Style::default().fg(Color::DarkGray)));
            spans.push(Span::styled(bag, Style::default().fg(Color::Gray)));
        }

        let paragraph = Paragraph::new(Line::from(spans)).style(Self::status_bar_style());
        f.render_widget(paragraph, area);
    }

    fn render_tabs(
        &self,
        f: &mut ratatui::Frame,
        area: Rect,
        show_trajectory_tab: bool,
        show_logs_tab: bool,
        selected_tab: Tab,
    ) {
        if Self::visible_tab_count(show_trajectory_tab, show_logs_tab) <= 1 {
            return;
        }

        let mut tabs = vec![(1, Tab::Stats, "Stats")];
        if show_trajectory_tab {
            tabs.push((2, Tab::Trajectory, "Trajectory"));
        }
        if show_logs_tab {
            tabs.push((if show_trajectory_tab { 3 } else { 2 }, Tab::Logs, "Logs"));
        }

        let mut spans = Vec::new();
        for (number, tab, title) in tabs {
            let style = if tab == selected_tab {
                Style::default()
                    .fg(Color::Black)
                    .bg(Color::Cyan)
                    .add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(Color::Gray)
            };
            spans.push(Span::styled(format!(" {number}:{title} "), style));
            spans.push(Span::raw(" "));
        }
        f.render_widget(
            Paragraph::new(Line::from(spans)).style(Self::status_bar_style()),
            area,
        );
    }

    fn render_stats_tab(&self, f: &mut ratatui::Frame, area: Rect, stats: &TuiStats) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4), // Header info
                Constraint::Length(7), // State
                Constraint::Length(4), // Navigation
                Constraint::Min(6),    // Performance
            ])
            .split(area);

        self.render_header(f, chunks[0], stats);
        self.render_state(f, chunks[1], stats);
        self.render_navigation(f, chunks[2], stats);
        self.render_performance(f, chunks[3], stats);
    }

    fn render_trajectory_tab(
        &self,
        f: &mut ratatui::Frame,
        area: Rect,
        stats: &TuiStats,
        left_divider: bool,
        show_rt_stat: bool,
    ) {
        match self.trajectory_pane_mode {
            TrajectoryPaneMode::Split => {
                let chunks = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
                    .split(area);

                // Left: Top-down view (X-Y)
                self.render_trajectory_view(
                    f,
                    chunks[0],
                    stats,
                    TrajectoryView {
                        title: "Top (X-Y)",
                        axis_h: 0,
                        axis_v: 1,
                        show_top_stats: false,
                        show_left_divider: left_divider,
                        show_rt_stat,
                    },
                );

                // Right: Side view (X-Z) + shared top stats overlay
                self.render_trajectory_view(
                    f,
                    chunks[1],
                    stats,
                    TrajectoryView {
                        title: "Side (X-Z)",
                        axis_h: 0,
                        axis_v: 2,
                        show_top_stats: true,
                        show_left_divider: true,
                        show_rt_stat,
                    },
                );
            }
            TrajectoryPaneMode::LeftOnly => {
                self.render_trajectory_view(
                    f,
                    area,
                    stats,
                    TrajectoryView {
                        title: "Top (X-Y)",
                        axis_h: 0,
                        axis_v: 1,
                        show_top_stats: true,
                        show_left_divider: left_divider,
                        show_rt_stat,
                    },
                );
            }
            TrajectoryPaneMode::RightOnly => {
                self.render_trajectory_view(
                    f,
                    area,
                    stats,
                    TrajectoryView {
                        title: "Side (X-Z)",
                        axis_h: 0,
                        axis_v: 2,
                        show_top_stats: true,
                        show_left_divider: left_divider,
                        show_rt_stat,
                    },
                );
            }
        }
    }

    fn render_trajectory_view(
        &self,
        f: &mut ratatui::Frame,
        area: Rect,
        stats: &TuiStats,
        view: TrajectoryView<'_>,
    ) {
        let TrajectoryView {
            title,
            axis_h,
            axis_v,
            show_top_stats,
            show_left_divider,
            show_rt_stat,
        } = view;
        let panel_borders = if show_left_divider {
            Borders::LEFT
        } else {
            Borders::NONE
        };

        if stats.trajectory.is_empty() {
            let block = Block::default()
                .borders(panel_borders)
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
            let range = self.follow_view_range_m; // View range around current position
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
        let mode_label = if self.view_mode == ViewMode::FollowCurrent {
            format!("{} {:.0}m", mode_str, self.follow_view_range_m)
        } else {
            mode_str.to_string()
        };

        let (rt_color, rt_status) = get_realtime_color_and_status(stats.real_time_factor);
        let vel_mag = (stats.linear_velocity[0].powi(2)
            + stats.linear_velocity[1].powi(2)
            + stats.linear_velocity[2].powi(2))
        .sqrt();

        let range_h = max_h - min_h;
        let range_v = max_v - min_v;

        // Axis labels
        let axis_h_label = ["X", "Y", "Z"][axis_h];
        let axis_v_label = ["X", "Y", "Z"][axis_v];

        let canvas = Canvas::default()
            .block(
                Block::default()
                    .borders(panel_borders)
                    .border_style(Style::default().fg(Color::DarkGray))
                    .title(Span::styled(
                        format!(" {} [{}] ", title, mode_label),
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

                // Scale labels on edges
            });

        f.render_widget(canvas, area);

        // Overlay the corner stats in screen space so they stay tightly packed
        // and keep a stable left/right pairing.
        if area.width > 2 && area.height > 7 {
            let inner_width = area.width.saturating_sub(2);
            if show_top_stats {
                let top_x = area.x + 1;
                let top_w = inner_width;
                let top_right_w = inner_width.min(30);
                let top_right_x = area.x + 1 + inner_width.saturating_sub(top_right_w);

                // Left: pose readout
                let top_left_line = Line::from(vec![
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
                ]);

                f.render_widget(
                    Paragraph::new(top_left_line).alignment(Alignment::Left),
                    Rect {
                        x: top_x,
                        y: area.y + 1,
                        width: top_w,
                        height: 1,
                    },
                );

                // Right: velocity and optionally real-time factor panel.
                // When stats/info is visible alongside trajectory, omit both here.
                let mut top_right_lines = Vec::new();
                if show_rt_stat {
                    top_right_lines.push(Line::from(vec![
                        Span::styled(
                            "VEL",
                            Style::default()
                                .fg(Color::Green)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(" ", Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            format!("{:.2} m/s", vel_mag),
                            Style::default()
                                .fg(Color::Green)
                                .add_modifier(Modifier::BOLD),
                        ),
                    ]));
                }
                if show_rt_stat {
                    top_right_lines.push(Line::from(vec![
                        Span::styled(
                            "RT",
                            Style::default().fg(rt_color).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(" ", Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            format!("{:.2}x", stats.real_time_factor),
                            Style::default().fg(rt_color).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(format!(" {}", rt_status), Style::default().fg(rt_color)),
                    ]));
                }

                for (idx, line) in top_right_lines.into_iter().enumerate() {
                    let y = area.y + 1 + idx as u16;
                    f.render_widget(
                        Paragraph::new(line).alignment(Alignment::Right),
                        Rect {
                            x: top_right_x,
                            y,
                            width: top_right_w,
                            height: 1,
                        },
                    );
                }
            }

            let axis_color = |axis: &str| match axis {
                "X" => Color::Red,
                "Y" => Color::Green,
                "Z" => Color::Cyan,
                _ => Color::DarkGray,
            };

            let left_legend_label = Line::from(vec![
                Span::styled(axis_v_label, Style::default().fg(axis_color(axis_v_label))),
                Span::styled(" ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1}m / {:.1}m", max_v, min_v),
                    Style::default().fg(Color::White),
                ),
            ]);
            let right_legend_label = Line::from(vec![
                Span::styled(axis_h_label, Style::default().fg(axis_color(axis_h_label))),
                Span::styled(" ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1}m / {:.1}m", max_h, min_h),
                    Style::default().fg(Color::White),
                ),
            ]);
            let range_label = format!("Δ{:.1}m × Δ{:.1}m", range_h, range_v);

            let col_width = inner_width.min(24);
            let left_x = area.x + 1;
            let right_x = area.x + 1 + inner_width.saturating_sub(col_width);
            let legend_y = area.y + area.height.saturating_sub(1);
            let delta_y = legend_y.saturating_sub(1);

            f.render_widget(
                Paragraph::new(left_legend_label).alignment(Alignment::Left),
                Rect {
                    x: left_x,
                    y: legend_y,
                    width: col_width,
                    height: 1,
                },
            );
            f.render_widget(
                Paragraph::new(right_legend_label).alignment(Alignment::Right),
                Rect {
                    x: right_x,
                    y: legend_y,
                    width: col_width,
                    height: 1,
                },
            );

            f.render_widget(
                Paragraph::new(Line::from(Span::styled(
                    range_label,
                    Style::default().fg(Color::DarkGray),
                )))
                .alignment(Alignment::Right),
                Rect {
                    x: right_x,
                    y: delta_y,
                    width: col_width,
                    height: 1,
                },
            );
        }
    }

    fn render_header(&self, f: &mut ratatui::Frame, area: Rect, stats: &TuiStats) {
        let vendor = match stats.lidar_vendor {
            Some(LidarVendor::Ouster) => "Ouster",
            Some(LidarVendor::Velodyne) => "Velodyne",
            Some(LidarVendor::Livox) => "Livox",
            Some(LidarVendor::LivoxMid360) => "Livox Mid-360",
            Some(LidarVendor::Robosense) => "Robosense",
            Some(LidarVendor::Hesai) => "Hesai",
            Some(LidarVendor::Blickfeld) => "Blickfeld",
            Some(LidarVendor::SickMultiscan) => "SickMultiScan",
            Some(LidarVendor::NoTimestamp) => "No Timestamp",
            None => "Unknown",
        };

        let elapsed = format_duration(stats.elapsed_secs);
        let sensor_rate_str = if stats.measured_sensor_rate_hz > 0.0 {
            format!(" @ {:.1} Hz", stats.measured_sensor_rate_hz)
        } else {
            String::new()
        };

        let mut info_line = vec![
            Span::styled("Backend     ", Style::default().fg(Color::DarkGray)),
            Span::styled(&stats.backend, Style::default().fg(Color::White)),
        ];

        if let Some(accel) = &stats.accelerator {
            info_line.push(Span::raw(" + "));
            let color = if accel.to_lowercase().contains("metal") {
                Color::Green
            } else if accel.to_lowercase().contains("cuda") {
                Color::Cyan
            } else {
                Color::Magenta
            };
            info_line.push(Span::styled(
                accel,
                Style::default().fg(color).add_modifier(Modifier::BOLD),
            ));

            if let Some(details) = &stats.accelerator_details {
                info_line.push(Span::raw(" ("));
                info_line.push(Span::styled(details, Style::default().fg(Color::Gray)));
                info_line.push(Span::raw(")"));
            }
        }

        let lines = vec![
            Line::from(vec![
                Span::styled("Elapsed     ", Style::default().fg(Color::DarkGray)),
                Span::styled(elapsed, Style::default().fg(Color::White)),
                Span::styled("  sensor ", Style::default().fg(Color::DarkGray)),
                Span::styled(vendor, Style::default().fg(Color::Cyan)),
                Span::styled(sensor_rate_str, Style::default().fg(Color::Cyan)),
            ]),
            Line::from(info_line),
        ];

        let block = Block::default()
            .borders(Borders::NONE)
            .border_style(Style::default().fg(Color::DarkGray))
            .padding(Padding::left(STATS_LEFT_PADDING))
            .title(Span::styled(
                " Info ",
                Style::default()
                    .fg(Color::Green)
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
        let vel_abs_ms = (lv[0].powi(2) + lv[1].powi(2) + lv[2].powi(2)).sqrt();
        let vel_abs_kmh = vel_abs_ms * 3.6;

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
                Span::styled("Speed |v|   ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:^12}", format!("{:.2} m/s", vel_abs_ms)),
                    Style::default().fg(Color::Green),
                ),
                Span::styled(" // ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:^12}", format!("{:.1} km/h", vel_abs_kmh)),
                    Style::default().fg(Color::Cyan),
                ),
            ]),
        ];

        let block = Block::default()
            .borders(Borders::NONE)
            .border_style(Style::default().fg(Color::DarkGray))
            .padding(Padding::left(STATS_LEFT_PADDING))
            .title(Span::styled(
                " State ",
                Style::default()
                    .fg(Color::Green)
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
            .borders(Borders::NONE)
            .border_style(Style::default().fg(Color::DarkGray))
            .padding(Padding::left(STATS_LEFT_PADDING))
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
        ];
        if stats.system_stats {
            lines.extend([
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
            ]);
        }

        {
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
            .borders(Borders::NONE)
            .border_style(Style::default().fg(Color::DarkGray))
            .padding(Padding::left(STATS_LEFT_PADDING))
            .title(Span::styled(
                " Performance ",
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ));
        let paragraph = Paragraph::new(lines).block(block);
        f.render_widget(paragraph, area);
    }

    fn is_minot_log(entry: &LogEntry) -> bool {
        entry.message.starts_with("[Minot]")
    }

    fn render_logs_tab(&self, f: &mut ratatui::Frame, area: Rect) {
        let entries = self.log_buffer.snapshot();
        let visible: Vec<(usize, &LogEntry)> = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| self.show_minot_logs || !Self::is_minot_log(e))
            .collect();
        let title = if self.show_minot_logs {
            " Logs "
        } else {
            " Logs [minot hidden] "
        };
        mt_log::pane::render(f, area, &visible, &self.log_view, title);
    }

    fn render_help_bar_context(&self, f: &mut ratatui::Frame, area: Rect, ctx: HelpBarContext) {
        let HelpBarContext {
            active_tab,
            logs_integrated,
            show_trajectory_tab,
            show_logs_tab,
            trajectory_visible,
        } = ctx;
        f.render_widget(Block::default().style(Self::status_bar_style()), area);

        let tab_hint = match (show_trajectory_tab, show_logs_tab) {
            (true, true) => "1-3",
            (true, false) => "1-2",
            (false, true) => "1-2",
            (false, false) => "1",
        };

        let left = match active_tab {
            Tab::Stats => {
                // With a single tab there is nothing to switch to.
                let mut spans = if show_trajectory_tab || show_logs_tab {
                    vec![
                        Span::styled(
                            " Tab",
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("/", Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            tab_hint,
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(" Switch  ", Style::default().fg(Color::DarkGray)),
                    ]
                } else {
                    vec![Span::raw(" ")]
                };

                if trajectory_visible {
                    spans.extend([
                        Span::styled(
                            "f",
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(" Follow/Fit  ", Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            "+/-",
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(" Zoom  ", Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            "v",
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(" Pane Layout  ", Style::default().fg(Color::DarkGray)),
                    ]);
                }

                if logs_integrated {
                    spans.extend([
                        Span::styled(
                            "j/k",
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(" Scroll Logs  ", Style::default().fg(Color::DarkGray)),
                        Span::styled(
                            "PgUp/PgDn",
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
                            "m",
                            Style::default()
                                .fg(Color::Yellow)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(" Minot  ", Style::default().fg(Color::DarkGray)),
                    ]);
                }

                Line::from(spans)
            }
            Tab::Trajectory => Line::from(vec![
                Span::styled(
                    " Tab",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled("/", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    tab_hint,
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
                Span::styled(" Toggle Follow/Fit  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "+/-",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Zoom  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "v",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Pane Layout  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "Ctrl+C",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Stop", Style::default().fg(Color::DarkGray)),
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
                    tab_hint,
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
                    "m",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Minot  ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    "Ctrl+C",
                    Style::default()
                        .fg(Color::Yellow)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(" Stop", Style::default().fg(Color::DarkGray)),
            ]),
        };

        let right = Line::from(vec![
            Span::styled(
                " ? ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Help  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                "q/Esc",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(" Quit", Style::default().fg(Color::DarkGray)),
        ]);
        f.render_widget(
            Paragraph::new(left)
                .alignment(Alignment::Left)
                .style(Self::status_bar_style()),
            area,
        );
        f.render_widget(
            Paragraph::new(right)
                .alignment(Alignment::Right)
                .style(Self::status_bar_style()),
            area,
        );
    }

    fn render_help_bar_merged(&self, f: &mut ratatui::Frame, area: Rect) {
        f.render_widget(Block::default().style(Self::status_bar_style()), area);

        let left = Line::from(vec![
            Span::styled(
                " f ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Follow/Fit  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                " +/- ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Zoom  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                " v ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Pane Layout  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                " j/k ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Scroll Logs  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                " m ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Minot", Style::default().fg(Color::DarkGray)),
        ]);
        let right = Line::from(vec![
            Span::styled(
                " ? ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Help  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                " q/Esc ",
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Quit", Style::default().fg(Color::DarkGray)),
        ]);
        f.render_widget(
            Paragraph::new(left)
                .alignment(Alignment::Left)
                .style(Self::status_bar_style()),
            area,
        );
        f.render_widget(
            Paragraph::new(right)
                .alignment(Alignment::Right)
                .style(Self::status_bar_style()),
            area,
        );
    }
}

fn get_realtime_color_and_status(rt_factor: f64) -> (Color, &'static str) {
    if rt_factor <= 0.0 {
        (Color::Gray, "initializing")
    } else if rt_factor < 0.8 {
        (Color::Red, "critical")
    } else if rt_factor < 0.95 {
        (Color::Yellow, "warning")
    } else if rt_factor < 1.5 {
        (Color::Green, "ok")
    } else if rt_factor < 3.0 {
        (Color::Green, "good")
    } else {
        (Color::Cyan, "excellent")
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_stats() -> TuiStats {
        TuiStats {
            mode: "Offline".to_owned(),
            backend: "CPU".to_owned(),
            lidar_vendor: Some(LidarVendor::Hesai),
            target_lidar_rate_hz: 10.0,
            position: [12.0, -3.5, 0.4],
            orientation: [1.0, 0.0, 0.0, 0.0],
            real_time_factor: 1.8,
            elapsed_secs: 42.0,
            trajectory: (0..500)
                .map(|i| {
                    let t = i as f64 * 0.05;
                    [t.cos() * 10.0, t.sin() * 6.0, 0.1 * t]
                })
                .collect(),
            ..TuiStats::default()
        }
    }

    #[test]
    fn renders_at_every_layout_and_after_every_key() {
        let stats = sample_stats();
        let mut view = StatsView::new(Arc::new(LogBuffer::new("test")));
        view.refresh_dynamic_layout_thresholds(&stats);
        let keys = [
            Key::Tab,
            Key::Char('2'),
            Key::Char('f'),
            Key::Char('v'),
            Key::Char('+'),
            Key::Char('-'),
            Key::Char('3'),
            Key::Up,
            Key::PageDown,
            Key::Home,
            Key::End,
            Key::Char('m'),
            Key::Char('?'),
            Key::Char('1'),
        ];
        for (width, height, elsewhere) in [(60, 20), (100, 30), (140, 34), (200, 50)]
            .into_iter()
            .flat_map(|(w, h)| [(w, h, false), (w, h, true)])
        {
            view.set_trajectory_elsewhere(elsewhere);
            let area = Rect::new(0, 0, width, height);
            let mut terminal =
                ratatui::Terminal::new(ratatui::backend::TestBackend::new(width, height)).unwrap();
            for key in keys {
                let press = KeyPress { key, ctrl: false };
                assert_eq!(view.handle_key(press, area), KeyAction::Continue);
                terminal.draw(|f| view.render(f, &stats)).unwrap();
            }
        }
        let quit = KeyPress {
            key: Key::Char('c'),
            ctrl: true,
        };
        assert_eq!(
            view.handle_key(quit, Rect::new(0, 0, 100, 30)),
            KeyAction::Exit
        );
    }

    #[test]
    fn tabbed_layout_has_no_empty_spacer_rows() {
        let area = Rect::new(0, 0, 100, 30);
        let with_tabs = StatsView::split_tabbed_layout(area, true);
        assert_eq!(with_tabs.header, Rect::new(0, 0, 100, 1));
        assert_eq!(with_tabs.tabs, Rect::new(0, 1, 100, 1));
        assert_eq!(with_tabs.content.y, 2);
        assert_eq!(with_tabs.help, Rect::new(0, 29, 100, 1));

        let without_tabs = StatsView::split_tabbed_layout(area, false);
        assert_eq!(without_tabs.tabs.height, 0);
        assert_eq!(without_tabs.content.y, 1);
    }

    #[test]
    fn realtime_status_uses_plain_text() {
        for factor in [0.0, 0.5, 0.9, 1.0, 2.0, 4.0] {
            let (_, status) = get_realtime_color_and_status(factor);
            assert!(status.is_ascii(), "status contains decoration: {status}");
        }
    }
}
