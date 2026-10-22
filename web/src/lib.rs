//! The Pelorus browser demo.
//!
//! The page runs this module twice. A Web Worker replays a bag through
//! [`pelorus_wasm::McapBag`] and [`pelorus_wasm::Odometer`] and posts every
//! scan as its [`toArray`](pelorus_wasm::Scan::to_array). The main thread hands
//! those to [`WebApp::push_scan`], which keeps the statistics and draws the
//! Pelorus status view with Ratzilla.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;

use mt_log::{LogBuffer, LogLevel};
use pelorus_core::LidarVendor;
use pelorus_tui::{Key, KeyAction, KeyPress, StatsView, TuiStats};
use pelorus_wasm::SCAN_ARRAY_LEN;
use ratzilla::backend::canvas::CanvasBackendOptions;
use ratzilla::backend::webgl2::WebGl2BackendOptions;
use ratzilla::event::{KeyCode, KeyEvent};
use ratzilla::ratatui::backend::Backend;
use ratzilla::ratatui::buffer::Buffer;
use ratzilla::ratatui::style::Color;
use ratzilla::ratatui::{Terminal, layout::Rect};
use ratzilla::{CanvasBackend, FontAtlasConfig, WebEventHandler, WebGl2Backend, WebRenderer};
use serde::Deserialize;
use wasm_bindgen::prelude::*;

/// Keeps the odometer and the bag reader in the module the worker loads.
pub use pelorus_wasm;

const MAX_TRAJECTORY_POINTS: usize = 10_000;
/// Scans the sensor rate is measured over.
const RATE_WINDOW: usize = 100;
/// Processing times the average and maximum are taken over.
const TIMING_WINDOW: usize = 1000;
/// A pose change below this does not extend the drawn trajectory.
const MIN_TRAJECTORY_STEP_M: f64 = 0.01;

/// Background of the page, which the view is drawn on.
const PAGE_BACKGROUND: Color = Color::Rgb(11, 12, 15);
/// Text without a colour of its own.
const PAGE_TEXT: Color = Color::Rgb(200, 202, 207);
const FONT_FAMILIES: [&str; 5] = [
    "JetBrains Mono",
    "Fira Code",
    "Menlo",
    "Consolas",
    "monospace",
];
const DEFAULT_FONT_SIZE_PX: f32 = 14.0;

/// The page's colour for a named terminal colour.
///
/// Ratzilla draws named colours with the dim VGA palette. A terminal shows
/// them brighter, so the view is recoloured to match what the CLI looks like.
fn page_color(color: Color, reset: Color) -> Color {
    match color {
        Color::Reset => reset,
        Color::Black => PAGE_BACKGROUND,
        Color::Red => Color::Rgb(224, 108, 117),
        Color::Green => Color::Rgb(152, 195, 121),
        Color::Yellow => Color::Rgb(229, 192, 123),
        Color::Blue => Color::Rgb(97, 175, 239),
        Color::Magenta => Color::Rgb(198, 120, 221),
        Color::Cyan => Color::Rgb(86, 182, 194),
        Color::Gray => Color::Rgb(171, 178, 191),
        Color::DarkGray => Color::Rgb(118, 126, 140),
        Color::LightRed => Color::Rgb(240, 140, 140),
        Color::LightGreen => Color::Rgb(175, 220, 145),
        Color::LightYellow => Color::Rgb(240, 212, 150),
        Color::LightBlue => Color::Rgb(135, 195, 250),
        Color::LightMagenta => Color::Rgb(220, 155, 240),
        Color::LightCyan => Color::Rgb(125, 210, 222),
        Color::White => Color::Rgb(232, 236, 242),
        other => other,
    }
}

fn recolor(buffer: &mut Buffer) {
    for cell in &mut buffer.content {
        let (fg, bg) = (cell.fg, cell.bg);
        cell.set_fg(page_color(fg, PAGE_TEXT));
        cell.set_bg(page_color(bg, PAGE_BACKGROUND));
    }
}

fn now_ms() -> f64 {
    js_sys::Date::now()
}

fn three(values: &[f64]) -> [f64; 3] {
    [values[0], values[1], values[2]]
}

struct State {
    stats: TuiStats,
    /// The statistics a replay starts from.
    initial_stats: TuiStats,
    view: StatsView,
    started_ms: f64,
    last_position: Option<[f64; 3]>,
    scan_stamps: VecDeque<f64>,
    processing_ms: VecDeque<f64>,
    /// When the recent scans arrived, for the processing rate.
    scan_arrivals_ms: VecDeque<f64>,
    /// Size of the last drawn frame, for key handling.
    area: Rect,
    quit: bool,
}

/// What a key press asks the page to do.
enum PageAction {
    None,
    Quit,
    Restart,
}

/// The page's callbacks for [`PageAction`].
struct PageCallbacks {
    on_quit: Option<js_sys::Function>,
    on_restart: Option<js_sys::Function>,
}

impl PageCallbacks {
    /// Run outside any borrow of [`State`], because the callbacks call back
    /// into the [`WebApp`].
    fn run(&self, action: PageAction) {
        let callback = match action {
            PageAction::None => return,
            PageAction::Quit => &self.on_quit,
            PageAction::Restart => &self.on_restart,
        };
        if let Some(callback) = callback {
            let _ = callback.call0(&JsValue::NULL);
        }
    }
}

impl State {
    /// Forget the replay so far. The view keeps its tab and zoom.
    fn reset(&mut self) {
        self.stats = self.initial_stats.clone();
        self.started_ms = now_ms();
        self.last_position = None;
        self.scan_stamps.clear();
        self.processing_ms.clear();
        self.scan_arrivals_ms.clear();
    }

    fn push_scan(&mut self, values: &[f64], processing: f64) {
        let stamp = values[0];
        let position = three(&values[1..4]);
        let stats = &mut self.stats;
        stats.orientation = [values[4], values[5], values[6], values[7]];
        stats.linear_velocity = three(&values[8..11]);
        stats.angular_velocity = three(&values[11..14]);
        stats.accel_bias = three(&values[14..17]);
        stats.gyro_bias = three(&values[17..20]);

        match self.last_position {
            Some(previous) => {
                let step = (0..3)
                    .map(|i| (position[i] - previous[i]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if step >= MIN_TRAJECTORY_STEP_M {
                    stats.distance_traveled += step;
                    self.last_position = Some(position);
                    stats.trajectory.push(position);
                }
            }
            None => {
                self.last_position = Some(position);
                stats.trajectory.push(position);
            }
        }
        if stats.trajectory.len() > MAX_TRAJECTORY_POINTS {
            stats.trajectory.remove(0);
        }
        stats.position = position;
        stats.distance_to_origin = position.iter().map(|v| v * v).sum::<f64>().sqrt();

        if self.scan_stamps.back().is_none_or(|last| stamp > *last) {
            self.scan_stamps.push_back(stamp);
            if self.scan_stamps.len() > RATE_WINDOW {
                self.scan_stamps.pop_front();
            }
        }
        if let (Some(first), Some(last)) = (self.scan_stamps.front(), self.scan_stamps.back())
            && last > first
        {
            stats.measured_sensor_rate_hz = (self.scan_stamps.len() - 1) as f64 / (last - first);
        }

        self.processing_ms.push_back(processing);
        if self.processing_ms.len() > TIMING_WINDOW {
            self.processing_ms.pop_front();
        }
        let average = self.processing_ms.iter().sum::<f64>() / self.processing_ms.len() as f64;
        stats.last_comp_time_ms = processing;
        stats.avg_comp_time_ms = average;
        stats.max_comp_time_ms = self.processing_ms.iter().copied().fold(0.0, f64::max);
        // Scans can arrive in bursts, so the rate is taken over a window.
        self.scan_arrivals_ms.push_back(now_ms());
        if self.scan_arrivals_ms.len() > RATE_WINDOW {
            self.scan_arrivals_ms.pop_front();
        }
        if let (Some(first), Some(last)) =
            (self.scan_arrivals_ms.front(), self.scan_arrivals_ms.back())
            && last > first
        {
            stats.processing_freq_hz =
                (self.scan_arrivals_ms.len() - 1) as f64 * 1000.0 / (last - first);
        }
        let sensor_rate = if stats.measured_sensor_rate_hz > 0.0 {
            stats.measured_sensor_rate_hz
        } else {
            stats.target_lidar_rate_hz
        };
        if sensor_rate > 0.0 && average > 0.0 {
            stats.real_time_factor = 1000.0 / sensor_rate / average;
        }
    }

    fn handle_key(&mut self, event: KeyEvent) -> PageAction {
        if self.quit {
            return PageAction::None;
        }
        if event.code == KeyCode::Char(' ') {
            return PageAction::Restart;
        }
        let Some(key) = key_press(&event) else {
            return PageAction::None;
        };
        if self.view.handle_key(key, self.area) == KeyAction::Exit {
            self.quit = true;
            return PageAction::Quit;
        }
        PageAction::None
    }
}

/// The view's key for a browser key event, or `None` for keys it ignores.
fn key_press(event: &KeyEvent) -> Option<KeyPress> {
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
        ctrl: event.ctrl,
    })
}

/// The status view in the page, fed with the scans the worker posts.
#[wasm_bindgen]
pub struct WebApp {
    state: Rc<RefCell<State>>,
    logs: Arc<LogBuffer>,
}

#[wasm_bindgen]
impl WebApp {
    /// Draw the status view into the element `elementId`, which it fills.
    ///
    /// `bag` names the replayed bag in the header. `lidarVendor` is what the
    /// worker's [`lidarVendor`](pelorus_wasm::Odometer::lidar_vendor) returned.
    /// `onQuit` runs once when the user presses `q`, `Esc` or Ctrl+C.
    /// `onRestart` runs when the user presses Space to replay from the start.
    /// `fontSize` is the cell font size in CSS pixels, 14 without it.
    #[wasm_bindgen(constructor)]
    pub fn new(
        element_id: &str,
        bag: &str,
        lidar_vendor: Option<String>,
        on_quit: Option<js_sys::Function>,
        on_restart: Option<js_sys::Function>,
        font_size: Option<f32>,
    ) -> Result<WebApp, JsError> {
        console_error_panic_hook::set_once();
        let logs = Arc::new(LogBuffer::new("pelorus"));
        let stats = TuiStats {
            mode: "Browser".to_owned(),
            bagfile: Some(bag.to_owned()),
            backend: "WebAssembly".to_owned(),
            lidar_vendor: lidar_vendor.and_then(|name| parse_vendor(&name)),
            target_lidar_rate_hz: 10.0,
            num_cores: 1,
            ..TuiStats::default()
        };
        let state = Rc::new(RefCell::new(State {
            initial_stats: stats.clone(),
            stats,
            view: StatsView::new(Arc::clone(&logs)),
            started_ms: now_ms(),
            last_position: None,
            scan_stamps: VecDeque::new(),
            processing_ms: VecDeque::new(),
            scan_arrivals_ms: VecDeque::new(),
            area: Rect::new(0, 0, 80, 24),
            quit: false,
        }));
        let callbacks = PageCallbacks {
            on_quit,
            on_restart,
        };

        // WebGL2 draws every cell on an exact grid, which the braille trajectory
        // plot needs, and follows the page when the window is resized. The 2D
        // canvas also keeps the grid but stays at the size it started with.
        let webgl = WebGl2Backend::new_with_options(
            WebGl2BackendOptions::new()
                .grid_id(element_id)
                .font_atlas_config(FontAtlasConfig::dynamic(
                    &FONT_FAMILIES,
                    font_size.unwrap_or(DEFAULT_FONT_SIZE_PX),
                ))
                .canvas_padding_color(PAGE_BACKGROUND)
                .disable_auto_css_resize(),
        );
        match webgl {
            Ok(backend) => run(backend, Rc::clone(&state), callbacks)?,
            Err(error) => {
                logs.push(
                    LogLevel::Warn,
                    "pelorus",
                    format!(
                        "WebGL2 is not available ({error}). The view keeps its size when the window is resized."
                    ),
                );
                let backend = CanvasBackend::new_with_options(
                    CanvasBackendOptions::new().grid_id(element_id),
                )
                .map_err(|e| JsError::new(&e.to_string()))?;
                run(backend, Rc::clone(&state), callbacks)?;
            }
        }
        Ok(WebApp { state, logs })
    }

    /// Account one scan the worker posted as its
    /// [`toArray`](pelorus_wasm::Scan::to_array). `processingMs` is how long
    /// the worker spent on the cloud that produced it.
    #[wasm_bindgen(js_name = pushScan)]
    pub fn push_scan(&self, values: &[f64], processing_ms: f64) -> Result<(), JsError> {
        if values.len() != SCAN_ARRAY_LEN {
            return Err(JsError::new(&format!(
                "a scan has {SCAN_ARRAY_LEN} values, got {}",
                values.len()
            )));
        }
        self.state.borrow_mut().push_scan(values, processing_ms);
        Ok(())
    }

    /// Leave the TUI's trajectory panes out while the page draws the
    /// trajectory in its 3D view.
    #[wasm_bindgen(js_name = setTrajectoryElsewhere)]
    pub fn set_trajectory_elsewhere(&self, elsewhere: bool) {
        self.state
            .borrow_mut()
            .view
            .set_trajectory_elsewhere(elsewhere);
    }

    /// Forget the replay so far, for a replay from the start. The logs and the
    /// view's tab and zoom stay.
    pub fn restart(&self) {
        self.state.borrow_mut().reset();
    }

    /// Show `message` in the log pane. `level` is `error`, `warn`, `info`,
    /// `debug` or `trace`.
    pub fn log(&self, level: &str, message: &str) {
        let level = match level {
            "error" => LogLevel::Error,
            "warn" => LogLevel::Warn,
            "debug" => LogLevel::Debug,
            "trace" => LogLevel::Trace,
            _ => LogLevel::Info,
        };
        self.logs.push(level, "pelorus", message);
    }
}

/// Draw `state` with `backend` on every animation frame and pass it the keys.
fn run<B>(backend: B, state: Rc<RefCell<State>>, callbacks: PageCallbacks) -> Result<(), JsError>
where
    B: Backend + WebEventHandler + 'static,
{
    let mut terminal = Terminal::new(backend).map_err(|e| JsError::new(&e.to_string()))?;
    let keys = Rc::clone(&state);
    terminal
        .on_key_event(move |event| {
            let action = keys.borrow_mut().handle_key(event);
            callbacks.run(action);
        })
        .map_err(|e| JsError::new(&e.to_string()))?;
    terminal.draw_web(move |frame| {
        let mut state = state.borrow_mut();
        let state = &mut *state;
        state.stats.elapsed_secs = (now_ms() - state.started_ms) / 1000.0;
        state.view.refresh_dynamic_layout_thresholds(&state.stats);
        state.view.render(frame, &state.stats);
        recolor(frame.buffer_mut());
        state.area = frame.area();
    });
    Ok(())
}

fn parse_vendor(name: &str) -> Option<LidarVendor> {
    LidarVendor::deserialize(
        serde::de::value::StrDeserializer::<serde::de::value::Error>::new(name),
    )
    .ok()
}
