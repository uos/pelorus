use crossterm::{
    event::{self, Event, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, Tabs},
};
use std::io;
use std::sync::{Arc, Mutex};

pub struct OutputCapture {
    minot_lines: Arc<Mutex<Vec<String>>>,
    pelorus_lines: Arc<Mutex<Vec<String>>>,
}

impl OutputCapture {
    pub fn new() -> Self {
        Self {
            minot_lines: Arc::new(Mutex::new(Vec::new())),
            pelorus_lines: Arc::new(Mutex::new(Vec::new())),
        }
    }

    pub fn add_minot_line(&self, line: String) {
        let mut lines = self.minot_lines.lock().unwrap();
        lines.push(line);
        let len = lines.len();
        if len > 1000 {
            lines.drain(0..len - 1000);
        }
    }

    pub fn add_pelorus_line(&self, line: String) {
        let mut lines = self.pelorus_lines.lock().unwrap();
        lines.push(line);
        let len = lines.len();
        if len > 1000 {
            lines.drain(0..len - 1000);
        }
    }

    pub fn get_minot_lines(&self) -> Vec<String> {
        self.minot_lines.lock().unwrap().clone()
    }

    pub fn get_pelorus_lines(&self) -> Vec<String> {
        self.pelorus_lines.lock().unwrap().clone()
    }
}

pub struct CaptureLogger {
    capture: Arc<OutputCapture>,
}

impl CaptureLogger {
    pub fn new(capture: Arc<OutputCapture>) -> Self {
        Self { capture }
    }
}

impl log::Log for CaptureLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        metadata.level() <= log::max_level()
    }

    fn log(&self, record: &log::Record) {
        if self.enabled(record.metadata()) {
            let line = format!("[{}] {}", record.level(), record.args());
            self.capture.add_pelorus_line(line);
        }
    }

    fn flush(&self) {}
}

pub struct OutputTui {
    capture: Arc<OutputCapture>,
    selected_tab: usize,
    stop_rx: Option<std::sync::mpsc::Receiver<()>>,
    shutdown_tx: Option<std::sync::mpsc::Sender<()>>,
    bagfile_name: String,
}

impl OutputTui {
    pub fn new(capture: Arc<OutputCapture>, bagfile_name: String) -> Self {
        Self {
            capture,
            bagfile_name,
            selected_tab: 0,
            stop_rx: None,
            shutdown_tx: None,
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
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        loop {
            terminal.draw(|f| {
                let size = f.area();

                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([Constraint::Length(3), Constraint::Min(0)].as_ref())
                    .split(size);

                let tabs = Tabs::new(vec!["Pelorus", "Minot"])
                    .block(Block::default().borders(Borders::ALL).title("Offline"))
                    .select(self.selected_tab)
                    .style(Style::default().fg(Color::White))
                    .highlight_style(
                        Style::default()
                            .fg(Color::Yellow)
                            .add_modifier(Modifier::BOLD),
                    );
                f.render_widget(tabs, chunks[0]);

                let lines = if self.selected_tab == 0 {
                    self.capture.get_pelorus_lines()
                } else {
                    self.capture.get_minot_lines()
                };

                let items: Vec<ListItem> = lines
                    .iter()
                    .map(|line| {
                        let content = Line::from(Span::raw(line.clone()));
                        ListItem::new(content)
                    })
                    .collect();

                let list = List::new(items)
                    .block(
                        Block::default()
                            .borders(Borders::ALL)
                            .title(self.bagfile_name.as_str()),
                    )
                    .style(Style::default().fg(Color::White));

                f.render_widget(list, chunks[1]);
            })?;

            // TODO do it like minot
            if event::poll(std::time::Duration::from_millis(50))? {
                if let Event::Key(key) = event::read()? {
                    match key.code {
                        KeyCode::Char('q') | KeyCode::Esc => {
                            if let Some(tx) = &self.shutdown_tx {
                                let _ = tx.send(());
                            }
                            break;
                        }
                        KeyCode::Tab | KeyCode::Char(' ') => {
                            self.selected_tab = (self.selected_tab + 1) % 2;
                        }
                        KeyCode::Char('1') => self.selected_tab = 0,
                        KeyCode::Char('2') => self.selected_tab = 1,
                        _ => {}
                    }
                    if key.code == KeyCode::Char('c')
                        && key.modifiers.contains(KeyModifiers::CONTROL)
                    {
                        if let Some(tx) = &self.shutdown_tx {
                            let _ = tx.send(());
                        }
                        break;
                    }
                }
            }

            // external
            if let Some(rx) = &self.stop_rx {
                if rx.try_recv().is_ok() {
                    // External shutdown requested; forward to main via shutdown_tx and exit
                    if let Some(tx) = &self.shutdown_tx {
                        let _ = tx.send(());
                    }
                    break;
                }
            }
            // TODO remove
            std::thread::sleep(std::time::Duration::from_millis(50));
        }

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        Ok(())
    }
}

impl Drop for OutputTui {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let mut stdout = io::stdout();
        let _ = execute!(stdout, LeaveAlternateScreen);
    }
}
