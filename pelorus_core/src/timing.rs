use std::{collections::VecDeque, time::Duration};

use std::ops::{AddAssign, Div, Sub};

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SlidingWindowFilter<T: Default> {
    window_size: f64,
    buffer: VecDeque<T>,
    pub mean: T,
}

impl<T: Default + Sub<T> + AddAssign<<T as Div<f64>>::Output> + Div<f64>> SlidingWindowFilter<T>
where
    T: Default + Sub<Output = T> + AddAssign<<T as Div<f64>>::Output> + Div<f64, Output = T> + Copy,
{
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size: window_size as f64,
            buffer: VecDeque::with_capacity(window_size),
            mean: Default::default(),
        }
    }

    pub fn update(&mut self, new_value: T) -> T {
        if self.window_size < 2.0 {
            return new_value; // Window size is too small
        }

        self.buffer.push_back(new_value);

        // Fill the buffer (init)
        if self.buffer.len() <= self.window_size as usize {
            self.mean += new_value / self.window_size;
            return new_value;
        }

        // Calc mean and throw away head of queue
        self.mean +=
            (*self.buffer.back().unwrap() - *self.buffer.front().unwrap()) / self.window_size;
        self.buffer.pop_front();
        self.mean
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct EvaluationFormular {
    name: String,
    active: bool,
    accumulated_time: Duration,
    num_measurements: u64,
    last_measurement: Duration,
    max_time: Duration,
    sum_time: Duration,
    filter: SlidingWindowFilter<f64>,
    mean_for_variance: f64,
    m2: f64,
}

impl EvaluationFormular {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            active: true,
            accumulated_time: Duration::from_secs(0),
            num_measurements: 0,
            last_measurement: Duration::from_secs(0),
            max_time: Duration::from_secs(0),
            sum_time: Duration::from_secs(0),
            filter: SlidingWindowFilter::new(10),
            mean_for_variance: 0.0,
            m2: 0.0,
        }
    }

    pub fn impact_score(&self) -> f64 {
        // total time spent in this task
        self.filter.mean * self.num_measurements as f64
    }

    pub fn variance(&self) -> f64 {
        if self.num_measurements < 2 {
            return 0.0;
        }
        self.m2 / (self.num_measurements - 1) as f64
    }

    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }
}

const MAX_HISTOGRAM_SIZE: u64 = 10;

#[derive(Debug, PartialEq, Clone)]
pub struct RuntimeEvaluator {
    evaluations: Vec<EvaluationFormular>,
    histogram: Vec<u64>,
    start_time: std::time::Instant,
}

impl Default for RuntimeEvaluator {
    fn default() -> Self {
        Self {
            evaluations: Vec::new(),
            histogram: Vec::new(),
            start_time: std::time::Instant::now(),
        }
    }
}

impl RuntimeEvaluator {
    pub fn new() -> Self {
        Self {
            evaluations: Vec::new(),
            histogram: vec![0, MAX_HISTOGRAM_SIZE],
            start_time: std::time::Instant::now(),
        }
    }

    pub fn merge(&mut self, other: &RuntimeEvaluator) {
        self.pause();
        for e in other.evaluations.iter() {
            let mut found = false;
            for self_e in self.evaluations.iter_mut() {
                if e.name == self_e.name {
                    found = true;
                    if self_e.active {
                        self_e.accumulated_time += e.accumulated_time;
                    } else {
                        self_e.accumulated_time = e.accumulated_time;
                        self_e.num_measurements = e.num_measurements;
                        self_e.last_measurement = e.last_measurement;
                        self_e.max_time = e.max_time;
                        self_e.sum_time = e.sum_time;
                        self_e.filter = e.filter.clone();
                        self_e.mean_for_variance = e.mean_for_variance;
                        self_e.m2 = e.m2;
                    }
                }
            }

            if !found {
                self.evaluations.push(e.clone());
            }
        }

        self.resume();
    }

    pub fn clear(&mut self) {
        self.evaluations.clear();
    }

    pub fn start(&mut self, name: &str) -> anyhow::Result<()> {
        self.pause();

        let mut found = false;
        for e in self.evaluations.iter_mut() {
            if e.name == name {
                found = true;
                // since we are async most of the time, this can happen but it is not important enough to throw an error
                // if e.active {
                //     return Err(anyhow::anyhow!("Evaluation already active"));
                // }
                e.active = true;
                e.accumulated_time = Duration::from_secs(0);
            }
        }

        if !found {
            self.evaluations.push(EvaluationFormular::new(name));
        }

        self.resume();

        Ok(())
    }

    pub fn reset(&mut self, name: &str) -> anyhow::Result<()> {
        for e in self.evaluations.iter_mut() {
            if e.name == name {
                e.active = false;
                return Ok(());
            }
        }

        Err(anyhow::anyhow!("Evaluation not found"))
    }

    /// Pauses the time measurements. The time that has past since the last resume is accumulated for every currently measured task.
    /// This method is called at the beginning of every public method.
    fn pause(&mut self) {
        let stop = std::time::Instant::now();
        let duration = stop - self.start_time;

        for evaluation in self.evaluations.iter_mut().filter(|x| x.active) {
            evaluation.accumulated_time += duration;
        }
    }

    /// Resumes the time measurements by simply storing the current time so that it can be used at the next pause.
    /// This method is called at the end of every public method.
    fn resume(&mut self) {
        self.start_time = std::time::Instant::now();
    }

    pub fn stop(&mut self, name: &str) -> anyhow::Result<()> {
        self.pause();

        for e in self.evaluations.iter_mut() {
            if e.name == name {
                // since we are async most of the time, this can happen but it is not important enough to throw an error
                // if !e.active {
                //     return Err(anyhow::anyhow!("Evaluation not active"));
                // }

                let acc_time = e.accumulated_time;
                e.active = false;
                e.num_measurements += 1;
                e.last_measurement = acc_time;
                e.sum_time += acc_time;

                let micros = acc_time.as_micros() as f64;
                e.filter.update(micros);

                let delta = micros - e.mean_for_variance;
                e.mean_for_variance += delta / e.num_measurements as f64;
                let delta2 = micros - e.mean_for_variance;
                e.m2 += delta * delta2;

                if acc_time > e.max_time && e.num_measurements != 1 {
                    e.max_time = acc_time;
                }

                if name == "total" {
                    let time_ms = acc_time.as_micros() as f64;

                    let bucket = time_ms / MAX_HISTOGRAM_SIZE as f64;
                    let bucket = bucket.min(self.histogram.len() as f64 - 1.0);
                    self.histogram[bucket as usize] += 1;
                }

                self.resume();
                return Ok(());
            }
        }

        Err(anyhow::anyhow!("Evaluation not found"))
    }

    pub fn get_last_total_time_ms(&self) -> Option<f64> {
        for e in self.evaluations.iter() {
            if e.name == "total" && e.num_measurements > 0 {
                return Some(e.last_measurement.as_micros() as f64 / 1000.0);
            }
        }
        None
    }

    pub fn render(&mut self) -> String {
        self.pause();

        // Filter out active evaluations
        let mut sorted_evals: Vec<_> = self
            .evaluations
            .iter()
            .filter(|ef| !ef.active)
            .cloned()
            .collect();

        // determine window size from first evaluation or default, then build table headers
        let window_n = sorted_evals
            .get(0)
            .map(|e| e.filter.window_size as usize)
            .unwrap_or(10);
        let table: Vec<String> = vec![
            "task".to_string(),
            format!("impact (rel)"),
            format!("avg(-{})", window_n),
            "avg(*)".to_string(),
            "std_dev".to_string(),
        ];
        let field_count: usize = table.len();

        // compute raw impacts and max for normalization
        let impacts: Vec<f64> = sorted_evals.iter().map(|e| e.impact_score()).collect();
        let max_impact = impacts.iter().cloned().fold(0. / 0., f64::max).max(1.0);

        // Pre-compute impact scores for all entries
        let impact_map: std::collections::HashMap<String, f64> = sorted_evals
            .iter()
            .map(|e| (e.name.clone(), e.impact_score()))
            .collect();

        // Hierarchical sort with impact-based ordering within each level
        sorted_evals.sort_by(|a, b| {
            let a_parts: Vec<&str> = a.name.split('/').collect();
            let b_parts: Vec<&str> = b.name.split('/').collect();

            // Compare each level of the hierarchy
            for depth in 0..a_parts.len().min(b_parts.len()) {
                // At each level, first check if the path components are the same
                if a_parts[depth] != b_parts[depth] {
                    // Different at this level - need to sort by impact at this level
                    let a_path_at_depth = a_parts[..=depth].join("/");
                    let b_path_at_depth = b_parts[..=depth].join("/");

                    // Find the impact for these paths
                    let a_impact = impact_map
                        .get(&a_path_at_depth)
                        .copied()
                        .unwrap_or_else(|| impact_map.get(&a.name).copied().unwrap_or(0.0));

                    let b_impact = impact_map
                        .get(&b_path_at_depth)
                        .copied()
                        .unwrap_or_else(|| impact_map.get(&b.name).copied().unwrap_or(0.0));

                    // Sort by impact (highest first) at this level
                    return b_impact
                        .partial_cmp(&a_impact)
                        .unwrap_or(std::cmp::Ordering::Equal);
                }
            }

            // One is a parent of the other - parent comes first
            a_parts.len().cmp(&b_parts.len())
        });

        // Calculate width considering indentation; cap widths to keep table compact
        let mut name_width = 0usize;
        for ef in sorted_evals.iter() {
            let depth = ef.name.matches('/').count();
            let indent = depth * 2;
            let display_name = ef.name.split('/').last().unwrap_or(&ef.name);
            name_width = std::cmp::max(name_width, display_name.len() + indent);
        }
        name_width = std::cmp::max(name_width, table[0].len());
        let name_width = std::cmp::min(std::cmp::max(name_width, 10), 24); // cap between 10 and 24

        // determine numeric column width based on headers and default
        let mut max_numeric_header = 0usize;
        for i in 1..field_count {
            max_numeric_header = std::cmp::max(max_numeric_header, table[i].len());
        }
        let num_width = std::cmp::max(8usize, max_numeric_header); // narrower numeric columns

        let mut ss = String::new();
        ss.push('\n');
        // print header: name column then numeric columns
        ss.push_str(&format!(
            "{:name_width$}",
            table[0],
            name_width = name_width
        ));
        ss.push_str(" | ");
        for i in 1..field_count {
            // center numeric header for readability
            let label = &table[i];
            let padding = num_width.saturating_sub(label.len());
            let left = padding / 2;
            let right = padding - left;
            ss.push_str(&" ".repeat(left));
            ss.push_str(label);
            ss.push_str(&" ".repeat(right));
            if i + 1 != field_count {
                ss.push_str(" | ");
            } else {
                ss.push('\n');
            }
        }

        // separator line (match header separators)
        ss.push_str(&format!("{:->name_width$}", "", name_width = name_width));
        ss.push_str(" | ");
        for i in 1..field_count {
            ss.push_str(&format!("{:->num_width$}", "", num_width = num_width));
            if i + 1 != field_count {
                ss.push_str(" | ");
            } else {
                ss.push('\n');
            }
        }

        // Compute total over top-level (highest hierarchy) entries
        let mut total_avg_micros: u128 = 0;
        let mut total_run_avg_micros: u128 = 0;
        let mut total_variance: f64 = 0.0; // sum of variances
        for ef in sorted_evals.iter().filter(|e| !e.name.contains('/')) {
            let avg = (ef.sum_time / ef.num_measurements as u32).as_micros() as u128;
            let run_avg = ef.filter.mean as u128;
            total_avg_micros += avg;
            total_run_avg_micros += run_avg;
            let sd = ef.std_dev(); // f64 microseconds
            total_variance += sd * sd;
        }
        let total_stddev = total_variance.sqrt();

        // Render special TOTAL row (highlighted)
        let total_name = "total";
        ss.push_str(&format!(
            "{:name_width$}",
            total_name,
            name_width = name_width
        ));
        ss.push_str(" | ");
        // impact column: show '-' centered
        let dash = "-".to_string();
        let padding = num_width.saturating_sub(dash.len());
        let left = padding / 2;
        let right = padding - left;
        ss.push_str(&" ".repeat(left));
        // magenta highlight for total values
        ss.push_str(&format!("\x1b[95m{}\x1b[0m", dash));
        ss.push_str(&" ".repeat(right));
        ss.push_str(" | ");
        // avg(-N) -> total_run_avg_micros
        {
            let micros = total_run_avg_micros as u64;
            if micros >= 1000 {
                let ms = micros / 1000;
                let us = micros % 1000;
                let plain_text = format!("{}'{:03}", ms, us);
                let colored_text = format!("\x1b[95m{}'{:03}\x1b[0m", ms, us);
                let padding = num_width.saturating_sub(plain_text.len());
                ss.push_str(&" ".repeat(padding));
                ss.push_str(&colored_text);
            } else {
                let plain_text = format!("{}", micros);
                let colored_text = format!("\x1b[95m{}\x1b[0m", micros);
                let padding = num_width.saturating_sub(plain_text.len());
                ss.push_str(&" ".repeat(padding));
                ss.push_str(&colored_text);
            }
        }
        ss.push_str(" | ");
        // avg(*) total
        {
            let micros = total_avg_micros as u64;
            if micros >= 1000 {
                let ms = micros / 1000;
                let us = micros % 1000;
                let plain_text = format!("{}'{:03}", ms, us);
                let colored_text = format!("\x1b[95m{}'{:03}\x1b[0m", ms, us);
                let padding = num_width.saturating_sub(plain_text.len());
                ss.push_str(&" ".repeat(padding));
                ss.push_str(&colored_text);
            } else {
                let plain_text = format!("{}", micros);
                let colored_text = format!("\x1b[95m{}\x1b[0m", micros);
                let padding = num_width.saturating_sub(plain_text.len());
                ss.push_str(&" ".repeat(padding));
                ss.push_str(&colored_text);
            }
        }
        ss.push_str(" | ");
        // std_dev total
        {
            let micros = total_stddev as u64;
            if micros >= 1000 {
                let ms = micros / 1000;
                let us = micros % 1000;
                let plain_text = format!("{}'{:03}", ms, us);
                let colored_text = format!("\x1b[95m{}'{:03}\x1b[0m", ms, us);
                let padding = num_width.saturating_sub(plain_text.len());
                ss.push_str(&" ".repeat(padding));
                ss.push_str(&colored_text);
            } else {
                let plain_text = format!("{}", micros);
                let colored_text = format!("\x1b[95m{}\x1b[0m", micros);
                let padding = num_width.saturating_sub(plain_text.len());
                ss.push_str(&" ".repeat(padding));
                ss.push_str(&colored_text);
            }
        }
        ss.push('\n');

        // Now render individual rows
        for ef in sorted_evals.iter() {
            let avg = ef.sum_time / ef.num_measurements as u32;
            let run_avg = Duration::from_micros(ef.filter.mean as u64);
            let std_dev = Duration::from_micros(ef.std_dev() as u64);
            let impact = ef.impact_score() as u64;
            // Swapped: run_avg comes before avg
            let values = [Duration::from_micros(impact), run_avg, avg, std_dev];

            // Calculate indentation based on hierarchy depth
            let depth = ef.name.matches('/').count();
            let indent = "  ".repeat(depth);
            let display_name = ef.name.split('/').last().unwrap_or(&ef.name);
            let formatted_name = format!("{}{}", indent, display_name);

            ss.push_str(&format!(
                "{:name_width$}",
                formatted_name,
                name_width = name_width
            ));
            ss.push_str(" | ");

            // Print all timing values with colors in numeric columns
            for (i, value) in values.iter().enumerate() {
                let micros = value.as_micros() as u64;

                if i == 0 {
                    // impact: show relative multiplier normalized to max_impact
                    let rel = (micros as f64) / max_impact;
                    let mut rel_text = format!("{:.2}", rel.max(0.0));
                    // omit leading zero for values < 1: 0.33 -> .33
                    if rel_text.starts_with('0') {
                        rel_text = rel_text.trim_start_matches('0').to_string();
                    }
                    rel_text.push('x');
                    ss.push_str(&format!("{:>num_width$}", rel_text, num_width = num_width));
                } else {
                    // timing columns: format with tick and colors
                    if micros >= 1000 {
                        let ms = micros / 1000;
                        let us = micros % 1000;
                        let plain_text = format!("{}'{:03}", ms, us);
                        let colored_text =
                            format!("\x1b[36m{}\x1b[0m'\x1b[92m{:03}\x1b[0m", ms, us);
                        // pad plain_text to num_width
                        let padding = num_width.saturating_sub(plain_text.len());
                        ss.push_str(&" ".repeat(padding));
                        ss.push_str(&colored_text);
                    } else {
                        let plain_text = format!("{}", micros);
                        let colored_text = format!("\x1b[92m{}\x1b[0m", micros);
                        let padding = num_width.saturating_sub(plain_text.len());
                        ss.push_str(&" ".repeat(padding));
                        ss.push_str(&colored_text);
                    }
                }

                if i + 1 == values.len() {
                    ss.push('\n');
                } else {
                    ss.push_str(" | ");
                }
            }
        }

        // Histogram of total time
        let total_index = self.evaluations.iter().position(|x| x.name == "total");
        if let Some(total_index) = total_index {
            let form = &self.evaluations[total_index];

            //                  columns with padding    +    separators     -       start of line
            // approximate line length: name column + numeric columns and separators
            let line_length = name_width + (field_count - 1) * (num_width + 3) - "10-20: ".len();

            for i in 0..self.histogram.len() {
                ss.push_str(&format!("{:2}-", i * MAX_HISTOGRAM_SIZE as usize));
                if i < self.histogram.len() - 1 {
                    ss.push_str(&format!("{:2}", (i + 1) * MAX_HISTOGRAM_SIZE as usize));
                } else {
                    ss.push_str("  ");
                }
                let count = (self.histogram[i] as f64 * line_length as f64
                    / form.num_measurements as f64)
                    .ceil() as usize;
                ss.push_str(&format!(": {}", "=".repeat(count)));
                ss.push('\n');
            }
        }

        self.resume();
        ss
    }
}
