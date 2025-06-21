use std::{collections::VecDeque, time::Duration};

use std::ops::{AddAssign, Div, Sub};

#[derive(Debug, Clone)]
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
        return self.mean;
    }
}

#[derive(Debug, Clone)]
pub struct EvaluationFormular {
    name: String,
    active: bool,
    accumulated_time: Duration,
    num_measurements: u64,
    last_measurement: Duration,
    min_time: Duration,
    max_time: Duration,
    sum_time: Duration,
    filter: SlidingWindowFilter<f64>,
}

impl EvaluationFormular {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            active: true,
            accumulated_time: Duration::from_secs(0),
            num_measurements: 0,
            last_measurement: Duration::from_secs(0),
            min_time: Duration::from_secs(0),
            max_time: Duration::from_secs(0),
            sum_time: Duration::from_secs(0),
            filter: SlidingWindowFilter::new(10),
        }
    }
}

const MAX_HISTOGRAM_SIZE: u64 = 10;

#[derive(Debug, Clone)]
pub struct RuntimeEvaluator {
    evaluations: Vec<EvaluationFormular>,
    histogram: Vec<u64>,
    start_time: std::time::Instant,
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
                        self_e.min_time = e.min_time;
                        self_e.max_time = e.max_time;
                        self_e.sum_time = e.sum_time;
                        self_e.filter = e.filter.clone();
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
                e.filter.update(acc_time.as_millis() as f64);
                if acc_time < e.min_time {
                    e.min_time = acc_time;
                }

                if acc_time > e.max_time && e.num_measurements != 1 {
                    e.max_time = acc_time;
                }

                if name == "total" {
                    let time_ms = acc_time.as_millis() as f64;

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

    pub fn render(&mut self) -> String {
        self.pause();

        const FIELD_COUNT: usize = 7;
        let table = vec!["task", "count", "last", "min", "max", "avg", "run_avg"];

        let mut width = 0;
        for ef in self.evaluations.iter() {
            width = std::cmp::max(width, ef.name.len());
        }
        for i in 0..FIELD_COUNT {
            width = std::cmp::max(width, table[i].len());
        }

        let mut ss = String::new();
        ss.push_str("\n");
        for i in 0..FIELD_COUNT {
            ss.push_str(&format!("{:width$}", table[i], width = width));
            if i == FIELD_COUNT - 1 {
                ss.push_str("\n");
            } else {
                ss.push_str(" | ");
            }
        }

        for i in 0..FIELD_COUNT {
            ss.push_str(&format!("{:width$}", "", width = width));
            if i == FIELD_COUNT - 1 {
                ss.push_str("-\n");
            } else {
                ss.push_str("-+-");
            }
        }

        for ef in self.evaluations.iter() {
            if ef.active {
                continue;
            }

            let avg = ef.sum_time / ef.num_measurements as u32;
            let run_avg = Duration::from_millis(ef.filter.mean as u64);
            let values = [ef.last_measurement, ef.min_time, ef.max_time, avg, run_avg];

            ss.push_str(&format!("{:width$}", ef.name, width = width));
            ss.push_str(" | ");
            ss.push_str(&format!("{:width$}", ef.num_measurements, width = width));
            ss.push_str(" | ");
            for i in 0..FIELD_COUNT - 2 {
                ss.push_str(&format!(
                    "{:width$}",
                    values[i].as_millis() as u64,
                    width = width
                ));
                if i == FIELD_COUNT - 3 {
                    ss.push_str("\n");
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
            let line_length = FIELD_COUNT * (width + 2) + (FIELD_COUNT - 1) - "10-20: ".len();

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
                ss.push_str("\n");
            }
        }

        self.resume();
        ss
    }
}
