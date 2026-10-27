//! Rate limiting for repeated log messages.
//!
//! Without `std` the clock never advances (see [`crate::clock`]), so each message
//! is logged once.

use crate::Instant;
use crate::sync::Mutex;
use alloc::vec::Vec;
use core::time::Duration;

struct Debouncer {
    default_secs: f64,
    /// When each message was last logged. Few distinct messages exist, so a list is enough.
    logged: Vec<(&'static str, Instant)>,
}

static DEBOUNCER: Mutex<Debouncer> = Mutex::new(Debouncer {
    default_secs: 1.0,
    logged: Vec::new(),
});

pub fn set_default_debounce_secs(secs: f64) {
    DEBOUNCER.lock().default_secs = secs;
}

pub fn get_default_debounce_secs() -> f64 {
    DEBOUNCER.lock().default_secs
}

pub fn should_log_once(key: &'static str, debounce_secs: f64) -> bool {
    let mut debouncer = DEBOUNCER.lock();
    let now = Instant::now();
    match debouncer
        .logged
        .iter_mut()
        .find(|(logged_key, _)| *logged_key == key)
    {
        Some((_, last)) => {
            if now.duration_since(*last) > Duration::from_secs_f64(debounce_secs) {
                *last = now;
                true
            } else {
                false
            }
        }
        None => {
            debouncer.logged.push((key, now));
            true
        }
    }
}

pub fn should_log_once_default(key: &'static str) -> bool {
    should_log_once(key, get_default_debounce_secs())
}
