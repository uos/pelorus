use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::Mutex;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;
use std::sync::atomic::{AtomicU64, Ordering};

static DEBOUNCER: Lazy<Mutex<HashMap<u64, Instant>>> = Lazy::new(|| Mutex::new(HashMap::new()));
static DEFAULT_DEBOUNCE_NANOS: AtomicU64 = AtomicU64::new(1_000_000_000); // 1 second

fn hash_key(key: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write(key.as_bytes());
    hasher.finish()
}

pub fn set_default_debounce_secs(secs: f64) {
    let ns = (secs * 1e9) as u64;
    DEFAULT_DEBOUNCE_NANOS.store(ns, Ordering::Relaxed);
}

pub fn get_default_debounce_secs() -> f64 {
    let ns = DEFAULT_DEBOUNCE_NANOS.load(Ordering::Relaxed);
    (ns as f64) / 1e9
}

pub fn should_log_once(key: &str, debounce_secs: f64) -> bool {
    let h = hash_key(key);
    let mut map = DEBOUNCER.lock().unwrap();
    let now = Instant::now();
    match map.get(&h) {
        Some(&last) => {
            if now.duration_since(last) > Duration::from_secs_f64(debounce_secs) {
                map.insert(h, now);
                true
            } else {
                false
            }
        }
        None => {
            map.insert(h, now);
            true
        }
    }
}

pub fn should_log_once_default(key: &str) -> bool {
    let secs = get_default_debounce_secs();
    should_log_once(key, secs)
}