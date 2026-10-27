//! The monotonic clock behind every runtime measurement.
//!
//! With `std`, [`Instant`] is the system clock, and it also works in the browser,
//! where `std::time::Instant` panics. Without `std` there is no clock. Every
//! instant is then the same moment, so measured durations are zero. Pelorus reads
//! the clock only for diagnostics and for choosing between sequential and parallel
//! execution, so odometry results do not depend on it.

#[cfg(feature = "std")]
pub use web_time::Instant;

#[cfg(not(feature = "std"))]
pub use self::no_clock::Instant;

#[cfg(not(feature = "std"))]
mod no_clock {
    use core::ops::{Add, Sub};
    use core::time::Duration;

    /// A point in time on a clock that never advances.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct Instant;

    impl Instant {
        pub fn now() -> Self {
            Self
        }

        pub fn elapsed(&self) -> Duration {
            Duration::ZERO
        }

        pub fn duration_since(&self, _earlier: Self) -> Duration {
            Duration::ZERO
        }

        pub fn saturating_duration_since(&self, _earlier: Self) -> Duration {
            Duration::ZERO
        }

        pub fn checked_duration_since(&self, _earlier: Self) -> Option<Duration> {
            Some(Duration::ZERO)
        }

        pub fn checked_add(&self, _duration: Duration) -> Option<Self> {
            Some(Self)
        }

        pub fn checked_sub(&self, _duration: Duration) -> Option<Self> {
            Some(Self)
        }
    }

    impl Add<Duration> for Instant {
        type Output = Self;

        fn add(self, _duration: Duration) -> Self {
            self
        }
    }

    impl Sub<Duration> for Instant {
        type Output = Self;

        fn sub(self, _duration: Duration) -> Self {
            self
        }
    }

    impl Sub for Instant {
        type Output = Duration;

        fn sub(self, _earlier: Self) -> Duration {
            Duration::ZERO
        }
    }
}
