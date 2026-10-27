//! Locks that work with and without `std`.
//!
//! With `std` these wrap the operating system's locks. Without `std` they are spin
//! locks, which is enough because Pelorus then runs on a single thread and a lock
//! is never contended. A poisoned lock panics, like `lock().unwrap()` on a std lock.

#[cfg(feature = "std")]
mod imp {
    pub type MutexGuard<'a, T> = std::sync::MutexGuard<'a, T>;
    pub type RwLockReadGuard<'a, T> = std::sync::RwLockReadGuard<'a, T>;
    pub type RwLockWriteGuard<'a, T> = std::sync::RwLockWriteGuard<'a, T>;

    #[derive(Debug, Default)]
    pub struct Mutex<T: ?Sized>(std::sync::Mutex<T>);

    impl<T> Mutex<T> {
        pub const fn new(value: T) -> Self {
            Self(std::sync::Mutex::new(value))
        }

        pub fn into_inner(self) -> T {
            self.0.into_inner().unwrap()
        }
    }

    impl<T: ?Sized> Mutex<T> {
        pub fn lock(&self) -> MutexGuard<'_, T> {
            self.0.lock().unwrap()
        }

        /// Lock even after a thread panicked while holding the lock.
        pub fn lock_ignoring_poison(&self) -> MutexGuard<'_, T> {
            self.0
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
        }

        pub fn get_mut(&mut self) -> &mut T {
            self.0.get_mut().unwrap()
        }
    }

    #[derive(Debug, Default)]
    pub struct RwLock<T: ?Sized>(std::sync::RwLock<T>);

    impl<T> RwLock<T> {
        pub const fn new(value: T) -> Self {
            Self(std::sync::RwLock::new(value))
        }
    }

    impl<T: ?Sized> RwLock<T> {
        pub fn read(&self) -> RwLockReadGuard<'_, T> {
            self.0.read().unwrap()
        }

        pub fn write(&self) -> RwLockWriteGuard<'_, T> {
            self.0.write().unwrap()
        }

        pub fn get_mut(&mut self) -> &mut T {
            self.0.get_mut().unwrap()
        }
    }

    #[derive(Debug, Default)]
    pub struct OnceLock<T>(std::sync::OnceLock<T>);

    impl<T> OnceLock<T> {
        pub const fn new() -> Self {
            Self(std::sync::OnceLock::new())
        }

        pub fn get(&self) -> Option<&T> {
            self.0.get()
        }

        pub fn get_or_init(&self, init: impl FnOnce() -> T) -> &T {
            self.0.get_or_init(init)
        }

        pub fn take(&mut self) -> Option<T> {
            self.0.take()
        }
    }
}

#[cfg(not(feature = "std"))]
mod imp {
    pub type MutexGuard<'a, T> = spin::MutexGuard<'a, T>;
    pub type RwLockReadGuard<'a, T> = spin::RwLockReadGuard<'a, T>;
    pub type RwLockWriteGuard<'a, T> = spin::RwLockWriteGuard<'a, T>;

    #[derive(Debug, Default)]
    pub struct Mutex<T: ?Sized>(spin::Mutex<T>);

    impl<T> Mutex<T> {
        pub const fn new(value: T) -> Self {
            Self(spin::Mutex::new(value))
        }

        pub fn into_inner(self) -> T {
            self.0.into_inner()
        }
    }

    impl<T: ?Sized> Mutex<T> {
        pub fn lock(&self) -> MutexGuard<'_, T> {
            self.0.lock()
        }

        /// Spin locks cannot be poisoned, so this is [`Self::lock`].
        pub fn lock_ignoring_poison(&self) -> MutexGuard<'_, T> {
            self.0.lock()
        }

        pub fn get_mut(&mut self) -> &mut T {
            self.0.get_mut()
        }
    }

    #[derive(Debug, Default)]
    pub struct RwLock<T: ?Sized>(spin::RwLock<T>);

    impl<T> RwLock<T> {
        pub const fn new(value: T) -> Self {
            Self(spin::RwLock::new(value))
        }
    }

    impl<T: ?Sized> RwLock<T> {
        pub fn read(&self) -> RwLockReadGuard<'_, T> {
            self.0.read()
        }

        pub fn write(&self) -> RwLockWriteGuard<'_, T> {
            self.0.write()
        }

        pub fn get_mut(&mut self) -> &mut T {
            self.0.get_mut()
        }
    }

    #[derive(Debug, Default)]
    pub struct OnceLock<T>(spin::Once<T>);

    impl<T> OnceLock<T> {
        pub const fn new() -> Self {
            Self(spin::Once::new())
        }

        pub fn get(&self) -> Option<&T> {
            self.0.get()
        }

        pub fn get_or_init(&self, init: impl FnOnce() -> T) -> &T {
            self.0.call_once(init)
        }

        pub fn take(&mut self) -> Option<T> {
            core::mem::take(&mut self.0).try_into_inner()
        }
    }
}

pub use imp::*;
