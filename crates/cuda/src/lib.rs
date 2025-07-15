pub mod ffi;

pub fn t() -> i32 {
    let num_elements = 1024;

    unsafe { ffi::allocate_memory_on_gpu(num_elements as i32) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(t(), 0);
    }
}
