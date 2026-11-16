//! Runs the no_std odometry on QEMU's `virt` machine with a core like the
//! Leszy, RV32IMF without atomic and compressed instructions, built for the
//! target in `pelorus_leszy_qemu/riscv32imf-unknown-none-elf.json`.
//!
//! QEMU loads the configuration (postcard, in the precision of this build) at
//! 0x9FF0_0000 and an input stream of `pelorus-nostd-check convert` at
//! 0xA000_0000. The program writes the scan lines of `pelorus-nostd-check run`
//! to the UART and ends QEMU through its test device, with 0 when every record
//! was processed. A panic or a trap, such as an illegal instruction, ends it
//! with 3 or 2. `mask nostd-check` shows how to run it. On other targets the
//! binary only says so.

#![cfg_attr(target_arch = "riscv32", no_std, no_main)]

#[cfg(target_arch = "riscv32")]
mod firmware;

#[cfg(not(target_arch = "riscv32"))]
fn main() {
    eprintln!("pelorus-leszy-qemu runs on QEMU's riscv32 virt machine, see `mask nostd-check`");
    std::process::exit(1);
}
