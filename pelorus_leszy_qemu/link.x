/* QEMU's virt machine starts at the beginning of its RAM. The heap reaches up to
   the configuration that QEMU loads at 0x9FF00000, followed by the stream. */
ENTRY(_start)
SECTIONS {
  . = 0x80000000;
  .text : { KEEP(*(.text.start)) *(.text .text.*) }
  .rodata : ALIGN(16) { *(.rodata .rodata.* .srodata .srodata.*) }
  .data : ALIGN(16) { *(.data .data.* .sdata .sdata.*) }
  .bss (NOLOAD) : ALIGN(16) { _bss_start = .; *(.bss .bss.* .sbss .sbss.* COMMON) _bss_end = .; }
  . = ALIGN(16);
  . += 0x800000;
  _stack_top = .;
  _heap_start = .;
}
