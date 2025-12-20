fn main() {
    // On macOS, Python extension modules generally rely on the Python interpreter to
    // provide the CPython API symbols at load time. The linker therefore must allow
    // unresolved `_Py*` symbols when producing the shared library.
    //
    // PyO3 normally handles this via its `extension-module` feature, but in some
    // build setups (notably workspace builds and/or certain toolchains) the flag
    // isn't propagated, resulting in linker errors like `Undefined symbols: _Py...`.
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-arg=-undefined");
        println!("cargo:rustc-link-arg=dynamic_lookup");
    }
}
