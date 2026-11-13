// The size example aborts on panic, so nothing reads its unwind tables. Fat
// LTO merges in core's debug info and unwind tables, which give every
// function one even with `force-unwind-tables=no`, so the linker drops them.
fn main() {
    println!("cargo::rerun-if-changed=no-unwind.ld");
    let size = std::env::var_os("CARGO_FEATURE_SIZE").is_some();
    if size && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux") {
        let script = concat!(env!("CARGO_MANIFEST_DIR"), "/no-unwind.ld");
        println!("cargo::rustc-link-arg=-Wl,--no-eh-frame-hdr,-T,{script}");
    }
}
