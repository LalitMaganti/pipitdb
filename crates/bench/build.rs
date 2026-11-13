// The size example aborts on panic, so nothing reads its unwind tables. Fat
// LTO merges in core's debug info and unwind tables, which give every
// function one even with `force-unwind-tables=no`, so the linker drops them.
fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    let size = std::env::var_os("CARGO_FEATURE_SIZE").is_some();
    let linux = std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("linux");
    let Some(out) = std::env::var_os("OUT_DIR") else { return };
    if size && linux {
        // In the build's own directory, which its cached output can point to.
        let script = std::path::Path::new(&out).join("no-unwind.ld");
        let discard = "SECTIONS { /DISCARD/ : { *(.eh_frame) } } INSERT AFTER .text;\n";
        if let Err(error) = std::fs::write(&script, discard) {
            panic!("can't write {}: {error}", script.display());
        }
        println!("cargo::rustc-link-arg=-Wl,--no-eh-frame-hdr,-T,{}", script.display());
    }
}
