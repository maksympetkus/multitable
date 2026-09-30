//! Detects a nightly compiler and emits `nightly_compiler` so the SIMD paths in `src/utils.rs`
//! turn on by themselves, and turns the `MT_*` build-time knobs below into cfgs.
use std::env;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(nightly_compiler)");
    println!("cargo:rerun-if-env-changed=RUSTC");

    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let is_nightly = Command::new(rustc)
        .arg("--version")
        .output()
        .is_ok_and(|out| String::from_utf8_lossy(&out.stdout).contains("nightly"));

    if is_nightly {
        println!("cargo:rustc-cfg=nightly_compiler");
    }

    // Picks which table `benches/comparison.rs`'s `Table` alias resolves to, read at build time so
    // a changed value only takes effect on a rebuild.
    println!("cargo:rustc-check-cfg=cfg(mt_bench_plain)");
    println!("cargo:rerun-if-env-changed=MT_BENCH_TABLE");
    match env::var("MT_BENCH_TABLE").as_deref() {
        Ok("plain") => println!("cargo:rustc-cfg=mt_bench_plain"),
        Ok("filtered") | Err(_) => {}
        Ok(other) => panic!("MT_BENCH_TABLE must be `plain` or `filtered`, got `{other}`"),
    }
}
