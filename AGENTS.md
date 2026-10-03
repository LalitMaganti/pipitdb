# pipitdb

- Follow [TigerStyle](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md).
- The kernel crate is `no_std` + `alloc`. No new dependencies without asking.
- Keep code simple and easy to read.
- Only add a comment when the code isn't obvious. Keep it short and plain.
- One small PR at a time.
- In the kernel, use `check!` instead of `assert!`: it costs a few bytes in release builds.
- Prefer lookup tables to branches. Build them as a `static` with a `const fn`, so they are computed at compile time.
- Prefer a few smoke tests over testing every combination.
- Add benchmarks for hot code to `crates/bench`: wall-clock ones with criterion, and instruction counts with gungraun, which CI checks against `crates/bench/instructions`.
- Use new APIs from `crates/size`, which CI checks against a size budget per architecture.
- Commit titles are prefixed with the area, e.g. `kernel:`.

Before sending a PR:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release --target wasm32-unknown-unknown
```
