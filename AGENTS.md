# pipitdb

- Follow [TigerStyle](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md).
- The kernel crate is `no_std` + `alloc`. No new dependencies without asking.
- One small PR at a time; small, focused tests.
- Commit titles are prefixed with the area, e.g. `kernel:`.

Before sending a PR:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release --target wasm32-unknown-unknown
```
