# pipitdb

- Follow [TigerStyle](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md).
- The kernel and pipesql crates are `no_std` + `alloc`. No new dependencies without asking.
- Keep code simple and easy to read.
- Don't implement something poorly: leave it out until it can be done well.
- Only add a comment when the code isn't obvious. Keep it short and plain.
- Order parameters from longest-lived to shortest, e.g. `context, state, batch`.
- One small PR at a time.
- Avoid work: design for laziness first. Skip what statistics and dictionaries rule out, and don't read, decode or copy values until something needs them.
- One hard memory budget per query: every allocation counts against it. Over budget, operators spill; with nothing left to spill, the query fails with an error. Parallelism is sized from the budget, never the other way round.
- Design for spilling to remote storage from the start: state that grows with the data is partitioned and appended in blocks to a spill log, which can be local disk or object storage (staged locally, uploaded in large parts, sealed before it's read).
- Steps get every buffer they fill for a batch's column from `Context::column_buffer`, never from the allocator directly, so how column memory is found (e.g. pooling) can change in one place. Its contents are unspecified: write every byte that's read, and don't rely on zeroing.
- Threads are the executor's job, a bonus on a single thread that's already competitive: it runs copies of the work between pipeline breakers, so operators aren't written for threads. Breakers, such as sorts and aggregations, build small local results in each copy and then merge them.
- In the kernel and pipesql, use `check!` instead of `assert!`, and `at!`/`at_mut!` instead of indexing with `[]`: they cost a few bytes in release builds.
- Prefer lookup tables to branches. Build them as a `static` with a `const fn`, so they are computed at compile time.
- Prefer a few smoke tests over testing every combination.
- Add benchmarks for hot code to `crates/bench`: wall-clock ones with criterion, and instruction counts with gungraun, which CI checks against `crates/bench/instructions`.
- Use new APIs from `crates/bench/examples/size.rs`, which CI checks against a size budget per architecture.
- Commit titles are prefixed with the area, e.g. `kernel:`.

Before sending a PR:

```sh
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
cargo build --release --target wasm32-unknown-unknown --workspace --exclude pipitdb-s3
```
