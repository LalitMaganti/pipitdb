//! Benchmarks. Run with `cargo run --release -p pipitdb-bench [filter]`.

use std::hint::black_box;
use std::time::Instant;

use pipit_kernel::lexer::{Lexer, TokenKind};

fn main() {
    let filter = std::env::args().nth(1).unwrap_or_default();
    let typical = "FROM slice |> WHERE dur > 1000 AND name != 'binder transaction' \
        |> EXTEND ts + dur AS end_ts |> AGGREGATE count(*) AS n GROUP BY track_id\n"
        .repeat(100_000);
    let long = "-- a long comment explaining what the query below does, in some detail\n\
        FROM t |> WHERE description = 'a long string literal, as found in real data'\n"
        .repeat(100_000);

    let benchmarks: [(&str, &str); 2] = [("lexer/typical", &typical), ("lexer/long_text", &long)];
    for (name, source) in benchmarks {
        if name.contains(&filter) {
            report(name, source.len(), || count_tokens(source.as_bytes()));
        }
    }
}

/// Prints the best of ten runs of `run`, which processes `bytes` bytes.
fn report(name: &str, bytes: usize, mut run: impl FnMut() -> u32) {
    let mut best = f64::MAX;
    for _ in 0..10 {
        let start = Instant::now();
        black_box(run());
        best = best.min(start.elapsed().as_secs_f64());
    }
    let bytes = f64::from(u32::try_from(bytes).unwrap_or(u32::MAX));
    println!("{name:<20} {:>8.0} MB/s", bytes / best / 1e6);
}

fn count_tokens(source: &[u8]) -> u32 {
    let mut lexer = Lexer::new(black_box(source));
    let mut count = 0;
    while let Ok(token) = lexer.next_token() {
        if token.kind == TokenKind::End {
            break;
        }
        count += 1;
    }
    count
}
