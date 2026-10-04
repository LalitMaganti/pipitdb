//! Inputs and helpers shared by the benchmarks in `benches/`.

use pipit_kernel::allocator::Heap;
use pipit_kernel::lexer::{Lexer, TokenKind};
use pipit_kernel::parser::{parse_expression, parse_query};
use pipit_kernel::registry::Registry;

static REGISTRY: Registry = Registry::new(&[pipit_std::RELATIONAL]);

/// Queries shaped like real ones, repeated `count` times.
pub fn typical_queries(count: usize) -> String {
    "FROM slice |> WHERE dur > 1000 AND name != 'binder transaction' \
     |> EXTEND ts + dur AS end_ts |> AGGREGATE count(*) AS n GROUP BY track_id\n"
        .repeat(count)
}

/// Queries with long comments and string literals, repeated `count` times.
pub fn long_text(count: usize) -> String {
    "-- a long comment explaining what the query below does, in some detail\n\
     FROM t |> WHERE description = 'a long string literal, as found in real data'\n"
        .repeat(count)
}

/// A long `WHERE`-style expression mixing every operator.
pub fn expression(terms: usize) -> String {
    let term = "(dur * 2 + ts - 1000) / 3 >= 10.5 AND NOT name = 'binder transaction'";
    vec![term; terms].join(" OR ")
}

pub fn count_nodes(source: &[u8]) -> u32 {
    parse_expression(Heap, source).map_or(0, |ast| ast.node_count())
}

/// Three-stage queries, one per line.
pub fn queries(count: usize) -> String {
    "FROM slice |> WHERE dur > 1000 AND name != 'binder transaction' \
     |> SELECT ts, dur / 1000, name, count(*)\n"
        .repeat(count)
}

/// Parses each line as a query, as an embedder would.
pub fn count_query_nodes(source: &str) -> u32 {
    source
        .lines()
        .map(|query| {
            parse_query(Heap, &REGISTRY, query.as_bytes()).map_or(0, |ast| ast.node_count())
        })
        .sum()
}

pub fn count_tokens(source: &[u8]) -> u32 {
    let Ok(mut lexer) = Lexer::new(source) else { return 0 };
    let mut count = 0;
    while let Ok(token) = lexer.next_token() {
        if token.kind == TokenKind::End {
            break;
        }
        count += 1;
    }
    count
}
