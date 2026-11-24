//! `iif`, `ifnull` and `coalesce`: functions that choose between their
//! arguments, so compile to `CASE`, as SQL's does. An embedder registers
//! them as the set `CHOICES`.

use pipit_kernel::containers::SmallVec;
use pipit_kernel::query::Expr;

use crate::ast::Node;
use crate::compile::Compiler;
use crate::error::Error;
use crate::expression::{case_of, compile_arguments};
use crate::registry::FunctionRule;

/// `iif(c, x, y)` is `CASE WHEN c THEN x ELSE y END`, and `coalesce(a, b,
/// c)` is `CASE WHEN a IS NOT NULL THEN a WHEN b IS NOT NULL THEN b ELSE c
/// END`, whose branches test that their values aren't null. `ifnull` is
/// `coalesce` of two.
pub const CHOICES: &[FunctionRule] = &[
    FunctionRule { name: "iif", compile: |c, node, depth| iif(c, (node, depth)) },
    FunctionRule { name: "ifnull", compile: |c, node, depth| coalesce(c, (node, depth), 2) },
    FunctionRule {
        name: "coalesce",
        compile: |c, node, depth| coalesce(c, (node, depth), u32::MAX),
    },
];

/// The `iif` call `node`.
fn iif(compiler: &mut Compiler<'_, '_>, (node, depth): (Node, usize)) -> Result<Expr, Error> {
    let values = compile_arguments(compiler, (node, depth), 3, 3)?;
    let branch = [(Some(*at!(values, 0)), *at!(values, 1))];
    case_of(compiler, node, &branch, *at!(values, 2))
}

/// The `coalesce` call `node`, of at most `most` arguments.
fn coalesce(
    compiler: &mut Compiler<'_, '_>,
    (node, depth): (Node, usize),
    most: u32,
) -> Result<Expr, Error> {
    let values = compile_arguments(compiler, (node, depth), 2, most)?;
    // At least two, as `compile_arguments` checked.
    let Some((otherwise, tested)) = values.split_last() else { pipit_kernel::failed!() };
    let mut branches = SmallVec::fixed(compiler.env(), tested.len())?;
    for &value in tested {
        branches.push((None, value))?;
    }
    case_of(compiler, node, &branches, *otherwise)
}
