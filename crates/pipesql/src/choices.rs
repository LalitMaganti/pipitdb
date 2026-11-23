//! `iif`, `ifnull` and `coalesce`: functions that choose between their
//! arguments, so compile to `CASE`, as SQL's does. An embedder registers
//! them as the set `CHOICES`.

use pipit_kernel::expression::{ExprBuilder, PRESENT};
use pipit_kernel::slow_vec::SlowVec;

use crate::ast::Node;
use crate::compile::Compiler;
use crate::error::Error;
use crate::expression::{compile_arguments, push_case};
use crate::registry::FunctionRule;

/// `iif(c, x, y)` is `CASE WHEN c THEN x ELSE y END`, and `coalesce(a, b,
/// c)` is `CASE WHEN a IS NOT NULL THEN a WHEN b IS NOT NULL THEN b ELSE c
/// END`, which `PRESENT` branches test. `ifnull` is `coalesce` of two.
pub const CHOICES: &[FunctionRule] = &[
    FunctionRule { name: "iif", compile: |c, node, depth, nodes| iif(c, (node, depth), nodes) },
    FunctionRule {
        name: "ifnull",
        compile: |c, node, depth, nodes| coalesce(c, (node, depth), nodes, 2),
    },
    FunctionRule {
        name: "coalesce",
        compile: |c, node, depth, nodes| coalesce(c, (node, depth), nodes, u32::MAX),
    },
];

/// Appends the nodes of the `iif` call `node`.
fn iif(
    compiler: &mut Compiler<'_, '_>,
    (node, depth): (Node, usize),
    nodes: &mut ExprBuilder,
) -> Result<(), Error> {
    let values = compile_arguments(compiler, (node, depth), nodes, 3, 3)?;
    let branch = [(*at!(values, 0), *at!(values, 1))];
    push_case(compiler, node, nodes, &branch, *at!(values, 2))
}

/// Appends the nodes of the `coalesce` call `node`, of at most `most`
/// arguments.
fn coalesce(
    compiler: &mut Compiler<'_, '_>,
    (node, depth): (Node, usize),
    nodes: &mut ExprBuilder,
    most: u32,
) -> Result<(), Error> {
    let values = compile_arguments(compiler, (node, depth), nodes, 2, most)?;
    // At least two, as `compile_arguments` checked.
    let Some((otherwise, tested)) = values.split_last() else { pipit_kernel::failed!() };
    let mut branches = SlowVec::fixed(compiler.env(), tested.len())?;
    for &value in tested {
        branches.push((PRESENT, value))?;
    }
    push_case(compiler, node, nodes, &branches, *otherwise)
}
