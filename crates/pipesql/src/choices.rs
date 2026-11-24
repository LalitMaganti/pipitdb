//! `iif`, `ifnull` and `coalesce`: functions that choose between their
//! arguments, so compile to `CASE`, as SQL's does. An embedder registers
//! them as the set `CHOICES`.

use pipit_kernel::containers::SmallVec;
use pipit_kernel::query::Value;

use crate::ast::Node;
use crate::compile::Compiler;
use crate::error::Error;
use crate::expression::compile_arguments;
use crate::registry::FunctionRule;

pub const CHOICES: &[FunctionRule] = &[
    FunctionRule { name: "iif", compile: |c, node, depth| iif(c, (node, depth)) },
    FunctionRule { name: "ifnull", compile: |c, node, depth| coalesce(c, (node, depth), 2) },
    FunctionRule {
        name: "coalesce",
        compile: |c, node, depth| coalesce(c, (node, depth), u32::MAX),
    },
];

/// `iif(c, a, b)`: `a` where `c` is true, else `b`.
fn iif(compiler: &mut Compiler<'_, '_>, (node, depth): (Node, usize)) -> Result<Value, Error> {
    let values = compile_arguments(compiler, (node, depth), 3, 3)?;
    let branch = [(Some(*at!(values, 0)), *at!(values, 1))];
    Ok(compiler.builder.case(&branch, *at!(values, 2))?)
}

/// `coalesce(a, b, …)`: the first argument that isn't null.
fn coalesce(
    compiler: &mut Compiler<'_, '_>,
    (node, depth): (Node, usize),
    most: u32,
) -> Result<Value, Error> {
    let values = compile_arguments(compiler, (node, depth), 2, most)?;
    // At least two, as `compile_arguments` checked.
    let Some((otherwise, tested)) = values.split_last() else { pipit_kernel::failed!() };
    let mut branches = SmallVec::fixed(compiler.env(), tested.len())?;
    for &value in tested {
        branches.push((None, value))?;
    }
    Ok(compiler.builder.case(&branches, *otherwise)?)
}
