//! The stages a parser knows.
//!
//! Each stage is described by a `Rule`: its keyword and the items after it.
//! Rules come in sets, and a `Registry` composes sets. Built as a `static`, a
//! registry's mistakes, such as two rules with one keyword, are build errors.

use pipit_kernel::expression::ExprBuilder;

use crate::ast::Node;
use crate::compile::Compiler;
use crate::error::Error;
use crate::expression::{CallFunction, call_function, no_function};
use crate::keywords::{KeywordTable, same_word};
use pipit_kernel::settings::build_setting;

/// How many keywords a registry has room for. Set at build time with
/// `PIPIT_STAGE_SLOTS`.
pub const STAGE_SLOTS: usize = build_setting(option_env!("PIPIT_STAGE_SLOTS"), 64);

/// How many items a rule can have.
pub const ITEMS_MAX: usize = 8;

/// Where a stage can go: first in a query, or after `|>`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Point {
    /// First in a query, as `FROM`.
    Source,
    /// After `|>`.
    Stage,
}

/// What comes after a stage's keyword. Each item is one child of the stage's
/// node, in order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Item {
    /// Exactly one.
    One(Shared),
    /// One or more, separated by commas, under a `List` node.
    List(Shared),
    /// If the query has these words next, a list of the rest after them, as
    /// `List` does; if not, an empty `List`. Words are matched ignoring case.
    Clause(&'static [&'static str], Shared),
    /// These words, then a list, as `List` does: `Clause`, but required.
    After(&'static [&'static str], Shared),
    /// This word, if the query has it next, as `OUTER` in `LEFT OUTER JOIN`:
    /// an empty `List` either way.
    Optional(&'static str),
    /// If the query has these words next, a list, as `List` does, in
    /// parentheses, as `USING (a, b)`; if not, an empty `List`.
    Parenthesized(&'static [&'static str], Shared),
}

/// The pieces of grammar stages are built from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shared {
    /// A name, as a `Name`.
    Name,
    /// An expression.
    Expr,
    /// An expression, then maybe `AS` and its name: a `Named` if there is
    /// one.
    Aliased,
    /// A name, `AS`, and a new name for it, as a `Named`.
    Renamed,
    /// A name, `=`, and an expression for it, as a `Named`.
    Assigned,
    /// An expression, then `ASC` or `DESC`, as a `SortKey`.
    Ordered,
}

#[derive(Clone, Copy, Debug)]
pub struct Rule {
    /// The word a stage starts with, matched ignoring case.
    pub keyword: &'static str,
    /// Where the stage can go.
    pub point: Point,
    /// What comes after the keyword.
    pub items: &'static [Item],
    /// Adds the stage, whose node is passed, to the plan being compiled.
    pub compile: fn(&mut Compiler<'_, '_>, Node) -> Result<(), Error>,
}

/// A function queries can call, which a crate adds by registering it.
#[derive(Clone, Copy)]
pub struct FunctionRule {
    /// What queries call it, matched ignoring case.
    pub name: &'static str,
    /// Appends the expression nodes computing the `Call` node passed,
    /// `depth` deep, its result last: steps the kernel has, as `iif`'s
    /// `CASE`, or a call of a function bound to the statement, which
    /// `call_bound` appends.
    pub compile: fn(&mut Compiler<'_, '_>, Node, usize, &mut ExprBuilder) -> Result<(), Error>,
}

/// Rules, found by keyword, and functions, found by name. A rule's id is
/// `set << 8 | index in set`.
pub struct Registry {
    /// The rules, in sets.
    sets: &'static [&'static [Rule]],
    /// Each rule's id, by its keyword.
    keywords: KeywordTable<STAGE_SLOTS>,
    /// The functions, in sets.
    functions: &'static [&'static [FunctionRule]],
    /// Compiles a call of a function, which only a registry with functions
    /// can: so one without doesn't pay for the code.
    pub(crate) call: CallFunction,
}

impl Registry {
    pub const fn new(sets: &'static [&'static [Rule]]) -> Registry {
        // Ids fit in 16 bits, and the keyword table reserves 0xffff.
        assert!(sets.len() < 256, "too many rule sets");
        let mut keywords = KeywordTable::new();
        let mut set = 0;
        while set < sets.len() {
            assert!(sets[set].len() <= 256, "too many rules in a set");
            let mut index = 0;
            while index < sets[set].len() {
                let rule = &sets[set][index];
                assert!(rule.items.len() <= ITEMS_MAX, "too many items in a rule");
                #[expect(clippy::cast_possible_truncation, reason = "checked above")]
                let id = (set << 8 | index) as u16;
                keywords = keywords.insert(rule.keyword.as_bytes(), id);
                index += 1;
            }
            set += 1;
        }
        Registry { sets, keywords, functions: &[], call: no_function }
    }

    /// It, with the functions of `functions` too.
    pub const fn with_functions(self, functions: &'static [&'static [FunctionRule]]) -> Registry {
        Registry { functions, call: call_function, ..self }
    }

    /// The function called `name`, ignoring case.
    pub fn function(&self, name: &[u8]) -> Option<&'static FunctionRule> {
        self.functions
            .iter()
            .flat_map(|set| set.iter())
            .find(|f| same_word(f.name.as_bytes(), name))
    }

    /// The id of the rule whose keyword is `word`, ignoring case.
    pub fn find(&self, word: &[u8]) -> Option<u16> {
        self.keywords.get(word, |id| same_word(self.rule(id).keyword.as_bytes(), word))
    }

    pub fn rule(&self, id: u16) -> &'static Rule {
        let id = usize::from(id);
        at!(at!(self.sets, id >> 8), id & 0xff)
    }
}
