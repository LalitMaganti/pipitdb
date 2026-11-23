//! `Statement`: what a statement's plan holds for its runs to read, which
//! the physical plan takes from the logical one.

use crate::allocator::AllocError;
use crate::containers::SmallVec;
use crate::env::Env;
use crate::function::DynFunction;
use crate::string_pool::{Id, StringPool};

/// The most functions a statement binds.
pub const FUNCTIONS_MAX: usize = 1 << 8;

#[derive(Default)]
pub struct Statement {
    /// The strings it compares with: made with the first, as most statements
    /// have none.
    literals: Option<StringPool>,
    /// The functions its expressions call, by id: made with the first.
    functions: Option<SmallVec<DynFunction>>,
}

impl Statement {
    /// Adds `literal` to its literals, allocating from `env`, which
    /// must outlive it, and returns its id there.
    pub fn add_literal(&mut self, env: &Env, literal: &[u8]) -> Result<Id, AllocError> {
        let literals = match &mut self.literals {
            Some(literals) => literals,
            None => self.literals.insert(StringPool::new(env)?),
        };
        literals.intern_string(env, Some(literal))
    }

    /// The strings it compares with, if it has any.
    pub fn literals(&self) -> Option<&StringPool> {
        self.literals.as_ref()
    }

    /// Adds `function`, allocating from `env`, which must outlive it,
    /// and returns its id, or `None` if it has `FUNCTIONS_MAX`.
    pub fn add_function(
        &mut self,
        env: &Env,
        function: DynFunction,
    ) -> Result<Option<u32>, AllocError> {
        let functions = match &mut self.functions {
            Some(functions) => functions,
            None => self.functions.insert(SmallVec::new(env, FUNCTIONS_MAX)?),
        };
        if functions.len() == FUNCTIONS_MAX {
            return Ok(None);
        }
        functions.push(function)?;
        #[expect(clippy::cast_possible_truncation, reason = "at most `FUNCTIONS_MAX`")]
        Ok(Some(functions.len() as u32 - 1))
    }

    /// The functions its expressions call, by id.
    pub fn functions(&self) -> &[DynFunction] {
        self.functions.as_deref().unwrap_or_default()
    }
}
