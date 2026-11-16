//! `Statement`: what a statement's plan holds for its runs to read, which
//! the physical plan takes from the logical one.

use crate::allocator::{AllocError, Allocator};
use crate::string_pool::{Id, StringPool};

#[derive(Default)]
pub struct Statement {
    /// The strings it compares with: made with the first, as most statements
    /// have none.
    literals: Option<StringPool>,
}

impl Statement {
    /// Adds `literal` to its literals, allocating from `allocator`, which
    /// must outlive it, and returns its id there.
    pub fn add_literal(
        &mut self,
        allocator: &dyn Allocator,
        literal: &[u8],
    ) -> Result<Id, AllocError> {
        let literals = match &mut self.literals {
            Some(literals) => literals,
            None => self.literals.insert(StringPool::new(allocator)?),
        };
        literals.intern_string(allocator, Some(literal))
    }

    /// The strings it compares with, if it has any.
    pub fn literals(&self) -> Option<&StringPool> {
        self.literals.as_ref()
    }
}
