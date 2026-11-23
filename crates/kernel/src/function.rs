//! `Function`: a function expressions call, which any crate can add. A
//! frontend binds one to a statement, with what it needs computed once, such
//! as a compiled pattern, and fixes the types it takes and gives.

use crate::allocator::AllocError;
use crate::column::ColumnView;
use crate::containers::DynBox;
use crate::context::Context;
use crate::env::Env;
use crate::error::Error;

/// The most arguments a call has.
pub const ARGUMENTS_MAX: usize = 8;

pub trait Function {
    /// Computes it at every row of a batch from its arguments' columns, of
    /// the types it was bound with: null where any argument is, and only
    /// there, as most of SQLite's functions are. It runs for rows the batch
    /// doesn't keep, so it fails only for memory.
    fn run(&self, context: &mut Context, arguments: &[&ColumnView]) -> Result<ColumnView, Error>;
}

/// A `Function` of any type, owned in memory from an `Env`.
pub struct DynFunction {
    function: DynBox<dyn Function>,
}

impl DynFunction {
    pub fn new<T: Function + 'static>(env: &Env, function: T) -> Result<DynFunction, AllocError> {
        let function = DynBox::<dyn Function>::new(env, function, |function| function)?;
        Ok(DynFunction { function })
    }

    pub fn run(
        &self,
        context: &mut Context,
        arguments: &[&ColumnView],
    ) -> Result<ColumnView, Error> {
        self.function.run(context, arguments)
    }
}
