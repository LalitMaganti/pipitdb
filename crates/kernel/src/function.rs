//! `Function`: a function expressions call, which any crate can add. A
//! frontend binds one to a statement, with what it needs computed once, such
//! as a compiled pattern, and fixes the types it takes and gives.

#![allow(unsafe_code)]

use core::marker::PhantomData;
use core::ptr::NonNull;

use crate::allocator::AllocError;
use crate::column::ColumnView;
use crate::containers::{Box, ErasedBox};
use crate::context::Context;
use crate::env::Env;
use crate::erase::value_of;
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

/// A `Function` of any type, owned in memory from an `Env`, and its type's
/// functions.
pub struct DynFunction {
    function: ErasedBox,
    functions: &'static FunctionFunctions,
}

/// A function type's functions: one table for each type.
struct FunctionFunctions {
    run: Run,
}

type Run = unsafe fn(NonNull<()>, &mut Context, &[&ColumnView]) -> Result<ColumnView, Error>;

/// Holds `T`'s table.
struct Functions<T>(PhantomData<T>);

impl<T: Function + 'static> Functions<T> {
    const TABLE: FunctionFunctions = FunctionFunctions {
        // SAFETY: only called with this function.
        run: |function, context, arguments| unsafe {
            value_of::<T>(function).run(context, arguments)
        },
    };
}

impl DynFunction {
    pub fn new<T: Function + 'static>(env: &Env, function: T) -> Result<DynFunction, AllocError> {
        let function = Box::new(env, function)?.erase();
        Ok(DynFunction { function, functions: &Functions::<T>::TABLE })
    }

    pub fn run(
        &self,
        context: &mut Context,
        arguments: &[&ColumnView],
    ) -> Result<ColumnView, Error> {
        // SAFETY: the function matches the function's type.
        unsafe { (self.functions.run)(self.function.as_ptr(), context, arguments) }
    }
}
