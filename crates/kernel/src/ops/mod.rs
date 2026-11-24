//! The core operations, each an extension of the [builder](crate::query::Builder)
//! as an operation outside the kernel is, built on its `add`: scanning,
//! filtering, joining, computing, aggregating, sorting and limiting rows.
//!
//! A builder made by [`Builder::new`](crate::query::Builder::new) checks
//! each operation as it's added: one that reads a column the rows it reads
//! don't have fails with [`Error::NotInScope`].

mod aggregate;
mod compute;
mod filter;
mod join;
mod limit;
mod scan;
mod sort;

pub use aggregate::{Aggregate, Function, Grouping};
pub use compute::{Compute, Extension};
pub use filter::Filter;
pub use join::{Join, JoinKind, Joining};
pub use limit::Limit;
pub use scan::Scan;
pub use sort::{Order, Sort, Sorting};

use crate::column::DataType;
use crate::error::{Error, TypeError};

/// Checks a condition of type `data_type` is an integer, true where it
/// isn't 0, as a comparison is.
pub(crate) fn check_condition(data_type: DataType) -> Result<(), Error> {
    match data_type {
        DataType::Int64 => Ok(()),
        DataType::String => Err(Error::WrongType(TypeError::Condition)),
        DataType::Float64 => Err(Error::WrongType(TypeError::DoubleCondition)),
    }
}
