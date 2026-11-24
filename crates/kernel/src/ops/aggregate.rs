//! Grouping rows, and aggregating each group's.

pub use crate::aggregate::Function;
use crate::aggregate::{Aggregate as AggregateOf, AggregateOp};
use crate::column::DataType;
use crate::containers::SmallVec;
use crate::error::{Error, TypeError};
use crate::plan::{ColumnId, DynOp};
use crate::query::{Builder, Column, Relation};

/// Rows being grouped, by keys and into aggregates added one at a time.
pub struct Grouping {
    /// The keys, in turn.
    keys: SmallVec<ColumnId>,
    /// The aggregates, in turn.
    aggregates: SmallVec<AggregateOf>,
}

/// Groups and aggregates rows.
pub trait Aggregate {
    /// Groups `grouping` by `column` too.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if `column` isn't of integers or strings, and
    /// [`Error::OutOfMemory`] if there are more keys than the grouping was
    /// made for.
    fn group_by(&mut self, grouping: &mut Grouping, column: Column) -> Result<(), Error>;

    /// A grouping by up to `keys` keys, which [`Aggregate::group_by`] adds, into up
    /// to `aggregates` aggregates, which [`Aggregate::aggregate`] adds.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    fn group(&mut self, keys: usize, aggregates: usize) -> Result<Grouping, Error>;

    /// A column of `grouping`: `function` of each group's
    /// values of `input`, of each distinct one if `distinct`, or of its rows
    /// if there's no input.
    ///
    /// # Errors
    ///
    /// [`Error::WrongType`] if `function` doesn't read `input`'s type, and
    /// [`Error::OutOfMemory`] if `env` has too little left.
    fn aggregate(
        &mut self,
        grouping: &mut Grouping,
        function: Function,
        input: Option<Column>,
        distinct: bool,
    ) -> Result<Column, Error>;

    /// The rows of `relation`, grouped by `grouping`.
    ///
    /// # Errors
    ///
    /// [`Error::OutOfMemory`] if `env` has too little left.
    fn grouped(&mut self, relation: Relation, grouping: Grouping) -> Result<Relation, Error>;
}

impl Aggregate for Builder<'_> {
    fn group_by(&mut self, grouping: &mut Grouping, column: Column) -> Result<(), Error> {
        if !matches!(self.column_type(column), DataType::Int64 | DataType::String) {
            return Err(Error::WrongType(TypeError::GroupKey));
        }
        Ok(grouping.keys.push(column.0)?)
    }

    fn group(&mut self, keys: usize, aggregates: usize) -> Result<Grouping, Error> {
        let keys = SmallVec::fixed(self.env, keys)?;
        Ok(Grouping { keys, aggregates: SmallVec::fixed(self.env, aggregates)? })
    }

    fn aggregate(
        &mut self,
        grouping: &mut Grouping,
        function: Function,
        input: Option<Column>,
        distinct: bool,
    ) -> Result<Column, Error> {
        let input = input.map(|column| column.0);
        let input_type =
            input.map_or(DataType::Int64, |id| at!(self.plan.columns, id as usize).data_type);
        if input.is_some() && !function.reads(input_type) {
            return Err(Error::WrongType(TypeError::Aggregated));
        }
        // A count is 0 for a group of nulls, where the others are null.
        let nullable = !matches!(function, Function::CountStar | Function::Count);
        let output_type = function.output_type(input_type);
        let binding = self.add_column(output_type, nullable)?.0;
        // The least or greatest of the distinct values is that of them all.
        let distinct = distinct && !matches!(function, Function::Min | Function::Max);
        grouping.aggregates.push(AggregateOf { function, input, distinct, binding })?;
        Ok(Column(binding))
    }

    fn grouped(&mut self, relation: Relation, grouping: Grouping) -> Result<Relation, Error> {
        let Grouping { keys, aggregates } = grouping;
        let aggregate = AggregateOp { keys, aggregates, answer: None };
        self.add(DynOp::new(self.env, aggregate)?, [relation])
    }
}
