//! `OrderByOp`: its child's rows sorted by keys, as `ORDER BY` does.

use crate::containers::SmallVec;
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::output_properties::{Arranged, DeriveContext, InputNeeds, OutputProperties};
use crate::plan::{ColumnId, Op, PlanNode, SortKey};

/// Outputs its child's rows, with the same columns, sorted stably by `keys`,
/// the first first. It needs its input sorted, which planning does only if
/// it isn't known to be, and only for as many rows as a LIMIT just above
/// reads. It reads every column, so each is made flat before it's held.
pub struct OrderByOp {
    pub keys: SmallVec<SortKey>,
}

impl<'c> Op<'c> for OrderByOp {
    /// A condition can be applied before the sort: sorting doesn't change
    /// which rows pass.
    fn moves_condition(&self, _: ColumnId) -> Option<u32> {
        Some(0)
    }

    /// Makes no columns, and needs its keys.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        for key in self.keys.iter() {
            needed.need(key.column);
        }
        Pruned::Keep
    }

    fn needs(&self, _: u32) -> InputNeeds<'_> {
        let arranged = Some(Arranged { grouped_by: &[], sorted_by: &self.keys });
        InputNeeds { arranged, ..InputNeeds::NONE }
    }

    fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
        Ok(c.input_copy(0)?)
    }

    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 1);
        lowering.lower_input(node, 0)
    }
}
