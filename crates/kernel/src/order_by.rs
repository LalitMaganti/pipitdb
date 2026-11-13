//! `OrderByOp`: its child's rows sorted by keys, as `ORDER BY` does.

use crate::allocator::{AllocError, Allocator};
use crate::lower::{self, LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::output_properties::{Arranged, DeriveContext, InputNeeds, OutputProperties};
use crate::plan::{Op, PlanNode};
use crate::slow_vec::SlowVec;

/// Outputs its child's rows, with the same columns, sorted stably by `keys`,
/// the first first. It needs its input sorted, which planning does only if
/// it isn't known to be, and only for as many rows as a LIMIT just above
/// reads. It reads every column, so each is made flat before it's held.
pub struct OrderByOp {
    pub keys: SlowVec<lower::SortKey>,
}

impl<'c> Op<'c> for OrderByOp {
    fn needs(&self, allocator: &dyn Allocator, _: u32) -> Result<InputNeeds<'_>, AllocError> {
        let mut needs = InputNeeds::none(allocator)?;
        needs.arranged = Some(Arranged { grouped_by: &[], sorted_by: &self.keys });
        Ok(needs)
    }

    fn derive(&self, c: &mut DeriveContext<'_, 'c>) -> Result<OutputProperties, LowerError> {
        Ok(*c.input(0))
    }

    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 1);
        lowering.lower_input(node, 0)
    }

    /// Makes no columns, and needs its keys.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        for key in self.keys.iter() {
            needed.need(key.column);
        }
        Pruned::Keep
    }
}
