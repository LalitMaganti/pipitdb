//! `OrderByOp`: its child's rows sorted by keys, as `ORDER BY` does.

use crate::lower::{self, LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::plan::{Op, PlanNode};
use crate::slow_vec::SlowVec;
use crate::sort::{SortKey, sort};
use crate::step::Step;

/// Outputs its child's rows, with the same columns, sorted by `keys`, the
/// first first, stably: as they are, if they're known to be already. Every
/// column it outputs is read, so each is made flat before it's held.
pub struct OrderByOp {
    pub keys: SlowVec<lower::SortKey>,
}

impl<'c> Op<'c> for OrderByOp {
    fn lower(
        &self,
        node: &PlanNode<'c>,
        lowering: &mut Lowering<'_, 'c>,
    ) -> Result<(), LowerError> {
        check!(node.children.len() == 1);
        lowering.lower(*at!(node.children, 0))?;
        // Rows already in this order need no sort.
        if lowering.query_arranged(&[], &self.keys) {
            return Ok(());
        }
        let allocator = lowering.allocator();
        let keys = self.keys.iter().map(|key| SortKey {
            position: lowering.query_position(key.column),
            data_type: lowering.query_type(key.column),
            descending: key.descending,
        });
        let keys = SlowVec::fixed_from(allocator, keys)?;
        lowering.add_breaker(Step::Operator(sort(allocator, keys)?))?;
        lowering.declare_sorted(&self.keys);
        Ok(())
    }

    /// Makes no columns, and needs its keys.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        for key in self.keys.iter() {
            needed.need(key.column);
        }
        Pruned::Keep
    }
}
