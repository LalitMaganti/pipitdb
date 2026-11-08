//! `OrderByOp`: its child's rows sorted by keys, as `ORDER BY` does.

use crate::lower::{self, LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::plan::{Op, PlanNode};
use crate::slow_vec::SlowVec;
use crate::sort::{SortKey, sort};
use crate::step::Step;
use crate::top_k::top_k;

/// Outputs its child's rows, with the same columns, sorted stably by `keys`,
/// the first first: as they are, if they're known to be in that order
/// already, and only as many as are read if a LIMIT reads only the first. It
/// reads every column, so each is made flat before it's held.
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
        let first_read = lowering.query_first_read();
        lowering.lower(*at!(node.children, 0))?;
        if lowering.query_arranged(&[], &self.keys) {
            return Ok(());
        }
        let allocator = lowering.allocator();
        let keys = self.keys.iter().map(|key| SortKey {
            column: lowering.query_position(key.column),
            descending: key.descending,
        });
        let keys = SlowVec::fixed_from(allocator, keys)?;
        // Only the first rows are read, as with LIMIT: keep only those.
        let step = match first_read {
            Some(count) => {
                lowering.declare_only_first_read();
                top_k(allocator, keys, count)?
            }
            None => sort(allocator, keys)?,
        };
        lowering.add_breaker(Step::Operator(step))?;
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
