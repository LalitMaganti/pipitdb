//! `OrderByOp`: its child's rows sorted by keys, as `ORDER BY` does.

use crate::column::Forms;
use crate::lower::{LowerError, Lowering};
use crate::optimize::{Needed, Pruned};
use crate::output_properties::{Arranged, DeriveContext, InputNeeds, OutputProperties};
use crate::plan::{ColumnId, Op, PlanNode, SortKey};
use crate::scannable::DynScannable;
use crate::slow_vec::SlowVec;

/// Outputs its child's rows, with the same columns, sorted stably by `keys`,
/// the first first. It needs its input sorted, which planning does only if
/// it isn't known to be, and only for as many rows as a LIMIT just above
/// reads. Its sort holds every column, and takes those it doesn't sort by
/// lazy too, loading them as it must: a `TopK` only for the rows it keeps.
pub struct OrderByOp<'c> {
    keys: SlowVec<SortKey>,
    /// What made the lazy columns its sort holds, if any.
    lazy: Option<&'c DynScannable<'c>>,
}

impl OrderByOp<'_> {
    pub fn new(keys: SlowVec<SortKey>) -> Self {
        OrderByOp { keys, lazy: None }
    }
}

impl<'c> Op<'c> for OrderByOp<'c> {
    /// Makes no columns, and needs its keys.
    fn prune(&mut self, needed: &mut Needed) -> Pruned {
        for key in self.keys.iter() {
            needed.need(key.column);
        }
        Pruned::Keep
    }

    /// Its keys, as they come. Its sort outputs its other columns flat,
    /// whatever form they come in.
    fn passes_through(&self, column: ColumnId) -> bool {
        self.keys.iter().any(|key| key.column == column)
    }

    /// Its keys flat, to sort by, and its other columns flat or lazy.
    fn accepts(&self, column: ColumnId) -> Forms {
        if self.keys.iter().any(|key| key.column == column) {
            Forms::FLAT
        } else {
            Forms::FLAT | Forms::LAZY
        }
    }

    fn holds_lazy(&mut self, scannable: &'c DynScannable<'c>) {
        self.lazy = Some(scannable);
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
        lowering.lower_input_holding(node, 0, self.lazy)
    }
}
