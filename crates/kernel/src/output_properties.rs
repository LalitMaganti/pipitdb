//! `OutputProperties`: what is known of the batches a plan node outputs, as
//! Perfetto's, and planning: before a plan is lowered, each node says what it
//! needs of its inputs' rows, the steps that meet those needs are chosen, and
//! the node derives what it outputs from that.

use crate::allocator::{AllocError, Allocator};
use crate::column::DataType;
use crate::lower::{LowerError, SortKey};
use crate::plan::{ColumnId, LogicalPlan, PlanNodeId};
use crate::row_batch::BATCH_COLUMNS_MAX;
use crate::slow_vec::SlowVec;

/// A layout known to hold for rows: grouped by the columns of its first
/// `grouped` keys, none meaning one group, and each group sorted by the
/// `sorted` keys after them. Rows sorted by (a, b) are also grouped by a, and
/// sorted by b within each group.
#[derive(Clone, Copy)]
struct Arrangement {
    grouped: u8,
    sorted: u8,
    keys: [SortKey; ARRANGEMENT_KEYS_MAX],
}

impl Arrangement {
    const NONE: Arrangement = Arrangement {
        grouped: 0,
        sorted: 0,
        keys: [SortKey { column: 0, descending: false }; ARRANGEMENT_KEYS_MAX],
    };

    fn grouped_by(&self) -> impl Iterator<Item = ColumnId> + '_ {
        at!(self.keys, ..usize::from(self.grouped)).iter().map(|key| key.column)
    }

    fn sorted_by(&self) -> &[SortKey] {
        let start = usize::from(self.grouped);
        at!(self.keys, start..start + usize::from(self.sorted))
    }
}

/// The most keys an arrangement has: one with more isn't kept, as knowing
/// less of the order is always safe.
pub(crate) const ARRANGEMENT_KEYS_MAX: usize = 8;

/// The most arrangements rows are known to have at once.
const ARRANGEMENTS_MAX: usize = 1 << 4;

/// Every layout known to hold, the first `len` of these.
#[derive(Clone, Copy)]
pub struct Arrangements {
    list: [Arrangement; ARRANGEMENTS_MAX],
    len: usize,
}

impl Arrangements {
    pub const NONE: Arrangements =
        Arrangements { list: [Arrangement::NONE; ARRANGEMENTS_MAX], len: 0 };

    fn known(&self) -> &[Arrangement] {
        at!(self.list, ..self.len)
    }

    /// Adds that rows are grouped by `keys`, and each group sorted by
    /// `sorted_by`. What there isn't room for is forgotten, which is safe:
    /// sort keys past the room left are dropped, as rows sorted by keys are
    /// sorted by their start too.
    pub fn push(&mut self, keys: &[ColumnId], sorted_by: &[SortKey]) {
        let Some(room) = ARRANGEMENT_KEYS_MAX.checked_sub(keys.len()) else {
            return;
        };
        let sorted_by = at!(sorted_by, ..sorted_by.len().min(room));
        let (Ok(grouped), Ok(sorted)) = (u8::try_from(keys.len()), u8::try_from(sorted_by.len()))
        else {
            return;
        };
        let mut arrangement = Arrangement { grouped, sorted, ..Arrangement::NONE };
        let grouped_by = keys.iter().map(|&column| SortKey { column, descending: false });
        for (slot, key) in
            arrangement.keys.iter_mut().zip(grouped_by.chain(sorted_by.iter().copied()))
        {
            *slot = key;
        }
        if let Some(slot) = self.list.get_mut(self.len) {
            *slot = arrangement;
            self.len += 1;
        }
    }
}

/// A column of a batch, and what it holds here.
#[derive(Clone, Copy)]
pub struct Column {
    pub id: ColumnId,
    pub data_type: DataType,
}

/// The most recent sort, while nothing has read its order: it can still be
/// undone. What was known before it is kept in its step.
#[derive(Clone, Copy)]
pub struct OpenSort {
    pub node: PlanNodeId,
    pub input: u32,
    pub step: usize,
}

/// Whether `a` holds the same columns as `b`, as often each.
fn is_permutation(a: &[ColumnId], b: &[ColumnId]) -> bool {
    let count = |list: &[ColumnId], column| list.iter().filter(|&&c| c == column).count();
    a.len() == b.len() && b.iter().all(|&column| count(a, column) == count(b, column))
}

/// What is known about the batches a node outputs: their columns and types,
/// and how the rows are arranged. Planning compares it with what each node
/// needs.
#[derive(Clone, Copy)]
pub struct OutputProperties {
    /// The columns of a batch, in order: the first `column_count` of these.
    columns: [Column; BATCH_COLUMNS_MAX as usize],
    /// How many of `columns` there are.
    column_count: u32,
    /// Every layout known to hold.
    pub arrangements: Arrangements,
    /// Set while a sort can still be undone.
    pub open_sort: Option<OpenSort>,
    /// The scan the rows are of, if each is one of its rows, kept or dropped
    /// by its own values alone, as a filter does. An op that keeps rows by
    /// where they are, as LIMIT does, clears it.
    pub scan: Option<PlanNodeId>,
}

impl OutputProperties {
    /// No columns, and nothing known.
    pub const NONE: OutputProperties = OutputProperties {
        columns: [Column { id: 0, data_type: DataType::Int64 }; BATCH_COLUMNS_MAX as usize],
        column_count: 0,
        arrangements: Arrangements::NONE,
        open_sort: None,
        scan: None,
    };

    /// Adds `column` at the next position in batches, which fail to hold
    /// more than `BATCH_COLUMNS_MAX`.
    pub fn push_column(&mut self, column: Column) -> Result<(), LowerError> {
        let Some(slot) = self.columns.get_mut(self.column_count as usize) else {
            return Err(LowerError::TooManyColumns);
        };
        *slot = column;
        self.column_count += 1;
        Ok(())
    }

    fn columns(&self) -> &[Column] {
        at!(self.columns, ..self.column_count as usize)
    }

    /// The position of `column` in a batch, if it has one.
    pub fn position_of(&self, column: ColumnId) -> Option<u32> {
        #[expect(clippy::cast_possible_truncation, reason = "at most `BATCH_COLUMNS_MAX`")]
        self.columns().iter().position(|c| c.id == column).map(|position| position as u32)
    }

    /// The type of `column`, which the batches must have.
    pub fn type_of(&self, column: ColumnId) -> DataType {
        let Some(position) = self.position_of(column) else { failed!() };
        at!(self.columns, position as usize).data_type
    }

    /// Whether the rows are grouped by `keys`, and each group sorted by
    /// `sorted_by`. Sorting by some keys first also groups by them, and a
    /// sort key on a grouped column is the same within a group, so means
    /// nothing.
    pub fn is_arranged(&self, keys: &[ColumnId], sorted_by: &[SortKey]) -> bool {
        let wanted = || sorted_by.iter().filter(|key| !keys.contains(&key.column));
        if keys.is_empty() && wanted().next().is_none() {
            return true;
        }
        'arrangements: for arrangement in self.arrangements.known() {
            if !arrangement.grouped_by().all(|column| keys.contains(&column)) {
                continue;
            }
            let grouped = |column| arrangement.grouped_by().any(|by| by == column);
            let mut sorted = arrangement.sorted_by().iter().filter(|key| !grouped(key.column));
            // Sort keys on the rest of `keys` group by them too.
            let mut missing = [0; ARRANGEMENT_KEYS_MAX];
            let mut len = 0;
            for column in arrangement.grouped_by() {
                (*at_mut!(missing, len), len) = (column, len + 1);
            }
            let mut next = sorted.next();
            while let Some(key) = next.filter(|key| keys.contains(&key.column)) {
                (*at_mut!(missing, len), len) = (key.column, len + 1);
                next = sorted.next();
            }
            if !is_permutation(at!(missing, ..len), keys) {
                continue;
            }
            for want in wanted() {
                if next != Some(want) {
                    continue 'arrangements;
                }
                next = sorted.next();
            }
            return true;
        }
        false
    }

    /// The longest start of a known sort of all rows that uses only
    /// `allowed`.
    pub fn sorted_by(&self, allowed: &[ColumnId]) -> &[SortKey] {
        let mut longest: &[SortKey] = &[];
        for arrangement in self.arrangements.known().iter().filter(|a| a.grouped == 0) {
            let keys = arrangement.sorted_by();
            let len = keys.iter().take_while(|key| allowed.contains(&key.column)).count();
            if len > longest.len() {
                longest = at!(keys, ..len);
            }
        }
        longest
    }

    /// Some rows were dropped by a step reading their order. Rows kept stay
    /// in order.
    pub fn drop_rows(&mut self) {
        self.open_sort = None;
    }
}

/// A column to hold a type.
#[derive(Clone, Copy)]
pub struct Type {
    pub column: ColumnId,
    pub data_type: DataType,
}

/// Grouped by `grouped_by` and sorted by `sorted_by` within each group.
#[derive(Clone, Copy)]
pub struct Arranged<'a> {
    pub grouped_by: &'a [ColumnId],
    pub sorted_by: &'a [SortKey],
}

/// Columns to hold types: the first `len` of these. Each is a column of the
/// input, needed as one type, so there are at most as many as a batch has
/// columns.
#[derive(Clone, Copy)]
pub struct Types {
    list: [Type; BATCH_COLUMNS_MAX as usize],
    len: usize,
}

impl Types {
    pub const NONE: Types = Types {
        list: [Type { column: 0, data_type: DataType::Int64 }; BATCH_COLUMNS_MAX as usize],
        len: 0,
    };

    /// Adds that `column` must hold `data_type`, unless that's known.
    pub fn push(&mut self, column: ColumnId, data_type: DataType) {
        let need = Type { column, data_type };
        if self.known().iter().any(|known| known.column == column && known.data_type == data_type) {
            return;
        }
        *at_mut!(self.list, self.len) = need;
        self.len += 1;
    }

    fn known(&self) -> &[Type] {
        at!(self.list, ..self.len)
    }
}

/// What a node needs of the rows of one of its inputs. All of it is
/// optional, and it is met in the order of these fields.
#[derive(Clone, Copy)]
pub struct InputNeeds<'a> {
    pub arranged: Option<Arranged<'a>>,
    pub types_then: Types,
}

impl InputNeeds<'_> {
    /// Nothing.
    pub const NONE: InputNeeds<'static> = InputNeeds { arranged: None, types_then: Types::NONE };
}

/// What it takes for an input to meet a node's needs, each step run in order
/// before the node reads the input.
pub enum InputStep {
    /// Sorts the rows by `keys`. Only the first `keep` rows in order are
    /// read, if given; if `publish`, a cutoff on the first key is published
    /// for a scan below to skip by. `unsorted` and `scan` are what was known
    /// of the rows before it, to undo it.
    Sort {
        keys: SlowVec<SortKey>,
        keep: Option<u32>,
        publish: bool,
        unsorted: Arrangements,
        scan: Option<PlanNodeId>,
    },
}

/// One input of a node, once planned.
pub struct PlannedInput {
    /// What runs on the input, in order, before the node reads it.
    pub steps: SlowVec<InputStep>,
    /// What is known of the rows after `steps`.
    pub rows: OutputProperties,
}

/// What is decided about a node before it is lowered.
pub struct PlannedNode {
    /// Each input's steps, and what is known of its rows after them.
    pub inputs: SlowVec<PlannedInput>,
    /// What is known of the rows it outputs.
    pub output: OutputProperties,
    /// For a scan: the column, and whether descending, a `TopK` above sorts
    /// by first and publishes a cutoff on.
    pub cutoff: Option<SortKey>,
}

/// By node, what's decided about it.
pub type Planned = SlowVec<PlannedNode>;

/// The most steps planned for one input: only a sort, so far.
const INPUT_STEPS_MAX: usize = 1;

/// What an op derives its output from.
pub struct DeriveContext<'p, 'c> {
    plan: &'p LogicalPlan<'c>,
    planned: &'p mut Planned,
    self_: PlanNodeId,
}

impl DeriveContext<'_, '_> {
    /// The id of the node being derived.
    pub fn node(&self) -> PlanNodeId {
        self.self_
    }

    /// What is known of input `i` once its needs are met.
    pub fn input(&self, i: u32) -> &OutputProperties {
        &at!(at!(self.planned, self.self_ as usize).inputs, i as usize).rows
    }

    /// `id` as a column of the output: as it is in an input which has it,
    /// else as the plan says.
    pub fn column_of(&self, id: ColumnId) -> Column {
        for input in at!(self.planned, self.self_ as usize).inputs.iter() {
            if input.rows.position_of(id).is_some() {
                return Column { id, data_type: input.rows.type_of(id) };
            }
        }
        Column { id, data_type: at!(self.plan.columns, id as usize).data_type }
    }

    /// Makes the open sort `sort` output only its first `rows` rows, so it
    /// is lowered as a `TopK`. If the rows it sorts are of a scan, and its
    /// first key is an `Int64`, the `TopK` publishes a cutoff on it for the
    /// scan to skip by.
    pub fn limit_sort(&mut self, sort: OpenSort, rows: u32) {
        let node = at_mut!(self.planned, sort.node as usize);
        let InputStep::Sort { keys, keep, publish, scan, .. } =
            at_mut!(at_mut!(node.inputs, sort.input as usize).steps, sort.step);
        *keep = Some(rows);
        let first = *at!(keys, 0);
        let Some(scan) = *scan else { return };
        if at!(self.plan.columns, first.column as usize).data_type != DataType::Int64 {
            return;
        }
        *publish = true;
        at_mut!(self.planned, scan as usize).cutoff = Some(first);
    }
}

/// Works out how one input of a node meets the node's needs.
struct InputPlanner<'p> {
    allocator: &'p dyn Allocator,
    node: PlanNodeId,
    input: u32,
    out: &'p mut PlannedInput,
}

impl InputPlanner<'_> {
    fn meet(&mut self, planned: &mut Planned, needs: &InputNeeds) -> Result<(), LowerError> {
        if let Some(arranged) = needs.arranged {
            self.arrange(planned, arranged.grouped_by, arranged.sorted_by)?;
        }
        self.types(needs.types_then.known())
    }

    fn types(&self, types: &[Type]) -> Result<(), LowerError> {
        for need in types {
            self.require_type(need.column, need.data_type)?;
        }
        Ok(())
    }

    fn require_type(&self, column: ColumnId, data_type: DataType) -> Result<(), LowerError> {
        if self.out.rows.type_of(column) != data_type {
            return Err(LowerError::WrongType(column));
        }
        Ok(())
    }

    fn arrange(
        &mut self,
        planned: &mut Planned,
        keys: &[ColumnId],
        sorted_by: &[SortKey],
    ) -> Result<(), AllocError> {
        if self.out.rows.is_arranged(keys, sorted_by)
            || (self.drop_open_sort(planned) && self.out.rows.is_arranged(keys, sorted_by))
        {
            return Ok(());
        }
        // Grouping isn't ported: no op needs its input grouped yet.
        check!(keys.is_empty());
        self.sort(sorted_by)
    }

    fn drop_open_sort(&mut self, planned: &mut Planned) -> bool {
        let rows = &mut self.out.rows;
        let Some(sort) = rows.open_sort.take() else { return false };
        // What was planned between the sort and here still says it's open, as
        // in Perfetto: only a LIMIT reads that, and it drops the open sort.
        let node = at_mut!(planned, sort.node as usize);
        let steps = &mut at_mut!(node.inputs, sort.input as usize).steps;
        // Only sorts are planned so far, and nothing after a sort before its
        // order is read, so it's the last step.
        check!(sort.step + 1 == steps.len());
        let Some(InputStep::Sort { unsorted, scan, .. }) = steps.pop() else { failed!() };
        (rows.arrangements, rows.scan) = (unsorted, scan);
        true
    }

    fn sort(&mut self, keys: &[SortKey]) -> Result<(), AllocError> {
        let rows = &mut self.out.rows;
        let sort = OpenSort { node: self.node, input: self.input, step: self.out.steps.len() };
        let keys_copy = SlowVec::fixed_from(self.allocator, keys.iter().copied())?;
        self.out.steps.push(InputStep::Sort {
            keys: keys_copy,
            keep: None,
            publish: false,
            unsorted: rows.arrangements,
            scan: rows.scan.take(),
        })?;
        rows.arrangements = Arrangements::NONE;
        rows.arrangements.push(&[], keys);
        rows.open_sort = Some(sort);
        Ok(())
    }
}

/// Plans each node of `plan`, by id, as Perfetto's `RunBoundPasses` does.
/// A node's inputs come before it, so each is planned after them.
pub fn plan_nodes(
    allocator: &dyn Allocator,
    plan: &LogicalPlan<'_>,
) -> Result<Planned, LowerError> {
    let mut planned = SlowVec::fixed(allocator, plan.nodes.len())?;
    for id in 0..plan.nodes.len() {
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_NODES_MAX`")]
        plan_node_inputs(allocator, plan, &mut planned, id as PlanNodeId)?;
    }
    Ok(planned)
}

/// Plans how each input of node `id` meets its needs, and derives its
/// output. Everything below `id` must already be planned.
fn plan_node_inputs(
    allocator: &dyn Allocator,
    plan: &LogicalPlan<'_>,
    planned: &mut Planned,
    id: PlanNodeId,
) -> Result<(), LowerError> {
    let node = at!(plan.nodes, id as usize);
    let mut inputs = SlowVec::fixed(allocator, node.children.len())?;
    for &child in node.children.iter() {
        check!(child < id);
        let rows = at!(planned, child as usize).output;
        inputs.push(PlannedInput { steps: SlowVec::new(allocator, INPUT_STEPS_MAX)?, rows })?;
    }
    for (i, out) in (0..).zip(inputs.iter_mut()) {
        let mut input = InputPlanner { allocator, node: id, input: i, out };
        input.meet(planned, &node.op.needs(i))?;
    }
    planned.push(PlannedNode { inputs, output: OutputProperties::NONE, cutoff: None })?;
    let mut derive = DeriveContext { plan, planned, self_: id };
    let output = node.op.derive(&mut derive)?;
    at_mut!(planned, id as usize).output = output;
    Ok(())
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::*;

    #[test]
    fn keeps_the_start_of_sorts_too_long_to_hold() {
        let mut rows = OutputProperties::NONE;
        let key = |column| SortKey { column, descending: false };
        let sorted_by: Vec<SortKey> = (0..10).map(key).collect();
        rows.arrangements.push(&[10], &sorted_by);
        assert!(rows.is_arranged(&[10], &sorted_by[..7]));
        assert!(!rows.is_arranged(&[10], &sorted_by[..8]));
        // Grouping by too many keys says nothing of grouping by fewer.
        let keys: Vec<ColumnId> = (0..9).collect();
        rows.arrangements.push(&keys, &[]);
        assert!(!rows.is_arranged(&keys[..8], &[]));
    }

    #[test]
    fn groups_only_by_the_keys_asked_for() {
        let mut rows = OutputProperties::NONE;
        let key = |column| SortKey { column, descending: false };
        // Two keys on one column group by it once, not by two columns.
        rows.arrangements.push(&[], &[key(0), key(0)]);
        assert!(!rows.is_arranged(&[0, 1], &[]));
    }

    #[test]
    fn needs_each_type_of_a_column_once() {
        let mut types = Types::NONE;
        for _ in 0..2 * BATCH_COLUMNS_MAX {
            types.push(0, DataType::Int64);
        }
        assert_eq!(types.known().len(), 1);
    }
}
