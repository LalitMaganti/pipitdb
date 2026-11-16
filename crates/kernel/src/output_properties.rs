//! `OutputProperties`: what is known of the batches a plan node outputs, as
//! Perfetto's, and planning: before a plan is lowered, each node says what it
//! needs of its inputs' rows, the steps that meet those needs are chosen, and
//! the node derives what it outputs from that.

use crate::allocator::{AllocError, Allocator};
use crate::column::DataType;
use crate::lower::LowerError;
use crate::plan::{ColumnId, LogicalPlan, PLAN_COLUMNS_MAX, PlanNodeId, SortKey};
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
    /// Whether it can hold a null.
    pub nullable: bool,
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
pub struct OutputProperties {
    /// The columns of a batch, in order: made with the first.
    columns: Option<SlowVec<Column>>,
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
        columns: None,
        arrangements: Arrangements::NONE,
        open_sort: None,
        scan: None,
    };

    /// A copy, its columns allocated from `allocator`.
    pub fn clone_in(&self, allocator: &dyn Allocator) -> Result<OutputProperties, AllocError> {
        let columns = match &self.columns {
            Some(columns) => {
                let mut copy = SlowVec::new(allocator, PLAN_COLUMNS_MAX)?;
                copy.extend_from_slice(columns)?;
                Some(copy)
            }
            None => None,
        };
        Ok(OutputProperties { columns, ..*self })
    }

    /// Adds `column` at the next position in batches, allocating from
    /// `allocator`. Batches hold at most as many columns as a plan has.
    pub fn push_column(
        &mut self,
        allocator: &dyn Allocator,
        column: Column,
    ) -> Result<(), LowerError> {
        let columns = match &mut self.columns {
            Some(columns) => columns,
            None => self.columns.insert(SlowVec::new(allocator, PLAN_COLUMNS_MAX)?),
        };
        if columns.len() == PLAN_COLUMNS_MAX {
            return Err(LowerError::TooManyColumns);
        }
        Ok(columns.push(column)?)
    }

    fn columns(&self) -> &[Column] {
        self.columns.as_deref().unwrap_or_default()
    }

    /// How many columns its batches have.
    pub fn column_count(&self) -> u32 {
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_COLUMNS_MAX`")]
        let count = self.columns().len() as u32;
        count
    }

    /// The position of `column` in a batch, if it has one.
    pub fn position_of(&self, column: ColumnId) -> Option<u32> {
        #[expect(clippy::cast_possible_truncation, reason = "at most `PLAN_COLUMNS_MAX`")]
        self.columns().iter().position(|c| c.id == column).map(|position| position as u32)
    }

    /// The type of `column`, which the batches must have.
    pub fn type_of(&self, column: ColumnId) -> DataType {
        let Some(position) = self.position_of(column) else { failed!() };
        at!(self.columns(), position as usize).data_type
    }

    /// Whether `column` can hold a null.
    pub fn nullable_of(&self, column: ColumnId) -> bool {
        let Some(position) = self.position_of(column) else { failed!() };
        at!(self.columns(), position as usize).nullable
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

/// What a node needs of the rows of one of its inputs. All of it is
/// optional, and it is met in the order of these fields.
#[derive(Clone, Copy)]
pub struct InputNeeds<'a> {
    pub arranged: Option<Arranged<'a>>,
    /// Columns of the input that must hold a type, each once.
    pub types_then: &'a [Type],
}

impl InputNeeds<'_> {
    /// Nothing.
    pub const NONE: InputNeeds<'static> = InputNeeds { arranged: None, types_then: &[] };
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
    allocator: &'p dyn Allocator,
}

impl<'p> DeriveContext<'p, '_> {
    /// The id of the node being derived.
    pub fn node(&self) -> PlanNodeId {
        self.self_
    }

    /// What outputs' columns are allocated from.
    pub fn allocator(&self) -> &'p dyn Allocator {
        self.allocator
    }

    /// What is known of input `i` once its needs are met.
    pub fn input(&self, i: u32) -> &OutputProperties {
        &at!(at!(self.planned, self.self_ as usize).inputs, i as usize).rows
    }

    /// A copy of `input(i)`, for an output that starts as its input.
    pub fn input_copy(&self, i: u32) -> Result<OutputProperties, AllocError> {
        self.input(i).clone_in(self.allocator)
    }

    /// `id` as a column of the output: as it is in an input which has it,
    /// else as the plan says.
    pub fn column_of(&self, id: ColumnId) -> Column {
        for input in at!(self.planned, self.self_ as usize).inputs.iter() {
            if input.rows.position_of(id).is_some() {
                let (data_type, nullable) = (input.rows.type_of(id), input.rows.nullable_of(id));
                return Column { id, data_type, nullable };
            }
        }
        let column = at!(self.plan.columns, id as usize);
        Column { id, data_type: column.data_type, nullable: column.nullable }
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
        self.types(needs.types_then)
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
        let rows = at!(planned, child as usize).output.clone_in(allocator)?;
        inputs.push(PlannedInput { steps: SlowVec::new(allocator, INPUT_STEPS_MAX)?, rows })?;
    }
    for (i, out) in (0..).zip(inputs.iter_mut()) {
        let mut input = InputPlanner { allocator, node: id, input: i, out };
        input.meet(planned, &node.op.needs(i))?;
    }
    planned.push(PlannedNode { inputs, output: OutputProperties::NONE, cutoff: None })?;
    let mut derive = DeriveContext { plan, planned, self_: id, allocator };
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
}
