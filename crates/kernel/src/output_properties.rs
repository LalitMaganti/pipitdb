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
const ARRANGEMENT_KEYS_MAX: usize = 8;

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

/// The most recent sort, while nothing has read its order, and what was
/// known before it: it can still be undone.
#[derive(Clone, Copy)]
pub struct OpenSort {
    /// Where its step is.
    pub at: SortAt,
    arrangements: Arrangements,
}

/// A sort step: the node and input it's planned for, and which of that
/// input's steps it is.
#[derive(Clone, Copy)]
pub struct SortAt {
    node: PlanNodeId,
    input: u32,
    step: usize,
}

/// What is known about the batches a node outputs: their columns, and how
/// the rows are arranged. Planning reads this to decide which steps an input
/// needs.
pub struct OutputProperties {
    /// The columns of a batch, in order.
    columns: SlowVec<ColumnId>,
    /// Every layout known to hold.
    pub arrangements: Arrangements,
    /// Set while a sort can still be undone.
    pub open_sort: Option<OpenSort>,
}

impl OutputProperties {
    /// No columns, and nothing known.
    pub fn new(allocator: &dyn Allocator) -> Result<OutputProperties, AllocError> {
        Ok(OutputProperties {
            columns: SlowVec::new(allocator, BATCH_COLUMNS_MAX as usize)?,
            arrangements: Arrangements::NONE,
            open_sort: None,
        })
    }

    pub fn copy(&self, allocator: &dyn Allocator) -> Result<OutputProperties, AllocError> {
        let mut columns = SlowVec::new(allocator, BATCH_COLUMNS_MAX as usize)?;
        columns.extend_from_slice(&self.columns)?;
        Ok(OutputProperties { columns, ..*self })
    }

    /// Adds `column` at the next position in batches, which fail to hold
    /// more than `BATCH_COLUMNS_MAX`.
    pub fn push_column(&mut self, column: ColumnId) -> Result<(), LowerError> {
        if self.columns.len() == BATCH_COLUMNS_MAX as usize {
            return Err(LowerError::TooManyColumns);
        }
        Ok(self.columns.push(column)?)
    }

    /// The position of `column` in a batch, if it has one.
    pub fn position_of(&self, column: ColumnId) -> Option<u32> {
        #[expect(clippy::cast_possible_truncation, reason = "at most `BATCH_COLUMNS_MAX`")]
        self.columns.iter().position(|&id| id == column).map(|position| position as u32)
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
            let mut covered = usize::from(arrangement.grouped);
            let mut next = sorted.next();
            while next.is_some_and(|key| keys.contains(&key.column)) {
                covered += 1;
                next = sorted.next();
            }
            if covered != keys.len() {
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

/// What a node needs of the rows of one of its inputs: sorted by
/// `sorted_by`, if given. Perfetto's needs grouping, types and trees too,
/// which no op here needs yet.
#[derive(Default)]
pub struct InputNeeds<'a> {
    pub sorted_by: Option<&'a [SortKey]>,
}

/// A step run on an input before the node reads it, to meet its needs.
pub enum InputStep {
    /// Sorts the rows by `keys`, and outputs only the first `keep` of them,
    /// if given.
    Sort { keys: SlowVec<SortKey>, keep: Option<u32> },
}

/// One input of a node, once planned.
pub struct PlannedInput {
    pub steps: SlowVec<InputStep>,
    /// What is known of the rows after `steps`.
    pub rows: OutputProperties,
}

/// What is decided about a node before it is lowered.
pub struct PlannedNode {
    pub inputs: SlowVec<PlannedInput>,
    pub output: OutputProperties,
}

/// By node, what's decided about it; `None` for nodes the plan doesn't
/// reach.
pub type Planned = SlowVec<Option<PlannedNode>>;

/// The most steps planned for one input.
const INPUT_STEPS_MAX: usize = 1 << 2;

/// What an op derives its output from.
pub struct DeriveContext<'p, 'c> {
    plan: &'p LogicalPlan<'c>,
    allocator: &'p dyn Allocator,
    planned: &'p mut Planned,
    node: PlanNodeId,
}

impl<'p> DeriveContext<'p, '_> {
    pub fn allocator(&self) -> &'p dyn Allocator {
        self.allocator
    }

    /// The id of the node being derived.
    pub fn node(&self) -> PlanNodeId {
        self.node
    }

    /// What is known of input `i` once its needs are met.
    pub fn input(&self, i: u32) -> &OutputProperties {
        let Some(node) = at!(self.planned, self.node as usize) else { failed!() };
        &at!(node.inputs, i as usize).rows
    }

    /// What `column` holds.
    pub fn query_type(&self, column: ColumnId) -> DataType {
        at!(self.plan.columns, column as usize).data_type
    }

    /// Makes the sort at `at` output only its first `rows` rows, so it is
    /// lowered as a `TopK`.
    pub fn limit_sort(&mut self, at: SortAt, rows: u32) {
        let InputStep::Sort { keep, .. } = step_mut(self.planned, at);
        *keep = Some(rows);
    }
}

fn step_mut(planned: &mut Planned, at: SortAt) -> &mut InputStep {
    let Some(node) = at_mut!(planned, at.node as usize) else { failed!() };
    at_mut!(at_mut!(node.inputs, at.input as usize).steps, at.step)
}

/// The step at `at`.
pub fn step(planned: &Planned, at: SortAt) -> &InputStep {
    let Some(node) = at!(planned, at.node as usize) else { failed!() };
    at!(at!(node.inputs, at.input as usize).steps, at.step)
}

/// Plans each node `plan` reaches, children before parents: how each input
/// meets its needs, and what the node derives it outputs.
pub fn plan_nodes(
    allocator: &dyn Allocator,
    plan: &LogicalPlan<'_>,
) -> Result<Planned, LowerError> {
    let mut planned = SlowVec::fixed_from(allocator, (0..plan.nodes.len()).map(|_| None))?;
    plan_tree(allocator, plan, &mut planned, plan.root)?;
    Ok(planned)
}

/// Plans node `id` after the nodes below it, as lowering lowers them.
fn plan_tree(
    allocator: &dyn Allocator,
    plan: &LogicalPlan<'_>,
    planned: &mut Planned,
    id: PlanNodeId,
) -> Result<(), LowerError> {
    for &child in at!(plan.nodes, id as usize).children.iter() {
        plan_tree(allocator, plan, planned, child)?;
    }
    plan_node_inputs(allocator, plan, planned, id)
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
    for (input, &child) in (0..).zip(node.children.iter()) {
        let Some(child) = at!(planned, child as usize) else { failed!() };
        let rows = child.output.copy(allocator)?;
        let mut out = PlannedInput { steps: SlowVec::new(allocator, INPUT_STEPS_MAX)?, rows };
        let at = SortAt { node: id, input, step: 0 };
        meet(allocator, planned, at, &mut out, &node.op.needs(input))?;
        inputs.push(out)?;
    }
    let output = OutputProperties::new(allocator)?;
    *at_mut!(planned, id as usize) = Some(PlannedNode { inputs, output });
    let mut derive = DeriveContext { plan, allocator, planned, node: id };
    let output = node.op.derive(&mut derive)?;
    let Some(node) = at_mut!(planned, id as usize) else { failed!() };
    node.output = output;
    Ok(())
}

/// Adds the steps `out`, the input of `at`'s node and input, needs to meet
/// `needs`.
fn meet(
    allocator: &dyn Allocator,
    planned: &mut Planned,
    at: SortAt,
    out: &mut PlannedInput,
    needs: &InputNeeds,
) -> Result<(), AllocError> {
    let Some(sorted_by) = needs.sorted_by else { return Ok(()) };
    let rows = &mut out.rows;
    if rows.is_arranged(&[], sorted_by)
        || (drop_open_sort(planned, rows) && rows.is_arranged(&[], sorted_by))
    {
        return Ok(());
    }
    let sort =
        OpenSort { at: SortAt { step: out.steps.len(), ..at }, arrangements: rows.arrangements };
    let keys = SlowVec::fixed_from(allocator, sorted_by.iter().copied())?;
    out.steps.push(InputStep::Sort { keys, keep: None })?;
    rows.arrangements = Arrangements::NONE;
    rows.arrangements.push(&[], sorted_by);
    rows.open_sort = Some(sort);
    Ok(())
}

/// Undoes the open sort of `rows`, if any, restoring what was known before
/// it.
fn drop_open_sort(planned: &mut Planned, rows: &mut OutputProperties) -> bool {
    let Some(sort) = rows.open_sort.take() else { return false };
    let Some(node) = at_mut!(planned, sort.at.node as usize) else { failed!() };
    // Nothing is planned after a sort before its order is read, so it's last.
    let steps = &mut at_mut!(node.inputs, sort.at.input as usize).steps;
    check!(sort.at.step + 1 == steps.len());
    steps.pop();
    rows.arrangements = sort.arrangements;
    true
}
