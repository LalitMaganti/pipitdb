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

/// A column of a batch, and what it holds here.
#[derive(Clone, Copy)]
pub struct Column {
    pub id: ColumnId,
    pub data_type: DataType,
}

/// The most recent sort, while nothing has read its order, and what was
/// known before it: it can still be undone.
#[derive(Clone, Copy)]
pub struct OpenSort {
    node: PlanNodeId,
    input: u32,
    step: usize,
}

/// What is known about the batches a node outputs: their columns and types,
/// and how the rows are arranged. Planning compares it with what each node
/// needs.
#[derive(Clone, Copy)]
pub struct OutputProperties {
    /// The columns of a batch, in order: the first `column_count` of these.
    columns: [Column; BATCH_COLUMNS_MAX as usize],
    column_count: u32,
    /// Every layout known to hold.
    pub arrangements: Arrangements,
    /// Set while a sort can still be undone.
    pub open_sort: Option<OpenSort>,
}

impl OutputProperties {
    /// No columns, and nothing known.
    pub const NONE: OutputProperties = OutputProperties {
        columns: [Column { id: 0, data_type: DataType::Int64 }; BATCH_COLUMNS_MAX as usize],
        column_count: 0,
        arrangements: Arrangements::NONE,
        open_sort: None,
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
pub struct InputNeeds<'a> {
    pub types_first: SlowVec<Type>,
    pub arranged: Option<Arranged<'a>>,
    pub types_then: SlowVec<Type>,
}

impl InputNeeds<'_> {
    /// Nothing.
    pub fn none(allocator: &dyn Allocator) -> Result<InputNeeds<'static>, AllocError> {
        Ok(InputNeeds {
            types_first: SlowVec::fixed(allocator, 0)?,
            arranged: None,
            types_then: SlowVec::fixed(allocator, 0)?,
        })
    }
}

/// What it takes for an input to meet a node's needs, each step run in order
/// before the node reads the input.
pub enum InputStep {
    /// Sorts the rows by `keys`. Only the first `keep` rows in order are
    /// read, if given.
    Sort { keys: SlowVec<SortKey>, keep: Option<u32>, unsorted: Arrangements },
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

/// By node, what's decided about it.
pub type Planned = SlowVec<PlannedNode>;

/// The most steps planned for one input.
const INPUT_STEPS_MAX: usize = 1 << 2;

/// What an op derives its output from.
pub struct DeriveContext<'p, 'c> {
    plan: &'p LogicalPlan<'c>,
    allocator: &'p dyn Allocator,
    planned: &'p mut Planned,
    self_: PlanNodeId,
}

impl<'p> DeriveContext<'p, '_> {
    pub fn allocator(&self) -> &'p dyn Allocator {
        self.allocator
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
    /// is lowered as a `TopK`.
    pub fn limit_sort(&mut self, sort: OpenSort, rows: u32) {
        let node = at_mut!(self.planned, sort.node as usize);
        let InputStep::Sort { keep, .. } =
            at_mut!(at_mut!(node.inputs, sort.input as usize).steps, sort.step);
        *keep = Some(rows);
    }
}

/// The step that `sort` is.
pub fn sort_step(planned: &Planned, sort: OpenSort) -> &InputStep {
    at!(at!(at!(planned, sort.node as usize).inputs, sort.input as usize).steps, sort.step)
}

/// Works out how one input of a node meets the node's needs.
struct InputPlanner<'p> {
    allocator: &'p dyn Allocator,
    planned: &'p mut Planned,
    node: PlanNodeId,
    input: u32,
    out: PlannedInput,
}

impl<'p> InputPlanner<'p> {
    fn new(
        allocator: &'p dyn Allocator,
        planned: &'p mut Planned,
        node: PlanNodeId,
        input: u32,
        rows: &OutputProperties,
    ) -> Result<InputPlanner<'p>, AllocError> {
        let out = PlannedInput { steps: SlowVec::new(allocator, INPUT_STEPS_MAX)?, rows: *rows };
        Ok(InputPlanner { allocator, planned, node, input, out })
    }

    fn meet(&mut self, needs: &InputNeeds) -> Result<(), LowerError> {
        self.types(&needs.types_first)?;
        if let Some(arranged) = needs.arranged {
            self.arrange(arranged.grouped_by, arranged.sorted_by)?;
        }
        self.types(&needs.types_then)
    }

    fn take(self) -> PlannedInput {
        self.out
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

    fn arrange(&mut self, keys: &[ColumnId], sorted_by: &[SortKey]) -> Result<(), AllocError> {
        if self.out.rows.is_arranged(keys, sorted_by)
            || (self.drop_open_sort() && self.out.rows.is_arranged(keys, sorted_by))
        {
            return Ok(());
        }
        // Grouping isn't ported: no op needs its input grouped yet.
        check!(keys.is_empty());
        self.sort(sorted_by)
    }

    fn drop_open_sort(&mut self) -> bool {
        let rows = &mut self.out.rows;
        let Some(sort) = rows.open_sort.take() else { return false };
        let steps =
            &mut at_mut!(at_mut!(self.planned, sort.node as usize).inputs, sort.input as usize)
                .steps;
        // Only sorts are planned so far, and nothing after a sort before its
        // order is read, so it's the last step.
        check!(sort.step + 1 == steps.len());
        let Some(InputStep::Sort { unsorted, .. }) = steps.pop() else { failed!() };
        rows.arrangements = unsorted;
        true
    }

    fn sort(&mut self, keys: &[SortKey]) -> Result<(), AllocError> {
        let rows = &mut self.out.rows;
        let sort = OpenSort { node: self.node, input: self.input, step: self.out.steps.len() };
        let keys_copy = SlowVec::fixed_from(self.allocator, keys.iter().copied())?;
        self.out.steps.push(InputStep::Sort {
            keys: keys_copy,
            keep: None,
            unsorted: rows.arrangements,
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
    for (i, &child) in (0..).zip(node.children.iter()) {
        check!(child < id);
        let rows = at!(planned, child as usize).output;
        let mut input = InputPlanner::new(allocator, planned, id, i, &rows)?;
        input.meet(&node.op.needs(allocator, i)?)?;
        inputs.push(input.take())?;
    }
    planned.push(PlannedNode { inputs, output: OutputProperties::NONE })?;
    let mut derive = DeriveContext { plan, allocator, planned, self_: id };
    let output = node.op.derive(&mut derive)?;
    at_mut!(planned, id as usize).output = output;
    Ok(())
}
