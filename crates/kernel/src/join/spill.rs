//! A join's spilling. Once its build side can't be held, its build rows,
//! those held and those still to come, are written to a log a partition, by
//! four bits of their keys' hash, and then so are its probe rows. Once the
//! probe side ends, each partition's build rows are held and its probe rows
//! joined with them in turn. A partition whose build rows again can't be
//! held is split by the next four bits.

use super::{Building, Built, HashBuild, Join, Keys, Probing, last_of};
use crate::column::{ColumnView, Form, Forms};
use crate::containers::{Array, Bitmap, SmallVec};
use crate::context::Context;
use crate::env::Env;
use crate::error::Error;
use crate::group_table::NONE;
use crate::murmur_hash;
use crate::pipeline::with_built;
use crate::row_batch::{BATCH_COLUMNS_MAX, BATCH_ROWS_MAX, RowBatch};
use crate::row_store::RowStore;
use crate::selection::Selection;
use crate::sort_runs::batch_bytes;
use crate::spill::{LogId, SpillStore, read_batch, write_batch};
use crate::step::{Progress, Sink};

use super::direct::Sequence;

/// How many partitions rows are written to.
const PARTITIONS: usize = 16;

/// The place among a side's logs of its rows with a null key, which match
/// none, so are never joined: kept only if output.
const NULLS: usize = PARTITIONS;

/// How many times a partition is split, using all of a hash's bits, before
/// the join gives up: its rows may all have one key.
const LEVELS_MAX: u32 = 16;

/// A partition's log, of batches of a side's columns.
#[derive(Clone, Copy)]
pub(super) struct Partition {
    /// The log.
    log: LogId,
    /// How many batches it holds.
    batches: u32,
}

/// A side's rows written to a log a partition.
pub(super) struct Partitions {
    /// The store the logs are in.
    store: &'static dyn SpillStore,
    /// Which bits of a hash pick the partition rows are written to.
    level: u32,
    /// Each partition, if any rows were written to it, then the rows with a
    /// null key, if kept.
    logs: [Option<Partition>; PARTITIONS + 1],
    /// Whether rows with a null key are kept.
    keeps_nulls: bool,
    /// How many columns each batch has.
    width: usize,
}

impl Drop for Partitions {
    fn drop(&mut self) {
        for partition in self.logs.iter().flatten() {
            self.store.delete(partition.log);
        }
    }
}

impl Partitions {
    fn new(store: &'static dyn SpillStore, level: u32, keeps_nulls: bool) -> Partitions {
        Partitions { store, level, logs: [None; PARTITIONS + 1], keeps_nulls, width: 0 }
    }

    /// Writes the kept rows of `batch`, whose columns are flat, each to the
    /// partition its keys, at `keys`, hash to.
    fn write(&mut self, batch: &RowBatch, keys: &[u32]) -> Result<(), Error> {
        let selection = batch.selection();
        let count = selection.len() as usize;
        if count == 0 {
            return Ok(());
        }
        self.width = batch.column_count() as usize;
        let mut columns = [batch.column(0); BATCH_COLUMNS_MAX as usize];
        for (c, column) in columns.iter_mut().enumerate().take(self.width) {
            #[expect(clippy::cast_possible_truncation, reason = "at most 64")]
            let c = c as u32;
            *column = batch.column(c);
        }
        let columns = at!(columns, ..self.width);
        let mut partitions = [0_u8; BATCH_ROWS_MAX as usize];
        for (i, partition) in at_mut!(partitions, ..count).iter_mut().enumerate() {
            #[expect(clippy::cast_possible_truncation, reason = "a row of a batch")]
            let row = selection.kept().row(i as u32);
            *partition = partition_of(batch, keys, row, self.level);
        }
        let mut rows = [0_u16; BATCH_ROWS_MAX as usize];
        for p in (0..=PARTITIONS).filter(|&p| p != NULLS || self.keeps_nulls) {
            let mut written = 0;
            for (i, &partition) in at!(partitions, ..count).iter().enumerate() {
                #[expect(clippy::cast_possible_truncation, reason = "a row of a batch")]
                let row = selection.kept().row(i as u32) as u16;
                *at_mut!(rows, written) = row;
                written += usize::from(usize::from(partition) == p);
            }
            if written == 0 {
                continue;
            }
            let slot = at_mut!(self.logs, p);
            let partition = match slot {
                Some(partition) => partition,
                None => slot.insert(Partition { log: self.store.create()?, batches: 0 }),
            };
            #[expect(clippy::cast_possible_truncation, reason = "a batch's rows")]
            let rows_count = written as u32;
            write_batch(self.store, partition.log, columns, at!(rows, ..written), rows_count)?;
            partition.batches += 1;
        }
        Ok(())
    }

    /// Seals every log, so it can be read.
    fn seal(&self) -> Result<(), Error> {
        for partition in self.logs.iter().flatten() {
            self.store.seal(partition.log)?;
        }
        Ok(())
    }
}

/// Which partition row `row` of `batch` goes to at `level`, by its keys at
/// `keys`, integers: four bits of their hash, the next four a level, or
/// `NULLS` if one is null.
fn partition_of(batch: &RowBatch, keys: &[u32], row: u32, level: u32) -> u8 {
    let mut hash = murmur_hash::SEED;
    for &key in keys {
        let column = batch.column(key);
        let word = match column.form() {
            #[expect(clippy::cast_possible_truncation, reason = "at most 16")]
            _ if column.is_null(row) => return NULLS as u8,
            Form::Sequence(first) => (first + i64::from(row)).cast_unsigned(),
            Form::Flat => at!(column.values().int64s(), row as usize).cast_unsigned(),
            Form::Constant => at!(column.values().int64s(), 0).cast_unsigned(),
            Form::Dictionary(indices) => {
                let value = *at!(indices, row as usize) as usize;
                at!(column.values().int64s(), value).cast_unsigned()
            }
        };
        hash = murmur_hash::combine(hash, word);
    }
    (murmur_hash::mix(hash) >> (60 - 4 * (level % 16)) & 15) as u8
}

/// A copy of `batch`'s columns, flat, sharing those that are.
fn flat(context: &mut Context, batch: &RowBatch) -> Result<RowBatch<'static>, Error> {
    let mut out = RowBatch::new();
    out.reset(batch.row_count());
    for c in 0..batch.column_count() {
        let mut column = batch.column(c).clone();
        column.make_in(context, Forms::FLAT)?;
        let Ok(()) = out.push_column(column) else { failed!() };
    }
    out.selection_mut().clone_from(batch.selection());
    Ok(out)
}

/// Reads batch `at` of `width` columns from `partition`'s log, returning
/// where the next starts.
fn read(
    context: &mut Context,
    store: &dyn SpillStore,
    partition: Partition,
    at: u64,
    width: usize,
    batch: &mut RowBatch<'static>,
) -> Result<u64, Error> {
    let mut columns = [const { None }; BATCH_COLUMNS_MAX as usize];
    let mut c = 0;
    let next = read_batch(context, store, partition.log, at, width, |column| {
        *at_mut!(columns, c) = Some(column);
        c += 1;
        Ok(())
    })?;
    let Some(first) = at!(columns, 0) else { failed!() };
    batch.reset(first.row_count());
    for column in at_mut!(columns, ..width) {
        let Some(column) = column.take() else { failed!() };
        let Ok(()) = batch.push_column(column) else { failed!() };
    }
    Ok(next)
}

impl HashBuild {
    /// Whether it can spill: it has keys, integers, and holds no strings in
    /// the run's pool, which every partition would add to.
    pub(super) fn spills(&self, env: &Env) -> bool {
        let integers = self.keys.as_ref().is_some_and(|keys| keys.strings == 0);
        env.spill().is_some() && integers && self.strings.is_none()
    }

    /// Writes `batch`'s rows, numbered from `first`, to partitions if the
    /// build side spilled, or now can't hold them, spilling the rows held,
    /// and returns whether it did.
    pub(super) fn spill_batch(
        &self,
        context: &mut Context,
        building: &mut Building,
        batch: &RowBatch,
        first: u32,
    ) -> Result<bool, Error> {
        let Some(keys) = &self.keys else { failed!() };
        let Some(built) = &mut building.built else { failed!() };
        if built.spilled.is_none() {
            let env = context.env();
            let taken = env.used().saturating_sub(building.base);
            if first == 0 || env.leaves_room(3 * taken + 2 * batch_bytes(batch)) {
                return Ok(false);
            }
            let nulls = self.keeps_nulls;
            spill_held(context, built, keys, self.held, first, (building.level, nulls))?;
        }
        let Some(partitions) = &mut built.spilled else { failed!() };
        let mut columns = RowBatch::new();
        columns.reset(batch.row_count());
        let held = at!(self.build_columns, ..self.held).iter();
        for &c in held.chain(at!(keys.build, ..keys.count)) {
            let mut column = batch.column(c).clone();
            column.make_in(context, Forms::FLAT)?;
            let Ok(()) = columns.push_column(column) else { failed!() };
        }
        columns.selection_mut().clone_from(batch.selection());
        let key_columns = key_columns(self.held);
        partitions.write(&columns, at!(key_columns, ..keys.count))?;
        Ok(true)
    }

    /// The build side that holds a partition's rows, written as `consume_spilling`
    /// writes them: its build columns, then its keys.
    fn of_partition(&self) -> HashBuild {
        let mut build_columns = [0; BATCH_COLUMNS_MAX as usize];
        for (c, column) in (0..).zip(build_columns.iter_mut()) {
            *column = c;
        }
        let keys = self.keys.map(|keys| {
            let mut build = keys.build;
            let columns = key_columns(self.held);
            at_mut!(build, ..keys.count).copy_from_slice(at!(columns, ..keys.count));
            Keys { build, ..keys }
        });
        HashBuild { build_columns, held: self.held, keys, strings: None, keeps_nulls: false }
    }
}

/// Where the keys are in the batches a build side that can spill holds:
/// after its `held` build columns.
fn key_columns(held: usize) -> SmallKeys {
    #[expect(clippy::cast_possible_truncation, reason = "at most a batch's columns")]
    core::array::from_fn(|k| (held + k) as u32)
}

/// Writes the `count` rows `built` holds, each its `held` build columns and
/// then `keys`, to partitions at `level`, keeping those with a null key if
/// `keeps_nulls`, and frees what held them. The keys
/// aren't held as columns, but found from how they were hashed.
fn spill_held(
    context: &mut Context,
    built: &mut Built,
    keys: &Keys,
    held: usize,
    count: u32,
    (level, keeps_nulls): (u32, bool),
) -> Result<(), Error> {
    let env = context.env();
    let Some(store) = env.spill() else { failed!() };
    let mut partitions = Partitions::new(store, level, keeps_nulls);
    // Each row's group, by walking each group's rows from its last.
    let mut group_of = Array::zeroed(env, 0)?;
    if let (Sequence::Broken, Some(table)) = (built.sequence, &built.table) {
        group_of = Array::zeroed(env, count as usize)?;
        group_of.fill(NONE);
        #[expect(clippy::cast_possible_truncation, reason = "below `GROUPS_MAX`")]
        for group in 0..table.len() as u32 {
            let mut row = last_of(table, group);
            while row != NONE {
                *at_mut!(group_of, row as usize) = group;
                row = *at!(built.before, row as usize);
            }
        }
    }
    let mut rows = [0_u32; BATCH_ROWS_MAX as usize];
    let mut batch = RowBatch::new();
    for first in (0..count).step_by(BATCH_ROWS_MAX as usize) {
        let take = (count - first).min(BATCH_ROWS_MAX);
        if held == 0 {
            batch.reset(take);
        } else {
            for (row, slot) in (first..).zip(at_mut!(rows, ..take as usize)) {
                *slot = row;
            }
            let selection = Selection::all(take);
            built.gather(context, &mut batch, at!(rows, ..take as usize), &selection)?;
        }
        for k in 0..keys.count {
            let key = |row: u32| match built.sequence {
                Sequence::From(min) => Some(min + i64::from(row)),
                Sequence::Held => Some(*at!(built.keys, row as usize)),
                Sequence::Empty | Sequence::Broken => {
                    let (Some(table), group) = (&built.table, *at!(group_of, row as usize)) else {
                        return None;
                    };
                    (group != NONE).then(|| table.key(group, k))
                }
            };
            let column = ColumnView::from_int64s(context, (first..first + take).map(key))?;
            let Ok(()) = batch.push_column(column) else { failed!() };
        }
        let key_columns = key_columns(held);
        partitions.write(&batch, at!(key_columns, ..keys.count))?;
    }
    batch.reset(0);
    *built = Built {
        rows: RowStore::new(env)?,
        count: built.count,
        table: None,
        before: SmallVec::new(env, 1 << 31)?,
        repeats: false,
        direct: None,
        sequence: Sequence::Broken,
        keys: SmallVec::new(env, 1 << 31)?,
        strings: None,
        spilled: Some(partitions),
    };
    Ok(())
}

/// Where a side's keys are, as many as a join has.
type SmallKeys = [u32; super::JOIN_KEYS_MAX];

/// A pair of partitions, of build rows and of probe rows, to join.
#[derive(Clone, Copy)]
struct Pair {
    build: Option<Partition>,
    probe: Option<Partition>,
    level: u32,
}

/// What a join that spilled keeps while it probes and then joins each pair
/// of partitions.
pub(super) struct ProbeSpill {
    /// The probe rows written, of the level being written.
    probe: Partitions,
    /// The build side that holds a partition, once the probe side ended.
    build: Option<HashBuild>,
    /// The store the logs are in.
    store: &'static dyn SpillStore,
    /// How many columns a build partition's batches have.
    build_width: usize,
    /// The build rows with a null key, to output once the probe side ended,
    /// if the join outputs them, and how many of its batches were read.
    nulls: Option<(Partition, u32, u64)>,
    /// The pairs left to join.
    pending: SmallVec<Pair>,
    /// The pair whose build rows are being held, until they are.
    taken: Option<Pair>,
    /// The pair being joined.
    current: Option<Current>,
}

/// A pair being joined: its build rows held, and its next probe batch.
struct Current {
    built: Built,
    probe: Option<Partition>,
    /// How many of its probe batches were read.
    read: u32,
    /// Where its next probe batch starts.
    at: u64,
    /// Its probe batch being joined, if one is.
    input: Option<RowBatch<'static>>,
}

impl ProbeSpill {
    /// Where probe rows go once `build`, a build side's partitions, spilled.
    fn new(env: &Env, build: &Partitions) -> Result<ProbeSpill, Error> {
        Ok(ProbeSpill {
            probe: Partitions::new(build.store, build.level, false),
            build: None,
            store: build.store,
            build_width: 0,
            nulls: None,
            pending: SmallVec::new(env, 1 << 20)?,
            taken: None,
            current: None,
        })
    }
}

impl Drop for ProbeSpill {
    fn drop(&mut self) {
        let current = self.current.iter().map(|current| current.probe);
        let nulls = self.nulls.iter().map(|&(partition, ..)| Some(partition));
        let pairs = self.pending.iter().chain(&self.taken);
        let pairs = pairs.flat_map(|pair| [pair.build, pair.probe]);
        for partition in current.chain(nulls).chain(pairs).flatten() {
            self.store.delete(partition.log);
        }
    }
}

impl Join {
    /// Writes `input`'s rows to the partitions of a build side that spilled,
    /// but for those with a null key, which match none: output at once if it
    /// keeps them.
    pub(super) fn spill_probe(
        &self,
        context: &mut Context,
        built: &Built,
        probing: &mut Probing,
        input: &RowBatch,
        output: &mut RowBatch,
    ) -> Result<Progress, Error> {
        let Some(keys) = &self.keys else { failed!() };
        let Some(partitions) = &built.spilled else { failed!() };
        let spill = match &mut probing.spill {
            Some(spill) => spill,
            None => probing.spill.insert(ProbeSpill::new(context.env(), partitions)?),
        };
        let keys = at!(keys.probe, ..keys.count);
        spill.probe.write(&flat(context, input)?, keys)?;
        if self.keeps_probe() {
            output.copy_from(input);
            let null = |row: u16| keys.iter().any(|&key| input.column(key).is_null(row.into()));
            output.selection_mut().retain(null);
            for null in probing.nulls.iter() {
                let Ok(()) = output.push_column(null.slice(0, input.row_count())) else {
                    failed!()
                };
            }
        }
        Ok(Progress::NeedInput)
    }

    /// As `finish`, for a join whose build side spilled: joins each pair of
    /// partitions, a batch at a time. `None` if it didn't spill.
    pub(super) fn finish_spilled(
        &self,
        context: &mut Context,
        probing: &mut Probing,
        output: &mut RowBatch,
    ) -> Result<Option<Progress>, Error> {
        if probing.spill.is_none() && !self.keeps_build {
            return Ok(None);
        }
        if probing.spill.as_ref().is_none_or(|spill| spill.build.is_none())
            && !self.start_pairs(context, probing)?
        {
            return Ok(None);
        }
        if let Some(progress) = self.build_nulls(context, probing, output)? {
            return Ok(Some(progress));
        }
        loop {
            let Some(spill) = &mut probing.spill else { failed!() };
            let mut current = match spill.current.take() {
                Some(current) => current,
                None => match self.next_pair(context, probing)? {
                    Some(current) => current,
                    None => return Ok(Some(Progress::NeedInput)),
                },
            };
            let Some(spill) = &mut probing.spill else { failed!() };
            if current.input.is_none() {
                if let Some(probe) = current.probe.filter(|probe| current.read < probe.batches) {
                    let mut batch = RowBatch::new();
                    current.at = read(
                        context,
                        spill.store,
                        probe,
                        current.at,
                        spill.probe.width,
                        &mut batch,
                    )?;
                    current.read += 1;
                    current.input = Some(batch);
                } else {
                    if let Some(probe) = current.probe {
                        spill.store.delete(probe.log);
                    }
                    if !self.keeps_build {
                        continue;
                    }
                    let progress = self.unmatched(context, &mut current.built, probing, output)?;
                    let more = matches!(progress, Progress::MoreOutput);
                    let Some(spill) = &mut probing.spill else { failed!() };
                    if more {
                        current.probe = None;
                        spill.current = Some(current);
                    }
                    if output.row_count() > 0 {
                        return Ok(Some(Progress::MoreOutput));
                    }
                    continue;
                }
            }
            let Some(input) = current.input.take() else { failed!() };
            let progress = self.probe(context, &mut current.built, probing, &input, output)?;
            let Some(spill) = &mut probing.spill else { failed!() };
            match progress {
                Progress::MoreOutput => current.input = Some(input),
                Progress::NeedInput => {}
                // Nothing can be joined: no probe batch is read again.
                Progress::Done => current.read = current.probe.map_or(0, |probe| probe.batches),
            }
            spill.current = Some(current);
            if !output.selection().is_empty() {
                return Ok(Some(Progress::MoreOutput));
            }
            output.reset(0);
        }
    }

    /// Outputs the next batch of build rows with a null key, with nulls for
    /// the probe side's columns, if any are left.
    fn build_nulls(
        &self,
        context: &mut Context,
        probing: &mut Probing,
        output: &mut RowBatch,
    ) -> Result<Option<Progress>, Error> {
        let Some(spill) = &mut probing.spill else { failed!() };
        let Some((partition, read, at)) = &mut spill.nulls else { return Ok(None) };
        let mut batch = RowBatch::new();
        *at = super::spill::read(
            context,
            spill.store,
            *partition,
            *at,
            spill.build_width,
            &mut batch,
        )?;
        *read += 1;
        if *read == partition.batches {
            spill.store.delete(partition.log);
            spill.nulls = None;
        }
        let count = batch.row_count();
        output.reset(count);
        for null in probing.probe_nulls.iter() {
            let Ok(()) = output.push_column(null.slice(0, count)) else { failed!() };
        }
        for c in 0..self.outputs {
            #[expect(clippy::cast_possible_truncation, reason = "at most a batch's columns")]
            let column = batch.column(c as u32).clone();
            let Ok(()) = output.push_column(column) else { failed!() };
        }
        Ok(Some(Progress::MoreOutput))
    }

    /// Readies the pairs of the build side's partitions and the probe side's,
    /// once the probe side ended. False if the build side didn't spill.
    fn start_pairs(&self, context: &mut Context, probing: &mut Probing) -> Result<bool, Error> {
        let taken = with_built(context, self.build, |_, sink, built| {
            let Some(built) = built else { failed!() };
            let Some(partitions) = built.spilled.take() else { return Ok(None) };
            Ok::<_, Error>(Some((partitions, sink.of_partition())))
        })??;
        let Some((mut build, sink)) = taken else { return Ok(false) };
        let env = context.env();
        let spill = match &mut probing.spill {
            Some(spill) => spill,
            None => probing.spill.insert(ProbeSpill::new(env, &build)?),
        };
        build.seal()?;
        spill.probe.seal()?;
        spill.build_width = build.width;
        spill.nulls = at_mut!(build.logs, NULLS).take().map(|partition| (partition, 0, 0));
        for p in 0..PARTITIONS {
            let (build, probe) =
                (at_mut!(build.logs, p).take(), at_mut!(spill.probe.logs, p).take());
            spill.pending.push(Pair { build, probe, level: spill.probe.level })?;
        }
        spill.build = Some(sink);
        Ok(true)
    }

    /// The next pair of partitions to join, its build rows held, or `None`
    /// if none is left. A pair whose build rows can't be held is split.
    fn next_pair(
        &self,
        context: &mut Context,
        probing: &mut Probing,
    ) -> Result<Option<Current>, Error> {
        loop {
            let Some(spill) = &mut probing.spill else { failed!() };
            let Some(pair) = spill.pending.pop() else { return Ok(None) };
            spill.taken = Some(pair);
            let nothing = (pair.build.is_none() && !self.keeps_probe())
                || (pair.probe.is_none() && !self.keeps_build);
            if nothing {
                for partition in [pair.build, pair.probe].into_iter().flatten() {
                    spill.store.delete(partition.log);
                }
                spill.taken = None;
                continue;
            }
            let Some(sink) = &spill.build else { failed!() };
            let (store, width) = (spill.store, spill.build_width);
            let mut building = Sink::new_local(sink, context)?;
            building.level = pair.level + 1;
            if let Some(build) = pair.build {
                let mut batch = RowBatch::new();
                let mut at = 0;
                for _ in 0..build.batches {
                    at = read(context, store, build, at, width, &mut batch)?;
                    Sink::consume(sink, context, &mut building, &batch)?;
                }
                batch.reset(0);
                store.delete(build.log);
                spill.taken = Some(Pair { build: None, ..pair });
            }
            let mut built = None;
            Sink::combine(sink, context, &mut built, &mut building)?;
            Sink::finalize(sink, context, &mut built)?;
            let Some(mut built) = built else { failed!() };
            if let Some(split) = built.spilled.take() {
                if pair.level + 1 >= LEVELS_MAX {
                    return Err(Error::OutOfMemory);
                }
                self.split_probe(context, probing, pair, split)?;
                continue;
            }
            let Some(spill) = &mut probing.spill else { failed!() };
            spill.taken = None;
            probing.started = false;
            probing.unmatched = 0;
            if self.keeps_build {
                probing.build_matched = Bitmap::new(context.env(), built.count as usize)?;
            }
            return Ok(Some(Current { built, probe: pair.probe, read: 0, at: 0, input: None }));
        }
    }

    /// Splits `pair`'s probe rows by the next bits, as its build rows were
    /// into `split`, and adds the pairs they make.
    fn split_probe(
        &self,
        context: &mut Context,
        probing: &mut Probing,
        pair: Pair,
        mut split: Partitions,
    ) -> Result<(), Error> {
        let Some(keys) = &self.keys else { failed!() };
        let Some(spill) = &mut probing.spill else { failed!() };
        let mut probe = Partitions::new(spill.store, split.level, false);
        if let Some(rows) = pair.probe {
            let mut batch = RowBatch::new();
            let mut at = 0;
            for _ in 0..rows.batches {
                at = read(context, spill.store, rows, at, spill.probe.width, &mut batch)?;
                probe.write(&batch, at!(keys.probe, ..keys.count))?;
            }
            spill.store.delete(rows.log);
        }
        spill.taken = None;
        split.seal()?;
        probe.seal()?;
        for p in 0..PARTITIONS {
            let (build, rows) = (at_mut!(split.logs, p).take(), at_mut!(probe.logs, p).take());
            spill.pending.push(Pair { build, probe: rows, level: split.level })?;
        }
        Ok(())
    }
}
