//! `Selection`: which rows of a batch are kept.

use core::mem::MaybeUninit;

use crate::row_batch::BATCH_ROWS_MAX;

/// The rows a batch keeps: all of them, none, or a selection of them by
/// index. Filters narrow it in place, so a batch's columns are never copied.
pub struct Selection {
    /// How many rows are kept: all of them, if `all`.
    len: u32,
    all: bool,
    /// How many rows the batch has: every kept row is below it.
    rows: u32,
    // Unless `all`, the first `len` are the kept rows, in increasing order;
    // only those are written. Kept whatever the state, so narrowing never
    // sets it up again.
    indices: [MaybeUninit<u16>; BATCH_ROWS_MAX as usize],
}

/// What a `Selection` keeps.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kept<'a> {
    All,
    None,
    /// These rows, in increasing order.
    Select(&'a [u16]),
}

impl Selection {
    /// All of `rows` rows.
    pub fn all(rows: u32) -> Selection {
        check!(rows <= BATCH_ROWS_MAX);
        Selection {
            len: rows,
            all: true,
            rows,
            indices: [MaybeUninit::uninit(); BATCH_ROWS_MAX as usize],
        }
    }

    pub fn kept(&self) -> Kept<'_> {
        match (self.len, self.all) {
            (0, _) => Kept::None,
            (_, true) => Kept::All,
            (len, false) => {
                let indices = at!(self.indices, ..len as usize);
                // SAFETY: unless `all`, the first `len` indices were written.
                Kept::Select(unsafe { &*(core::ptr::from_ref(indices) as *const [u16]) })
            }
        }
    }

    /// How many rows the batch has. Every kept row is below it, so a filter
    /// can check its column covers them once, rather than each row.
    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// How many rows are kept.
    pub fn len(&self) -> u32 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Keeps the rows `keep` says to, in order, writing over the indices as it
    /// goes: it never writes past where it has read.
    #[expect(clippy::cast_possible_truncation, reason = "rows are below `BATCH_ROWS_MAX`")]
    pub fn retain(&mut self, mut keep: impl FnMut(u16) -> bool) {
        let len = self.len as usize;
        check!(len <= self.indices.len());
        let indices = self.indices.as_mut_ptr().cast::<u16>();
        let mut kept = 0;
        // Two loops, so neither checks `all` for each row. In both, `kept`
        // never passes `i`, which is below `len`, so every index is in bounds.
        if self.all {
            for i in 0..len {
                let row = i as u16;
                // SAFETY: see above.
                unsafe { indices.add(kept).write(row) };
                kept += usize::from(keep(row));
            }
        } else {
            for i in 0..len {
                // SAFETY: see above; the first `len` indices were written.
                let row = unsafe { indices.add(i).read() };
                // SAFETY: see above.
                unsafe { indices.add(kept).write(row) };
                kept += usize::from(keep(row));
            }
        }
        // Every row still kept stays `all`, which needs no indices.
        self.all &= kept == self.len as usize;
        self.len = kept as u32;
    }

    /// Drops the rows `removed` keeps, which must be among these.
    pub fn subtract(&mut self, removed: &Selection) {
        check!(removed.rows == self.rows);
        match removed.kept() {
            Kept::None => {}
            Kept::All => self.len = 0,
            Kept::Select(removed) => {
                // Both are in increasing order, so one pass over each.
                let mut next = 0;
                self.retain(|row| {
                    while next < removed.len() && *at!(removed, next) < row {
                        next += 1;
                    }
                    removed.get(next) != Some(&row)
                });
            }
        }
    }

    /// Adds the rows `added` keeps, which must be none of these.
    #[expect(clippy::cast_possible_truncation, reason = "at most `BATCH_ROWS_MAX` rows")]
    pub fn union(&mut self, added: &Selection) {
        check!(added.rows == self.rows);
        let added = match added.kept() {
            Kept::None => return,
            Kept::All => {
                check!(self.len == 0);
                self.reset(self.rows);
                return;
            }
            Kept::Select(added) => added,
        };
        // Disjoint from all rows, `added` would be empty, which it isn't.
        check!(!self.all);
        let (mine, theirs) = (self.len as usize, added.len());
        let total = mine + theirs;
        check!(total <= self.rows as usize);
        let indices = self.indices.as_mut_ptr().cast::<u16>();
        // Merges from the back, so no index is overwritten before it's read:
        // the next write is always past the next unread one of `self`.
        let (mut i, mut j) = (mine, theirs);
        for out in (0..total).rev() {
            // SAFETY: `i` and `out` are below `total`, within the indices, and
            // the first `mine` were written; `j` is within `added`.
            unsafe {
                let take_mine =
                    j == 0 || (i > 0 && indices.add(i - 1).read() > *added.get_unchecked(j - 1));
                let row = if take_mine {
                    i -= 1;
                    indices.add(i).read()
                } else {
                    j -= 1;
                    *added.get_unchecked(j)
                };
                indices.add(out).write(row);
            }
        }
        self.len = total as u32;
    }

    /// Keeps the rows `keep` says to, and writes the others to `rejected`,
    /// in one pass.
    #[expect(clippy::cast_possible_truncation, reason = "rows are below `BATCH_ROWS_MAX`")]
    pub fn partition(&mut self, rejected: &mut Selection, mut keep: impl FnMut(u16) -> bool) {
        let len = self.len as usize;
        check!(len <= self.indices.len());
        rejected.rows = self.rows;
        let indices = self.indices.as_mut_ptr().cast::<u16>();
        let others = rejected.indices.as_mut_ptr().cast::<u16>();
        let (mut kept, mut out) = (0, 0);
        // As in `retain`: two loops, so neither checks `all` for each row, and
        // `kept` and `out` never pass `i`, below `len`.
        let mut put = |row: u16| {
            let keeps = keep(row);
            // SAFETY: see above.
            unsafe {
                indices.add(kept).write(row);
                others.add(out).write(row);
            }
            kept += usize::from(keeps);
            out += usize::from(!keeps);
        };
        if self.all {
            (0..len).for_each(|i| put(i as u16));
        } else {
            // SAFETY: see above; the first `len` indices were written.
            (0..len).for_each(|i| put(unsafe { indices.add(i).read() }));
        }
        self.all &= kept == len;
        self.len = kept as u32;
        rejected.all = false;
        rejected.len = out as u32;
    }

    /// All of `rows` rows, at `selection`, in place: the indices aren't
    /// written.
    ///
    /// # Safety
    ///
    /// `selection` must be valid for writes of a `Selection`, and is then one.
    pub(crate) unsafe fn init(selection: *mut Selection, rows: u32) {
        check!(rows <= BATCH_ROWS_MAX);
        // SAFETY: upheld by the caller. The indices need no writing, as none
        // are read until written.
        unsafe {
            (&raw mut (*selection).len).write(rows);
            (&raw mut (*selection).all).write(true);
            (&raw mut (*selection).rows).write(rows);
        }
    }

    /// Back to all of `rows` rows.
    pub(crate) fn reset(&mut self, rows: u32) {
        check!(rows <= BATCH_ROWS_MAX);
        self.len = rows;
        self.all = true;
        self.rows = rows;
    }
}

impl Clone for Selection {
    fn clone(&self) -> Selection {
        let mut selection = Selection::all(0);
        selection.clone_from(self);
        selection
    }

    /// Copies only the indices in use: none, if all rows are kept.
    fn clone_from(&mut self, other: &Selection) {
        self.len = other.len;
        self.all = other.all;
        self.rows = other.rows;
        if !other.all {
            let len = other.len as usize;
            at_mut!(self.indices, ..len).copy_from_slice(at!(other.indices, ..len));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(selection: &Selection) -> alloc::vec::Vec<u16> {
        match selection.kept() {
            Kept::All => (0..u16::try_from(selection.rows()).unwrap()).collect(),
            Kept::None => alloc::vec::Vec::new(),
            Kept::Select(rows) => rows.to_vec(),
        }
    }

    #[test]
    fn subtracts_and_unites() {
        let mut odd = Selection::all(10);
        odd.retain(|row| row % 2 == 1);
        let mut low = odd.clone();
        low.retain(|row| row < 5);

        let mut high = odd.clone();
        high.subtract(&low);
        assert_eq!(rows(&high), [5, 7, 9]);
        high.union(&low);
        assert_eq!(rows(&high), [1, 3, 5, 7, 9]);

        let mut everything = Selection::all(10);
        everything.subtract(&odd);
        assert_eq!(rows(&everything), [0, 2, 4, 6, 8]);
        everything.union(&odd);
        assert_eq!(rows(&everything), (0..10).collect::<alloc::vec::Vec<_>>());
    }

    #[test]
    fn partitions_in_one_pass() {
        let mut kept = Selection::all(10);
        let mut rejected = Selection::all(0);
        kept.partition(&mut rejected, |row| row % 3 == 0);
        assert_eq!(
            (rows(&kept), rows(&rejected)),
            ([0, 3, 6, 9].into(), [1, 2, 4, 5, 7, 8].into())
        );
        kept.partition(&mut rejected, |row| row > 3);
        assert_eq!((rows(&kept), rows(&rejected)), ([6, 9].into(), [0, 3].into()));
    }

    #[test]
    fn narrows_in_place() {
        let mut selection = Selection::all(10);
        assert_eq!((selection.len(), selection.kept()), (10, Kept::All));
        selection.retain(|_| true);
        assert_eq!(selection.kept(), Kept::All);
        selection.retain(|row| row % 2 == 0);
        assert_eq!(selection.kept(), Kept::Select(&[0, 2, 4, 6, 8]));
        selection.retain(|row| row > 3);
        assert_eq!(selection.kept(), Kept::Select(&[4, 6, 8]));
        selection.retain(|_| false);
        assert_eq!((selection.len(), selection.kept()), (0, Kept::None));
    }
}
