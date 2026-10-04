//! `Selection`: which rows of a batch are kept.

use core::mem::MaybeUninit;

use crate::row_batch::BATCH_ROWS_MAX;

/// The rows a batch keeps: all of them, none, or a selection of them by
/// index. Filters narrow it in place, so a batch's columns are never copied.
pub struct Selection {
    /// How many rows are kept: all of them, if `all`.
    len: u32,
    all: bool,
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
        let mut kept = 0;
        for i in 0..self.len as usize {
            let row = if self.all {
                i as u16
            } else {
                // SAFETY: unless `all`, the first `len` indices were written.
                unsafe { at!(self.indices, i).assume_init() }
            };
            at_mut!(self.indices, kept).write(row);
            kept += usize::from(keep(row));
        }
        // Every row still kept stays `all`, which needs no indices.
        self.all &= kept == self.len as usize;
        self.len = kept as u32;
    }

    /// Back to all of `rows` rows.
    pub(crate) fn reset(&mut self, rows: u32) {
        check!(rows <= BATCH_ROWS_MAX);
        self.len = rows;
        self.all = true;
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
        if !other.all {
            let len = other.len as usize;
            at_mut!(self.indices, ..len).copy_from_slice(at!(other.indices, ..len));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
