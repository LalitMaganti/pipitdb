//! `InlineVec`: up to `N` values, held in place rather than allocated.

use core::ops::{Deref, DerefMut};

use crate::allocator::AllocError;

/// A short list with a known most, such as a node's children or a sort's
/// keys: held in place, so there's no memory to allocate or free.
#[derive(Clone, Copy)]
pub struct InlineVec<T: Copy + Default, const N: usize> {
    values: [T; N],
    len: u32,
}

impl<T: Copy + Default, const N: usize> InlineVec<T, N> {
    pub fn new() -> InlineVec<T, N> {
        InlineVec { values: [T::default(); N], len: 0 }
    }

    /// `values`, failing as running out of memory if there are more than
    /// `N`, as a full `SlowVec` does.
    pub fn from_slice(values: &[T]) -> Result<InlineVec<T, N>, AllocError> {
        let mut list = InlineVec::new();
        let Some(room) = list.values.get_mut(..values.len()) else { return Err(AllocError) };
        room.copy_from_slice(values);
        #[expect(clippy::cast_possible_truncation, reason = "at most `N`")]
        let len = values.len() as u32;
        list.len = len;
        Ok(list)
    }

    /// Adds `value` at the end, failing if it holds `N`.
    pub fn push(&mut self, value: T) -> Result<(), AllocError> {
        let Some(slot) = self.values.get_mut(self.len as usize) else { return Err(AllocError) };
        *slot = value;
        self.len += 1;
        Ok(())
    }

    /// Keeps the values `keep` says to, in order.
    pub fn retain(&mut self, mut keep: impl FnMut(&T) -> bool) {
        let mut kept = 0;
        for i in 0..self.len as usize {
            let value = *at!(self.values, i);
            if keep(&value) {
                *at_mut!(self.values, kept) = value;
                kept += 1;
            }
        }
        #[expect(clippy::cast_possible_truncation, reason = "at most `N`")]
        let kept = kept as u32;
        self.len = kept;
    }
}

impl<T: Copy + Default, const N: usize> Default for InlineVec<T, N> {
    fn default() -> InlineVec<T, N> {
        InlineVec::new()
    }
}

impl<T: Copy + Default, const N: usize> Deref for InlineVec<T, N> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        at!(self.values, ..self.len as usize)
    }
}

impl<T: Copy + Default, const N: usize> DerefMut for InlineVec<T, N> {
    fn deref_mut(&mut self) -> &mut [T] {
        at_mut!(self.values, ..self.len as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holds_up_to_its_most() {
        let mut list = InlineVec::<u32, 3>::from_slice(&[1, 2]).unwrap();
        assert!(list.push(3).is_ok());
        assert!(list.push(4).is_err());
        list.retain(|&value| value != 2);
        assert_eq!(*list, [1, 3]);
        assert!(InlineVec::<u32, 1>::from_slice(&[1, 2]).is_err());
    }
}
