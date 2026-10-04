//! `KeywordTable`: a case-insensitive table from keywords to small values,
//! built by `const fn`s so it can be a `static` computed at compile time.
//!
//! It is a hash table with linear probing. Slots hold a keyword's hash, not
//! the keyword, so `get` takes a check that the value found is for the word.
//! Building fails if two keywords hash the same (including the same keyword
//! twice), or if finding a keyword would take more than `PROBES_MAX` probes.

/// The most slots a lookup checks.
pub const PROBES_MAX: usize = 4;

const EMPTY: u16 = u16::MAX;

#[derive(Clone, Copy)]
struct Slot {
    hash: u32,
    value: u16,
}

pub struct KeywordTable<const SLOTS: usize> {
    slots: [Slot; SLOTS],
}

impl<const SLOTS: usize> KeywordTable<SLOTS> {
    pub const fn new() -> Self {
        const { assert!(SLOTS.is_power_of_two()) };
        KeywordTable { slots: [Slot { hash: 0, value: EMPTY }; SLOTS] }
    }

    pub const fn insert(mut self, keyword: &[u8], value: u16) -> Self {
        assert!(value != EMPTY, "keyword values must be below u16::MAX");
        let hash = hash(keyword);
        let mut probe = 0;
        while probe < PROBES_MAX {
            let slot = (hash as usize).wrapping_add(probe) % SLOTS;
            if self.slots[slot].value == EMPTY {
                self.slots[slot] = Slot { hash, value };
                return self;
            }
            assert!(
                self.slots[slot].hash != hash,
                "keyword added twice, or two keywords hash the same"
            );
            probe += 1;
        }
        panic!("keyword table too full: use more slots");
    }

    /// The value for `word`. `is_word` checks that a value found is for
    /// `word`, as different words can share a hash.
    pub fn get(&self, word: &[u8], is_word: impl Fn(u16) -> bool) -> Option<u16> {
        let hash = hash(word);
        for probe in 0..PROBES_MAX {
            let found = self.slots[(hash as usize).wrapping_add(probe) % SLOTS];
            if found.value == EMPTY {
                return None;
            }
            if found.hash == hash && is_word(found.value) {
                return Some(found.value);
            }
        }
        None
    }
}

impl<const SLOTS: usize> Default for KeywordTable<SLOTS> {
    fn default() -> Self {
        Self::new()
    }
}

/// FNV-1a over `word`, lowercased.
const fn hash(word: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    let mut i = 0;
    while i < word.len() {
        hash = (hash ^ word[i].to_ascii_lowercase() as u32).wrapping_mul(0x0100_0193);
        i += 1;
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORDS: [&[u8]; 3] = [b"from", b"where", b"select"];

    static TABLE: KeywordTable<8> =
        KeywordTable::new().insert(WORDS[0], 0).insert(WORDS[1], 1).insert(WORDS[2], 2);

    fn get(word: &[u8]) -> Option<u16> {
        TABLE.get(word, |value| WORDS[usize::from(value)].eq_ignore_ascii_case(word))
    }

    #[test]
    #[should_panic(expected = "keyword added twice")]
    fn rejects_a_keyword_twice() {
        let _ = KeywordTable::<8>::new().insert(b"from", 0).insert(b"FROM", 1);
    }

    #[test]
    #[should_panic(expected = "too full")]
    fn rejects_a_full_table() {
        let _ = KeywordTable::<2>::new().insert(b"a", 0).insert(b"b", 1).insert(b"c", 2);
    }

    #[test]
    fn finds_keywords_ignoring_case() {
        assert_eq!(get(b"from"), Some(0));
        assert_eq!(get(b"WHERE"), Some(1));
        assert_eq!(get(b"SeLeCt"), Some(2));
        assert_eq!(get(b"order"), None);
        assert_eq!(get(b""), None);
    }
}
