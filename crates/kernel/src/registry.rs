//! The stages a parser knows.
//!
//! Each stage is described by a `Rule`: its keyword and the items after it.
//! Rules come in sets, and a `Registry` composes sets. Built as a `static`, a
//! registry's mistakes, such as two rules with one keyword, are build errors.

use pipitdb_keywords::KeywordTable;

use crate::settings::build_setting;

/// How many keywords a registry has room for. Set at build time with
/// `PIPIT_STAGE_SLOTS`.
pub const STAGE_SLOTS: usize = build_setting(option_env!("PIPIT_STAGE_SLOTS"), 64);

/// How many items a rule can have.
pub const ITEMS_MAX: usize = 8;

/// Where a stage can go: first in a query, or after `|>`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Point {
    Source,
    Stage,
}

/// What comes after a stage's keyword. Each item is one child of the stage's
/// node, in order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Item {
    One(Shared),
    /// One or more, separated by commas, under a `List` node.
    List(Shared),
}

/// The pieces of grammar stages are built from.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Shared {
    Name,
    Expr,
}

#[derive(Clone, Copy, Debug)]
pub struct Rule {
    pub keyword: &'static str,
    pub point: Point,
    pub items: &'static [Item],
}

/// Rules, found by keyword. A rule's id is `set << 8 | index in set`.
pub struct Registry {
    sets: &'static [&'static [Rule]],
    keywords: KeywordTable<STAGE_SLOTS>,
}

impl Registry {
    pub const fn new(sets: &'static [&'static [Rule]]) -> Registry {
        // Ids fit in 16 bits, and the keyword table reserves 0xffff.
        assert!(sets.len() < 256, "too many rule sets");
        let mut keywords = KeywordTable::new();
        let mut set = 0;
        while set < sets.len() {
            assert!(sets[set].len() <= 256, "too many rules in a set");
            let mut index = 0;
            while index < sets[set].len() {
                let rule = &sets[set][index];
                assert!(rule.items.len() <= ITEMS_MAX, "too many items in a rule");
                #[expect(clippy::cast_possible_truncation, reason = "checked above")]
                let id = (set << 8 | index) as u16;
                keywords = keywords.insert(rule.keyword.as_bytes(), id);
                index += 1;
            }
            set += 1;
        }
        Registry { sets, keywords }
    }

    /// The id of the rule whose keyword is `word`, ignoring case.
    pub fn find(&self, word: &[u8]) -> Option<u16> {
        self.keywords.get(word, |id| self.rule(id).keyword.as_bytes().eq_ignore_ascii_case(word))
    }

    pub fn rule(&self, id: u16) -> &'static Rule {
        let id = usize::from(id);
        at!(at!(self.sets, id >> 8), id & 0xff)
    }
}
