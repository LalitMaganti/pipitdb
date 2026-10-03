//! The pipitdb microkernel.
//!
//! Holds only what every build needs: the data model, the pipeline driver,
//! the module registry and expression parsing. Operators, sources and
//! functions live in modules that register with it.

#![no_std]

extern crate alloc;

pub mod allocator;
pub mod buffer;
