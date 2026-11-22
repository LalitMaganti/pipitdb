//! The tables, sources and operators pipitdb comes with. They use the same
//! API as any extension.

#![no_std]

#[macro_use]
extern crate pipit_kernel;

pub mod table;

#[cfg(test)]
mod join_tests;
#[cfg(test)]
mod nested_loop_join_tests;
