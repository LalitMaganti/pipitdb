//! PipeSQL: query text, parsed into an `Ast`, with errors that can be
//! rendered against the text.

#![no_std]

extern crate alloc;

#[macro_use]
extern crate pipit_kernel;

pub mod ast;
pub mod diagnostics;
pub mod error;
pub mod keywords;
pub mod lexer;
pub mod parser;
pub mod registry;
mod settings;
pub mod stages;
