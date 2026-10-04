//! PipeSQL: query text, parsed into an `Ast`.

#![no_std]

extern crate alloc;

#[macro_use]
extern crate pipit_kernel;

pub mod ast;
pub mod error;
pub mod keywords;
pub mod lexer;
pub mod parser;
pub mod registry;
mod settings;
pub mod stages;
