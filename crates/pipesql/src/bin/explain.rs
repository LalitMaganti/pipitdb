//! Explains a compact pipitdb error: `pipit-explain <error> [query file]`.
//! Reads the query from stdin if no file is given.

use std::io::Read;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(Some(error)) =
        args.next().map(|text| pipit_pipesql::diagnostics::parse_compact(&text))
    else {
        eprintln!("usage: pipit-explain pipit:E0007:2+0:5 [query file]");
        return ExitCode::FAILURE;
    };
    match read_query(args.next()) {
        Ok((source, name)) => {
            print!("{}", pipit_pipesql::diagnostics::render(&error, &source, &name));
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("{message}");
            ExitCode::FAILURE
        }
    }
}

/// The query's text and name, from `path`, or stdin without one.
fn read_query(path: Option<String>) -> Result<(String, String), String> {
    if let Some(path) = path {
        let source =
            std::fs::read_to_string(&path).map_err(|e| format!("can't read {path}: {e}"))?;
        return Ok((source, path));
    }
    let mut source = String::new();
    std::io::stdin().read_to_string(&mut source).map_err(|e| format!("can't read stdin: {e}"))?;
    Ok((source, "stdin".into()))
}
