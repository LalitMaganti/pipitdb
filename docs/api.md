# Public API

pipitdb is two layers. The kernel compiles queries and runs them: it turns
data into data, and knows nothing of any query language. The `pipitdb`
crate is a database built on it: connections, statements and, in time, a
catalog, storage and transactions. Each layer makes public only what its
users need, so everything else can change freely.

This is the design. Most of it doesn't exist yet; Migration says in what
order it's built.

## Layers

1. **`pipitdb`: the database.** The traditional architecture of a database
   engine: `Database`, `Connection`, `Statement`, `Rows`. It speaks
   PipeSQL, through the `pipesql` crate, and bundles the tables and
   functions it ships with. Embedders, such as Perfetto, use this layer.
2. **`pipesql`: a front end.** It parses PipeSQL and builds kernel queries.
   Its registry of stages is how its syntax is extended. Another language
   would be another front end, on the same kernel.
3. **Kernel extensions.** `parquet`, `operators`, `regex`, `file` and `s3`
   add tables, functions and I/O through the kernel's extension points.
4. **The kernel.** Compiled data to data transformations, designed for
   SQL's semantics but tied to no language.

## The kernel

From outside, a query goes through three stages:

1. **A query** is what a front end builds: scans, filters, joins and
   aggregates over expressions, and any custom operations. It describes
   what's wanted, not how to run it.
2. **A compiled query** is what the kernel makes of it, once. It is opaque
   but for its result columns and their types. How it was compiled, such as
   optimizing, planning joins and lowering to pipelines, is not part of the
   API, so it can change without breaking anyone.
3. **Rows** are one run of a compiled query. Each run has its own memory
   and budget. `next` returns the next batch, borrowed until the next call.

```rust
let mut query = Query::new(&runtime);
let scan = query.scan(table, &columns)?;
let kept = query.filter(scan, condition)?;
query.output(kept, &outputs)?;
let compiled = query.compile(&runtime)?;
let mut rows = compiled.run(&runtime)?;
while let Some(batch) = rows.next()? {
    // Read the batch's columns.
}
```

### Runtime

What runs draw on, shared by every query: the allocator memory comes from,
the budget each run may take, and in time where spills go, local disk or
object storage, and how I/O is done. A run allocates within its own budget,
from the runtime's allocator; over budget, it spills, and with nothing left
to spill, it fails.

### Errors

Building and compiling a query can fail, such as on a type error or a
column the query doesn't have. Each thing added to a query returns a handle,
and a compile error names the handle it's about, so a front end can point at
the text it came from. PipeSQL turns handles back into spans of the query.

### Extension points

Everything a front end or an engine plugs in goes through these:

- **Tables:** `Scannable`, which reads columns in batches, and `Catalog`,
  which finds tables by name.
- **Functions:** `Function`, a scalar function of columns.
- **Operations:** a custom operation describes itself as data: the columns
  it reads and outputs, and the conditions it holds. It also gives the steps
  it runs as. The kernel decides every rewrite from these descriptions; an
  operation never calls the optimizer.
- **Syntax:** a front end's own, such as PipeSQL's registry of stages.

Each extension point is a table of functions, one for each type that
implements it, so it can become `#[repr(C)]` with `extern "C"` functions,
for extensions in other languages.

### Data

`DataType`, `Buffer`, `ColumnView` and the batches `Rows` returns: what
extensions read and write, and what results are made of.

### Private

Everything else is `pub(crate)`: logical and physical plans, the optimizer,
the join planner, pipelines, steps, lowering, `OutputProperties`, `Env`,
containers and arenas.

## The database

```rust
let database = Database::new(Heap, query_budget)?;
let mut connection = database.connect()?;
connection.register("hits", ParquetTable::open(&database, files)?)?;
let statement = connection.prepare("FROM hits |> WHERE CounterID = 62")?;
let mut rows = statement.execute()?;
while let Some(batch) = rows.next()? {
    // Read the batch's columns.
}
```

- **`Database`** owns what its connections share: the runtime, the tables
  and functions it ships with, and in time its catalog and storage. Until it
  has storage, it holds no tables of its own.
- **`Connection`** holds what one user's session needs: its settings, and
  tables registered on it, which live only as long as it does. In time, it
  holds its transaction.
- **`Statement`** is a PipeSQL query, compiled once, to run any number of
  times.
- **`Rows`** are one run of a statement: the kernel's `Rows`.

DDL, storage and transactions belong to this layer. They come later; the
types are named so they fit without renaming anything.

## Lifetimes and threads

- A runtime, and a database, are shared, by any number of threads.
- A connection belongs to one thread at a time.
- A compiled query, and a statement, can be run by any number of threads at
  once, each run with its own memory.
- Rows belong to one thread, and borrow what they run.

## C ABI

Each public type is an opaque handle: `pipit_database`, `pipit_connection`,
`pipit_statement` and `pipit_rows` for the database, and `pipit_runtime`,
`pipit_query`, `pipit_compiled` and `pipit_rows` for the kernel. A batch's
columns are read by index. An error is a code and a message. Extensions
register their tables of functions.

## Open questions

- **Features.** Whether `pipitdb` takes features, such as `pipesql`,
  `parquet`, `file` and `s3`, so a small embedding takes only what it needs.
- **Results.** Whether batches are read through typed getters only, or can
  also be exported as Arrow.
- **Brands.** Handles into a query's or a run's arena can't be made outside
  it, and those of a query can't be used in a run. Branding each arena with
  a lifetime would make using another arena's handles impossible too, at the
  cost of a lifetime on the types that hold them. It's left out until it's
  needed.

## Migration

1. The kernel's `Runtime`, `Query`, compiled query and `Rows`, with every
   caller moved onto them.
2. Everything else in the kernel made `pub(crate)`. Tests that use its
   internals move into the kernel, or become PipeSQL tests.
3. The `pipitdb` crate: `Database`, `Connection`, `Statement` and `Rows`,
   over the kernel and `pipesql`.
4. Arenas for a query's and a run's memory, created only by `Query` and
   `run`.
