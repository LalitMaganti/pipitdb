//! Running queries.
//!
//! A front end, such as PipeSQL, builds a [`Query`]: what to compute, from
//! which tables. A [`Session`] [prepares](Session::prepare) it once,
//! choosing how to compute it. The [`Prepared`] query then
//! [runs](Prepared::run) as many times as wanted, and each run's result
//! comes out as [`Rows`], a batch at a time.
//!
//! ```
//! use pipit_kernel::env::Env;
//! use pipit_kernel::error::Error;
//! use pipit_kernel::session::{Query, Session};
//!
//! /// Runs `query` twice, and returns how many rows each run made.
//! fn count_twice<'e>(env: &'e Env, query: Query<'e>) -> Result<[u64; 2], Error> {
//!     let mut session = Session::new(env);
//!     let prepared = session.prepare(query)?;
//!     for column in prepared.columns() {
//!         println!("{}", prepared.name(*column));
//!     }
//!     let mut counts = [0; 2];
//!     for count in &mut counts {
//!         let mut rows = prepared.run(&mut session)?;
//!         while let Some(batch) = rows.next()? {
//!             *count += u64::from(batch.selection().len());
//!         }
//!     }
//!     Ok(counts)
//! }
//! ```
//!
//! # Memory
//!
//! Everything a session, its queries and their runs allocate comes from
//! the [`Env`] they're made with, and counts against its limit. They borrow
//! it, so it outlives them all.
//!
//! # One run at a time
//!
//! A session prepares any number of queries, but runs one at a time: a run
//! borrows the session mutably until its [`Rows`] are dropped.
//!
//! # Sessions
//!
//! A prepared query runs only in the session that prepared it. Running it
//! in another fails with [`Error::WrongSession`].

use core::marker::PhantomData;
use core::sync::atomic::{AtomicU32, Ordering};

use crate::allocator::AllocError;
use crate::containers::Array;
use crate::env::Env;
use crate::error::Error;
use crate::lower::{OutputColumn, PhysicalPlan};
use crate::pipeline::Execution;
use crate::plan::LogicalPlan;
use crate::row_batch::RowBatch;

/// The next session's id.
static SESSIONS: AtomicU32 = AtomicU32::new(0);

/// Prepares queries and runs them, one at a time: what a connection to a
/// database runs its queries in.
pub struct Session<'e> {
    /// Where queries' and runs' memory comes from.
    env: &'e Env,
    /// Which session it is, so a query prepared in another isn't run in it.
    id: u32,
    /// The memory each run lays its state out in, kept from run to run and
    /// grown when a run needs more.
    memory: Array<u8>,
}

impl<'e> Session<'e> {
    /// A session whose queries and runs take their memory from `env`.
    pub fn new(env: &'e Env) -> Session<'e> {
        let memory = Array::empty(env);
        Session { env, id: SESSIONS.fetch_add(1, Ordering::Relaxed), memory }
    }

    /// Chooses how to compute `query`, once, so it can run any number of
    /// times in this session.
    ///
    /// # Errors
    ///
    /// [`Error::TooManyColumns`], [`Error::WrongType`] or [`Error::TooDeep`]
    /// if the query can't be computed as built, and [`Error::OutOfMemory`]
    /// if preparing it takes more memory than the session's `Env` has left.
    pub fn prepare(&self, mut query: Query<'e>) -> Result<Prepared<'e>, Error> {
        let plan = PhysicalPlan::from_logical(self.env, &mut query.plan)?;
        Ok(Prepared { plan, session: self.id })
    }
}

/// What a front end asks to compute: a query, built but not yet prepared.
pub struct Query<'e> {
    plan: LogicalPlan<'e>,
}

impl<'e> Query<'e> {
    /// The query `build` makes, with memory from `env`.
    pub fn new<E: From<AllocError>>(
        env: &'e Env,
        build: impl FnOnce(&'e Env) -> Result<LogicalPlan<'e>, E>,
    ) -> Result<Query<'e>, E> {
        Ok(Query { plan: build(env)? })
    }

    /// The plan the front end built.
    pub fn plan(&self) -> &LogicalPlan<'e> {
        &self.plan
    }
}

/// A query, prepared to run in the session that prepared it. How it runs
/// is the session's choice; what it makes is its [columns](Prepared::columns).
pub struct Prepared<'e> {
    /// The query, optimized and lowered.
    plan: PhysicalPlan<'e>,
    /// The session it was prepared in, which is the only one it runs in.
    session: u32,
}

impl Prepared<'_> {
    /// Starts a run of it in `session`, which runs nothing else until the
    /// [`Rows`] are dropped.
    ///
    /// # Errors
    ///
    /// [`Error::WrongSession`] if it was prepared in another session, and
    /// [`Error::OutOfMemory`] if starting the run takes more memory than
    /// the session's `Env` has left.
    pub fn run<'r>(&'r self, session: &'r mut Session<'_>) -> Result<Rows<'r>, Error> {
        if session.id != self.session {
            return Err(Error::WrongSession);
        }
        let execution = self.plan.start_in(session.env, &mut session.memory)?;
        Ok(Rows { execution, session: PhantomData })
    }

    /// The columns its runs make, in order. Each says where it is in the
    /// batches [`Rows::next`] lends.
    pub fn columns(&self) -> &[OutputColumn] {
        self.plan.columns()
    }

    /// The name of `column`.
    pub fn name(&self, column: OutputColumn) -> &str {
        self.plan.name(column)
    }
}

/// One run of a prepared query: its result, a batch at a time.
pub struct Rows<'r> {
    /// The run's state, and the batch `next` lends.
    execution: Execution<'r>,
    /// The session, borrowed mutably for as long as the run lasts.
    session: PhantomData<&'r mut ()>,
}

impl<'r> Rows<'r> {
    /// The next batch of the result, or `None` once the run has made them
    /// all. The batch is lent until the next call.
    ///
    /// Only the rows its [selection](RowBatch::selection) keeps are part of
    /// the result. Its columns are where [`Prepared::columns`] says.
    ///
    /// # Errors
    ///
    /// Why the run failed, such as [`Error::OutOfMemory`] if it takes more
    /// memory than the session's `Env` has left. A failed run fails every
    /// call after.
    #[expect(clippy::should_implement_trait, reason = "it lends a batch, which `Iterator` can't")]
    pub fn next(&mut self) -> Result<Option<&RowBatch<'r>>, Error> {
        self.execution.next_batch()
    }
}
