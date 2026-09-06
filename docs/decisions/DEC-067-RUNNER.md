---
id: DEC-067-RUNNER
status: accepted
generation: 5
date: 2026-08-30
amended: 2026-09-06
owner: senior-registrar
---
# Asynchronous production execution and storage

**Question:** How are long production jobs executed and persisted without coupling the engine to a target or site path?

## Answer

Jobs expected to outlive an interactive turn run asynchronously on the configured batch platform. Resource requests come from campaign configuration and a recorded preflight estimate. Hardware is selected from measured behavior of the exact production kernel; no CPU or GPU class is assumed globally.

Each run has a unique target-neutral ID and receives resolved configuration paths through CLI arguments or environment variables. Target IDs are manifest metadata. Production source must not encode target names, site usernames, project paths, resource sizes, chain counts, or convergence thresholds.

High-frequency I/O uses node-local `/scratch/kinuv-$USER/<run_id>`. Durable products use `${KINUV_RUN_ROOT}/<run_id>` on the project `/arc` volume. `$HOME` is not a production data tier.

Scratch contains JIT caches, temporary arrays, verbose progress streams, and staging files. Durable storage contains the run manifest, bounded logs, validated checkpoints, compact draws, posterior summaries, gate reports, and required figures. Raw inputs, duplicated visibility tables, JIT caches, and unbounded platform polling dumps are not copied into each run.

Checkpoint promotion is scratch-first and atomic: close, validate, copy to a temporary durable path, fsync, verify, and rename. Workers preserve the last valid checkpoint and failure record on controlled exit or termination. Platform logs are sampled into bounded job-owned records before provider expiry; unless deployment evidence guarantees longer retention, snapshot them within 1 hour.

Completion is a manifest state, not the existence of a file. Successful computation may still end as `UNMIXED`, `UNCALIBRATED`, or `REJECTED`. Only a verified dossier with Registrar and Consultant sign-off becomes `PROMOTED`.

Run names, launch commands, environments, resource profiles, retry rules, sampler settings, and numeric convergence gates belong in deployment or campaign configuration. Campaign-specific runbooks may record historical commands but do not redefine this decision.
