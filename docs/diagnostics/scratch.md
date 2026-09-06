# Scratch, checkpoints, and durable storage

This runbook implements the storage policy in the field guide and `DEC-067-RUNNER`.

## Tiers

- Use `/scratch/kinuv-$USER/<run_id>` for JIT caches, temporary arrays, staging files, and high-volume progress output.
- Use `${KINUV_RUN_ROOT}/<run_id>` on `/arc` for manifests, bounded logs, validated checkpoints, compact posterior products, gate reports, and required figures.
- Reference calibrated inputs by stable URI and checksum. Do not duplicate them into every run.
- Never use `$HOME` for production data or credentials embedded in manifests.

`KINUV_RUN_ROOT` and the run ID are deployment inputs. Code must not synthesize them from a target name or a developer-specific path.

## Worker startup

1. Create the scratch directory with restrictive permissions.
2. Set `TMPDIR`, `TMP`, `TEMP`, and compilation-cache variables before importing numerical backends.
3. Create the durable run directory and write the initial manifest atomically.
4. Record host, scheduler/session ID, environment, code revision, configuration checksums, input checksums, expected resources, and current state.

## Checkpoint protocol

1. Write a complete checkpoint to scratch using a temporary name.
2. Close it and validate shape, dtype, finiteness, configuration identity, and expected keys.
3. Copy to a temporary file in the durable checkpoint directory.
4. Flush and fsync the file, then fsync its parent directory.
5. Verify size or checksum and atomically rename to the final checkpoint name.
6. Update the manifest only after the durable checkpoint exists.

Independent chains and retries have distinct identifiers. Do not merge checkpoints from different code commits, configuration hashes, data hashes, or parameter charts.

## Logs and monitoring

Keep verbose sampler output on scratch. Periodically overwrite-copy a bounded worker log to durable storage and maintain a compact structured status record. Avoid append loops, repeated full scheduler dumps, and per-sample writes on network storage.

Record explicit states such as `RUNNING`, `CHECKPOINTED`, `FAILED`, `INTERRUPTED`, `UNMIXED`, `UNCALIBRATED`, `VERIFIED`, and `PROMOTED`. A scheduler's success state proves process exit only.

## Exit and cleanup

On success or controlled failure, promote the final valid checkpoint, bounded logs, environment record, and failure or completion reason. Verify durable artifacts before deleting scratch. Interrupted or invalid partial files retain a non-promotion state and are never treated as complete draws.

Archive superseded durable runs only after their scientific conclusions and checksums are preserved. Never overwrite a promoted run directory.
