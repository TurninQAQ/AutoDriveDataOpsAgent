# AutoDriveDataOpsAgent V2.0.0 Operations Runbook

## Release profile

This runbook operates the `single-node-simulated` profile. The Agent,
LangGraph runtime, V2 safety/runtime, gateway, RAG, persistence, and Qwen
Provider are real. The stage and GPU environments are intentionally
simulated.

```text
AUTODRIVE_ENVIRONMENT=single-node-simulated
PLATFORM_STAGE_RUNTIME=mock
PLATFORM_GPU_RUNTIME=simulated
AUTODRIVE_GATEWAY_BACKEND=in_process
AUTODRIVE_PLATFORM_ENDPOINT=http://127.0.0.1:8765/mcp
AUTODRIVE_PROVIDER=qwen
AUTODRIVE_MODEL=qwen3.7-plus-2026-05-26
AUTODRIVE_PROVIDER_STRUCTURED_MODE=json_schema
AUTODRIVE_PROVIDER_THINKING_MODE=auto
PLATFORM_RAG_EMBED_PROVIDER=local
```

`auto` selects `enable_thinking=false` for the validated Qwen 3.7 Plus
strict JSON-Schema Agent path. The provider secret is supplied at runtime
through `DASHSCOPE_API_KEY`; its value is never part of this profile.

## Paths and topology

The source tree is:

```text
/home/ubuntu/project/AutoDriveDataOpsAgent
```

The external runtime root is:

```text
/home/ubuntu/project/autodrive_dataops_runtimev2
```

The canonical launcher is:

```text
/home/ubuntu/project/autodrive_dataops_runtimev2/bin/start-v2-gateway
```

It loads the non-secret profile from
`autodrive_dataops_runtimev2/config/single_node_simulated.env`, changes to
the V2 source directory, and starts exactly one localhost gateway on port
`8765`. Port `8766` and the deleted `deploy_ci_cloud_agent` tree are not part
of this release.

Airflow/PostgreSQL are backing services supplied by the existing
`autodrive_dataops_runtime` installation. Agent state is persisted separately
in SQLite at:

```text
/home/ubuntu/project/autodrive_dataops_runtimev2/state/autodrive.sqlite3
```

The runtime is single-instance. Do not run two V2 processes against the same
SQLite state or place the state on a network filesystem.

## Secret handling

The credential file is outside the repository:

```text
/home/ubuntu/project/auth/ali.api
```

It must remain mode `600`, owned by the service account, and must never be
printed, copied into the repository, passed on a command line, logged, put in
SQLite, or included in a release artifact. Load it only into the service
process environment using a protected operator shell. Disable shell tracing
before loading it.

## Start, stop, restart, and status

Start backing services when they are not already healthy:

```bash
cd /home/ubuntu/project/autodrive_dataops_runtime
bin/platform start
```

Start the canonical V2 gateway:

```bash
/home/ubuntu/project/autodrive_dataops_runtimev2/bin/start-v2-gateway
```

The launcher is a foreground process. A deployment supervisor may own its
stdout/stderr; do not start a second copy. Find the exact process with:

```bash
pgrep -af 'deploy_ci_cloud_agentv2.platform.http_gateway'
ss -lntp | grep ':8765'
```

Stop only the PID returned by the exact gateway command, then stop backing
services when the whole single-node profile is being shut down:

```bash
kill <gateway-pid>
cd /home/ubuntu/project/autodrive_dataops_runtime
bin/platform stop
```

For a clean restart, stop the exact gateway PID, verify port `8765` is free,
run `bin/platform start`, and invoke the canonical V2 launcher again. Do not
use the legacy `deploy_ci_cloud_agent` path or a temporary `8766` bridge.

## Health and readiness

Health is liveness: it proves the V2 host can answer a local health request.
Readiness is configuration and local-state readiness: it checks the provider
configuration, referenced secret presence, platform endpoint, sealed tool
catalog, and writable SQLite state. Neither command performs a platform
WRITE.

```bash
curl --fail --silent http://127.0.0.1:8765/health
autodrive-agent health
# Load the secret only in the current protected process, then:
autodrive-agent ready
```

Also check backing services when required by the deployment:

```bash
curl --fail --silent http://127.0.0.1:8080/api/v2/monitor/health
curl --fail --silent http://127.0.0.1:8081/execution/health
pg_isready -h 127.0.0.1 -p 5432
```

## Logs and monitoring

The current gateway launcher is foreground-oriented and writes stdout/stderr
to its owning supervisor/terminal. Airflow owns its logs under the
`autodrive_dataops_runtime` installation. The V2 runtime persists structured
events in SQLite; event types expose provider failures, decision rejection,
budget exhaustion, approval interruption, mutation lifecycle,
`OUTCOME_UNKNOWN`, reconciliation, and CompletionGate results.

For a foreground gateway, inspect its supervisor output. For Airflow:

```bash
tail -n 200 /home/ubuntu/project/autodrive_dataops_runtime/airflow/logs/scheduler.log
tail -n 200 /home/ubuntu/project/autodrive_dataops_runtime/airflow/logs/api_server.log
tail -n 200 /home/ubuntu/project/autodrive_dataops_runtime/airflow/logs/execution_api_server.log
```

There is no separate metrics/alerting service in v2.0.0. Health/readiness,
SQLite event inspection, provider error typing, and disk checks are
implemented operational signals. Alert thresholds for gateway failure,
repeated provider failures/timeouts, schema rejection, disk pressure, SQLite
unwritability, Airflow/PostgreSQL failure, unreconciled unknown outcomes, and
budget exhaustion are documented signals for the host supervisor/operator.
Configure bounded supervisor log retention if the foreground process is
daemonized; do not allow unbounded file logging.

## SQLite backup and restore

The database uses WAL mode. Use SQLite's online backup API while the service
is running; never make a backup by copying only the main database file.

Example safe backup procedure (the destination must be outside the source
tree and retained according to the host policy):

```bash
python - <<'PY'
import sqlite3
source = '/home/ubuntu/project/autodrive_dataops_runtimev2/state/autodrive.sqlite3'
destination = '/home/ubuntu/project/autodrive_dataops_runtimev2/backups/autodrive.sqlite3'
with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
    src.backup(dst)
PY
```

Validate a backup before retention:

```bash
python - <<'PY'
import sqlite3
path = '/home/ubuntu/project/autodrive_dataops_runtimev2/backups/autodrive.sqlite3'
with sqlite3.connect(path) as db:
    assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    print([row[0] for row in db.execute("select name from sqlite_master where type='table' order by name")])
PY
```

Restore is a controlled service operation: stop V2, preserve the current
database, place a validated backup at the configured state path, start V2,
then run health/readiness and a READ-only smoke. Never manually edit approval,
checkpoint, event, or execution-claim rows. The release restore drill uses a
temporary isolated copy and must not overwrite the live database.

The simulated Airflow/PostgreSQL state is backing-service state, not the V2
Agent checkpoint database. A PostgreSQL/Airflow backup is not required to
restore the Agent state in this release; if the simulated platform state must
be preserved for a separate operational reason, use the existing PostgreSQL
and Airflow backup procedures rather than copying live files.

## OUTCOME_UNKNOWN and incidents

Never replay a consumed WRITE after `OUTCOME_UNKNOWN`. Inspect the durable
event/checkpoint state, run the READ-only reconciliation command for the
thread, and create a new transaction and approval for any later mutation.

For provider timeout or HTTP failure, inspect the typed provider event and
network/service health before changing configuration. For gateway failure,
check the exact process and port, then restart the canonical launcher. For
SQLite failure, stop unsafe execution and restore only from a validated
backup. For `BUDGET_EXHAUSTED`, inspect the trajectory and evidence rather
than automatically increasing budgets.

## Upgrade and rollback

Upgrade:

1. Record the current tag and wheel checksum.
2. Create and validate an online SQLite backup.
3. Stop the V2 gateway.
4. Install the new wheel and apply only the documented non-secret profile.
5. Start the canonical launcher.
6. Run health, ready, gateway health, and a READ-only smoke.
7. Confirm SQLite persistence before declaring success.

Rollback to v2.0.0 uses the retained `v2.0.0` wheel/tag, the same stop/
install/start/health/ready/READ sequence, and configuration compatibility
checks. Application rollback and database rollback are separate decisions;
never automatically roll SQLite backwards unless an explicitly compatible
schema migration requires it.

## Retention and limitations

Retain the v2.0.0 wheel, SHA-256, release manifest, tag, release notes,
deployment documentation, and validated SQLite backups according to the host
retention policy. Logs and transient test artifacts may be rotated after
evidence is archived. The canonical corpus, 30-case JSONL, and optional dense
sidecar are release assets; the sidecar is reusable and should not be rebuilt
on every startup.

This product does not claim physical multi-GPU, multi-node, or non-mock
AutoDrive validation. Those are `OUT_OF_SCOPE`; the supported release target
is the single-node simulated environment described above.
