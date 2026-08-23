# AutoDriveDataOpsAgent V2 deployment

The first production target is a single active runtime instance using a local
SQLite database. SQLite WAL/checkpoint semantics are not an active-active or
network-filesystem coordination protocol.

Build and run the image from the repository root:

```bash
docker build -t autodrive-dataops-agent-v2 .
docker run --rm --read-only --tmpfs /tmp \
  -v /srv/autodrive-dataops:/var/lib/autodrive-dataops \
  -e DASHSCOPE_API_KEY \
  autodrive-dataops-agent-v2 ready
```

The runtime volume must be local and writable. Run exactly one active process
against one SQLite database. Do not deploy multiple replicas, NFS-backed state,
or multiple workers until the persistence architecture is replaced in a later
phase.

The host CLI exposes `health`, `ready`, `invoke`, `pending`, `approve`,
`reject`, and `reconcile`. It never calls a WRITE tool directly; approval and
all mutation safety checks remain inside the Runtime API.
