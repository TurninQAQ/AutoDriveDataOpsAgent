# AutoDriveDataOpsAgent V3.9 deployment

V3.9 targets one active process on a single node with local SQLite state. It is not an active-active or NFS coordination design.

```bash
docker build -t autodrive-dataops-agent:v3.9 .
docker run --rm -p 8080:8080 \
  -v /srv/autodrive-dataops:/var/lib/autodrive-dataops \
  -e QWEN_API_KEY \
  autodrive-dataops-agent:v3.9
```

The default container command is `autodrive-agent serve`. Use `autodrive-agent mcp-serve` when exposing the MCP host separately. Keep the runtime volume writable by UID 10001 and run a single active process against a given SQLite database.
