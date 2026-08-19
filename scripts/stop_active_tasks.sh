#!/usr/bin/env bash
set -euo pipefail

API_BASE="${AIRFLOW_API_BASE:-http://127.0.0.1:${AIRFLOW_PORT:-8080}}"
DAG_ID="batch_pipeline_universal"
SCHEDULER_DAG_ID="scheduler_all"
SINCE=""
YES=0
STOP_CONTAINERS=1
IMAGE_PREFIXES=(
  "172.16.201.100:5000/data_parser:"
  "172.16.201.100:5000/offline_mapping:"
  "172.16.201.100:5000/pointcloud_coloration:"
  "172.16.201.100:5000/sam31:"
  "172.16.201.100:5000/label_od:"
  "172.16.201.100:5000/label_occ:"
)

usage() {
  cat <<'EOF'
Usage:
  stop_active_tasks.sh [options] --yes

Stops active Airflow work for the batch pipeline:
  1. pauses scheduler_all and batch_pipeline_universal
  2. marks queued/running/scheduled DagRuns as failed
  3. tries to stop matching docker containers

Options:
  --dag-id ID               Target pipeline DAG. Default: batch_pipeline_universal
  --scheduler-dag-id ID     Scheduler DAG to pause/stop first. Default: scheduler_all
  --since TIME              Only stop DagRuns queued/logical after TIME.
                            Example: 2026-07-01T17:30:00+08:00
  --api-base URL            Airflow API base. Default: http://127.0.0.1:\${AIRFLOW_PORT:-8080}
  --no-stop-containers      Do not stop docker containers; only stop Airflow DagRuns.
  --image-prefix PREFIX     Extra docker image prefix to stop. Can be repeated.
  --yes                     Required to make changes. Without it, dry-run only.
  -h, --help                Show this help.

Auth:
  Uses AIRFLOW_API_TOKEN if set.
  Otherwise uses AIRFLOW_API_USER/AIRFLOW_API_PASSWORD.
  If password is not set, it tries /home/cidi/airflow/simple_auth_manager_passwords.json.generated.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dag-id)
      DAG_ID="${2:?missing value for --dag-id}"
      shift 2
      ;;
    --scheduler-dag-id)
      SCHEDULER_DAG_ID="${2:?missing value for --scheduler-dag-id}"
      shift 2
      ;;
    --since)
      SINCE="${2:?missing value for --since}"
      shift 2
      ;;
    --api-base)
      API_BASE="${2:?missing value for --api-base}"
      shift 2
      ;;
    --no-stop-containers)
      STOP_CONTAINERS=0
      shift
      ;;
    --image-prefix)
      IMAGE_PREFIXES+=("${2:?missing value for --image-prefix}")
      shift 2
      ;;
    --yes)
      YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

python3 - "$API_BASE" "$DAG_ID" "$SCHEDULER_DAG_ID" "$SINCE" "$YES" "$STOP_CONTAINERS" "${IMAGE_PREFIXES[@]}" <<'PY'
import json
import os
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

api_base = sys.argv[1].rstrip("/")
dag_id = sys.argv[2]
scheduler_dag_id = sys.argv[3]
since_raw = sys.argv[4]
apply_changes = sys.argv[5] == "1"
stop_containers = sys.argv[6] == "1"
image_prefixes = sys.argv[7:]

ACTIVE_STATES = {"queued", "running", "scheduled"}
PASSWORD_FILE = os.environ.get(
    "AIRFLOW_PASSWORD_FILE",
    "/home/cidi/airflow/simple_auth_manager_passwords.json.generated",
)


def parse_dt(value):
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    value = value.replace(" ", "T")
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    if "T" not in value:
        value += "T00:00:00"
    time_part = value.split("T", 1)[1]
    if "+" not in time_part and "-" not in time_part:
        value += "+00:00"
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def load_password(user):
    password = os.environ.get("AIRFLOW_API_PASSWORD")
    if password:
        return password
    try:
        with open(PASSWORD_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get(user)
    except FileNotFoundError:
        return None


def request_json(method, path, payload=None, token=None, ok=(200, 201, 204)):
    body = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(
        f"{api_base}{path}",
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            if resp.status not in ok:
                raise RuntimeError(f"{method} {path} returned HTTP {resp.status}: {raw!r}")
            if not raw:
                return {}
            return json.loads(raw.decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {path} failed HTTP {exc.code}: {raw}") from exc


def get_token():
    token = os.environ.get("AIRFLOW_API_TOKEN")
    if token:
        return token
    user = os.environ.get("AIRFLOW_API_USER", "chang.fy")
    password = load_password(user)
    if not password:
        raise RuntimeError(
            "Airflow API password not found. Set AIRFLOW_API_PASSWORD or AIRFLOW_API_TOKEN."
        )
    data = request_json(
        "POST",
        "/auth/token",
        payload={"username": user, "password": password},
        token=None,
    )
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Airflow token response did not contain access_token")
    return token


def quote(value):
    return urllib.parse.quote(value, safe="")


def pause_dag(token, dag):
    if not dag:
        return
    print(f"[INFO] Pause DAG: {dag}")
    if apply_changes:
        request_json("PATCH", f"/api/v2/dags/{quote(dag)}", {"is_paused": True}, token)


def list_dag_runs(token, dag):
    runs = []
    limit = 100
    offset = 0
    while True:
        path = f"/api/v2/dags/{quote(dag)}/dagRuns?limit={limit}&offset={offset}"
        data = request_json("GET", path, token=token)
        batch = data.get("dag_runs", [])
        runs.extend(batch)
        total = data.get("total_entries")
        if total is None:
            if len(batch) < limit:
                break
        elif len(runs) >= int(total):
            break
        offset += limit
    return runs


def run_time(run):
    for key in ("queued_at", "start_date", "logical_date"):
        value = run.get(key)
        if value:
            try:
                return parse_dt(value)
            except Exception:
                pass
    return None


def filter_active_runs(runs, since_dt):
    selected = []
    for run in runs:
        state = (run.get("state") or "").lower()
        if state not in ACTIVE_STATES:
            continue
        if since_dt is not None:
            dt = run_time(run)
            if dt is None or dt < since_dt:
                continue
        selected.append(run)
    return selected


def dag_run_id(run):
    return run.get("dag_run_id") or run.get("run_id")


def fail_runs(token, dag, runs):
    if not runs:
        print(f"[INFO] No active DagRuns to stop for {dag}")
        return 0
    print(f"[INFO] Stop {len(runs)} active DagRuns for {dag}:")
    for run in runs[:20]:
        print(f"  - {dag_run_id(run)} state={run.get('state')}")
    if len(runs) > 20:
        print(f"  ... and {len(runs) - 20} more")

    if not apply_changes:
        return 0

    count = 0
    for run in runs:
        run_id = dag_run_id(run)
        if not run_id:
            print(f"[WARN] Skip DagRun without run id: {run}")
            continue
        request_json(
            "PATCH",
            f"/api/v2/dags/{quote(dag)}/dagRuns/{quote(run_id)}",
            {"state": "failed"},
            token,
        )
        count += 1
    return count


def stop_matching_containers():
    if not stop_containers:
        print("[INFO] Docker container stop disabled")
        return 0
    try:
        output = subprocess.check_output(
            ["docker", "ps", "--format", "{{.ID}} {{.Image}} {{.Names}}"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=30,
        )
    except Exception as exc:
        print(f"[WARN] Could not list docker containers: {exc}")
        return 0

    targets = []
    for line in output.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) < 2:
            continue
        container_id, image = parts[0], parts[1]
        if any(image.startswith(prefix) for prefix in image_prefixes):
            targets.append(container_id)

    if not targets:
        print("[INFO] No matching docker containers to stop")
        return 0

    print(f"[INFO] Stop {len(targets)} matching docker containers: {' '.join(targets)}")
    if not apply_changes:
        return 0

    try:
        subprocess.check_call(["docker", "stop"] + targets, timeout=120)
    except Exception as exc:
        print(f"[WARN] docker stop failed: {exc}")
        return 0
    return len(targets)


def main():
    since_dt = parse_dt(since_raw) if since_raw else None
    print(f"[INFO] Airflow API: {api_base}")
    print(f"[INFO] Target DAG: {dag_id}")
    print(f"[INFO] Scheduler DAG: {scheduler_dag_id}")
    print(f"[INFO] Since: {since_dt.isoformat() if since_dt else 'not set'}")
    print(f"[INFO] Mode: {'apply' if apply_changes else 'dry-run'}")

    token = get_token()

    if not apply_changes:
        print("[WARN] Dry-run only. Re-run with --yes to apply changes.")

    pause_dag(token, scheduler_dag_id)
    scheduler_runs = filter_active_runs(list_dag_runs(token, scheduler_dag_id), since_dt)
    stopped_scheduler = fail_runs(token, scheduler_dag_id, scheduler_runs)

    pause_dag(token, dag_id)
    target_runs = filter_active_runs(list_dag_runs(token, dag_id), since_dt)
    stopped_target = fail_runs(token, dag_id, target_runs)

    stopped_containers = stop_matching_containers()

    print("[INFO] Summary:")
    print(f"  scheduler_dagruns_failed={stopped_scheduler}")
    print(f"  target_dagruns_failed={stopped_target}")
    print(f"  containers_stopped={stopped_containers}")
    if apply_changes:
        print("[INFO] DAGs are left paused. Unpause manually before restarting.")


if __name__ == "__main__":
    main()
PY
