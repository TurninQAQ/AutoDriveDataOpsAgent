from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


class AirflowGateway:
    """Low-level boundary for Airflow CLI and REST transport.

    It deliberately contains no task/queue business rules. Higher-level runtime
    orchestration remains in task_manager during V0.1.
    """

    def __init__(
        self,
        airflow_bin: str,
        airflow_home: str,
        run_home: str | Path,
        api_timeout_sec: int = 10,
    ):
        self.airflow_bin = airflow_bin
        self.airflow_home = airflow_home
        self.run_home = Path(run_home)
        self.api_timeout_sec = int(api_timeout_sec)

    def run_cli(self, args, check=True, extra_env=None):
        cmd = [self.airflow_bin] + list(args)
        env = os.environ.copy()
        env["HOME"] = str(self.run_home)
        env["AIRFLOW_HOME"] = self.airflow_home
        if extra_env:
            env.update({str(key): str(value) for key, value in extra_env.items()})
        for key in (
            "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN",
            "AIRFLOW__CORE__SQL_ALCHEMY_CONN",
        ):
            if not env.get(key):
                env.pop(key, None)
        result = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if check and result.returncode != 0:
            raise RuntimeError(
                "Command failed: {}\nstdout:\n{}\nstderr:\n{}".format(
                    " ".join(cmd), result.stdout, result.stderr
                )
            )
        return result

    def request_json(self, method, api_base, path, payload=None, token=None, ok=(200, 201, 204)):
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(
            f"{api_base.rstrip('/')}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=self.api_timeout_sec) as resp:
                raw = resp.read()
                if resp.status not in ok:
                    raise RuntimeError(f"{method} {path} returned HTTP {resp.status}: {raw!r}")
                if not raw:
                    return {}
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed HTTP {exc.code}: {raw}") from exc
