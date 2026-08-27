from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class AirflowReadGateway:
    """Read-only Airflow 3 REST API gateway.

    This gateway intentionally exposes only GET-style runtime evidence plus token
    acquisition. It never mutates DagRun or TaskInstance state.
    """

    def __init__(
        self,
        api_base: str,
        user: str = "admin",
        password: str = "",
        token: str = "",
        password_file: str | Path | None = None,
        timeout_sec: int = 10,
    ):
        self.api_base = api_base.rstrip("/")
        self.user = user
        self.password = password
        self._token = token
        self.password_file = Path(password_file) if password_file else None
        self.timeout_sec = int(timeout_sec)

    @staticmethod
    def _quote(value: Any) -> str:
        return urllib.parse.quote(str(value), safe="")

    def _password(self) -> str:
        if self.password:
            return self.password
        if not self.password_file or not self.password_file.is_file():
            return ""
        try:
            data = json.loads(self.password_file.read_text(encoding="utf-8"))
        except Exception:
            return ""
        return str(data.get(self.user) or "") if isinstance(data, dict) else ""

    def token(self) -> str:
        if self._token:
            return self._token
        password = self._password()
        if not password:
            raise RuntimeError(
                "Airflow API credential unavailable; set AIRFLOW_API_TOKEN or AIRFLOW_API_PASSWORD"
            )
        data = self._request_json(
            "POST",
            "/auth/token",
            payload={"username": self.user, "password": password},
            use_auth=False,
        )
        token = str(data.get("access_token") or "")
        if not token:
            raise RuntimeError("Airflow token response did not contain access_token")
        self._token = token
        return token

    def _request_raw(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        use_auth: bool = True,
    ) -> tuple[bytes, str]:
        body = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if use_auth:
            headers["Authorization"] = f"Bearer {self.token()}"
        req = urllib.request.Request(
            f"{self.api_base}{path}", data=body, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as response:
                return response.read(), str(response.headers.get("content-type") or "")
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed HTTP {exc.code}: {raw}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"{method} {path} failed: {exc}") from exc

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        use_auth: bool = True,
    ) -> dict[str, Any]:
        raw, _ = self._request_raw(method, path, payload=payload, use_auth=use_auth)
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{method} {path} did not return JSON") from exc
        if not isinstance(data, dict):
            raise RuntimeError(f"{method} {path} returned non-object JSON")
        return data

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "/api/v2/monitor/health")

    def get_dag(self, dag_id: str) -> dict[str, Any]:
        """Return DAG metadata. A missing DAG is surfaced as the gateway HTTP error."""
        return self._request_json("GET", f"/api/v2/dags/{self._quote(dag_id)}")

    def list_dag_runs(self, dag_id: str, limit: int = 100) -> list[dict[str, Any]]:
        page_size = min(max(1, int(limit)), 100)
        offset = 0
        runs: list[dict[str, Any]] = []
        while len(runs) < limit:
            path = (
                f"/api/v2/dags/{self._quote(dag_id)}/dagRuns"
                f"?limit={page_size}&offset={offset}"
            )
            data = self._request_json("GET", path)
            batch = data.get("dag_runs") or []
            if not isinstance(batch, list):
                batch = []
            runs.extend(item for item in batch if isinstance(item, dict))
            total = data.get("total_entries")
            if len(batch) < page_size or (total is not None and len(runs) >= int(total)):
                break
            offset += page_size
        def sort_key(item: dict[str, Any]):
            return str(
                item.get("run_after")
                or item.get("start_date")
                or item.get("logical_date")
                or item.get("dag_run_id")
                or item.get("run_id")
                or ""
            )
        runs.sort(key=sort_key, reverse=True)
        return runs[:limit]

    def list_task_instances(
        self, dag_id: str, run_id: str, limit: int = 200
    ) -> list[dict[str, Any]]:
        page_size = min(max(1, int(limit)), 100)
        offset = 0
        items: list[dict[str, Any]] = []
        while len(items) < limit:
            path = (
                f"/api/v2/dags/{self._quote(dag_id)}/dagRuns/{self._quote(run_id)}"
                f"/taskInstances?limit={page_size}&offset={offset}"
            )
            data = self._request_json("GET", path)
            batch = data.get("task_instances") or []
            if not isinstance(batch, list):
                batch = []
            items.extend(item for item in batch if isinstance(item, dict))
            total = data.get("total_entries")
            if len(batch) < page_size or (total is not None and len(items) >= int(total)):
                break
            offset += page_size
        return items[:limit]

    def get_task_log(
        self,
        dag_id: str,
        run_id: str,
        task_id: str,
        try_number: int,
        map_index: int = -1,
    ) -> str:
        query = ""
        if int(map_index) != -1:
            query = f"?map_index={int(map_index)}"
        path = (
            f"/api/v2/dags/{self._quote(dag_id)}/dagRuns/{self._quote(run_id)}"
            f"/taskInstances/{self._quote(task_id)}/logs/{int(try_number)}{query}"
        )
        raw, content_type = self._request_raw("GET", path)
        text = raw.decode("utf-8", errors="replace")
        if "json" in content_type.lower():
            try:
                data = json.loads(text)
                if isinstance(data, dict):
                    for key in ("content", "log", "message"):
                        if key in data:
                            value = data[key]
                            return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
                if isinstance(data, str):
                    return data
            except json.JSONDecodeError:
                pass
        return text
