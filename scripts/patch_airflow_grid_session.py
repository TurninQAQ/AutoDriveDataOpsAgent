#!/usr/bin/env python3
"""Patch Airflow 3.2.0 grid streaming session leak.

Airflow 3.2.0's /ui/grid/ti_summaries endpoint returns a StreamingResponse
whose generator reuses the request-scoped SessionDep after the route function
has returned. The session can reopen a DB connection outside FastAPI's cleanup
scope, leaving PostgreSQL connections idle in transaction after heavy UI grid
refreshes.

This patch makes the route materialize NDJSON rows inside an explicit
create_session() context and only starts streaming after the DB session is
closed.
"""

from __future__ import annotations

import argparse
import importlib.util
import py_compile
import stat
from pathlib import Path


PATCH_MARKER = '# DEPLOY_CI_CLOUD_GRID_SESSION_PATCH = "2026-08-04"'

OLD_IMPORT = "from airflow.models.taskinstancehistory import TaskInstanceHistory\n"
NEW_IMPORT = (
    "from airflow.models.taskinstancehistory import TaskInstanceHistory\n"
    "from airflow.utils.session import create_session\n"
)

OLD_FUNCTION = '''def get_grid_ti_summaries_stream(
    dag_id: str,
    session: SessionDep,
    run_ids: Annotated[list[str] | None, Query()] = None,
) -> StreamingResponse:
    """
    Stream TI summaries for multiple Dag runs as NDJSON (one JSON line per run).

    Each line is a serialized ``GridTISummaries`` object emitted as soon as that
    run's task instances have been processed, so the client can render columns
    progressively without waiting for all runs to complete.

    The serialized Dag structure is loaded once and reused for all runs that
    share the same ``dag_version_id``, avoiding repeated deserialization.
    """

    def _generate() -> Generator[str, None, None]:
        serdag_cache: dict = {}
        for run_id in run_ids or []:
            tis = session.execute(
                select(
                    TaskInstance.task_id,
                    TaskInstance.state,
                    TaskInstance.dag_version_id,
                    TaskInstance.start_date,
                    TaskInstance.end_date,
                    DagVersion.version_number,
                )
                .outerjoin(DagVersion, TaskInstance.dag_version_id == DagVersion.id)
                .where(TaskInstance.dag_id == dag_id)
                .where(TaskInstance.run_id == run_id)
                .order_by(TaskInstance.task_id)
            ).all()
            if not tis:
                continue
            version_id = tis[0].dag_version_id
            if version_id not in serdag_cache:
                serdag_cache[version_id] = _get_serdag(dag_id, version_id, session)
            summary = _build_ti_summaries(dag_id, run_id, tis, session, serdag=serdag_cache[version_id])
            yield GridTISummaries.model_validate(summary).model_dump_json() + "\\n"

    return StreamingResponse(content=_generate(), media_type="application/x-ndjson")
'''

NEW_FUNCTION = f'''{PATCH_MARKER}
def get_grid_ti_summaries_stream(
    dag_id: str,
    run_ids: Annotated[list[str] | None, Query()] = None,
) -> StreamingResponse:
    """
    Stream TI summaries for multiple Dag runs as NDJSON (one JSON line per run).

    Each line is materialized while an explicit DB session is open, then the
    response streams from memory after the session is closed. This avoids
    leaking idle-in-transaction connections when UI grid requests are refreshed
    heavily or clients disconnect during a streaming response.
    """

    lines: list[str] = []
    with create_session(scoped=False) as session:
        serdag_cache: dict = {{}}
        for run_id in run_ids or []:
            tis = session.execute(
                select(
                    TaskInstance.task_id,
                    TaskInstance.state,
                    TaskInstance.dag_version_id,
                    TaskInstance.start_date,
                    TaskInstance.end_date,
                    DagVersion.version_number,
                )
                .outerjoin(DagVersion, TaskInstance.dag_version_id == DagVersion.id)
                .where(TaskInstance.dag_id == dag_id)
                .where(TaskInstance.run_id == run_id)
                .order_by(TaskInstance.task_id)
            ).all()
            if not tis:
                continue
            version_id = tis[0].dag_version_id
            if version_id not in serdag_cache:
                serdag_cache[version_id] = _get_serdag(dag_id, version_id, session)
            summary = _build_ti_summaries(dag_id, run_id, tis, session, serdag=serdag_cache[version_id])
            lines.append(GridTISummaries.model_validate(summary).model_dump_json() + "\\n")

    return StreamingResponse(content=iter(lines), media_type="application/x-ndjson")
'''


def find_airflow_grid_file() -> Path:
    spec = importlib.util.find_spec("airflow.api_fastapi.core_api.routes.ui.grid")
    if spec is None or spec.origin is None:
        raise RuntimeError("cannot find airflow.api_fastapi.core_api.routes.ui.grid")
    return Path(spec.origin)


def patch_text(text: str) -> tuple[str, str]:
    if PATCH_MARKER in text:
        return text, "already_patched"
    if OLD_FUNCTION not in text:
        raise RuntimeError("Airflow grid.py does not match expected 3.2.0 ti_summaries implementation")

    patched = text.replace(OLD_FUNCTION, NEW_FUNCTION)
    if "from airflow.utils.session import create_session\n" not in patched:
        if OLD_IMPORT not in patched:
            raise RuntimeError("Airflow grid.py import block does not match expected layout")
        patched = patched.replace(OLD_IMPORT, NEW_IMPORT)
    return patched, "patched"


def patch_file(path: Path, check: bool = False) -> str:
    original = path.read_text(encoding="utf-8")
    patched, status = patch_text(original)
    if check:
        if status != "already_patched":
            raise RuntimeError(f"Airflow grid session patch is not applied: {path}")
        return status
    if status == "patched":
        backup = path.with_suffix(path.suffix + ".deploy_ci_cloud.bak")
        tmp_path = path.with_suffix(path.suffix + ".deploy_ci_cloud.tmp")
        mode = stat.S_IMODE(path.stat().st_mode)
        tmp_path.write_text(patched, encoding="utf-8")
        tmp_path.chmod(mode)
        try:
            py_compile.compile(str(tmp_path), doraise=True)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        tmp_path.replace(path)
    return status


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patch Airflow UI grid streaming DB session leak")
    parser.add_argument("--target", default=None, help="grid.py path; defaults to installed Airflow grid.py")
    parser.add_argument("--check", action="store_true", help="verify the patch is already applied")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    target = Path(args.target) if args.target else find_airflow_grid_file()
    status = patch_file(target, check=args.check)
    print(f"airflow_grid_session_patch={status} target={target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
