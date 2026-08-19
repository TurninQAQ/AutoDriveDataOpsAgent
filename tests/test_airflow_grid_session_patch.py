#!/usr/bin/env python3
import importlib
import sys
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def main():
    patcher = importlib.import_module("scripts.patch_airflow_grid_session")

    source = (
        "from __future__ import annotations\n"
        "from collections.abc import Generator\n"
        "from typing import Annotated\n"
        "from fastapi import Query\n"
        "from fastapi.responses import StreamingResponse\n"
        "from airflow.api_fastapi.common.db.common import SessionDep\n"
        "from airflow.models.taskinstancehistory import TaskInstanceHistory\n"
        "\n"
        "@grid_router.get('/ti_summaries/{dag_id}')\n"
        + patcher.OLD_FUNCTION
    )

    patched, status = patcher.patch_text(source)
    assert status == "patched"
    assert patcher.PATCH_MARKER in patched
    assert "from airflow.utils.session import create_session\n" in patched
    assert "def _generate()" not in patched
    assert "session: SessionDep" not in patched
    assert "with create_session(scoped=False) as session:" in patched
    assert "return StreamingResponse(content=iter(lines)" in patched

    patched_again, status_again = patcher.patch_text(patched)
    assert status_again == "already_patched"
    assert patched_again == patched

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "grid.py"
        target.write_text(source, encoding="utf-8")

        file_status = patcher.patch_file(target)
        assert file_status == "patched"
        assert target.with_suffix(".py.deploy_ci_cloud.bak").exists()

        checked = patcher.patch_file(target, check=True)
        assert checked == "already_patched"


if __name__ == "__main__":
    main()
