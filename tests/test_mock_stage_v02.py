from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK_STAGE = REPO_ROOT / "scripts" / "mock_stage.py"
VALIDATOR = REPO_ROOT / "scripts" / "validate_json.py"


def run_mock(tmp_path: Path, result: str, stage: str = "segment", duration: float = 0):
    return subprocess.run(
        [
            sys.executable,
            str(MOCK_STAGE),
            "--stage", stage,
            "--dataset-path", str(tmp_path),
            "--dataset-name", "clip_001",
            "--duration-sec", str(duration),
            "--result", result,
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )


def validate(tmp_path: Path, stage: str = "segment"):
    return subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--root-dir", str(tmp_path),
            "--dataset", "clip_001",
            "--task-suffix", stage,
            "--min-date", "2020-01-01",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )


def test_mock_stage_success_is_accepted_by_existing_validator(tmp_path: Path):
    result = run_mock(tmp_path, "success")
    assert result.returncode == 0
    validation = validate(tmp_path)
    assert validation.returncode == 0, validation.stdout + validation.stderr
    payload = json.loads((tmp_path / "clip_001" / "results_segment.json").read_text())
    assert payload["status"] == "success"
    assert payload["mock"] is True


def test_mock_stage_fail_returns_nonzero(tmp_path: Path):
    result = run_mock(tmp_path, "fail")
    assert result.returncode == 1
    validation = validate(tmp_path)
    assert validation.returncode != 0


def test_mock_stage_validate_fail_exits_zero_but_validator_rejects(tmp_path: Path):
    result = run_mock(tmp_path, "validate_fail")
    assert result.returncode == 0
    validation = validate(tmp_path)
    assert validation.returncode != 0
    assert "failed" in validation.stderr.lower()


def test_mock_stage_oom_has_diagnostic_log_and_nonzero_exit(tmp_path: Path):
    result = run_mock(tmp_path, "oom")
    assert result.returncode == 137
    assert "out of memory" in result.stderr.lower()


def test_mock_stage_timeout_is_killable_by_caller(tmp_path: Path):
    with pytest.raises(subprocess.TimeoutExpired):
        subprocess.run(
            [
                sys.executable,
                str(MOCK_STAGE),
                "--stage", "segment",
                "--dataset-path", str(tmp_path),
                "--dataset-name", "clip_001",
                "--result", "timeout",
            ],
            capture_output=True,
            text=True,
            timeout=0.2,
        )


def test_original_stage_shell_can_switch_to_mock_without_docker(tmp_path: Path):
    clip = tmp_path / "clip_001"
    clip.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PLATFORM_STAGE_RUNTIME": "mock",
            "MOCK_STAGE_RESULT_SEGMENT": "success",
            "DATASET_PATH": str(tmp_path),
            "DATASET_NAME": "clip_001",
            "DATA_DIR": str(clip),
            # Deliberately do not provide IMAGE_TAG/GPU_IDS/CONTAINER_NAME/checkpoint.
            # Mock mode must branch before real-runtime requirements are checked.
            "AIRFLOW_PYTHON": sys.executable,
        }
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / "run_segment.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (clip / "results_segment.json").is_file()


@pytest.mark.parametrize(
    ("script_name", "stage"),
    [
        ("run_precheck.sh", "precheck"),
        ("run_parser.sh", "parser"),
        ("run_segment.sh", "segment"),
        ("run_map.sh", "map"),
        ("run_od.sh", "od"),
        ("run_occ.sh", "occ"),
        ("run_coloration.sh", "coloration"),
    ],
)
def test_all_pipeline_stage_shells_execute_in_mock_mode(tmp_path: Path, script_name: str, stage: str):
    clip = tmp_path / "clip_001"
    clip.mkdir()
    env = os.environ.copy()
    env.update(
        {
            "PLATFORM_STAGE_RUNTIME": "mock",
            "DATASET_PATH": str(tmp_path),
            "DATASET_NAME": "clip_001",
            "DATA_DIR": str(clip),
            "AIRFLOW_PYTHON": sys.executable,
        }
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "scripts" / script_name)],
        env=env,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (clip / f"results_{stage}.json").is_file()
    validation = subprocess.run(
        [
            sys.executable,
            str(VALIDATOR),
            "--root-dir", str(tmp_path),
            "--dataset", "clip_001",
            "--task-suffix", stage,
            "--min-date", "2020-01-01",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert validation.returncode == 0, validation.stdout + validation.stderr


def test_dag_gpu_runtime_delegates_to_platform_core():
    # Airflow is intentionally not required in the local V0.2 environment, so
    # verify the deployed DAG source is wired to the tested core implementation.
    text = (REPO_ROOT / "dags" / "batch_pipeline_universal.py").read_text(encoding="utf-8")
    assert "create_gpu_runtime_from_env" in text
    assert "GPUAllocator" in text
    assert "platform_gpu_allocator().acquire" in text
    assert "platform_gpu_allocator().release" in text
