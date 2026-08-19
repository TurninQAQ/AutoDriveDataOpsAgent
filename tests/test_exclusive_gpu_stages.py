#!/usr/bin/env python3
import importlib
import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "dags"))


def assert_raises(expected_text, func):
    try:
        func()
    except Exception as exc:
        assert expected_text in str(exc), str(exc)
        return
    raise AssertionError("expected exception containing: {}".format(expected_text))


def base_gpu_config():
    return {
        "gpu_ids": "0,1",
        "gpu_stages": "segment,od,occ",
        "gpu_stage_memory_mb": {
            "segment": 1000,
            "od": 1000,
            "occ": 1000,
        },
        "gpu_wait_interval_sec": 1,
        "gpu_reservation_pending_sec": 1,
    }


def write_gpu_lock(lock_dir, gpu_id, reservations):
    lock_dir.mkdir(parents=True, exist_ok=True)
    (lock_dir / "gpu_{}.lock".format(gpu_id)).write_text(
        json.dumps({"reservations": reservations}, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def alive_reservation(stage, exclusive=False, required_mb=1000):
    return {
        "pid": os.getpid(),
        "stage": stage,
        "exclusive": exclusive,
        "required_mb": required_mb,
    }


def test_task_manager_config():
    tm = importlib.import_module("scripts.task_manager")
    stages = ["segment", "od", "occ"]

    config = base_gpu_config()
    normalized = tm.normalize_gpu_config(config, stages=stages)
    assert normalized["exclusive_gpu_stages"] == "segment,od,occ"
    assert normalized["exclusive_gpu_idle_used_max_mb"] == 512

    config = base_gpu_config()
    config["exclusive_gpu_stages"] = ""
    config["exclusive_gpu_idle_used_max_mb"] = "bad-value"
    normalized = tm.normalize_gpu_config(config, stages=stages)
    assert normalized["exclusive_gpu_stages"] == ""
    assert normalized["exclusive_gpu_idle_used_max_mb"] == 512

    config = base_gpu_config()
    config["exclusive_gpu_stages"] = "map"
    assert_raises(
        "exclusive_gpu_stages contains stages not listed in gpu_stages",
        lambda: tm.normalize_gpu_config(config, stages=stages + ["map"]),
    )

    config = base_gpu_config()
    config["gpu_stages"] = "segment,od,occ,map"
    config["gpu_stage_memory_mb"]["map"] = 1000
    config["exclusive_gpu_stages"] = "map"
    assert_raises(
        "exclusive_gpu_stages contains stages not listed in pipeline_stages",
        lambda: tm.normalize_gpu_config(config, stages=stages),
    )

    config = base_gpu_config()
    config["exclusive_gpu_stages"] = "segment"
    config["exclusive_gpu_idle_used_max_mb"] = -1
    assert_raises(
        "exclusive_gpu_idle_used_max_mb must be a non-negative integer",
        lambda: tm.normalize_gpu_config(config, stages=stages),
    )


def test_dag_runtime_gpu_locking():
    dag_runtime = importlib.import_module("batch_pipeline_universal")
    original_gpu_lock_dir = dag_runtime.GPU_LOCK_DIR
    original_query_gpu_memory_mb = dag_runtime.query_gpu_memory_mb

    try:
        dag_runtime.query_gpu_memory_mb = lambda gpu_id: (10000, 9900)

        with tempfile.TemporaryDirectory() as tmp:
            lock_dir = Path(tmp) / "locks_1"
            dag_runtime.GPU_LOCK_DIR = str(lock_dir)
            write_gpu_lock(
                lock_dir,
                "0",
                {"active": alive_reservation("od", exclusive=False)},
            )
            gpu_id, token = dag_runtime.acquire_gpu_from_pool(
                "0,1",
                stage="occ",
                required_mb=1000,
                wait_interval_sec=1,
                pending_sec=1,
                exclusive_gpu_stages=["occ"],
                exclusive_gpu_idle_used_max_mb=256,
            )
            assert gpu_id == "1"
            state = json.loads((lock_dir / "gpu_1.lock").read_text(encoding="utf-8"))
            assert state["reservations"][token]["exclusive"] is True

        with tempfile.TemporaryDirectory() as tmp:
            lock_dir = Path(tmp) / "locks_2"
            dag_runtime.GPU_LOCK_DIR = str(lock_dir)
            write_gpu_lock(
                lock_dir,
                "0",
                {"active": alive_reservation("segment", exclusive=True)},
            )
            gpu_id, token = dag_runtime.acquire_gpu_from_pool(
                "0,1",
                stage="od",
                required_mb=1000,
                wait_interval_sec=1,
                pending_sec=1,
                exclusive_gpu_stages=["segment"],
                exclusive_gpu_idle_used_max_mb=256,
            )
            assert gpu_id == "1"
            state = json.loads((lock_dir / "gpu_1.lock").read_text(encoding="utf-8"))
            assert state["reservations"][token]["exclusive"] is False

        with tempfile.TemporaryDirectory() as tmp:
            lock_dir = Path(tmp) / "locks_3"
            dag_runtime.GPU_LOCK_DIR = str(lock_dir)
            write_gpu_lock(
                lock_dir,
                "0",
                {"active": alive_reservation("od", exclusive=False, required_mb=1000)},
            )
            gpu_id, token = dag_runtime.acquire_gpu_from_pool(
                "0,1",
                stage="occ",
                required_mb=1000,
                wait_interval_sec=1,
                pending_sec=1,
                exclusive_gpu_stages=["segment"],
                exclusive_gpu_idle_used_max_mb=256,
            )
            assert gpu_id == "0"
            state = json.loads((lock_dir / "gpu_0.lock").read_text(encoding="utf-8"))
            assert state["reservations"][token]["exclusive"] is False
            assert len(state["reservations"]) == 2
    finally:
        dag_runtime.GPU_LOCK_DIR = original_gpu_lock_dir
        dag_runtime.query_gpu_memory_mb = original_query_gpu_memory_mb


def test_generator_output_fields():
    generator = importlib.import_module("scripts.tools.genarate_dataset_config")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        record_dir = tmp_path / "record"
        (record_dir / "clip_001").mkdir(parents=True)
        output_yaml = tmp_path / "generated.yaml"

        generator.generate_dataset_configs(
            str(record_dir),
            output_file=str(output_yaml),
            pipeline_stages=[["segment"], ["od"]],
            task_type="taska",
            priority=80,
            gpu_ids="0",
            gpu_stages="segment,od",
            exclusive_gpu_stages="od",
            exclusive_gpu_idle_used_max_mb=128,
            gpu_stage_memory_mb={"segment": 1000, "od": 2000},
            images={"image_segment": "segment:test", "image_od": "od:test"},
            tier="medium",
            pool="custom_pool",
        )

        data = yaml.safe_load(output_yaml.read_text(encoding="utf-8"))
        assert data["task_type"] == "taska"
        assert data["priority"] == 80
        assert data["exclusive_gpu_stages"] == "od"
        assert data["exclusive_gpu_idle_used_max_mb"] == 128
        assert data["gpu_stages"] == "segment,od"
        assert data["datasets"][0]["tier"] == "medium"
        assert data["datasets"][0]["pool"] == "custom_pool"

        default_output_yaml = tmp_path / "generated_default.yaml"
        generator.generate_dataset_configs(
            str(record_dir),
            output_file=str(default_output_yaml),
            pipeline_stages=[["segment"], ["od"]],
            gpu_ids="0",
            gpu_stages="segment,od,occ",
            images={"image_segment": "segment:test", "image_od": "od:test"},
        )

        default_data = yaml.safe_load(default_output_yaml.read_text(encoding="utf-8"))
        assert default_data["exclusive_gpu_stages"] == "segment,od"
        assert default_data["gpu_stage_memory_mb"]["segment"] == 24000
        assert default_data["gpu_stage_memory_mb"]["od"] == 24000
        assert default_data["datasets"][0]["tier"] == "small"
        assert default_data["datasets"][0]["pool"] == "default_pool"


def main():
    test_task_manager_config()
    test_dag_runtime_gpu_locking()
    test_generator_output_fields()


if __name__ == "__main__":
    main()
