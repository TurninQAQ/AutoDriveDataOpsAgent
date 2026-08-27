from __future__ import annotations

import json
import re
import subprocess
import time

def info(message):
    print(f"[INFO] {message}", flush=True)

def warn(message):
    print(f"[WARN] {message}", flush=True)

def inspect_running_containers():
    try:
        output = subprocess.check_output(
            ["docker", "ps", "-q"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=30,
        )
    except Exception as exc:
        warn(f"Could not list docker containers: {exc}")
        return []

    container_ids = [line.strip() for line in output.splitlines() if line.strip()]
    if not container_ids:
        return []

    inspected = []
    for container_id in container_ids:
        try:
            raw = subprocess.check_output(
                ["docker", "inspect", container_id],
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                timeout=30,
            )
            data = json.loads(raw)
            if data:
                inspected.append(data[0])
        except Exception as exc:
            warn(f"Could not inspect docker container {container_id}: {exc}")
    return inspected

def container_text(container):
    config = container.get("Config") or {}
    host_config = container.get("HostConfig") or {}
    mounts = container.get("Mounts") or []
    parts = [
        container.get("Id", ""),
        container.get("Name", ""),
        config.get("Image", ""),
        " ".join(config.get("Env") or []),
        " ".join(config.get("Cmd") or []),
        " ".join(config.get("Entrypoint") or []),
        " ".join(host_config.get("Binds") or []),
        " ".join(str(mount.get("Source", "")) for mount in mounts),
        " ".join(str(mount.get("Destination", "")) for mount in mounts),
    ]
    return "\n".join(str(part) for part in parts if part)

def safe_container_part(value, fallback):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip("-._") or fallback

def task_container_prefix(task_name):
    return f"/airflow-task-{safe_container_part(task_name, 'task')}--"

def container_matches_dataset(container, dataset_name):
    safe_dataset = safe_container_part(dataset_name, "dataset")
    name = str(container.get("Name") or "")
    if f"--{safe_dataset}--" in name:
        return True

    # Long dataset names may be truncated in the Docker name. Fall back to an
    # exact token-like match in command, environment and mount metadata without
    # treating clip_001 as a match for clip_0010.
    pattern = r"(?<![A-Za-z0-9_.-]){}(?![A-Za-z0-9_.-])".format(
        re.escape(str(dataset_name))
    )
    return re.search(pattern, container_text(container)) is not None


def task_containers(task_name, dataset_names=None):
    """Return running containers owned by task without requiring task config.

    V0.8 verification must still inspect containers after delete_task removed the
    task YAML. Ownership therefore starts with the exact task container-name
    prefix and optionally applies the same dataset token matcher used by normal
    lifecycle management.
    """
    prefix = task_container_prefix(task_name)
    selected = list(dataset_names or [])
    matches = []
    for container in inspect_running_containers():
        name = str(container.get("Name") or "")
        if not name.startswith(prefix):
            continue
        if selected and not any(container_matches_dataset(container, item) for item in selected):
            continue
        matches.append(container)
    return matches

def matching_containers(task_name, config, dataset_names):
    prefix = task_container_prefix(task_name)
    matches = []
    for container in inspect_running_containers():
        name = str(container.get("Name") or "")
        if not name.startswith(prefix):
            continue
        for dataset_name in dataset_names:
            if container_matches_dataset(container, dataset_name):
                matches.append(container)
                break
    return matches

def container_absent(container_id):
    result = subprocess.run(
        ["docker", "inspect", container_id],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return False
    message = f"{result.stdout}\n{result.stderr}"
    message_lower = message.lower()
    if "no such object" in message_lower or "no such container" in message_lower:
        return True
    raise RuntimeError(f"Unable to confirm container state for {container_id}: {message.strip()}")

def stop_containers(task_name, config, dataset_names, apply_changes):
    containers = matching_containers(task_name, config, dataset_names)
    if not containers:
        info("No matching docker containers found")
        return 0

    return stop_container_objects(containers, apply_changes)

def stop_container_objects(containers, apply_changes, title="Matching docker containers"):
    labels = [
        f"{container.get('Id', '')[:12]}:{str(container.get('Name') or '').lstrip('/')}"
        for container in containers
        if container.get("Id")
    ]
    info(f"{title}: {' '.join(labels)}")
    if not apply_changes:
        return 0

    stopped = 0
    for container in containers:
        container_id = container.get("Id")
        if not container_id:
            continue
        try:
            subprocess.run(
                ["docker", "stop", "-t", "10", container_id],
                capture_output=True,
                text=True,
                timeout=120,
            )
            subprocess.run(
                ["docker", "rm", "-f", container_id],
                capture_output=True,
                text=True,
                timeout=120,
            )
            deadline = time.time() + 60
            while time.time() < deadline:
                if container_absent(container_id):
                    break
                time.sleep(2)
            else:
                raise RuntimeError(f"Container still exists after stop/remove: {container_id}")
            stopped += 1
        except Exception as exc:
            warn(f"docker stop failed for {container_id[:12]}: {exc}")
    return stopped

def managed_task_containers():
    containers = []
    for container in inspect_running_containers():
        name = str(container.get("Name") or "").lstrip("/")
        if name.startswith("airflow-task-"):
            containers.append(container)
    return containers

def stop_all_task_containers(apply_changes):
    containers = managed_task_containers()
    if not containers:
        info("No managed task docker containers found")
        return 0
    return stop_container_objects(
        containers,
        apply_changes,
        title="Managed task docker containers",
    )



class DockerGateway:
    """Object-oriented Docker boundary for services and future MCP tools.

    Module-level functions are intentionally retained for CLI compatibility.
    """

    def inspect_running(self):
        return inspect_running_containers()

    def task_containers(self, task_name, dataset_names=None):
        return task_containers(task_name, dataset_names)

    def matching(self, task_name, config, dataset_names):
        return matching_containers(task_name, config, dataset_names)

    def stop_matching(self, task_name, config, dataset_names, apply_changes):
        return stop_containers(task_name, config, dataset_names, apply_changes)

    def managed(self):
        return managed_task_containers()

    def stop_all_managed(self, apply_changes):
        return stop_all_task_containers(apply_changes)
