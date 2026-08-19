from __future__ import annotations

from ..gateways.docker import DockerGateway


class DockerService:
    def __init__(self, gateway: DockerGateway | None = None):
        self.gateway = gateway or DockerGateway()

    def matching_containers(self, task_name, config, dataset_names):
        return self.gateway.matching(task_name, config, dataset_names)

    def stop_task_containers(self, task_name, config, dataset_names, apply_changes):
        return self.gateway.stop_matching(task_name, config, dataset_names, apply_changes)

    def stop_all_managed_containers(self, apply_changes):
        return self.gateway.stop_all_managed(apply_changes)
