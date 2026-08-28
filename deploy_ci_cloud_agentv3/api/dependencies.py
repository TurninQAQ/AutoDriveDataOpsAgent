from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from deploy_ci_cloud_agentv3.api.events import EventBroker
from deploy_ci_cloud_agentv3.persistence.audit_store import AuditStore
from deploy_ci_cloud_agentv3.persistence.run_store import RunStore


@dataclass
class AppServices:
    runtime: Any
    run_store: RunStore
    audit_store: AuditStore
    broker: EventBroker
