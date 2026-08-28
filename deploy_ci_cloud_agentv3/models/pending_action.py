from __future__ import annotations

from typing import Any
from pydantic import BaseModel, ConfigDict, Field

from deploy_ci_cloud_agentv3.models.common import sha256_json


def compute_pending_action_fingerprint(
    *,
    proposal_id: str,
    action: str,
    args: dict[str, Any],
    artifact: dict[str, Any] | None,
    precondition: dict[str, Any],
    action_precondition: dict[str, Any] | None = None,
) -> str:
    """Hash every runtime-owned field that can affect admission or execution semantics."""
    return sha256_json(
        {
            "proposal_id": proposal_id,
            "action": action,
            "args": args,
            "artifact": artifact,
            "precondition": precondition,
            "action_precondition": action_precondition or {},
        }
    )


class PendingAction(BaseModel):
    """Frozen action approved by a human reviewer.

    The idempotency key is intentionally *not* stored here. WriteService derives it
    from the recomputed fingerprint at execution time so mutable workflow state
    cannot mint a fresh mutation-attempt key for an already-approved action. The
    runtime-generated ``proposal_id`` is part of the fingerprint: retries of the
    same approved action dedupe, while a later human approval with identical
    semantic content remains a distinct execution identity.

    ``action_precondition`` contains narrow, action-specific admission facts that are
    not appropriate for the global platform precondition. For resume, it records
    whether the approved target set came from a failed-run resolution and binds the
    verification baseline used later to prove a new DagRun.
    """

    model_config = ConfigDict(extra="forbid")

    proposal_id: str
    action: str
    args: dict[str, Any] = Field(default_factory=dict)
    reason: str
    expected_effect: str
    before: dict[str, Any] = Field(default_factory=dict)
    artifact: dict[str, Any] | None = None
    precondition: dict[str, Any] = Field(default_factory=dict)
    action_precondition: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str

    def recompute_fingerprint(self) -> str:
        return compute_pending_action_fingerprint(
            proposal_id=self.proposal_id,
            action=self.action,
            args=self.args,
            artifact=self.artifact,
            precondition=self.precondition,
            action_precondition=self.action_precondition,
        )
