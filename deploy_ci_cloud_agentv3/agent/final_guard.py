from __future__ import annotations

import json
from typing import Any

from deploy_ci_cloud_agentv3.models.final_response import FinalCandidate, FinalResponse
from deploy_ci_cloud_agentv3.models.write_result import WriteResult


class FinalGuard:
    """Fail-closed structured final guard. Runtime owns write outcome wording."""

    def build(self, candidate: FinalCandidate | dict[str, Any] | str, last_write_result: dict | None) -> FinalResponse:
        parsed = self._parse_candidate(candidate)
        if parsed is None:
            return FinalResponse(
                status="write_uncertain" if last_write_result else "informational",
                write_result_id=(last_write_result or {}).get("id") if isinstance(last_write_result, dict) else None,
                message="Final response was not emitted in the required structured format; no write-success claim was released.",
            )

        if not last_write_result:
            if parsed.status in {"informational", "incomplete"}:
                return FinalResponse(status=parsed.status, message=parsed.message)
            return FinalResponse(status="write_not_executed", message="No platform write was executed in this run.")

        result = WriteResult.model_validate(last_write_result)
        if result.status == "VERIFIED" and result.verified is True:
            return FinalResponse(status="write_verified", write_result_id=result.id, message=f"Verified write succeeded: {result.action}.")
        if result.status == "REJECTED":
            return FinalResponse(status="write_not_executed", write_result_id=result.id, message="The proposed write was rejected and was not executed.")
        if result.status == "PRECONDITION_FAILED":
            return FinalResponse(status="write_not_executed", write_result_id=result.id, message="The approved write was not executed because the platform state changed after review.")
        if result.status == "UNKNOWN_OUTCOME":
            return FinalResponse(status="write_uncertain", write_result_id=result.id, message="The write outcome could not be confirmed; the mutation was not blindly retried.")
        return FinalResponse(status="write_failed", write_result_id=result.id, message=f"The write was not verified ({result.status}).")

    @staticmethod
    def _parse_candidate(candidate: FinalCandidate | dict[str, Any] | str) -> FinalCandidate | None:
        if isinstance(candidate, FinalCandidate):
            return candidate
        try:
            if isinstance(candidate, str):
                value = json.loads(candidate)
            else:
                value = candidate
            return FinalCandidate.model_validate(value)
        except Exception:
            return None
