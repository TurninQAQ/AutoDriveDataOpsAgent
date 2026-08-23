"""Minimal production host CLI; all mutations remain behind Runtime approval."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid

from .config import ConfigurationError, RuntimeConfig
from .agent.runtime import invoke, reconcile, resume
from .host import build_production_context, health, pending_approval, readiness
from .safety.approval import ApprovalDecision, ResumeInput


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autodrive-agent")
    parser.add_argument("--config", help="optional strict JSON runtime config")
    parser.add_argument("--runtime-root", default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("health")
    sub.add_parser("ready")
    invoke_parser = sub.add_parser("invoke")
    invoke_parser.add_argument("user_input")
    invoke_parser.add_argument("--thread-id", default=None)
    pending_parser = sub.add_parser("pending")
    pending_parser.add_argument("--thread-id", required=True)
    for name, decision in (("approve", ApprovalDecision.APPROVE), ("reject", ApprovalDecision.REJECT)):
        item = sub.add_parser(name)
        item.set_defaults(decision=decision)
        item.add_argument("--thread-id", required=True)
        item.add_argument("--approval-request-id", required=True)
        item.add_argument("--transaction-id", required=True)
        item.add_argument("--fingerprint", required=True)
    reconcile_parser = sub.add_parser("reconcile")
    reconcile_parser.add_argument("--thread-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = RuntimeConfig.from_env(
            runtime_root=args.runtime_root or "/home/ubuntu/project/autodrive_dataops_runtimev2",
            config_path=args.config,
        )
        if args.command == "health":
            _print(health(config))
            return 0
        if args.command == "ready":
            result = readiness(config)
            _print(result)
            return 0 if result["status"] == "ready" else 1
        context = build_production_context(config)
        if args.command == "pending":
            _print(pending_approval(thread_id=args.thread_id, context=context))
            return 0
        if args.command == "invoke":
            result = asyncio.run(invoke(args.user_input, thread_id=args.thread_id or f"thread_{uuid.uuid4().hex}", system_context=context))
        elif args.command in {"approve", "reject"}:
            result = asyncio.run(resume(
                thread_id=args.thread_id,
                resume_input=ResumeInput(
                    decision=args.decision,
                    approval_request_id=args.approval_request_id,
                    transaction_id=args.transaction_id,
                    fingerprint=args.fingerprint,
                ),
                system_context=context,
            ))
        elif args.command == "reconcile":
            result = asyncio.run(reconcile(thread_id=args.thread_id, system_context=context))
        else:  # pragma: no cover - argparse enforces this
            raise ConfigurationError("unknown command")
        _print(result)
        return 0
    except (ConfigurationError, ValueError, RuntimeError) as exc:
        _print({"status": "error", "error_type": type(exc).__name__, "message": str(exc)})
        return 2


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default))


def _json_default(value: object) -> object:
    if hasattr(value, "value"):
        return getattr(value, "value")
    if hasattr(value, "__dict__"):
        return {key: item for key, item in vars(value).items() if not key.startswith("_")}
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
