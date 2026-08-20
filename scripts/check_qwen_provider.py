#!/usr/bin/env python3
"""Run a synthetic structured-response preflight against the configured Qwen endpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from urllib.parse import urlparse

from platform_agent.provider_preflight import run_qwen_preflight


def endpoint_host(value: str) -> str:
    parsed = urlparse(value)
    return parsed.netloc or parsed.path.split("/", 1)[0]


async def main(args) -> int:
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    endpoint = os.environ.get("DASHSCOPE_OPENAI_BASE_URL", "").strip()
    if not api_key or not endpoint:
        print(json.dumps({
            "status": "BLOCKED_PROVIDER_PREFLIGHT",
            "failure_type": "missing_credentials_or_endpoint",
            "requests_attempted": 0,
            "requests_completed": 0,
        }, ensure_ascii=False))
        return 2

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=endpoint,
        timeout=max(0.001, args.timeout_sec),
    )
    result = await run_qwen_preflight(
        client,
        model=args.model,
        checks=args.checks,
        timeout_sec=args.timeout_sec,
    )
    payload = result.as_dict()
    payload["endpoint_host"] = endpoint_host(endpoint)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--checks", type=int, default=2)
    parser.add_argument("--timeout-sec", type=float, default=15.0)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main(parse_args())))
