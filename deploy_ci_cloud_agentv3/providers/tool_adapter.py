from __future__ import annotations

from deploy_ci_cloud_agentv3.mcp.client import MCPTool


def mcp_tools_to_native(tools: list[MCPTool]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]
