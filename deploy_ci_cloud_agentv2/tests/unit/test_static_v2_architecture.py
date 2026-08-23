import ast
from pathlib import Path

import tomllib


ROOT = Path(__file__).resolve().parents[2]


def _production_python():
    for path in ROOT.rglob("*.py"):
        if any(part in {"tests", "build", "__pycache__"} for part in path.parts):
            continue
        yield path


def test_v2_production_has_no_v1_runtime_import_or_second_semantic_authority():
    forbidden_names = {
        "Planner", "PlannerAgent", "IntentRouter", "SemanticRouter", "StrategyEngine",
        "DecisionEngine", "AdaptiveController", "SupervisorAgent", "AnswerJudge",
    }
    for path in _production_python():
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
            else:
                imports = []
            for name in imports:
                assert name != "deploy_ci_cloud_agent"
                assert not name.startswith("deploy_ci_cloud_agent.")
                if path.parent.name == "tools":
                    assert "provider" not in name.lower()
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                assert node.name not in forbidden_names


def test_v2_distribution_metadata_matches_complete_runtime_packages():
    with (ROOT / "pyproject.toml").open("rb") as fh:
        config = tomllib.load(fh)
    assert config["project"]["version"] == "2.0.0"
    assert config["project"]["dependencies"] == ["langgraph==1.2.11", "httpx>=0.28,<1"]
    packages = set(config["tool"]["setuptools"]["packages"])
    for suffix in ("safety", "memory", "verification", "evaluation"):
        assert f"deploy_ci_cloud_agentv2.{suffix}" in packages
