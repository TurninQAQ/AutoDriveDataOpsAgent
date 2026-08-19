#!/usr/bin/env python3
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    airflow_ctl = (REPO_ROOT / "scripts" / "airflow_ctl.sh").read_text(encoding="utf-8")
    platform = (REPO_ROOT / "platform").read_text(encoding="utf-8")

    assert 'start_api_server_instance "api_server" "API Server" "0.0.0.0" "$API_SERVER_PORT" "core"' in airflow_ctl
    assert 'start_api_server_instance "api_server" "API Server" "0.0.0.0" "$API_SERVER_PORT" "all"' not in airflow_ctl
    assert (
        'start_api_server_instance "execution_api_server" "Execution API Server" '
        '"127.0.0.1" "$AIRFLOW_EXECUTION_API_PORT" "execution"'
    ) in airflow_ctl
    assert "validate_runtime_api_config" in airflow_ctl
    assert 'configured_execution_url" != "$expected_execution_url"' in airflow_ctl

    load_env_body = platform.split("load_dot_env_for_install() {", 1)[1].split("\n}", 1)[0]
    assert "clear_install_airflow_env" in load_env_body
    assert load_env_body.index("clear_install_airflow_env") < load_env_body.index('source "$ENV_FILE"')
    assert "unset AIRFLOW__CORE__EXECUTION_API_SERVER_URL" in platform
    assert 'AIRFLOW__CORE__EXECUTION_API_SERVER_URL="${AIRFLOW_EXECUTION_API_BASE%/}/execution/"' in platform
    assert 'AIRFLOW__CORE__EXECUTION_API_SERVER_URL="${AIRFLOW__CORE__EXECUTION_API_SERVER_URL:-' not in platform
    assert 'AIRFLOW__CORE__EXECUTION_API_SERVER_URL="${AIRFLOW_EXECUTION_API_BASE%/}/execution/"' in airflow_ctl
    assert 'AIRFLOW__CORE__EXECUTION_API_SERVER_URL="${AIRFLOW__CORE__EXECUTION_API_SERVER_URL:-' not in airflow_ctl

    install_body = platform.split("cmd_install() {", 1)[1].split("\n}", 1)[0]
    assert "install_venv" in install_body
    assert "apply_airflow_patches" in install_body
    assert "render_airflow_cfg" in install_body
    assert install_body.index("install_venv") < install_body.index("apply_airflow_patches")
    assert "patch-airflow)" not in platform


if __name__ == "__main__":
    main()
