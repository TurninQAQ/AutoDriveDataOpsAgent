# Airflow 恢复配置

这个目录保存当前稳定版本使用的 Airflow 配置和登录认证文件。

- `airflow.cfg`：Airflow 主配置，包含 PostgreSQL 连接、Executor、DAG 路径和认证配置。
- `simple_auth_manager_passwords.json.generated`：Airflow 页面和 API 的登录密码文件。

当前 `airflow.cfg` 中 `dag_processor.refresh_interval = 90`，新生成 DAG 最多约 90 秒被扫描到。

恢复平台时，将这两个文件放回 `/home/cidi/airflow/`。
