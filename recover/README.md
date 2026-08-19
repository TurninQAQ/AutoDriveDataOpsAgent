# Airflow 恢复说明

本目录只保留恢复流程说明，不保存 Airflow 配置、数据库连接、加密密钥或登录密码。

运行时敏感文件位于 Runtime 外部配置目录，并由部署脚本生成或读取：

- `config/runtime_secrets.env`：Fernet、API 和 JWT 密钥，权限应为 `0600`。
- `airflow/airflow.cfg`：由 `config/airflow.cfg.base` 与 Runtime 配置生成，权限应为 `0600`。
- `airflow/simple_auth_manager_passwords.json.generated`：本机登录密码文件，权限应为 `0600`。
- `backups/`：部署或轮换前的本地备份，不应加入 Git。

恢复或轮换前，先确认 DagRun 数量为零并备份 Airflow 配置、密码文件和元数据库。随后使用
`./platform install` 或 `./platform deploy` 重新生成配置；Fernet 轮换需按部署报告中的双密钥迁移流程执行。
