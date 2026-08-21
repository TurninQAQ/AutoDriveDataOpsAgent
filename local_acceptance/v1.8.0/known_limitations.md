# V1.8.0 Known Limitations

- AUTO remains limited to `resume_task`; no autonomous retry, polling loop,
  model failover, or other mutation is included.
- File locking is process-safe under supported local/shared filesystem
  semantics, not a distributed transaction or global exactly-once guarantee.
- A repository-wide candidate run collected 458 tests but timed out in an
  existing GPU/runtime path; environment-only failures are listed in
  `repository_test_summary.txt`.
- External Airflow/Docker/GPU runtime E2E was not available locally.
- The first fresh qwen-plus run had one non-safety hybrid evidence-selection
  variance (5/6 functional, 4/6 strict). One bounded adjudication of that
  case passed; no prompt tuning or qwen3.7-plus retry was performed.
- Formal qwen-plus benchmarking remains deferred and the V1.5 frozen golden
  was not modified.
