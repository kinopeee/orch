# orch

CLIエージェントをDAGで実行する Python 製オーケストレーターです。

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

実行:

```bash
orch run examples/plan_basic.yaml
orch run examples/plan_parallel.yaml --max-parallel 2
```

状態確認:

```bash
orch status <run_id>
orch logs <run_id> --task inspect --tail 50
```

再開/中断:

```bash
orch resume <run_id>
orch resume <run_id> --failed-only
orch cancel <run_id>
```

## Plan スキーマ

```yaml
goal: "文字列（任意だが推奨）"
artifacts_dir: ".orch/artifacts"
tasks:
  - id: "inspect"
    cmd: ["python3", "tools/fake_agent.py", "inspect"]
    # cmd: "python3 tools/fake_agent.py inspect"
    depends_on: ["other_task_id"]
    cwd: "."
    env: {"KEY": "VALUE"}
    timeout_sec: 60
    retries: 2
    retry_backoff_sec: [1, 3, 10]
    outputs: ["dist/**", "report.json"]
```

## 終了コード

- `0`: 全タスク成功
- `2`: plan 検証エラー
- `3`: 実行失敗（FAILED または SKIPPED を含む）
- `4`: キャンセル終了

## 開発コマンド

```bash
ruff format --check .
ruff check .
mypy src
pytest
```
