# orch

CLIエージェントをDAGで実行する Python 製オーケストレーターです。

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`python -m venv` が `ensurepip is not available` で失敗する環境では、次の代替手順を使用できます。

```bash
python3 -m pip install --user virtualenv
python3 -m virtualenv .venv
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
  - id: "inspect"  # 1..128文字、英数字で開始し [A-Za-z0-9._-] のみ使用可（大文字小文字を区別せず一意）
    cmd: ["python3", "tools/fake_agent.py", "inspect"]
    # cmd: "python3 tools/fake_agent.py inspect"
    depends_on: ["other_task_id"]
    cwd: "."
    env: {"KEY": "VALUE"}  # KEY は非空かつ '=' を含まない文字列
    timeout_sec: 60  # 0より大きい有限数
    retries: 2
    retry_backoff_sec: [1, 3, 10]  # 0以上の有限数
    outputs: ["dist/**", "report.json"]
```

`artifacts_dir` を指定すると、`outputs` で収集した成果物を run 内 (`runs/<run_id>/artifacts/...`)
に保存するだけでなく、`artifacts_dir/<task_id>/...` にもコピーします。
相対パスは `--workdir` 基準で解決され、絶対パスはそのまま使用されます。
また、`outputs` はタスクが失敗した場合でも可能な範囲で収集されます（best-effort）。

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

## Release 0.1 DoD セルフチェック

以下を順に実行すると、Release 0.1 の主要DoDを手元で確認できます。

```bash
orch run examples/plan_basic.yaml
orch run examples/plan_parallel.yaml --max-parallel 2
orch run examples/plan_fail_retry.yaml
orch resume <run_id>
orch status <run_id>
orch logs <run_id> --task <task_id> --tail 50
orch cancel <running_run_id>
ruff format --check .
ruff check .
pytest
```

ワンコマンドで確認する場合は次を実行してください。

```bash
python tools/dod_check.py
```

CI で実行する runtime smoke（実行系のみ検証）は次です。

```bash
python tools/dod_check.py --skip-quality-gates
```

実行結果を他の `.orch` 実行履歴と分離したい場合は `--home` を使います。

```bash
python tools/dod_check.py --home .orch_dod
```

機械可読な結果が必要な場合は `--json` を付けると最後に JSON summary を出力します。

```bash
python tools/dod_check.py --json
```

JSON をファイルとして保存したい場合は `--json-out` を使います。

```bash
python tools/dod_check.py --json-out /tmp/orch_dod_summary.json
```

- `plan_fail_retry.yaml` は失敗系確認用です（終了コード `3` が期待値）。
- `resume` は SUCCESS 済みタスクを再実行しません。
- `cancel` は実行中 run に対して実施してください（run 側終了コード `4`）。
