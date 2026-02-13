# orch: CLIエージェント・オーケストレーター 実装指示書（Python / 0ベース）

> **目的**：YAMLで定義されたタスク（依存関係つき）を、外部CLI（=エージェント）として並列実行し、ログ・状態・成果物を保存し、失敗/中断後に再開でき、最終レポートを生成する **オーケストレーションCLI** を完成させる。  
> **この書類は**：環境構築 → 実装 → テスト → ドキュメント → CIまで、実装者（またはコーディングエージェント）が迷わず完遂できるよう、仕様・設計・タスク分解・完了条件を“完全に”具体化した指示書。

---

## 1. スコープ

### 1.1 できること（Release 0.1の必須要件）
- `orch run plan.yaml`  
  - YAMLプランを読み込み、依存関係に基づいてタスクを実行する
  - `--max-parallel N` で同時実行数を制御する
  - タスクの stdout/stderr をログファイルに **ストリーミング保存** する（メモリ肥大しない）
  - 実行状態を `.orch/runs/<run_id>/state.json` に保存する（原子的書き込み）
  - 実行終了時に `report/final_report.md` を生成する
- `orch status <run_id>`  
  - タスク状態を表形式（または `--json`）で表示する
- `orch logs <run_id> [--task <id>] [--tail N]`  
  - タスク別ログを閲覧できる（末尾N行）
- `orch resume <run_id> [--failed-only]`  
  - 中断/失敗した実行を再開できる（成功済みはスキップ）
- `orch cancel <run_id>`  
  - 実行中のrunをキャンセルできる（可能な限り実行中プロセスも停止）

### 1.2 強化点（必須として取り込む改善）
長時間自走テストに効くため、以下は **0.1で必須** とする（MVP以上）：
- **cmdは原則 list[str] を推奨**（安全・移植性）  
  - 互換のため string も許可するが、内部で `shlex.split()` して `create_subprocess_exec()` を使う（shellを避ける）
- **timeout と retry**（タスク単位）
  - `timeout_sec`, `retries`, `retry_backoff_sec` をサポート
- **依存失敗の自動SKIP伝播**
  - 依存タスクが `FAILED/CANCELED/SKIPPED` の場合、下流は `SKIPPED` にして runを終わらせられる（ハングしない）
- **runディレクトリのロック**
  - 同一runに対する `resume/cancel/status` の同時実行で state が壊れないように、簡易ロックファイルを実装
- **`--dry-run`**
  - 実行せずに検証 + 実行順（DAG）を出す

### 1.3 非目標（0.1ではやらない）
- 分散実行（複数マシン/キュー/ワーカー）
- LLM連携による自動タスク分割（planner）
- GUI/Web UI
- Dockerサンドボックス・リソース制限（CPU/mem制御）
- スケジュール実行（cron相当）

---

## 2. 用語
- **Plan**：YAMLで定義された実行計画（goal + tasks）
- **Task**：外部コマンド（エージェント）を実行する単位
- **DAG**：依存関係有向非巡回グラフ
- **Run**：Planを実行した1回分の記録（run_id）
- **State**：Runの状態（state.json）
- **Artifacts**：タスクが生成した成果物を集約保存したもの

---

## 3. リポジトリ初期化（環境設定）

### 3.1 前提
- Python **3.11以上**
- OS：Linux/macOS を主対象（Windowsはベストエフォート）

### 3.2 リポジトリ作成
```bash
mkdir orch && cd orch
git init
```

### 3.3 仮想環境（確実に動く標準手順）
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

> uv/poetryを使いたい場合は後で置き換え可能。まずは venv+pip を“正”にする。

### 3.4 `pyproject.toml` 作成（この内容で固定）
> **指示**：下記をコピペし、依存はこのセットで進める（余計な追加をしない）。

```toml
[project]
name = "orch"
version = "0.1.0"
description = "CLI agent task orchestrator (DAG, parallel, resume, logs, report)"
requires-python = ">=3.11"
readme = "README.md"
license = {text = "MIT"}
authors = [{name = "Your Name"}]
dependencies = [
  "typer>=0.12",
  "rich>=13.7",
  "PyYAML>=6.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.0",
  "pytest-asyncio>=0.23",
  "ruff>=0.5",
  "mypy>=1.10",
  "types-PyYAML>=6.0",
]

[project.scripts]
orch = "orch.cli:app"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-q"

[tool.ruff]
line-length = 100
target-version = "py311"
fix = true

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = []

[tool.mypy]
python_version = "3.11"
strict = true
mypy_path = ["src"]
warn_unused_configs = true
disallow_any_generics = true
```

依存を入れる：
```bash
pip install -e ".[dev]"
```

### 3.5 ディレクトリ構成（最初に作る）
```bash
mkdir -p src/orch
mkdir -p tests examples tools .github/workflows
touch src/orch/__init__.py
```

---

## 4. 最終的なディレクトリ構成（0.1ターゲット）
> **指示**：以下の構成を最終形として実装する（不要な階層は増やさない）。

```
src/orch/
  __init__.py
  cli.py

  config/
    __init__.py
    schema.py
    loader.py

  dag/
    __init__.py
    build.py
    validate.py

  exec/
    __init__.py
    runner.py
    capture.py
    retry.py
    timeout.py
    cancel.py

  state/
    __init__.py
    model.py
    store.py
    lock.py

  report/
    __init__.py
    render_md.py
    summarize.py

  util/
    __init__.py
    ids.py
    time.py
    paths.py
    tail.py
    errors.py

tools/
  fake_agent.py

examples/
  plan_basic.yaml
  plan_parallel.yaml
  plan_fail_retry.yaml

tests/
  test_plan_loader.py
  test_dag_validate.py
  test_state_store.py
  test_runner_integration.py
  test_resume_integration.py
  test_cancel_integration.py
```

---

## 5. CLI仕様（コマンド・オプション・終了コード）

### 5.1 CLI共通仕様
- CLIは `typer` を使用し、エントリポイント `orch` を提供する
- 出力は原則人間可読（rich）。ただし `--json` をつけた場合はJSONを標準出力に出す
- 重要：**runの結果で終了コードを変える**
  - `0`：runが完全成功（全タスクSUCCESS）
  - `2`：plan検証エラー（YAML不正、循環、未知依存など）
  - `3`：run失敗（FAILEDまたはSKIPPEDが存在）
  - `4`：キャンセル終了（CANCELED）

### 5.2 `orch run`
**Usage**
```bash
orch run path/to/plan.yaml \
  --max-parallel 4 \
  --home .orch \
  --workdir . \
  --fail-fast false \
  --dry-run false
```

**オプション**
- `--max-parallel N`（int, default=4）
- `--home PATH`（default=`.orch`）  
  - run保存のルート。`<home>/runs/<run_id>/...`
- `--workdir PATH`（default=`.`）  
  - タスクのデフォルト `cwd`
- `--fail-fast / --no-fail-fast`（default=`--no-fail-fast`）
- `--dry-run`（default false）
  - 実行はしない。plan検証 + 実行順（DAG）出力のみ。

### 5.3 `orch resume`
```bash
orch resume <run_id> --home .orch --max-parallel 4 --failed-only
```
- `--failed-only`：FAILEDタスクのみ再試行（依存上必要なものは再評価）
- 既にSUCCESSのタスクは絶対に再実行しない

### 5.4 `orch status`
```bash
orch status <run_id> --home .orch
orch status <run_id> --home .orch --json
```

### 5.5 `orch logs`
```bash
orch logs <run_id> --home .orch
orch logs <run_id> --home .orch --task build --tail 200
```

### 5.6 `orch cancel`
```bash
orch cancel <run_id> --home .orch
```
- cancelは **「キャンセル要求ファイル」** を runディレクトリに作成し、runnerがそれを検知して停止する方式を採用する（詳細は後述）

---

## 6. Plan（YAML）仕様

### 6.1 YAMLスキーマ（0.1）
> **指示**：このスキーマで実装し、READMEにもそのまま掲載する。

```yaml
goal: "文字列（任意だが推奨）"
artifacts_dir: ".orch/artifacts"         # 任意。各タスク成果物の集約先（run内にも保存）
tasks:
  - id: "inspect"                         # 必須・ユニーク
    cmd: ["python", "tools/fake_agent.py", "inspect"]   # 推奨: list[str]
    # cmd: "python tools/fake_agent.py inspect"         # 互換: stringも可（shlexでsplit）
    depends_on: ["other_task_id"]         # 任意
    cwd: "."                              # 任意（なければCLIの--workdir）
    env: {"KEY": "VALUE"}                 # 任意
    timeout_sec: 60                       # 任意
    retries: 2                            # 任意（default=0）
    retry_backoff_sec: [1, 3, 10]         # 任意（retries>0なら推奨）
    outputs: ["dist/**", "report.json"]   # 任意（glob。成功/失敗問わず収集可能にしてよい）
```

### 6.2 Plan検証ルール（必須）
- `tasks` は1件以上
- `id` はユニーク
- `depends_on` の参照先が存在する
- DAGが循環しない（cycleがない）
- `retries >= 0`
- `timeout_sec > 0`（指定された場合）
- `cmd` は list[str] または str（他の型はエラー）

---

## 7. Run保存仕様（ディレクトリ規約・ファイル仕様）

### 7.1 ルート
- `HOME`（既定 `.orch`）
  - `runs/<run_id>/...`

### 7.2 run_id
- 形式：`YYYYMMDD_HHMMSS_<6chars>`（例：`20260212_104512_a1b2c3`）
- `<6chars>` は `secrets.token_hex(3)` などでよい
- タイムゾーンはローカル（JST想定）ISO文字列で state に保存する

### 7.3 runディレクトリ構造
```
<home>/runs/<run_id>/
  plan.yaml                 # 実行に使ったプランをコピー（固定化）
  state.json                # RunState（原子的）
  cancel.request            # cancel要求（存在すればキャンセル）
  logs/
    <task_id>.out.log
    <task_id>.err.log
  artifacts/
    <task_id>/
      ...（outputsで集めたファイル）
  report/
    final_report.md
```

> **改善ポイント**：run開始時に plan を runディレクトリへコピーし、resume時は原則このコピーを参照（元planが変わっても影響しない）。

---

## 8. State仕様（state.json）

### 8.1 ステータス定義
- RunStatus：`PENDING | RUNNING | SUCCESS | FAILED | CANCELED`
- TaskStatus：`PENDING | READY | RUNNING | SUCCESS | FAILED | SKIPPED | CANCELED`

### 8.2 state.json のスキーマ（必須フィールド）
```json
{
  "run_id": "20260212_104512_a1b2c3",
  "created_at": "2026-02-12T10:45:12+09:00",
  "updated_at": "2026-02-12T10:45:40+09:00",
  "status": "RUNNING",
  "goal": "READMEを整備して…",
  "plan_relpath": "plan.yaml",
  "home": ".orch",
  "workdir": ".",
  "max_parallel": 4,
  "fail_fast": false,
  "tasks": {
    "inspect": {
      "status": "SUCCESS",
      "depends_on": [],
      "cmd": ["python", "tools/fake_agent.py", "inspect"],
      "cwd": ".",
      "env": null,
      "timeout_sec": null,
      "retries": 0,
      "retry_backoff_sec": [],
      "outputs": [],
      "attempts": 1,
      "started_at": "2026-02-12T10:45:13+09:00",
      "ended_at": "2026-02-12T10:45:14+09:00",
      "duration_sec": 1.2,
      "exit_code": 0,
      "timed_out": false,
      "canceled": false,
      "skip_reason": null,
      "stdout_path": "logs/inspect.out.log",
      "stderr_path": "logs/inspect.err.log",
      "artifact_paths": []
    }
  }
}
```

### 8.3 原子的書き込み（必須）
- `state.json.tmp` に書く → `os.replace(tmp, state.json)`
- 書き込みは “状態遷移のたび” に行う（RUNNING開始/終了/失敗/スキップ/キャンセル）

---

## 9. ロック（Runディレクトリの排他）

### 9.1 目的
- `resume` と `cancel` が同時に走っても state が壊れない
- `status` が読み取り中に書き込み中でも破損を避ける

### 9.2 実装方式（簡易ロックファイル）
- run_dir に `.lock` を作る（排他的作成）
- 実装：`os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)`
- ロック取得できなければ：
  - `status`：再試行（短いバックオフ）→ダメならエラー
  - `run/resume/cancel`：エラー終了（「他プロセスが実行中」）

> 注意：クラッシュで lock が残る可能性あり。0.1では「一定時間以上古いlockは無視する」などの回復策を入れる（`mtime` で判定）。

---

## 10. 実行エンジン仕様（DAG / 並列 / 失敗伝播 / resume）

### 10.1 DAGスケジューリング（基本ルール）
- 依存が全て `SUCCESS` のタスクのみ実行可能
- 依存に `FAILED/CANCELED/SKIPPED` が含まれるタスクは **実行しない** → `SKIPPED`
- `--fail-fast` の場合：最初の `FAILED` 発生時点で、未開始タスクを全て `SKIPPED` にして終了

### 10.2 並列制御
- `asyncio.Semaphore(max_parallel)` を使用
- タスク実行中はセマフォを保持
- “readyキュー” に投入 → ワーカーがセマフォを取って実行

### 10.3 subprocess 実行（安全設計）
- 可能な限り `asyncio.create_subprocess_exec(*cmd_list, ...)` を使用
- `cmd` が string の場合は `shlex.split(cmd)` で list へ変換し exec
- stdout/stderr は PIPE を使い、**逐次読み取り** してファイルに追記
- タスクのログファイル：
  - `logs/<task_id>.out.log`
  - `logs/<task_id>.err.log`
  - リトライ時は同ファイルに追記し、区切り行を入れる  
    `===== attempt 2 / 3 =====`

### 10.4 timeout（タスク単位）
- `timeout_sec` が指定されたら、待機を `asyncio.wait_for(process.wait(), timeout)` で包む
- timeout発生時：
  1. `process.terminate()`
  2. 短い猶予後に `process.kill()`
  3. `timed_out=true`, `exit_code` は `None` or 特別値（stateでは `exit_code: null`）

### 10.5 retry（タスク単位）
- `retries` は「追加試行回数」（例：retries=2なら最大3回）
- バックオフ：
  - `retry_backoff_sec` が指定されていればその配列を使う
  - 足りない場合は最後の値を繰り返す、または指数（仕様固定）
- retry対象：
  - `FAILED` または `timed_out=true` の場合
  - `canceled=true` は retryしない（即中断）

### 10.6 SKIPPEDの伝播（ハング防止の必須設計）
**重要**：失敗した依存があるタスクは実行されないが、そこで止まるとDAGが “未完のまま” になってrunが終了できない。  
よって、依存が満たされないタスクは確実に `SKIPPED` に遷移させ、終端扱いとして後続の依存解決を進める。

**実装ルール**
- あるタスク `T` が ready になった（依存カウント0）時点で、依存タスクを参照し
  - 全てSUCCESS → 実行（RUNNING）
  - 1つでも SUCCESS 以外 → `SKIPPED`（skip_reasonに依存失敗を記録）
- `SKIPPED` になったタスクも終端扱いとして dependents の依存カウントを減らす  
  → これで下流も順次SKIPPEDになり、runが必ず終わる

### 10.7 resumeの仕様
- state.json を読み込んで現在のタスク状態を取得
- `SUCCESS` は絶対に再実行しない
- `RUNNING` が残っている場合：
  - 前回クラッシュで残存したものとして扱い、`FAILED` に落とすか `CANCELED` に落とすかを決める  
  - 0.1仕様：`RUNNING` は `FAILED` に落とし、`skip_reason="previous_run_interrupted"` を入れる（再実行可能にする）
- `--failed-only`：
  - `FAILED` のタスクのみ再実行対象（ただし依存整合のため必要なら上流の未完を再評価）
- resumeも、runと同様にDAGを再構築し、stateに基づいて未完のみ処理する

---

## 11. キャンセル仕様

### 11.1 cancel要求ファイル方式（採用）
- `orch cancel <run_id>` は run_dir に `cancel.request` を作成（内容は任意）
- runnerは以下のタイミングで `cancel.request` の存在をチェック：
  - 新しいタスクを開始する直前
  - 実行中タスクの監視ループ（一定間隔） ※0.1では簡易でも可

### 11.2 キャンセル時の挙動
- run status を `CANCELED` にする
- 実行中プロセスがあれば terminate/kill
- 未開始タスクは `CANCELED` または `SKIPPED`（仕様固定）
  - 0.1では未開始は `CANCELED` に統一し、`skip_reason="run_canceled"` を入れる

---

## 12. レポート仕様（report/final_report.md）

### 12.1 出力場所
- `<run_dir>/report/final_report.md`

### 12.2 必須内容（テンプレ固定）
- Run概要（run_id、goal、開始/終了、max_parallel、fail_fast、workdir）
- タスク結果一覧（表）
  - id / status / attempts / duration / exit_code / timed_out / logs
- FAILED/ SKIPPED/ CANCELED 詳細
  - stderrの末尾N行（例：50行）
  - skip_reason
- Artifacts一覧
  - `artifacts/<task_id>/...` のツリー（または一覧）

---

## 13. 偽エージェント（tools/fake_agent.py）仕様（必須）

### 13.1 目的
- 外部コマンドが不安定（失敗/遅延/大量ログ）でもorchが壊れないことを検証する
- 統合テストの安定化（再現性ある失敗を作る）

### 13.2 仕様
`python tools/fake_agent.py <subcommand> [options]`

共通オプション：
- `--sleep SEC`：実行前にsleep
- `--fail-rate P`：確率でexit 1
- `--fail-always`：常に失敗（テスト用）
- `--produce PATH`：成果物ファイルを作る（中身はJSONやテキストでよい）
- `--spam-bytes N`：stdoutにNバイト程度を吐く（ログ耐性）

subcommandは何でもよいが、examples用に最低3つ：
- `inspect`
- `build`
- `test`

---

## 14. 実装ガイド（モジュール単位の責務と関数シグネチャ）

> **指示**：ここに書いた責務と境界を守る（“なんでもcli.pyに書く”は禁止）。

### 14.1 `orch/util/*`
- `ids.py`
  - `def new_run_id(now: datetime) -> str`
- `time.py`
  - `def now_iso() -> str`
  - `def duration_sec(start: datetime, end: datetime) -> float`
- `paths.py`
  - `def run_dir(home: Path, run_id: str) -> Path`
  - `def ensure_run_layout(run_dir: Path) -> None`
- `tail.py`
  - `def tail_lines(path: Path, n: int) -> list[str]`（巨大ファイルでも破綻しない）
- `errors.py`
  - `class PlanError(Exception)` など（終了コード分岐に使う）

### 14.2 `orch/config/*`
- `schema.py`（dataclass）
  - `TaskSpec` / `PlanSpec`
- `loader.py`
  - `def load_plan(path: Path) -> PlanSpec`
  - `def normalize_cmd(cmd: str | list[str]) -> list[str]`
  - `def validate_plan(plan: PlanSpec) -> None`（ルールは6章）

### 14.3 `orch/dag/*`
- `build.py`
  - `def build_adjacency(plan: PlanSpec) -> tuple[dict[str, list[str]], dict[str, int]]`
    - dependents（隣接）と in_degree（依存数）を返す
- `validate.py`
  - `def assert_acyclic(task_ids: list[str], dependents: dict[str, list[str]], in_degree: dict[str, int]) -> None`
    - Kahn法で循環検出。循環ならPlanError

### 14.4 `orch/state/*`
- `model.py`（dataclass）
  - `TaskState`, `RunState`
- `store.py`
  - `def load_state(run_dir: Path) -> RunState`
  - `def save_state_atomic(run_dir: Path, state: RunState) -> None`
- `lock.py`
  - `@contextmanager def run_lock(run_dir: Path, stale_sec: int = 3600) -> Iterator[None]`

### 14.5 `orch/exec/*`
- `cancel.py`
  - `def cancel_requested(run_dir: Path) -> bool`
  - `def write_cancel_request(run_dir: Path) -> None`
- `capture.py`
  - `async def stream_to_file(stream: asyncio.StreamReader, file_path: Path) -> None`
- `timeout.py`
  - `async def wait_with_timeout(proc, timeout_sec: float | None) -> tuple[bool, int | None]`
- `retry.py`
  - `def backoff_for_attempt(attempt_idx: int, backoff: list[float]) -> float`
- `runner.py`
  - `async def run_plan(plan: PlanSpec, run_dir: Path, *, max_parallel: int, fail_fast: bool, workdir: Path, resume: bool, failed_only: bool) -> RunState`
  - `async def run_task(task: TaskSpec, run_dir: Path, *, attempt: int, default_cwd: Path) -> "TaskResult"`

### 14.6 `orch/report/*`
- `summarize.py`
  - `def build_summary(state: RunState, run_dir: Path) -> dict`
- `render_md.py`
  - `def render_markdown(summary: dict) -> str`

### 14.7 `orch/cli.py`
- `typer.Typer()` を `app` として定義
- ここは「引数パース→run_lock→ロード→実行→exit code」に集中  
  実処理は各モジュールへ委譲

---

## 18. 完了条件（Release 0.1 Definition of Done）
以下が全て満たされて「完成」とする：

- [ ] `orch run examples/plan_basic.yaml` が成功し終了コード0
- [ ] `orch run examples/plan_parallel.yaml --max-parallel 2` が動き、並列実行の痕跡がある
- [ ] 失敗を含むplanで run がハングせずに終了する（SKIP伝播が働く）
- [ ] `orch resume <run_id>` が成功し、成功済みは再実行されない
- [ ] `orch cancel <run_id>` が実行中runを止められる
- [ ] `orch status/logs` が実用レベルで見れる
- [ ] `report/final_report.md` が生成され、結果が読める
- [ ] `ruff format --check .` / `ruff check .` / `pytest` が通る
- [ ] READMEのQuickstartで再現できる
- [ ] CIが動く（任意だが推奨）

---

## 19. 実装時の注意（落とし穴を先に潰す）
- **SKIP伝播を必ず実装**：これがないと失敗時にrunが終わらず、自走テストが破綻する
- `cmd` は string を shell 実行しない（`create_subprocess_shell` は避ける）
- stdout/stderr を `communicate()` でまとめて受けない（大出力で死ぬ）  
  → 必ずストリーミング保存
- state.json は必ず atomic write
- lock は必須（cancel/resumeが絡むと壊れやすい）
- resume時、残存RUNNINGをどう扱うかを固定する（本書の仕様に従う）
