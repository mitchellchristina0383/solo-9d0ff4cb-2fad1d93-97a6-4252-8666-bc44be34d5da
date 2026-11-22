# 混合精度训练 · 单作业协调 API

FastAPI + SQLite 实现的单作业混合精度训练协调服务。任一进程梯度溢出即跳过整步，
动态损失缩放倍率从 16 起步、溢出时减半（最低 1）。

在此之上支持**试跑停步**：登记作业时可设单个权重的改变量上限 `max_weight_delta`。
梯度收齐后若有任一 `|Δw|` 超过上限，就停在原步、保留原权重和倍率，生成带编号的候选权重；
工程师调低本步学习率后用已受理梯度重算候选，再确认未超限的候选继续训练。

## 运行

```bash
docker compose up --build
# 打开 http://localhost:8000/docs 进行交互演示
```

本地开发（不用 Docker）：

```bash
pip install -r requirements.txt
DB_PATH=./coordinator.db uvicorn app.main:app --port 8000
```

SQLite 数据库落在 Compose 命名卷 `coordinator-data`（容器内 `/data/coordinator.db`），
服务/容器重启后，已受理的上报、权重、倍率、步决议和试跑候选全部保留。

## 实际实现

### 数据模型（SQLite，4 张表）

| 表 | 内容 | 关键约束 |
|---|---|---|
| `job` | 单行：进程数、学习率、当前权重(JSON)、当前倍率、当前步号、`max_delta`、`blocked` | `id = 1` 单例 |
| `reports` | 已受理的上报：步号、rank、类型(grad/overflow)、梯度(JSON)、受理时间 | `PRIMARY KEY(step, rank)` —— 每进程每步只采纳首报 |
| `decisions` | 每步决议：updated/skipped、本步倍率、新倍率、步后权重、平均梯度、是否试跑、候选号、本步实际 lr | `PRIMARY KEY(step)` —— 每步只有一份决议 |
| `candidates` | 试跑停步的候选：步号、候选号、本步 lr、候选权重、重算用平均梯度、最大 \|Δw\|、是否超限、状态 | `PRIMARY KEY(step, candidate)`；部分唯一索引保证每步至多一个 `current` |

`candidates.status` 取 `current`(当前可确认) / `superseded`(已被新候选替换) /
`confirmed`(已确认并落库)。

应用本身无内存状态，SQLite 是唯一事实来源，因此重启天然保留全部状态。
`init_db()` 会对旧库自动补列（`job.max_delta`、`job.blocked`、`decisions.trial/candidate/lr`），
旧卷可直接升级，无需迁移脚本。

### API

| 端点 | 说明 |
|---|---|
| `POST /job` | 登记作业：进程数、初始权重向量、学习率、可选 `max_weight_delta`。倍率初始化 16，步号从 0 起步。重复登记 409 |
| `GET /state` | 当前步号、权重、倍率、学习率、上限、`blocked`、`current_candidate`，及本步已受理上报 |
| `POST /report` | 按 `{step, rank, kind, grad?}` 上报已缩放梯度或溢出（语义见下） |
| `GET /steps/{step}` | 该步决议、带编号的候选列表（含 `confirmable`）、已受理上报 |
| `POST /steps/{step}/recompute` | `{"lr": 更小的值}`：用本步已受理梯度重算候选并替换当前候选 |
| `POST /steps/{step}/confirm` | `{"candidate": 编号}`：确认当前未超限候选，权重/决议/步进同时生效 |
| `DELETE /job` | 清空全部状态（仅演示/测试用） |
| `GET /healthz` | 健康检查 |

`max_weight_delta` 也接受 `max_delta` / `delta_limit` / `weight_delta_limit` 作为别名；
`recompute` 的 `lr` 也接受 `learning_rate`；`confirm` 的 `candidate` 也接受
`candidate_id` / `candidate_no`。请求体只写一个字段即可。

### `POST /report` 语义

- **旧步（step < 当前步）**：迟报/重试。不入库、不改状态，只返回该步决议（`late: true`）。
  试跑确认的步返回的决议带 `trial: true` 和当时的 `candidate`/`lr`。
- **未来步（step > 当前步）**：409 拒绝。
- **当前步已停步（`blocked`）**：仍按首报规则受理/拒绝（`INSERT OR IGNORE`），
  但**绝不**触发溢出跳步或自动更新，当前候选原样保留；响应 `blocked: true` 并带回当前候选。
- **当前步未停步**：首报入库后在同一事务内检查：
  - 本步存在任一 overflow → **立即**决议 `skipped`：跳过更新、倍率减半（`max(1, scale/2)`）、
    权重不变，进入下一步。不等其余进程。
  - 否则上报数 == 进程数 → 用已受理梯度算平均、除以本步倍率、`w -= lr * grad`：
    - 未设上限，或全部 `|Δw| <= max_delta` → 直接更新（普通步），倍率不变，进入下一步；
    - 任一 `|Δw| > max_delta` → **停步**：`job.step`、`job.weights`、`job.scale` 全部不动，
      `blocked=1`，写入候选 1（`over_limit: true`），不产生步决议。
- 决议与步进在同一 `BEGIN IMMEDIATE` 事务中提交。
- 参数校验：rank 越界、梯度维度与权重不符、`kind=grad` 缺梯度均返回 422。
- 边界判定用容差：`|Δw|` 与上限在浮点误差内相等（如 `0.1*1.5` 得 `0.15000000000000002`）
  视为未超限，不会误停步。

### 试跑：候选 / 重算 / 确认

- **候选编号**在步内从 1 单调递增；旧 `current` 被重算替换时变 `superseded`，
  历史候选保留可查。
- **重算** `POST /steps/{step}/recompute`：
  - 只允许对 `job.step == step` 且 `blocked` 的步操作，否则 409；
  - `lr` 必须**严格小于**当前候选的 lr（只能调低），否则 422；
  - 用**已受理的首报梯度**重算（不接受新的梯度输入），候选 = `mean(g)/scale*lr`；
  - 重算可能产生仍超限的候选（`over_limit: true`），它不可确认，可继续调低 lr 重算；
  - 校验失败或异常时整个事务回滚，原候选保持 `current`（`recompute` 失败不丢候选）。
- **确认** `POST /steps/{step}/confirm`：
  - 只接受 `current` 且 `over_limit=false` 的候选；超限候选 409，已被替换的
    `superseded` 候选号 409，不存在的候选号 404；
  - 权重更新、`decisions` 记录（`trial: true`、候选号、实际 lr）、步进
    `step+1` / `blocked=0` 在**同一个** `BEGIN IMMEDIATE` 事务内提交，要么全生效要么全不生效；
  - 确认成功后，**同号重试**返回当时落库的原决议（`already_confirmed: true`），不会重复步进；
  - 确认不改倍率（试跑停步不触发溢出的减半逻辑）。
- 并发：进程内 `threading.Lock` + SQLite `BEGIN IMMEDIATE`，且部分唯一索引
  `candidates_one_current` 保证每步最多一个 `current`；并发确认只会产生一次步进，
  并发重算 + 确认按先到者串行化：确认先赢则重算全部 409，重算先赢则确认全部 409
  且候选被替换。

### 并发与一致性

- 写路径：进程内 `threading.Lock` + SQLite `BEGIN IMMEDIATE` 事务，读-改-写整体串行。
- 首报唯一由 `reports` 主键 + `INSERT OR IGNORE` 保证；每步单决议由 `decisions` 主键保证；
  每步单当前候选由部分唯一索引保证。

## 验证结果

本环境**无 Docker**，Compose 配置未实际构建；服务在本地以 uvicorn（与镜像相同的
`fastapi==0.165.0` / `uvicorn==0.54.0` 版本）直接运行，用 `scripts/verify.py`
（仅标准库）分四阶段验证，**全部 PASS**，并连续跑 3 轮无抖动：

```bash
python3 scripts/verify.py phase1   # 基础主流程（32 项断言）
# 重启服务，用同一数据库
python3 scripts/verify.py phase2   # 重启后基础状态（9 项断言）
python3 scripts/verify.py phase3   # 试跑停步/候选/重算/确认/交错/并发（31 项断言）
# 再次重启服务，用同一数据库
python3 scripts/verify.py phase4   # 重启后候选仍在 + 继续重算/确认/竞态（17 项断言）
```

phase1（作业：2 进程、权重 `[1.0,-1.0]`、lr=0.1、不设上限）：
登记/重复登记 409；rank 越界/维度不符/未来步 4xx；step0 正常更新为 `[0.85,-1.05]`；
同 rank 重报被拒；step1 溢出立即跳步、倍率 16→8；溢出迟报只读；step2 按首报更新为
`[0.75,-1.05]`；step3–6 连续溢出倍率 8→4→2→1→1（下限）。

phase2：重启后 step7/scale1/权重与挂起上报保留；旧步重报返回持久化决议；当前步首报跨重启唯一；
补齐梯度后更新为 `[0.55,-1.25]`。

phase3（新作业：2 进程、`[1.0,-1.0]`、lr=0.1、`max_weight_delta=0.15`）：
- **边界**：`|Δw|` 恰为 0.15（浮点算得 0.15000000000000002）判为未超限，普通步直接更新为 `[0.85,-1.05]`；
- 超限步停住：`step` 仍为 1、权重 `[0.85,-1.05]`、倍率 16、`blocked=true`、候选 1 `over_limit`；
- 停步期间同 rank 重报被拒且候选保留；超限候选确认 409；
- 重算 lr 未调低 422；lr=0.01 得候选 2（仍超限、不可确认）；lr=0.0005 得候选 3（未超限、可确认）；
- 已被替换的候选 1/2 确认均 409，候选状态 `1→superseded, 2→superseded, 3→current`；
- **重算与确认交错**：候选 3 被重算替换为 4 后，再确认 3 得 409；
- 8 并发确认候选 4：全部 200、只一次 `already_confirmed=false`、只步进一次，决议一致；
- 确认后权重/步号/决议同时生效为 `[0.83,-1.052]`、step2、`blocked=false`；同号重试幂等；
- step2 未超限普通步直接更新为 `[0.73,-1.052]`；step3 有效溢出照常 16→8 跳步；
- step4 再次超限停住（候选 1），留待重启。

phase4（重启后）：
- step4 仍 `blocked`、原权重 `[0.73,-1.052]`、倍率 8、候选 1 可查且无决议；
- 重启后可继续重算为候选 2（未超限）；重算失败（lr 未调低）422 且候选 2 保留；
- 确认候选 2 生效为 `[0.65,-1.052]`、step5；同号重试幂等；重报旧步返回 `trial:true` 决议；
- 候选历史 `1/2/3 superseded, 4 confirmed` 完整；未来步仍 409；
- **真实竞态**：4 个确认线程 + 4 个重算线程同时打同一步，确认先赢时恰好一次决议/一次步进、
  重算全部 409；手工构造重算先赢时确认全部 409、候选被替换为新的 `current`、无决议；
  两种顺序都无 500，状态可继续处理。

另单独验证：旧版 schema 数据库（无新列）启动后自动补列并能正常读旧决议；
`DELETE /job` 后 `GET /state` 409；不存在的候选 404；`/docs` 与 `/openapi.json` 均 200，
OpenAPI 列出全部 8 个路径，可直接在页面演示上述流程。

## 未完成项 / 已知限制

- **Docker 未实测**：本环境无 Docker，`Dockerfile`/`docker-compose.yml` 经过静态检查
  （YAML 合法、依赖版本与本地实测一致）但未实际构建运行；本地等效命令（uvicorn + 同一 DB 路径）已全量验证。
- **单作业、单实例**：只支持一个作业；多副本水平扩展未做（SQLite 单写者，跨进程正确性依赖
  `BEGIN IMMEDIATE` 串行化，多实例需换 PostgreSQL 等外部库）。
- **无鉴权**：所有端点（含 `DELETE /job` 重置、候选确认）无认证授权，仅限可信内网/演示。
- **无进程存活跟踪**：进程掉线不上报时会永远等待，无超时/心跳/占位跳过机制；
  停步后也需人工操作，无自动超时/自动重算。
- **重算只接受首报梯度**：即停步时已受理的 `reports` 行；后续到达的迟报梯度不会并入候选计算
  （迟报语义本身要求不改状态）。
- **学习率只能逐步调低**：重算要求新 lr 严格小于当前候选 lr；如需调高须先重置作业。
- **倍率只降不升**：按需求只实现溢出减半（下限 1），未实现连续成功后升倍率的策略；确认试跑也不改倍率。
- **单权重上限为全局标量**：`max_weight_delta` 对所有权重维度取同一个上限，不支持逐维上限。
- **步号从 0 起步**、时间戳为 UTC（`datetime('now')`），如需 1 起步或本地时区可调整。
- 权重/梯度以 JSON 数组存 SQLite，适合演示与小向量；大规模向量需换二进制/外部存储。
