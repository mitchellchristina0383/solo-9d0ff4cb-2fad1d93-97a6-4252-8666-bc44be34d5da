"""混合精度训练 · 单作业协调 API.

协调 N 个训练进程的梯度/溢出上报:
- 作业登记: 进程数、初始权重向量、学习率; 可选单个权重的改变量上限;
  缩放倍率从 16 起步, 步号从 0 起步
- 每进程每步只采纳首报 (reports 表 PRIMARY KEY(step, rank) + INSERT OR IGNORE)
- 任一进程报溢出 -> 立即跳过更新, 倍率减半 (最低 1), 不等其余进程
- 梯度齐全且未设上限 / 全部权重改变量都在上限内 -> 按学习率直接更新一次
- 梯度齐全但任一 |Δw| 超过上限 -> 试跑停步:
    * 保持原步、原权重、原倍率, 不产生步决议
    * 在 candidates 表保存带编号的候选权重, 当前候选可在 /steps/{step} 查看
    * 工程师可用更小的本步学习率重算: 用已受理梯度重算, 新候选替换旧候选
    * 确认当前且未超限的候选时, 权重 / 步决议 / 步进在同一个写事务里生效
    * 确认成功后同候选号重试幂等返回原决议; 被替换的候选号不可再确认
- 全部状态存 SQLite, 服务重启后保留已受理上报、权重、倍率、步决议和候选
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
from contextlib import asynccontextmanager, contextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

DB_PATH = os.environ.get("DB_PATH", "/data/coordinator.db")
INITIAL_SCALE = 16.0
MIN_SCALE = 1.0

# 进程内锁: 串行化写路径。跨进程的正确性由 SQLite 的
# BEGIN IMMEDIATE 事务 + 唯一约束兜底, 因此单 worker 即可安全运行。
_lock = threading.RLock()


# ---------------------------------------------------------------------------
# 存储层
# ---------------------------------------------------------------------------

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


@contextmanager
def write_tx():
    """BEGIN IMMEDIATE 写事务: 读-改-写整体串行化, 异常即回滚."""
    conn = _connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


@contextmanager
def read_conn():
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def _ensure_column(conn: sqlite3.Connection, table: str, name: str, ddl: str) -> None:
    """给已有数据库补列, 让旧卷上的 SQLite 文件也能直接升级."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if name not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")


def init_db() -> None:
    directory = os.path.dirname(DB_PATH)
    if directory:
        os.makedirs(directory, exist_ok=True)
    # executescript 自行管理事务, 不走 write_tx
    conn = _connect()
    try:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS job (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                num_processes INTEGER NOT NULL,
                lr REAL NOT NULL,
                weights TEXT NOT NULL,        -- JSON 数组, 当前权重
                scale REAL NOT NULL,          -- 当前缩放倍率
                step INTEGER NOT NULL,        -- 当前步号
                max_delta REAL,               -- 单个权重改变量上限; NULL = 不启用试跑停步
                blocked INTEGER NOT NULL DEFAULT 0  -- 1 = 当前步因超限停住, 等待确认候选
            );
            CREATE TABLE IF NOT EXISTS reports (
                step INTEGER NOT NULL,
                rank INTEGER NOT NULL,
                kind TEXT NOT NULL CHECK (kind IN ('grad', 'overflow')),
                grad TEXT,                    -- JSON 数组; overflow 时为 NULL
                received_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (step, rank)      -- 每进程每步只采纳首报
            );
            CREATE TABLE IF NOT EXISTS decisions (
                step INTEGER PRIMARY KEY,     -- 每步只有一份决议
                outcome TEXT NOT NULL CHECK (outcome IN ('updated', 'skipped')),
                scale REAL NOT NULL,          -- 本步使用的倍率
                new_scale REAL NOT NULL,      -- 步结束后的倍率
                weights TEXT NOT NULL,        -- 步结束后的权重 (JSON)
                avg_grad TEXT,                -- 平均梯度 (JSON); 溢出跳过时为 NULL
                decided_at TEXT NOT NULL DEFAULT (datetime('now')),
                trial INTEGER NOT NULL DEFAULT 0,  -- 1 = 试跑候选确认产生的更新
                candidate INTEGER,                 -- 试跑确认时使用的候选编号
                lr REAL                            -- 本步实际使用的学习率
            );
            CREATE TABLE IF NOT EXISTS candidates (
                step INTEGER NOT NULL,
                candidate INTEGER NOT NULL,   -- 步内候选编号, 从 1 递增
                lr REAL NOT NULL,             -- 生成本候选时使用的本步学习率
                weights TEXT NOT NULL,        -- 候选权重 (JSON)
                avg_grad TEXT NOT NULL,       -- 重算所用的已受理平均梯度 (JSON)
                max_abs_delta REAL NOT NULL,  -- 本候选相对原权重的最大 |Δw|
                over_limit INTEGER NOT NULL,  -- 1 = 仍超过本作业的 max_delta
                status TEXT NOT NULL CHECK (status IN ('current', 'superseded', 'confirmed')),
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (step, candidate)
            );
            CREATE UNIQUE INDEX IF NOT EXISTS candidates_one_current
                ON candidates(step) WHERE status = 'current';
            """
        )
        # 旧版本数据库升级
        _ensure_column(conn, "job", "max_delta", "max_delta REAL")
        _ensure_column(conn, "job", "blocked", "blocked INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "decisions", "trial", "trial INTEGER NOT NULL DEFAULT 0")
        _ensure_column(conn, "decisions", "candidate", "candidate INTEGER")
        _ensure_column(conn, "decisions", "lr", "lr REAL")
    finally:
        conn.close()


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="Mixed-Precision Training Coordinator",
    version="2.0.0",
    description=(
        "单作业混合精度训练协调: 梯度汇总 / 溢出跳步 / 动态损失缩放 / "
        "权重改变量超限停步与候选确认。"
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# 请求模型
# ---------------------------------------------------------------------------

class JobConfig(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "num_processes": 2,
                "weights": [1.0, -1.0],
                "lr": 0.1,
                "max_weight_delta": 0.15,
            }
        },
    )

    num_processes: int = Field(gt=0, description="训练进程数")
    weights: list[float] = Field(min_length=1, description="初始权重向量")
    lr: float = Field(gt=0, description="全局学习率")
    max_weight_delta: Optional[float] = Field(
        default=None,
        gt=0,
        validation_alias=AliasChoices(
            "max_weight_delta", "max_delta", "delta_limit", "weight_delta_limit"
        ),
        description=(
            "可选的单个权重改变量上限 |Δw|。任一权重超过它时当前步停住, "
            "生成候选权重等待工程师调低本步学习率后确认; 不传则不做试跑停步。"
        ),
    )


class Report(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {"step": 0, "rank": 0, "kind": "grad", "grad": [16.0, 32.0]}
        }
    )

    step: int = Field(ge=0, description="上报所属步号")
    rank: int = Field(ge=0, description="进程编号, [0, num_processes)")
    kind: Literal["grad", "overflow"]
    grad: Optional[list[float]] = Field(default=None, description="已缩放梯度, kind=grad 时必填")


class RecomputeRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"example": {"lr": 0.0625}},
    )

    lr: float = Field(
        gt=0,
        validation_alias=AliasChoices("lr", "learning_rate"),
        description="本步新的学习率, 必须严格小于当前候选使用的学习率",
    )


class ConfirmRequest(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={"example": {"candidate": 2}},
    )

    candidate: int = Field(
        ge=1,
        validation_alias=AliasChoices("candidate", "candidate_id", "candidate_no"),
        description="要确认的当前候选编号",
    )


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _get_job(conn: sqlite3.Connection) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM job WHERE id = 1").fetchone()
    if row is None:
        raise HTTPException(409, "no job registered; POST /job first")
    return row


def _decision_dict(row: sqlite3.Row) -> dict:
    keys = row.keys()
    return {
        "step": row["step"],
        "outcome": row["outcome"],
        "scale": row["scale"],
        "new_scale": row["new_scale"],
        "weights": json.loads(row["weights"]),
        "avg_grad": json.loads(row["avg_grad"]) if row["avg_grad"] else None,
        "decided_at": row["decided_at"],
        "trial": bool(row["trial"]) if "trial" in keys else False,
        "candidate": row["candidate"] if "candidate" in keys else None,
        "lr": row["lr"] if "lr" in keys else None,
    }


def _candidate_dict(
    row: sqlite3.Row,
    *,
    blocked: bool,
    current_step: int,
    max_delta: Optional[float],
) -> dict:
    is_current = row["status"] == "current"
    over = bool(row["over_limit"])
    return {
        "step": row["step"],
        "candidate": row["candidate"],
        "lr": row["lr"],
        "weights": json.loads(row["weights"]),
        "avg_grad": json.loads(row["avg_grad"]),
        "max_abs_delta": row["max_abs_delta"],
        "max_weight_delta": max_delta,
        "over_limit": over,
        "status": row["status"],
        "current": is_current,
        "confirmable": bool(
            is_current and not over and blocked and row["step"] == current_step
        ),
        "created_at": row["created_at"],
    }


def _average_grads(grads: list[list[float]], dim: int) -> list[float]:
    n = len(grads)
    return [sum(g[i] for g in grads) / n for i in range(dim)]


def _candidate_weights(
    weights: list[float],
    grads: list[list[float]],
    lr: float,
    scale: float,
) -> tuple[list[float], list[float], float]:
    """按已受理梯度算候选: 平均 -> 除以本步倍率 -> w - lr * g."""
    avg = _average_grads(grads, len(weights))
    new_weights = [w - lr * (a / scale) for w, a in zip(weights, avg)]
    max_abs_delta = max(abs(nw - w) for nw, w in zip(new_weights, weights))
    return avg, new_weights, max_abs_delta


def _exceeds(max_abs_delta: float, max_delta: Optional[float]) -> bool:
    """是否严格超过上限。浮点误差内等于上限不算超过。

    例如 lr=0.1、梯度/倍率=1.5 时 |Δw| 算出来是 0.15000000000000002,
    不能因此把恰好等于 0.15 上限的步误判为超限。
    """
    if max_delta is None:
        return False
    return max_abs_delta > max_delta and not math.isclose(
        max_abs_delta, max_delta, rel_tol=1e-9, abs_tol=1e-12
    )


def _grads_for_step(conn: sqlite3.Connection, step: int) -> list[list[float]]:
    rows = conn.execute(
        "SELECT grad FROM reports WHERE step = ? AND kind = 'grad' ORDER BY rank",
        (step,),
    ).fetchall()
    return [json.loads(r["grad"]) for r in rows]


def _step_has_overflow(conn: sqlite3.Connection, step: int) -> bool:
    return conn.execute(
        "SELECT 1 FROM reports WHERE step = ? AND kind = 'overflow' LIMIT 1",
        (step,),
    ).fetchone() is not None


def _current_candidate(conn: sqlite3.Connection, step: int) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM candidates WHERE step = ? AND status = 'current'", (step,)
    ).fetchone()


def _create_candidate(
    conn: sqlite3.Connection,
    *,
    step: int,
    lr: float,
    avg: list[float],
    weights: list[float],
    max_abs_delta: float,
    over_limit: bool,
) -> sqlite3.Row:
    """把旧 current 候选降为 superseded, 写入新的 current 候选."""
    conn.execute(
        "UPDATE candidates SET status = 'superseded' WHERE step = ? AND status = 'current'",
        (step,),
    )
    number = conn.execute(
        "SELECT COALESCE(MAX(candidate), 0) + 1 AS n FROM candidates WHERE step = ?",
        (step,),
    ).fetchone()["n"]
    conn.execute(
        "INSERT INTO candidates"
        " (step, candidate, lr, weights, avg_grad, max_abs_delta, over_limit, status)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, 'current')",
        (step, number, lr, json.dumps(weights), json.dumps(avg),
         max_abs_delta, 1 if over_limit else 0),
    )
    return conn.execute(
        "SELECT * FROM candidates WHERE step = ? AND candidate = ?", (step, number)
    ).fetchone()


def _decide_skip(conn: sqlite3.Connection, job: sqlite3.Row) -> dict:
    """溢出: 跳过更新, 倍率减半 (最低 1), 权重不变, 进入下一步."""
    step, scale = job["step"], job["scale"]
    new_scale = max(MIN_SCALE, scale / 2)
    conn.execute(
        "INSERT INTO decisions"
        " (step, outcome, scale, new_scale, weights, avg_grad, trial, candidate, lr)"
        " VALUES (?, 'skipped', ?, ?, ?, NULL, 0, NULL, NULL)",
        (step, scale, new_scale, job["weights"]),
    )
    conn.execute("UPDATE job SET step = ?, scale = ? WHERE id = 1", (step + 1, new_scale))
    return {
        "step": step,
        "outcome": "skipped",
        "scale": scale,
        "new_scale": new_scale,
        "weights": json.loads(job["weights"]),
        "avg_grad": None,
        "trial": False,
        "candidate": None,
        "lr": None,
    }


def _decide_update(
    conn: sqlite3.Connection,
    job: sqlite3.Row,
    new_weights: list[float],
    avg: list[float],
) -> dict:
    """梯度齐全且未超限: 直接应用一次更新, 倍率不变, 进入下一步."""
    step, scale, lr = job["step"], job["scale"], job["lr"]
    conn.execute(
        "INSERT INTO decisions"
        " (step, outcome, scale, new_scale, weights, avg_grad, trial, candidate, lr)"
        " VALUES (?, 'updated', ?, ?, ?, ?, 0, NULL, ?)",
        (step, scale, scale, json.dumps(new_weights), json.dumps(avg), lr),
    )
    conn.execute(
        "UPDATE job SET step = ?, weights = ? WHERE id = 1",
        (step + 1, json.dumps(new_weights)),
    )
    return {
        "step": step,
        "outcome": "updated",
        "scale": scale,
        "new_scale": scale,
        "weights": new_weights,
        "avg_grad": avg,
        "trial": False,
        "candidate": None,
        "lr": lr,
    }


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "mixed-precision-training-coordinator",
        "docs": "/docs",
        "endpoints": [
            "POST /job", "GET /state", "POST /report", "GET /steps/{step}",
            "POST /steps/{step}/recompute", "POST /steps/{step}/confirm",
            "DELETE /job", "GET /healthz",
        ],
    }


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/job", status_code=201)
def create_job(cfg: JobConfig):
    """登记作业: 进程数、初始权重、学习率、可选单权重改变量上限。

    倍率从 16 起步, 步号从 0 起步。
    """
    with _lock, write_tx() as conn:
        if conn.execute("SELECT id FROM job WHERE id = 1").fetchone():
            raise HTTPException(409, "job already registered; DELETE /job to reset")
        conn.execute(
            "INSERT INTO job"
            " (id, num_processes, lr, weights, scale, step, max_delta, blocked)"
            " VALUES (1, ?, ?, ?, ?, 0, ?, 0)",
            (cfg.num_processes, cfg.lr, json.dumps(cfg.weights), INITIAL_SCALE,
             cfg.max_weight_delta),
        )
    return {
        "ok": True,
        "num_processes": cfg.num_processes,
        "weights": cfg.weights,
        "lr": cfg.lr,
        "max_weight_delta": cfg.max_weight_delta,
        "scale": INITIAL_SCALE,
        "step": 0,
        "blocked": False,
    }


@app.get("/state")
def get_state():
    """进程读取: 当前步号、权重、倍率、学习率、是否停步及本步已受理上报."""
    with read_conn() as conn:
        job = _get_job(conn)
        reports = conn.execute(
            "SELECT rank, kind, received_at FROM reports WHERE step = ? ORDER BY rank",
            (job["step"],),
        ).fetchall()
        current = _current_candidate(conn, job["step"]) if job["blocked"] else None
    return {
        "step": job["step"],
        "scale": job["scale"],
        "weights": json.loads(job["weights"]),
        "lr": job["lr"],
        "max_weight_delta": job["max_delta"],
        "num_processes": job["num_processes"],
        "blocked": bool(job["blocked"]),
        "current_candidate": current["candidate"] if current else None,
        "reports_received": [dict(r) for r in reports],
    }


@app.post("/report")
def post_report(rep: Report):
    """按步号和进程编号上报已缩放梯度或溢出。

    - 旧步 (step < 当前步): 迟报/重试, 只返回该步决议, 不改状态
    - 未来步 (step > 当前步): 409 拒绝
    - 当前步已因超限停住: 仍按首报规则受理/拒绝, 但绝不产生新候选或跳步,
      当前候选保留
    - 当前步: 首报入库; 有溢出立即跳步减半倍率; 梯度齐全时
      * 未设上限或全部 |Δw| <= 上限 -> 直接更新
      * 任一 |Δw| > 上限 -> 停步并生成编号候选, 等待人工确认
    """
    with _lock, write_tx() as conn:
        job = _get_job(conn)
        n_proc = job["num_processes"]
        weights = json.loads(job["weights"])
        if rep.rank >= n_proc:
            raise HTTPException(422, f"rank {rep.rank} out of range [0, {n_proc})")
        if rep.kind == "grad":
            if rep.grad is None:
                raise HTTPException(422, "grad is required when kind='grad'")
            if len(rep.grad) != len(weights):
                raise HTTPException(
                    422, f"grad length {len(rep.grad)} != weights length {len(weights)}"
                )
        cur = job["step"]

        # 旧步: 迟报或重试, 只返回该步决议
        if rep.step < cur:
            row = conn.execute(
                "SELECT * FROM decisions WHERE step = ?", (rep.step,)
            ).fetchone()
            return {
                "accepted": False,
                "late": True,
                "duplicate": False,
                "step": rep.step,
                "current_step": cur,
                "scale": job["scale"],
                "blocked": False,
                "candidate": None,
                "decision": _decision_dict(row) if row else None,
            }
        if rep.step > cur:
            raise HTTPException(
                409, f"report for future step {rep.step}; current step is {cur}"
            )

        # 当前步: 每进程每步只采纳首报
        cursor = conn.execute(
            "INSERT OR IGNORE INTO reports (step, rank, kind, grad) VALUES (?, ?, ?, ?)",
            (rep.step, rep.rank, rep.kind,
             json.dumps(rep.grad) if rep.kind == "grad" else None),
        )
        accepted = cursor.rowcount == 1

        # 已停步: 首报规则照旧, 但保留候选, 不做溢出跳步/自动更新
        if job["blocked"]:
            cand = _current_candidate(conn, cur)
            return {
                "accepted": accepted,
                "late": False,
                "duplicate": not accepted,
                "step": rep.step,
                "current_step": cur,
                "scale": job["scale"],
                "blocked": True,
                "candidate": _candidate_dict(
                    cand, blocked=True, current_step=cur, max_delta=job["max_delta"]
                ) if cand else None,
                "decision": None,
            }

        decision = None
        cand_row = None
        if _step_has_overflow(conn, cur):
            # 任一进程报溢出: 立即跳过更新, 倍率减半
            decision = _decide_skip(conn, job)
        else:
            cnt = conn.execute(
                "SELECT COUNT(*) AS c FROM reports WHERE step = ?", (cur,)
            ).fetchone()["c"]
            if cnt == n_proc:
                grads = _grads_for_step(conn, cur)
                avg, new_weights, max_abs_delta = _candidate_weights(
                    weights, grads, job["lr"], job["scale"]
                )
                over = _exceeds(max_abs_delta, job["max_delta"])
                if over:
                    # 试跑停步: 原步/原权重/原倍率不变, 只落候选
                    cand_row = _create_candidate(
                        conn,
                        step=cur,
                        lr=job["lr"],
                        avg=avg,
                        weights=new_weights,
                        max_abs_delta=max_abs_delta,
                        over_limit=True,
                    )
                    conn.execute("UPDATE job SET blocked = 1 WHERE id = 1")
                else:
                    decision = _decide_update(conn, job, new_weights, avg)

        new_job = conn.execute("SELECT * FROM job WHERE id = 1").fetchone()
        if cand_row is None and new_job["blocked"]:
            cand_row = _current_candidate(conn, new_job["step"])
        return {
            "accepted": accepted,
            "late": False,
            "duplicate": not accepted,
            "step": rep.step,
            "current_step": new_job["step"],
            "scale": new_job["scale"],
            "blocked": bool(new_job["blocked"]),
            "candidate": _candidate_dict(
                cand_row,
                blocked=bool(new_job["blocked"]),
                current_step=new_job["step"],
                max_delta=new_job["max_delta"],
            ) if cand_row else None,
            "decision": decision,
        }


@app.get("/steps/{step}")
def get_step(step: int):
    """查询某步的决议、候选权重和已受理上报.

    停住的步 decision 为 null, candidates 给出带编号的候选;
    confirmable=true 的候选可以被 POST /steps/{step}/confirm 确认。
    """
    with read_conn() as conn:
        job = _get_job(conn)
        dec = conn.execute(
            "SELECT * FROM decisions WHERE step = ?", (step,)
        ).fetchone()
        reports = conn.execute(
            "SELECT rank, kind, grad, received_at FROM reports"
            " WHERE step = ? ORDER BY rank",
            (step,),
        ).fetchall()
        cand_rows = conn.execute(
            "SELECT * FROM candidates WHERE step = ? ORDER BY candidate", (step,)
        ).fetchall()
        blocked = bool(job["blocked"]) and job["step"] == step
        current = next(
            (r for r in cand_rows if r["status"] == "current"), None
        )
    return {
        "step": step,
        "blocked": blocked,
        "max_weight_delta": job["max_delta"],
        "current_candidate": current["candidate"] if current else None,
        "decision": _decision_dict(dec) if dec else None,
        "candidates": [
            _candidate_dict(
                r, blocked=blocked, current_step=step, max_delta=job["max_delta"]
            )
            for r in cand_rows
        ],
        "reports": [
            {
                "rank": r["rank"],
                "kind": r["kind"],
                "grad": json.loads(r["grad"]) if r["grad"] else None,
                "received_at": r["received_at"],
            }
            for r in reports
        ],
    }


@app.post("/steps/{step}/recompute")
def recompute_candidate(step: int, req: RecomputeRequest):
    """用本步已受理的梯度与更小的学习率重算候选, 替换当前候选.

    - 只允许对当前停住的步操作
    - lr 必须严格小于当前候选的 lr (只能调低)
    - 成功: 旧 current 候选变 superseded, 新候选编号 +1 成为 current
    - 失败 (校验失败/计算异常): 事务回滚, 原候选保持 current
    """
    with _lock, write_tx() as conn:
        job = _get_job(conn)
        if job["step"] != step:
            raise HTTPException(
                409, f"step {step} is not the current step {job['step']}"
            )
        if not job["blocked"]:
            raise HTTPException(409, f"step {step} is not blocked on a trial candidate")
        current = _current_candidate(conn, step)
        if current is None:
            raise HTTPException(409, f"step {step} has no current candidate")
        if req.lr >= current["lr"]:
            raise HTTPException(
                422,
                f"lr must be strictly lower than current candidate lr {current['lr']}",
            )

        weights = json.loads(job["weights"])
        n_proc = job["num_processes"]
        grads = _grads_for_step(conn, step)
        if len(grads) != n_proc or _step_has_overflow(conn, step):
            raise HTTPException(409, "accepted gradients for this step are incomplete")
        for g in grads:
            if len(g) != len(weights):
                raise HTTPException(409, "accepted gradient dimension mismatch")

        avg, new_weights, max_abs_delta = _candidate_weights(
            weights, grads, req.lr, job["scale"]
        )
        over = _exceeds(max_abs_delta, job["max_delta"])
        row = _create_candidate(
            conn,
            step=step,
            lr=req.lr,
            avg=avg,
            weights=new_weights,
            max_abs_delta=max_abs_delta,
            over_limit=over,
        )
        return {
            "ok": True,
            "step": step,
            "blocked": True,
            "candidate": _candidate_dict(
                row, blocked=True, current_step=step, max_delta=job["max_delta"]
            ),
        }


@app.post("/steps/{step}/confirm")
def confirm_candidate(step: int, req: ConfirmRequest):
    """确认当前未超限的候选.

    权重、step 决议、进入下一步在同一个 BEGIN IMMEDIATE 事务里提交。
    - 仅 current 且 over_limit=false 的候选可确认
    - 已被替换 (superseded) 的候选号不可确认
    - 确认成功后, 同候选号再次请求幂等返回原决议, 不会重复步进
    """
    with _lock, write_tx() as conn:
        job = _get_job(conn)
        cand = conn.execute(
            "SELECT * FROM candidates WHERE step = ? AND candidate = ?",
            (step, req.candidate),
        ).fetchone()

        # 幂等重试: 该候选已是 confirmed, 返回当时落库的决议
        if cand is not None and cand["status"] == "confirmed":
            dec = conn.execute(
                "SELECT * FROM decisions WHERE step = ?", (step,)
            ).fetchone()
            if dec is None:
                raise HTTPException(409, f"step {step} has no decision to return")
            return {
                "confirmed": True,
                "already_confirmed": True,
                "step": step,
                "candidate": cand["candidate"],
                "decision": _decision_dict(dec),
                "current_step": job["step"],
                "current_weights": json.loads(job["weights"]),
                "scale": job["scale"],
            }

        if job["step"] != step:
            raise HTTPException(
                409, f"step {step} is not the current step {job['step']}"
            )
        if not job["blocked"]:
            raise HTTPException(409, f"step {step} is not blocked on a trial candidate")
        if cand is None:
            raise HTTPException(404, f"candidate {req.candidate} not found for step {step}")
        if cand["status"] == "superseded":
            raise HTTPException(
                409, f"candidate {req.candidate} has been superseded and cannot be confirmed"
            )
        if cand["status"] != "current":
            raise HTTPException(
                409, f"candidate {req.candidate} is not the current candidate"
            )
        if cand["over_limit"]:
            raise HTTPException(
                409,
                f"candidate {req.candidate} exceeds max_weight_delta "
                f"{job['max_delta']}; recompute with a lower lr first",
            )

        new_weights = json.loads(cand["weights"])
        avg = json.loads(cand["avg_grad"])
        scale = job["scale"]
        # 权重 / 步决议 / 步进 三件事在同一事务内生效
        conn.execute(
            "INSERT INTO decisions"
            " (step, outcome, scale, new_scale, weights, avg_grad, trial, candidate, lr)"
            " VALUES (?, 'updated', ?, ?, ?, ?, 1, ?, ?)",
            (step, scale, scale, json.dumps(new_weights), json.dumps(avg),
             cand["candidate"], cand["lr"]),
        )
        conn.execute(
            "UPDATE job SET step = ?, weights = ?, blocked = 0 WHERE id = 1",
            (step + 1, json.dumps(new_weights)),
        )
        conn.execute(
            "UPDATE candidates SET status = 'confirmed'"
            " WHERE step = ? AND candidate = ?",
            (step, cand["candidate"]),
        )
        dec = conn.execute("SELECT * FROM decisions WHERE step = ?", (step,)).fetchone()
        return {
            "confirmed": True,
            "already_confirmed": False,
            "step": step,
            "candidate": cand["candidate"],
            "decision": _decision_dict(dec),
            "current_step": step + 1,
            "current_weights": new_weights,
            "scale": scale,
        }


@app.delete("/job")
def reset_job():
    """清空作业与全部历史 (仅用于演示/测试重置)。"""
    with _lock, write_tx() as conn:
        conn.execute("DELETE FROM candidates")
        conn.execute("DELETE FROM reports")
        conn.execute("DELETE FROM decisions")
        conn.execute("DELETE FROM job")
    return {"ok": True}
