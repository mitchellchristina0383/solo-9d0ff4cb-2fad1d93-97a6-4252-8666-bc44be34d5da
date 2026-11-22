#!/usr/bin/env python3
"""协调 API 验证脚本 (仅依赖标准库)。

用法:
    python3 scripts/verify.py phase1 [BASE_URL]   # 主流程: 更新/溢出/迟报/重试/倍率下限
    # ... 重启服务 (同一数据库) ...
    python3 scripts/verify.py phase2 [BASE_URL]   # 重启后: 状态保留 + 重报旧步/当前步
    python3 scripts/verify.py phase3 [BASE_URL]   # 试跑停步: 上限/候选/重算/确认/交错
    # ... 再次重启服务 (同一数据库) ...
    python3 scripts/verify.py phase4 [BASE_URL]   # 重启后: 停步候选仍在 + 重算/确认/幂等

phase1 结束时的预期状态: step=7, scale=1, weights=[0.75, -1.05],
且 rank0 已对 step7 上报梯度 [1,1] (未齐, 无决议)。
"""
import json
import sys
import threading
import urllib.error
import urllib.request

PHASE = sys.argv[1] if len(sys.argv) > 1 else "phase1"
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
TOL = 1e-9
failures = []


def req(method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(r) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def check(name, cond, detail=""):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))
    if not cond:
        failures.append(name)


def vec_close(a, b):
    return a is not None and len(a) == len(b) and all(abs(x - y) < TOL for x, y in zip(a, b))


def phase1():
    # 重置并登记作业: 2 进程, 权重 [1.0, -1.0], lr=0.1
    req("DELETE", "/job")
    st, j = req("POST", "/job", {"num_processes": 2, "weights": [1.0, -1.0], "lr": 0.1})
    check("登记作业 201", st == 201 and j["scale"] == 16.0 and j["step"] == 0, f"status={st}")
    check("未设上限时不启用试跑停步", j["max_weight_delta"] is None and j["blocked"] is False)
    st, _ = req("POST", "/job", {"num_processes": 2, "weights": [1.0], "lr": 0.1})
    check("重复登记被拒 409", st == 409, f"status={st}")

    st, s = req("GET", "/state")
    check("初始状态 step0/scale16", s["step"] == 0 and s["scale"] == 16.0
          and vec_close(s["weights"], [1.0, -1.0]), json.dumps(s))

    # 参数校验
    st, _ = req("POST", "/report", {"step": 0, "rank": 5, "kind": "overflow"})
    check("rank 越界 422", st == 422, f"status={st}")
    st, _ = req("POST", "/report", {"step": 0, "rank": 0, "kind": "grad", "grad": [1.0]})
    check("梯度维度不符 422", st == 422, f"status={st}")
    st, _ = req("POST", "/report", {"step": 9, "rank": 0, "kind": "overflow"})
    check("未来步被拒 409", st == 409, f"status={st}")

    # --- step 0: 正常更新 ---
    st, r = req("POST", "/report", {"step": 0, "rank": 0, "kind": "grad", "grad": [16.0, 32.0]})
    check("s0 rank0 首报受理", st == 200 and r["accepted"] and r["decision"] is None)
    st, r = req("POST", "/report", {"step": 0, "rank": 0, "kind": "grad", "grad": [999.0, 999.0]})
    check("s0 rank0 重报被拒(首报唯一)", r["accepted"] is False and r["duplicate"] is True)
    st, r = req("POST", "/report", {"step": 0, "rank": 1, "kind": "grad", "grad": [32.0, -16.0]})
    # avg=[24,8], /16=[1.5,0.5], w=[1-0.15, -1-0.05]=[0.85,-1.05]
    check("s0 齐全触发更新", r["decision"] and r["decision"]["outcome"] == "updated")
    check("s0 更新后权重正确", vec_close(r["decision"]["weights"], [0.85, -1.05]),
          json.dumps(r["decision"]["weights"]))
    check("s0 倍率不变", r["decision"]["new_scale"] == 16.0 and r["current_step"] == 1)

    # --- step 1: 溢出立即跳步 ---
    st, r = req("POST", "/report", {"step": 1, "rank": 0, "kind": "overflow"})
    check("s1 溢出立即跳过", r["decision"]["outcome"] == "skipped" and r["current_step"] == 2)
    check("s1 倍率减半 16->8", r["decision"]["new_scale"] == 8.0)
    check("s1 权重不变", vec_close(r["decision"]["weights"], [0.85, -1.05]))

    # --- 溢出后的迟报: 只返回该步决议 ---
    st, r = req("POST", "/report", {"step": 1, "rank": 1, "kind": "grad", "grad": [1.0, 1.0]})
    check("s1 迟报标记 late", r["late"] is True and r["accepted"] is False)
    check("s1 迟报返回该步决议(skipped)", r["decision"] and r["decision"]["outcome"] == "skipped")
    st, s = req("GET", "/state")
    check("迟报不改状态", s["step"] == 2 and s["scale"] == 8.0
          and vec_close(s["weights"], [0.85, -1.05]))

    # --- 旧步重试: 返回该步决议 ---
    st, r = req("POST", "/report", {"step": 0, "rank": 0, "kind": "grad", "grad": [0.0, 0.0]})
    check("s0 重试返回 updated 决议", r["late"] and r["decision"]["outcome"] == "updated"
          and vec_close(r["decision"]["weights"], [0.85, -1.05]))

    # --- step 2: 首报为梯度, 之后的溢出重报不生效 ---
    st, r = req("POST", "/report", {"step": 2, "rank": 0, "kind": "grad", "grad": [8.0, 8.0]})
    check("s2 rank0 梯度首报受理", r["accepted"] is True)
    st, r = req("POST", "/report", {"step": 2, "rank": 0, "kind": "overflow"})
    check("s2 rank0 溢出重报被拒(首报唯一)", r["duplicate"] is True and r["decision"] is None)
    st, r = req("POST", "/report", {"step": 2, "rank": 1, "kind": "grad", "grad": [8.0, -8.0]})
    # avg=[8,0], /8=[1,0], w=[0.85-0.1, -1.05]=[0.75,-1.05]
    check("s2 按首报更新", r["decision"]["outcome"] == "updated"
          and vec_close(r["decision"]["weights"], [0.75, -1.05]))
    check("s2 倍率仍为 8", r["decision"]["new_scale"] == 8.0)

    # --- step 3..6: 连续溢出, 倍率 8->4->2->1->1(下限) ---
    expect = [(3, 4.0), (4, 2.0), (5, 1.0), (6, 1.0)]
    for step, new_scale in expect:
        st, r = req("POST", "/report", {"step": step, "rank": 0, "kind": "overflow"})
        check(f"s{step} 溢出, 倍率->{new_scale}",
              r["decision"]["outcome"] == "skipped"
              and r["decision"]["new_scale"] == new_scale
              and r["current_step"] == step + 1)
    st, s = req("GET", "/state")
    check("倍率下限 1, 权重未被溢出改动", s["scale"] == 1.0
          and vec_close(s["weights"], [0.75, -1.05]) and s["step"] == 7)

    # --- 步历史查询 ---
    st, h = req("GET", "/steps/1")
    kinds = [(x["rank"], x["kind"]) for x in h["reports"]]
    check("steps/1 决议为 skipped", h["decision"]["outcome"] == "skipped")
    check("steps/1 仅含已受理的溢出上报(迟报不入库)", kinds == [(0, "overflow")], json.dumps(kinds))

    # --- 留一个挂起上报, 供重启后验证 reports 表持久化 ---
    st, r = req("POST", "/report", {"step": 7, "rank": 0, "kind": "grad", "grad": [1.0, 1.0]})
    check("s7 rank0 首报受理(挂起)", r["accepted"] is True and r["decision"] is None)

    st, s = req("GET", "/state")
    check("phase1 终态 step7/scale1/w[0.75,-1.05]",
          s["step"] == 7 and s["scale"] == 1.0 and vec_close(s["weights"], [0.75, -1.05]))


def phase2():
    # 重启后: 作业状态保留
    st, s = req("GET", "/state")
    check("重启后状态保留 step7/scale1", s["step"] == 7 and s["scale"] == 1.0
          and vec_close(s["weights"], [0.75, -1.05]), json.dumps(s))
    check("重启后已受理上报保留(rank0 挂起)",
          any(r["rank"] == 0 and r["kind"] == "grad" for r in s["reports_received"]),
          json.dumps(s["reports_received"]))

    # 重启后重报旧步: 返回已持久化的决议
    st, r = req("POST", "/report", {"step": 1, "rank": 1, "kind": "grad", "grad": [1.0, 1.0]})
    check("重启后重报 s1 -> skipped 决议", r["late"] and r["decision"]["outcome"] == "skipped")
    st, r = req("POST", "/report", {"step": 0, "rank": 0, "kind": "grad", "grad": [16.0, 32.0]})
    check("重启后重报 s0 -> updated 决议", r["late"] and r["decision"]["outcome"] == "updated"
          and vec_close(r["decision"]["weights"], [0.85, -1.05]))

    # 重启后重报当前步同 rank: 首报(重启前受理)仍然唯一, 溢出重报不生效
    st, r = req("POST", "/report", {"step": 7, "rank": 0, "kind": "overflow"})
    check("重启后 s7 rank0 溢出重报被拒(首报跨重启唯一)",
          r["duplicate"] is True and r["decision"] is None)

    # 补齐梯度 -> 按 scale=1 更新: avg=[2,2]/1, w=[0.75-0.2, -1.05-0.2]=[0.55,-1.25]
    st, r = req("POST", "/report", {"step": 7, "rank": 1, "kind": "grad", "grad": [3.0, 3.0]})
    check("重启后 s7 齐全触发更新", r["decision"] and r["decision"]["outcome"] == "updated")
    check("s7 更新权重正确(scale=1)", vec_close(r["decision"]["weights"], [0.55, -1.25]),
          json.dumps(r["decision"]["weights"]))

    st, h = req("GET", "/steps/7")
    check("steps/7 决议已持久化", h["decision"]["outcome"] == "updated"
          and vec_close(h["decision"]["weights"], [0.55, -1.25]))
    st, h = req("GET", "/steps/0")
    check("steps/0 旧决议仍在", h["decision"]["outcome"] == "updated")


def _confirm_many(step, cand, n):
    results = []
    lock = threading.Lock()

    def worker():
        st, body = req("POST", f"/steps/{step}/confirm", {"candidate": cand})
        with lock:
            results.append((st, body))

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def phase3():
    # 新作业: 2 进程, 权重 [1.0, -1.0], lr=0.1, 单权重改变量上限 0.15
    req("DELETE", "/job")
    st, j = req("POST", "/job", {"num_processes": 2, "weights": [1.0, -1.0], "lr": 0.1,
                                 "max_weight_delta": 0.15})
    check("登记作业(带权重改变量上限) 201",
          st == 201 and j["max_weight_delta"] == 0.15 and j["blocked"] is False,
          f"status={st} body={json.dumps(j)}")

    # --- step0: |Δw| 恰好等于上限, 不算超过 -> 直接更新 ---
    st, r = req("POST", "/report", {"step": 0, "rank": 0, "kind": "grad", "grad": [16.0, 32.0]})
    check("s0 rank0 首报受理", r["accepted"] is True and r["decision"] is None)
    st, r = req("POST", "/report", {"step": 0, "rank": 1, "kind": "grad", "grad": [32.0, -16.0]})
    # avg=[24,8] /16=[1.5,0.5] *0.1 -> Δw=[0.15,0.05]; max=0.15 不 > 0.15
    check("s0 |Δw|=上限 -> 普通步直接更新",
          r["decision"] and r["decision"]["outcome"] == "updated"
          and r["decision"]["trial"] is False and r["blocked"] is False
          and vec_close(r["decision"]["weights"], [0.85, -1.05]), json.dumps(r))
    check("s0 步进到 1", r["current_step"] == 1)

    # --- step1: |Δw| 超限 -> 停步, 生成候选 1 ---
    st, r = req("POST", "/report", {"step": 1, "rank": 0, "kind": "grad",
                                    "grad": [1600.0, 160.0]})
    check("s1 rank0 首报受理(仍挂起)", r["accepted"] is True and r["decision"] is None)
    st, r = req("POST", "/report", {"step": 1, "rank": 1, "kind": "grad",
                                    "grad": [1600.0, 160.0]})
    # avg=[1600,160] /16=[100,10] *0.1 -> Δw=[10,1] 远超 0.15
    check("s1 超限 -> 停步不跳步", r["blocked"] is True and r["decision"] is None
          and r["current_step"] == 1)
    c1 = r["candidate"] or {}
    check("s1 生成候选 1 且标记超限",
          c1.get("candidate") == 1 and c1.get("over_limit") is True
          and c1.get("confirmable") is False, json.dumps(c1))

    st, s = req("GET", "/state")
    check("停步时保留原权重/原倍率/原步号",
          s["step"] == 1 and s["scale"] == 16.0 and s["blocked"] is True
          and s["current_candidate"] == 1 and vec_close(s["weights"], [0.85, -1.05]),
          json.dumps(s))

    # 重报: 按首报规则拒绝, 且保留当前候选
    st, r = req("POST", "/report", {"step": 1, "rank": 0, "kind": "grad",
                                    "grad": [1.0, 1.0]})
    check("停步时同 rank 重报被拒", r["duplicate"] is True)
    check("停步时重报保留候选 1",
          (r["candidate"] or {}).get("candidate") == 1 and r["current_step"] == 1,
          json.dumps(r))

    # 未超限的候选才能确认
    st, _ = req("POST", "/steps/1/confirm", {"candidate": 1})
    check("超限候选不可确认 409", st == 409, f"status={st}")

    # 重算: lr 必须严格调低
    st, _ = req("POST", "/steps/1/recompute", {"lr": 0.1})
    check("重算 lr 未调低 -> 422", st == 422, f"status={st}")
    # lr=0.01 -> Δw=[1.0,0.1], 仍超限
    st, r = req("POST", "/steps/1/recompute", {"lr": 0.01})
    c2 = r.get("candidate") or {}
    check("重算生成候选 2(仍超限)",
          st == 200 and c2.get("candidate") == 2 and c2.get("over_limit") is True
          and c2.get("confirmable") is False, json.dumps(r))
    st, _ = req("POST", "/steps/1/confirm", {"candidate": 2})
    check("仍超限的候选 2 不可确认 409", st == 409, f"status={st}")

    # lr=0.0005 -> Δw=[0.05,0.005], 未超限
    st, r = req("POST", "/steps/1/recompute", {"lr": 0.0005})
    c3 = r.get("candidate") or {}
    check("再次重算生成候选 3(未超限)",
          st == 200 and c3.get("candidate") == 3 and c3.get("over_limit") is False
          and c3.get("confirmable") is True, json.dumps(r))
    check("候选 3 权重正确(按新 lr 重算)",
          vec_close(c3.get("weights"), [0.8, -1.055]), json.dumps(c3.get("weights")))

    st, _ = req("POST", "/steps/1/confirm", {"candidate": 1})
    check("已被替换的候选 1 不可确认 409", st == 409, f"status={st}")
    st, _ = req("POST", "/steps/1/confirm", {"candidate": 2})
    check("已被替换的候选 2 不可确认 409", st == 409, f"status={st}")
    st, h = req("GET", "/steps/1")
    sts = {c["candidate"]: c["status"] for c in h["candidates"]}
    check("候选历史保留 1superseded/2superseded/3current",
          sts == {1: "superseded", 2: "superseded", 3: "current"}, json.dumps(sts))

    # 重算与确认交错: 先把候选 3 重算成候选 4, 再回头确认候选 3
    st, r = req("POST", "/steps/1/recompute", {"lr": 0.0002})
    check("交错: 重算再次替换为候选 4", r["candidate"]["candidate"] == 4)
    st, _ = req("POST", "/steps/1/confirm", {"candidate": 3})
    check("交错: 刚被替换的候选 3 确认失败 409", st == 409, f"status={st}")

    # 并发确认候选 4: 只应步进一次, 所有响应返回同一决议
    results = _confirm_many(1, 4, 8)
    oks = [(st, b) for st, b in results if st == 200]
    decs = {json.dumps(b["decision"]["weights"], sort_keys=True) for _, b in oks}
    check("并发确认全部成功且返回同一决议",
          len(oks) == 8 and len(decs) == 1, json.dumps([st for st, _ in results]))
    check("并发确认只产生一次步进",
          all(b["current_step"] == 2 for _, b in oks)
          and sum(1 for _, b in oks if b["already_confirmed"] is False) == 1,
          json.dumps([(st, b.get("already_confirmed"), b.get("current_step"))
                      for st, b in results]))

    st, r = req("POST", "/steps/1/confirm", {"candidate": 4})
    check("确认成功后同号重试幂等返回原决议",
          st == 200 and r["already_confirmed"] is True
          and vec_close(r["decision"]["weights"], [0.83, -1.052])
          and r["decision"]["trial"] is True and r["decision"]["candidate"] == 4
          and r["current_step"] == 2, json.dumps(r))

    st, s = req("GET", "/state")
    check("确认后权重/步号/倍率同时生效",
          s["step"] == 2 and s["scale"] == 16.0 and s["blocked"] is False
          and vec_close(s["weights"], [0.83, -1.052]), json.dumps(s))
    st, h = req("GET", "/steps/1")
    check("steps/1 决议记录试跑来源(候选 4)",
          h["decision"]["trial"] is True and h["decision"]["candidate"] == 4
          and h["decision"]["outcome"] == "updated" and h["blocked"] is False,
          json.dumps(h["decision"]))
    st, _ = req("POST", "/steps/1/recompute", {"lr": 0.0001})
    check("已决议的步不再可重算 409", st == 409, f"status={st}")

    # --- step2: 未超限普通步直接更新 ---
    st, _ = req("POST", "/report", {"step": 2, "rank": 0, "kind": "grad", "grad": [16.0, 16.0]})
    st, r = req("POST", "/report", {"step": 2, "rank": 1, "kind": "grad",
                                    "grad": [16.0, -16.0]})
    # avg=[16,0] /16=[1,0] *0.1 -> [0.1,0] -> w=[0.73,-1.052]
    check("s2 未超限普通步直接更新",
          r["decision"]["outcome"] == "updated" and r["decision"]["trial"] is False
          and vec_close(r["decision"]["weights"], [0.73, -1.052]), json.dumps(r["decision"]))

    # --- step3: 有效溢出照常跳步 ---
    st, r = req("POST", "/report", {"step": 3, "rank": 0, "kind": "overflow"})
    check("s3 溢出照常跳步减半 16->8",
          r["decision"]["outcome"] == "skipped" and r["decision"]["new_scale"] == 8.0
          and r["current_step"] == 4 and r["blocked"] is False)

    # --- step4: 再停一步, 留给重启后验证 ---
    st, _ = req("POST", "/report", {"step": 4, "rank": 0, "kind": "grad",
                                    "grad": [1600.0, 0.0]})
    st, r = req("POST", "/report", {"step": 4, "rank": 1, "kind": "grad",
                                    "grad": [1600.0, 0.0]})
    # avg=[1600,0] /8=[200,0] *0.1 -> Δw=[20,0]
    check("s4 超限停步, 候选 1 留待重启",
          r["blocked"] is True and r["current_step"] == 4
          and (r["candidate"] or {}).get("candidate") == 1
          and (r["candidate"] or {}).get("over_limit") is True, json.dumps(r))
    st, s = req("GET", "/state")
    check("phase3 终态 step4/blocked/scale8/w[0.73,-1.052]",
          s["step"] == 4 and s["blocked"] is True and s["scale"] == 8.0
          and s["current_candidate"] == 1 and vec_close(s["weights"], [0.73, -1.052]),
          json.dumps(s))


def phase4():
    # 重启后: 停住的步、原权重、候选都还在
    st, s = req("GET", "/state")
    check("重启后停步状态保留 step4/blocked/cand1",
          s["step"] == 4 and s["blocked"] is True and s["scale"] == 8.0
          and s["current_candidate"] == 1 and vec_close(s["weights"], [0.73, -1.052]),
          json.dumps(s))
    st, h = req("GET", "/steps/4")
    check("重启后候选可查(候选 1 超限)",
          len(h["candidates"]) == 1 and h["candidates"][0]["candidate"] == 1
          and h["candidates"][0]["over_limit"] is True
          and h["candidates"][0]["confirmable"] is False, json.dumps(h))
    check("重启后停步的步仍无决议", h["decision"] is None)

    # 重启后仍可继续重算
    st, r = req("POST", "/steps/4/recompute", {"lr": 0.0004})
    check("重启后可重算 -> 候选 2 未超限",
          st == 200 and r["candidate"]["candidate"] == 2
          and r["candidate"]["over_limit"] is False
          and r["candidate"]["confirmable"] is True, json.dumps(r))

    # 重算失败保留原候选: lr 未调低
    st, _ = req("POST", "/steps/4/recompute", {"lr": 0.0005})
    check("重启后重算 lr 未调低 -> 422 保留原候选", st == 422, f"status={st}")
    st, h = req("GET", "/steps/4")
    check("失败后当前候选仍是候选 2",
          h["current_candidate"] == 2
          and [c["candidate"] for c in h["candidates"] if c["current"]] == [2],
          json.dumps(h))

    # 重启后确认
    st, r = req("POST", "/steps/4/confirm", {"candidate": 2})
    check("重启后确认候选 2 成功",
          st == 200 and r["confirmed"] is True and r["current_step"] == 5
          and vec_close(r["decision"]["weights"], [0.65, -1.052])
          and r["decision"]["candidate"] == 2, json.dumps(r))

    st, r = req("POST", "/steps/4/confirm", {"candidate": 2})
    check("重启后同号重试幂等",
          st == 200 and r["already_confirmed"] is True
          and vec_close(r["decision"]["weights"], [0.65, -1.052])
          and r["current_step"] == 5, json.dumps(r))

    # 迟报旧步返回试跑决议
    st, r = req("POST", "/report", {"step": 4, "rank": 0, "kind": "grad",
                                    "grad": [1.0, 1.0]})
    check("重报已确认的步返回试跑决议",
          r["late"] is True and r["decision"]["trial"] is True
          and r["decision"]["candidate"] == 2, json.dumps(r["decision"]))

    # 完整候选历史
    st, h = req("GET", "/steps/1")
    sts = {c["candidate"]: c["status"] for c in h["candidates"]}
    check("重启后候选历史完整",
          sts == {1: "superseded", 2: "superseded", 3: "superseded",
                  4: "confirmed"}, json.dumps(sts))

    # 停步期间提交的未来步仍被拒绝, 回到当前步后正常
    st, _ = req("POST", "/report", {"step": 9, "rank": 0, "kind": "overflow"})
    check("未来步仍 409", st == 409, f"status={st}")

    st, s = req("GET", "/state")
    check("phase4 终态 step5/scale8/w[0.65,-1.052]",
          s["step"] == 5 and s["scale"] == 8.0 and s["blocked"] is False
          and vec_close(s["weights"], [0.65, -1.052]), json.dumps(s))

    # --- 真实交错竞态: 重算与确认同时打 ---
    st, h = req("GET", "/steps/5")
    st, r = req("POST", "/report", {"step": 5, "rank": 0, "kind": "grad",
                                    "grad": [1600.0, 0.0]})
    st, r = req("POST", "/report", {"step": 5, "rank": 1, "kind": "grad",
                                    "grad": [1600.0, 0.0]})
    check("s5 超限停步(竞态准备)",
          r["blocked"] is True and (r["candidate"] or {}).get("candidate") == 1,
          json.dumps(r))
    st, r = req("POST", "/steps/5/recompute", {"lr": 0.0004})
    check("s5 候选 2 可确认(竞态准备)",
          st == 200 and r["candidate"]["confirmable"] is True, json.dumps(r))

    outcomes = []
    lock = threading.Lock()

    def race_confirm():
        st, b = req("POST", "/steps/5/confirm", {"candidate": 2})
        with lock:
            outcomes.append(("confirm", st, b))

    def race_recompute():
        st, b = req("POST", "/steps/5/recompute", {"lr": 0.0002})
        with lock:
            outcomes.append(("recompute", st, b))

    threads = ([threading.Thread(target=race_confirm) for _ in range(4)]
               + [threading.Thread(target=race_recompute) for _ in range(4)])
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    fresh_confirms = [b for kind, st, b in outcomes
                      if kind == "confirm" and st == 200
                      and b.get("already_confirmed") is False]
    confirm_statuses = sorted(st for kind, st, _ in outcomes if kind == "confirm")
    recompute_statuses = sorted(st for kind, st, _ in outcomes if kind == "recompute")
    st, h = req("GET", "/steps/5")
    st, s = req("GET", "/state")
    if h["decision"] is not None:
        # 确认先赢: 恰好一次决议、一次步进, 重算全部被拒
        check("竞态(确认赢): 只产生一次决议/一次步进",
              h["decision"]["trial"] is True and len(fresh_confirms) == 1
              and s["step"] == 6 and s["blocked"] is False
              and recompute_statuses == [409, 409, 409, 409],
              json.dumps({"outcomes": [(k, st) for k, st, _ in outcomes], "state": s}))
        check("竞态(确认赢): 候选 2 confirmed, 无新候选",
              h["candidates"][-1]["candidate"] == 2
              and [c["status"] for c in h["candidates"]] == ["superseded", "confirmed"],
              json.dumps(h["candidates"]))
    else:
        # 重算先赢: 未确认, 候选被替换为 3, 原候选 2 的确认全部失败
        check("竞态(重算赢): 未产生决议, 仍停在 step5",
              len(fresh_confirms) == 0 and s["step"] == 5 and s["blocked"] is True
              and s["current_candidate"] == 3,
              json.dumps({"outcomes": [(k, st) for k, st, _ in outcomes], "state": s}))
        check("竞态(重算赢): 候选 2 被替换, 确认未成功",
              [c["status"] for c in h["candidates"]]
              == ["superseded", "superseded", "current"]
              and all(st == 409 for st in confirm_statuses),
              json.dumps(h["candidates"]))
    check("竞态: 状态可继续处理(无 500)",
          all(st < 500 for _, st, _ in outcomes), json.dumps(outcomes))


print(f"== verify {PHASE} against {BASE} ==")
if PHASE == "phase1":
    phase1()
elif PHASE == "phase2":
    phase2()
elif PHASE == "phase3":
    phase3()
elif PHASE == "phase4":
    phase4()
else:
    print(f"unknown phase: {PHASE}")
    sys.exit(2)

print(f"== {'ALL PASS' if not failures else f'{len(failures)} FAILURES: {failures}'} ==")
sys.exit(1 if failures else 0)
