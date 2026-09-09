# -*- coding: utf-8 -*-
"""会话记忆：跨会话/跨轮次练习摘要（runs/session_summary.json）

- 写：每次 rep 完成与 VLM 评审回填后，聚合当前轮 reps 增量覆盖写盘
      （直接关闭/崩溃也不丢已完成练习）
- 读：应用启动时读上一次练习摘要，作为先验——
      · last_metrics → 本轮第一次 rep 的改善对比基线
      · last_issue / top_issue → 本轮第一次 VLM 评审的"上一次主要问题"
"轮" = 应用启动或点"开始练习"重置后的一段练习；重置时已完成的 rep
聚合为新先验（同一天练多轮也连续记忆）。记忆属增强功能：任何读写
失败只打印日志，不影响主流程。
"""
import json
import os
import time

SUMMARY_FILE = os.path.join("runs", "session_summary.json")


def summarize_session(reps, started_at=None):
    """聚合一轮 reps → 摘要 dict；无 rep 返回 None"""
    if not reps:
        return None
    judged_total, judged_ok = {}, {}
    issue_counts = {}
    degraded = improved = 0
    for r in reps:
        for j in r.get("judged") or []:
            judged_total[j["name"]] = judged_total.get(j["name"], 0) + 1
            judged_ok[j["name"]] = judged_ok.get(j["name"], 0) + (1 if j["ok"] else 0)
        v = r.get("vlm") or {}
        if v.get("main_issue"):
            issue_counts[v["main_issue"]] = issue_counts.get(v["main_issue"], 0) + 1
        if v.get("degraded"):
            degraded += 1
        if (r.get("improvement") or {}).get("verdict") == "improved":
            improved += 1
    top_issue = max(issue_counts, key=issue_counts.get) if issue_counts else None
    last = reps[-1]
    return {
        "session_id": time.strftime("%Y%m%d_%H%M%S",
                                    time.localtime(started_at or time.time())),
        "rep_count": len(reps),
        "metric_pass_rate": {n: round(judged_ok[n] / judged_total[n], 2)
                             for n in judged_total},
        "top_issue": top_issue,
        "issue_counts": issue_counts,
        "improved_count": improved,
        "vlm_degraded_count": degraded,
        "last_metrics": last.get("metrics"),       # 下轮第一次 rep 的对比基线
        "last_issue": (last.get("vlm") or {}).get("main_issue"),
    }


def save_session_summary(reps, started_at=None, path=SUMMARY_FILE):
    """写盘（rep 完成与 VLM 回填后调用）。失败静默——记忆不影响主流程。"""
    try:
        s = summarize_session(reps, started_at)
        if s is None:
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"[会话记忆] 写入失败: {e}")
        return False


def load_prev_session(path=SUMMARY_FILE):
    """启动时读上一次练习摘要；无文件/损坏/无 rep 返回 None。
    utf-8-sig 兼容带 BOM 的文件。"""
    try:
        with open(path, encoding="utf-8-sig") as f:
            s = json.load(f)
        if s.get("rep_count"):
            return s
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[会话记忆] 读取失败: {e}")
    return None
