# -*- coding: utf-8 -*-
"""session_memory 单元测试：聚合 / 落盘 / 读回 / 跨会话改善对比。
运行：python _test_session_memory.py（在项目根目录）"""
import os

from coach.session_memory import (SUMMARY_FILE, load_prev_session,
                                  save_session_summary, summarize_session)

reps = [
    {"rep": 1, "ts": 100.0,
     "judged": [{"name": "下蹲深度", "value": "15%", "ok": False, "comment": "幅度不足"}],
     "metrics": {"knee_angle": 144.0, "knee_angle_l": 143.0, "knee_angle_r": 145.0,
                 "depth_ratio": 0.15, "torso_lean": 34.0, "l_r_diff": 3.0},
     "rule_level": "minor", "improvement": None,
     "vlm": {"main_issue": "测试-下蹲深度不足", "degraded": False}},
    {"rep": 2, "ts": 110.0,
     "judged": [{"name": "下蹲深度", "value": "62%", "ok": True, "comment": "达标"}],
     "metrics": {"knee_angle": 98.0, "knee_angle_l": 97.0, "knee_angle_r": 99.0,
                 "depth_ratio": 0.62, "torso_lean": 12.0, "l_r_diff": 2.0},
     "rule_level": "good",
     "improvement": {"verdict": "improved", "detail": "膝角 144.0°→98.0° 改善"},
     "vlm": {"main_issue": "测试-下蹲深度不足", "degraded": False}},
]

# 1) 聚合
s = summarize_session(reps, 0)
assert s["rep_count"] == 2
assert s["top_issue"] == "测试-下蹲深度不足"
assert s["improved_count"] == 1
assert s["metric_pass_rate"]["下蹲深度"] == 0.5
assert s["last_metrics"]["knee_angle"] == 98.0
assert s["last_issue"] == "测试-下蹲深度不足"
assert summarize_session([], 0) is None

# 2) 落盘 + 读回（模拟下次启动）
assert save_session_summary(reps, 0)
p = load_prev_session()
assert p and p["rep_count"] == 2
assert p["last_metrics"]["depth_ratio"] == 0.62
assert p["top_issue"] == "测试-下蹲深度不足"

# 3) 读回的 JSON 指标能直接喂 judge_improvement（跨会话对比路径）
from metrics.metrics_engine import judge_improvement
v, d = judge_improvement(p["last_metrics"],
                         {"knee_angle": 80.0, "depth_ratio": 0.7,
                          "torso_lean": 10.0, "l_r_diff": 2.0})
assert v == "improved", (v, d)

os.remove(SUMMARY_FILE)  # 清理测试产物
print("session_memory 测试通过 ✓")
