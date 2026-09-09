# -*- coding: utf-8 -*-
"""M2 冒烟测试：对测试视频跑 状态机+指标，验证 rep 计数和指标合理性"""
import cv2

from vision.pose_tracker import PoseTracker, L
from vision.skeleton_draw import angle_3d
from vision.state_machine import SquatStateMachine
from metrics.metrics_engine import MetricsEngine, judge_metrics
from coach.rule_engine import RuleEngine

cap = cv2.VideoCapture("test_squat.mp4")
tracker = PoseTracker()
sm = SquatStateMachine()
metrics = MetricsEngine()
print("视频帧数:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps:", cap.get(cv2.CAP_PROP_FPS))

rep_events = []
poses = {}   # frame_idx -> (lm, world) 底帧回查用
frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    detected, lm, world = tracker.process(frame)
    if not detected:
        sm.update(None, False)
        continue
    poses[frame_idx] = (lm, world)
    lk = angle_3d(world[L.LEFT_HIP], world[L.LEFT_KNEE], world[L.LEFT_ANKLE])
    rk = angle_3d(world[L.RIGHT_HIP], world[L.RIGHT_KNEE], world[L.RIGHT_ANKLE])
    knee = (lk + rk) / 2
    metrics.update_baseline(lm, world, knee)
    state, ev = sm.update(knee, True)
    if ev:
        # 指标用底帧姿态（而非 rep 完成时的站立帧）
        bottom_lm, bottom_world = poses[ev.frame_bottom]
        m = metrics.compute_rep_metrics(bottom_lm, bottom_world)
        judged = judge_metrics(m)
        level, summary = RuleEngine.rep_summary(judged)
        rep_events.append((ev.rep_index, m, level))
        print(f"[REP {ev.rep_index}] 膝角={m['knee_angle']}° 深度={m.get('depth_ratio')} "
              f"前倾={m['torso_lean']}° 对称差={m['l_r_diff']}° → {level}")

print(f"\n共检测 rep: {len(rep_events)}（视频含 15 次深蹲循环）")
cap.release()
tracker.close()

assert len(rep_events) == 15, f"rep 检测数量异常: {len(rep_events)}"
for idx, m, level in rep_events:
    assert 120 <= m["knee_angle"] <= 175, f"rep{idx} 膝角异常: {m['knee_angle']}"
    assert 0 <= m["torso_lean"] <= 90, f"rep{idx} 前倾角异常: {m['torso_lean']}"
print("M2 离线冒烟测试通过 ✓")
