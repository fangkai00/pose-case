# -*- coding: utf-8 -*-
"""M2 冒烟测试：对测试视频跑 状态机+指标，验证 rep 计数和指标合理性"""
import types

import cv2

from vision import state_machine as _sm_mod
from vision.pose_tracker import PoseTracker, L
from vision.skeleton_draw import angle_3d
from vision.state_machine import SquatStateMachine
from metrics.metrics_engine import MetricsEngine, judge_metrics
from coach.rule_engine import RuleEngine


class _VideoClock:
    """视频时间轴假时钟。

    状态机内部用 wall-clock 判定 rep 时长（min_rep_secs）与站立保持（stand_hold），
    本地 12fps 推理时一个 rep ≈1s 能过门限；但 CI CPU 推理快于实时回放，
    rep 真实耗时不足 0.5s 会被全部误杀。用视频帧率驱动时钟，与实机播放一致。
    """
    def __init__(self, fps):
        self.t = 0.0
        self.dt = 1.0 / fps

    def time(self):
        return self.t

    def tick(self):
        self.t += self.dt


cap = cv2.VideoCapture("test_squat.mp4")
FPS = cap.get(cv2.CAP_PROP_FPS) or 12.0
clock = _VideoClock(FPS)
_sm_mod.time = types.SimpleNamespace(time=clock.time)

tracker = PoseTracker()
sm = SquatStateMachine()
metrics = MetricsEngine()
print("视频帧数:", int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), "fps:", FPS)

rep_events = []
poses = {}   # frame_idx -> (lm, world) 底帧回查用
frame_idx = 0
detected_frames = 0
knee_min, knee_max = 999.0, 0.0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame_idx += 1
    clock.tick()
    detected, lm, world = tracker.process(frame)
    if not detected:
        sm.update(None, False)
        continue
    detected_frames += 1
    poses[frame_idx] = (lm, world)
    lk = angle_3d(world[L.LEFT_HIP], world[L.LEFT_KNEE], world[L.LEFT_ANKLE])
    rk = angle_3d(world[L.RIGHT_HIP], world[L.RIGHT_KNEE], world[L.RIGHT_ANKLE])
    knee = (lk + rk) / 2
    knee_min, knee_max = min(knee_min, knee), max(knee_max, knee)
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

print(f"\n实际读帧 {frame_idx}｜检出人体 {detected_frames} 帧｜膝角 {knee_min:.0f}–{knee_max:.0f}°")
print(f"共检测 rep: {len(rep_events)}（视频含 15 次深蹲循环；"
      f"状态机的抖动过滤会按设计取消过短/过浅的 rep）")
cap.release()
tracker.close()

# 视频含 15 个循环；min_rep_secs/min_rep_frames 抖动过滤会按设计取消部分短 rep
# （假时钟 12fps 下为 13 个）。取区间既容纳跨平台浮点微差，又能拦住真回归（0/5 个）。
assert 12 <= len(rep_events) <= 15, f"rep 检测数量异常: {len(rep_events)}"
assert detected_frames == frame_idx, f"人体检出异常: {detected_frames}/{frame_idx}"
for idx, m, level in rep_events:
    assert 120 <= m["knee_angle"] <= 175, f"rep{idx} 膝角异常: {m['knee_angle']}"
    assert 0 <= m["torso_lean"] <= 90, f"rep{idx} 前倾角异常: {m['torso_lean']}"
print("M2 离线冒烟测试通过 ✓")
