# -*- coding: utf-8 -*-
"""深蹲状态机：rep 检测 + 关键帧抓取（自适应阈值版）

状态转移：STAND → DESCEND → BOTTOM → ASCEND →(rep完成)→ STAND

自适应基线：
- 站立确认阈值 = min(硬阈值160°, 个人站立基线-10°)
  （有人站姿微屈膝 ~162°，用绝对 160° 会永远进不了 STAND；反之蹲不深的人 158° 也算他"站立"）
- 下降确认：膝角 < 站立确认阈值
- 底部确认：从本 rep 最低点回升 > REBOUND_ANGLE（最低膝角及其帧持续跟踪，
  回升只用于确认已越过最低点，不把回升帧当最低点）
- rep 完成：膝角回到站立确认阈值上方
- 人体丢失：动作中短暂丢失冻结 ≤person_loss_tol 秒，超时丢弃未完成动作重新就绪

每帧喂 update()，返回 (state, rep_event|None)。
"""
import time
from enum import Enum

import config


class SquatState(Enum):
    IDLE = "idle"          # 未检测到人
    STAND = "stand"         # 站立
    DESCEND = "descend"     # 下降中
    BOTTOM = "bottom"       # 到达最低点（已开始回升）
    ASCEND = "ascend"       # 起身中


class RepEvent:
    """一次完整深蹲的数据包"""

    def __init__(self, rep_index):
        self.rep_index = rep_index
        self.frame_stand = None      # 起始站立帧索引（关键帧：顶帧）
        self.frame_bottom = None    # 最低点帧索引（关键帧：底帧）
        self.bottom_knee_angle = None   # 底帧膝角（左右均值）
        self.ts = time.time()


class SquatStateMachine:
    def __init__(self, stand_angle=None, rebound_angle=None,
                 stand_hold=0.3, min_rep_frames=6, baseline_margin=10.0,
                 person_loss_tol=1.0, min_rep_secs=0.5):
        self.hard_stand_angle = stand_angle if stand_angle is not None else config.STAND_ANGLE
        self.rebound_angle = rebound_angle if rebound_angle is not None else config.REBOUND_ANGLE
        self.stand_hold = stand_hold          # 站立确认时长（秒），防抖
        self.min_rep_frames = min_rep_frames  # 一次 rep 最少帧数，过滤误检
        self.min_rep_secs = min_rep_secs      # 一次 rep 最短时长（秒），高帧率下防抖动误计
        self.baseline_margin = baseline_margin  # 基线余量（度）
        self.person_loss_tol = person_loss_tol  # 动作中人体短暂丢失的容忍时长（秒）

        self.state = SquatState.IDLE
        self._stand_since = None
        self._frame_idx = 0
        self._min_knee = None
        self._min_frame = None
        self._rep_start_frame = None
        self._rep_start_time = None
        self._stand_frame = None
        self._lost_since = None      # 人体丢失起始时刻（容忍计时用）
        self.rep_count = 0

        # 自适应站立基线：近直立（>145°）帧的滑动最大值
        self._baseline = None          # 个人站立膝角估计
        self._baselineEMA = None

    # ------------------------------------------------------------------
    @property
    def stand_threshold(self):
        """站立确认阈值：min(160, 基线-10)"""
        if self._baseline is None:
            return self.hard_stand_angle
        return min(self.hard_stand_angle, self._baseline - self.baseline_margin)

    def _update_baseline(self, knee_angle):
        """膝角 > 145° 视为近直立，更新基线（取近直立最大值，EMA 平滑）"""
        if knee_angle > 145:
            self._baseline = knee_angle if self._baseline is None else max(
                self._baseline, knee_angle)
            b = self._baseline
            self._baselineEMA = b if self._baselineEMA is None else \
                0.9 * self._baselineEMA + 0.1 * b

    # ------------------------------------------------------------------
    def update(self, knee_angle, person_detected=True, frame_idx=None):
        """每帧调用。knee_angle: 左右膝角均值（度）。
        frame_idx: 外部帧序号（摄像头帧号），传入则以其为准，
        保证 rep 事件中的关键帧索引能与外部帧缓存对齐。
        返回 (state, rep_event|None)
        """
        if not person_detected or knee_angle is None:
            # 动作进行中短暂丢失（检测闪烁）：冻结状态机，容忍 person_loss_tol 秒
            if self.state in (SquatState.DESCEND, SquatState.BOTTOM):
                if self._lost_since is None:
                    self._lost_since = time.time()
                if time.time() - self._lost_since < self.person_loss_tol:
                    return self.state, None      # 容忍期内冻结，不累计不丢弃
            # 站立/空闲状态或丢失超时：丢弃未完成动作，重新就绪
            self.state = SquatState.IDLE
            self._reset_cycle()
            return self.state, None

        self._lost_since = None
        self._frame_idx = frame_idx if frame_idx is not None else self._frame_idx + 1
        self._update_baseline(knee_angle)
        ev = None
        s = self.state
        thr = self.stand_threshold

        if s in (SquatState.IDLE, SquatState.STAND):
            if knee_angle > thr:
                if self._stand_since is None:
                    self._stand_since = time.time()
                    self._stand_frame = self._frame_idx
                elif time.time() - self._stand_since >= self.stand_hold:
                    self.state = SquatState.STAND
            else:
                self._begin_descend(knee_angle)
                self.state = SquatState.DESCEND

        elif s == SquatState.DESCEND:
            if knee_angle > thr:
                # 实际下降幅度 = 站立基线到底部的降幅；太小视为抖动取消
                base = self._baseline if self._baseline is not None else thr
                if self._min_knee is not None and (base - self._min_knee) < 20:
                    self.state = SquatState.STAND
                    self._stand_since = time.time()
                    self._reset_cycle()
                else:
                    ev = self._finish_rep()
                    self.state = SquatState.STAND
                    self._stand_since = time.time()
            elif self._min_knee is not None and (knee_angle - self._min_knee) > self.rebound_angle:
                self.state = SquatState.BOTTOM
            else:
                self._track_min(knee_angle)

        elif s == SquatState.BOTTOM:
            if knee_angle > thr:
                ev = self._finish_rep()
                self.state = SquatState.STAND
                self._stand_since = time.time()
            elif knee_angle < self._min_knee:
                self._track_min(knee_angle)  # 二次下蹲更深

        return self.state, ev

    # ------------------------------------------------------------------
    def _begin_descend(self, knee_angle):
        self._rep_start_frame = self._frame_idx
        self._rep_start_time = time.time()
        self._min_knee = knee_angle
        self._min_frame = self._frame_idx

    def _track_min(self, knee_angle):
        if self._min_knee is None or knee_angle < self._min_knee:
            self._min_knee = knee_angle
            self._min_frame = self._frame_idx

    def _finish_rep(self):
        """回到站立时调用：帧数与时长都足够才产出 rep 事件"""
        if self._rep_start_frame is None:
            return None
        duration_frames = self._frame_idx - self._rep_start_frame
        duration_secs = time.time() - self._rep_start_time if self._rep_start_time else 0
        if (duration_frames < self.min_rep_frames
                or duration_secs < self.min_rep_secs
                or self._min_knee is None):
            self._reset_cycle()
            return None
        self.rep_count += 1
        ev = RepEvent(self.rep_count)
        ev.frame_stand = self._stand_frame or self._rep_start_frame
        ev.frame_bottom = self._min_frame
        ev.bottom_knee_angle = self._min_knee
        self._reset_cycle()
        return ev

    def _reset_cycle(self):
        self._stand_since = None
        self._min_knee = None
        self._min_frame = None
        self._rep_start_frame = None
        self._rep_start_time = None
        self._stand_frame = None
