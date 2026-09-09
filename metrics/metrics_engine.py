# -*- coding: utf-8 -*-
"""指标引擎：4 个可解释指标（rep 结束时计算，实时逐帧算膝角）

1. 膝关节角度  bottom_knee_angle：最低点 hip-knee-ankle 角（度）
2. 下蹲深度    depth_ratio：髋部 y 下降量 / 站立腿长（%）
3. 躯干前倾角  torso_lean：底帧 肩中点-髋中点 连线与垂直线夹角（度）
4. 左右对称性 l_r_knee_diff：底帧左右膝角差（度）

深度用归一化坐标（画面 y），前后倾用 world 3D（不受身高距离影响）。
"""
import numpy as np

import config
from vision.pose_tracker import L
from vision.skeleton_draw import angle_3d


def _mid(a, b):
    return (np.asarray(a) + np.asarray(b)) / 2.0


class MetricsEngine:
    """帧间累积站姿基线 + rep 结束时汇总指标"""

    def __init__(self):
        self._stand_leg_len = None    # 站立时腿长（归一化 y：hip 到 ankle 距离）
        self._stand_hip_y = None      # 站立时髋部 y（归一化）

    # ---------------- 逐帧：更新站姿基线（只在膝角>145°近直立时）-------------
    def update_baseline(self, landmarks, world, knee_angle):
        if landmarks is None or knee_angle is None or knee_angle < 145:
            return
        hip_y = (landmarks[L.LEFT_HIP][1] + landmarks[L.RIGHT_HIP][1]) / 2
        ank_y = (landmarks[L.LEFT_ANKLE][1] + landmarks[L.RIGHT_ANKLE][1]) / 2
        leg_len = abs(hip_y - ank_y)
        if leg_len > 0.05:  # 有效腿长
            self._stand_leg_len = leg_len
            self._stand_hip_y = hip_y

    # ---------------- rep 结束：计算 4 指标 ---------------------------------
    def compute_rep_metrics(self, landmarks, world):
        """landmarks: 底帧归一化关键点 33x4；world: 底帧世界坐标 33x3
        返回 dict（全部可解释，单位明确）"""
        m = {}
        # 1) 膝角（左/右/均值）
        lk = angle_3d(world[L.LEFT_HIP], world[L.LEFT_KNEE], world[L.LEFT_ANKLE])
        rk = angle_3d(world[L.RIGHT_HIP], world[L.RIGHT_KNEE], world[L.RIGHT_ANKLE])
        m["knee_angle"] = round(float(lk + rk) / 2, 1)
        m["knee_angle_l"] = round(float(lk), 1)
        m["knee_angle_r"] = round(float(rk), 1)

        # 2) 下蹲深度：髋部下降量 / 站立腿长
        hip_y = float((landmarks[L.LEFT_HIP][1] + landmarks[L.RIGHT_HIP][1]) / 2)
        depth = None
        if self._stand_hip_y is not None and self._stand_leg_len:
            drop = hip_y - self._stand_hip_y
            depth = drop / self._stand_leg_len
        m["depth_ratio"] = round(float(max(0.0, depth)), 3) if depth is not None else None

        # 3) 躯干前倾角：肩中点-髋中点 连线 与 垂直向上方向 的夹角
        shoulder_mid = _mid(world[L.LEFT_SHOULDER], world[L.RIGHT_SHOULDER])
        hip_mid = _mid(world[L.LEFT_HIP], world[L.RIGHT_HIP])
        v_torso = shoulder_mid - hip_mid          # 髋指向肩
        v_up = np.array([0.0, -1.0, 0.0])         # 世界坐标 y 向下为正 → 向上为 -y
        cos = np.dot(v_torso, v_up) / (np.linalg.norm(v_torso) + 1e-9)
        m["torso_lean"] = round(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))), 1)

        # 4) 左右对称性：左右膝角差
        m["l_r_diff"] = round(abs(lk - rk), 1)

        return m


def judge_metrics(m):
    """规则初判（CP-wise 二值/分级），返回 [(指标名, 值, 达标与否, 中文短评)]"""
        # 深度判定
    depth_ok = m.get("depth_ratio") is not None and m["depth_ratio"] >= config.DEPTH_MIN_RATIO
    depth_str = f"{m['depth_ratio']*100:.0f}%" if m.get("depth_ratio") is not None else "N/A"
    # 膝角判定
    knee_ok = m.get("knee_angle") is not None and m["knee_angle"] <= config.KNEE_ANGLE_FULL
    # 前倾判定
    lean_ok = m.get("torso_lean") is not None and m["torso_lean"] <= config.TORSO_LEAN_MAX
    # 对称判定
    sym_ok = m.get("l_r_diff") is not None and m["l_r_diff"] <= config.L_R_DIFF_MAX

    return [
        ("下蹲深度", depth_str, depth_ok,
         "达标" if depth_ok else "幅度不足，蹲得更深"),
        ("膝关节角度", f"{m.get('knee_angle','?')}°", knee_ok,
         "达标" if knee_ok else "弯曲不够"),
        ("躯干前倾角", f"{m.get('torso_lean','?')}°", lean_ok,
         "挺直" if lean_ok else "过度前倾，挺胸收腹"),
        ("左右对称性", f"{m.get('l_r_diff','?')}°", sym_ok,
         "对称" if sym_ok else "左右不对称（侧拍参考，建议正面机位复测）"),
    ]


# (指标key, 中文名, 容差, 改善方向: +1=变大改善 / -1=变小改善, 格式化)
_IMP_RULES = [
    ("knee_angle", "膝角", config.IMP_TOL_KNEE, -1, "{:.1f}°"),
    ("depth_ratio", "深度", config.IMP_TOL_DEPTH, +1, "{:.0%}"),
    ("torso_lean", "前倾", config.IMP_TOL_LEAN, -1, "{:.1f}°"),
    ("l_r_diff", "对称差", config.IMP_TOL_SYM, -1, "{:.1f}°"),
]


def judge_improvement(prev_m, curr_m):
    """本地改善判定：两次 rep 指标对比，变化须超过容差才计入（避免抖动误判）。

    返回 (verdict, detail)
      verdict: "improved" 改善 / "no_change" 无明显变化 / "unknown" 无法判断
      detail:  逐指标变化说明，如 "膝角 143.9°→96.2°↓改善；深度 15%→62%↑改善"
    注意：仅判断"相比上次是否改善"，与"是否达到演示阈值"是两回事（分开显示）。
    """
    if not prev_m or not curr_m:
        return "unknown", "缺少可比对的两次指标"
    better = worse = compared = 0
    parts = []
    for key, name, tol, direction, fmt in _IMP_RULES:
        pv, cv = prev_m.get(key), curr_m.get(key)
        if pv is None or cv is None:
            continue
        delta = (cv - pv) * direction      # >0 表示朝改善方向变化
        compared += 1
        if delta > tol:
            better += 1
            parts.append(f"{name} {fmt.format(pv)}→{fmt.format(cv)} 改善")
        elif delta < -tol:
            worse += 1
            parts.append(f"{name} {fmt.format(pv)}→{fmt.format(cv)} 变差")
        else:
            parts.append(f"{name} {fmt.format(pv)}→{fmt.format(cv)} 持平")
    if compared == 0:
        return "unknown", "两次指标均缺失，无法判断"
    if better > worse:
        return "improved", "；".join(parts)
    return "no_change", "；".join(parts)
