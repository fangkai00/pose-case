# -*- coding: utf-8 -*-
"""骨架绘制：用 OpenCV 在帧上画 33 关键点骨架 + 实时角度标注

MediaPipe 1.x 无内置绘图模块，自行实现（颜色可自定义，重点关节高亮）。
"""
from collections import namedtuple

import cv2
import numpy as np

# 骨架连接（MediaPipe 33 点拓扑的常用子集）
Edge = namedtuple("Edge", ["a", "b"])
POSE_EDGES = [
    # 躯干
    Edge(11, 12), Edge(11, 23), Edge(12, 24), Edge(23, 24),
    # 左臂
    Edge(11, 13), Edge(13, 15),
    # 右臂
    Edge(12, 14), Edge(14, 16),
    # 左腿
    Edge(23, 25), Edge(25, 27), Edge(27, 31),
    # 右腿
    Edge(24, 26), Edge(26, 28), Edge(28, 32),
    # 头
    Edge(0, 11), Edge(0, 12),
]

# 重点关节（大圆点高亮）：肩/髋/膝/踝
KEY_JOINTS = {11, 12, 23, 24, 25, 26, 27, 28}

COLOR_EDGE = (255, 200, 60)      # BGR 金黄
COLOR_JOINT = (80, 220, 80)     # 绿
COLOR_KEY = (60, 160, 255)      # 橙红


def draw_skeleton(frame, landmarks, left_knee_angle=None, right_knee_angle=None):
    """在帧上绘制骨架。

    frame: BGR 图像（会被就地修改）
    landmarks: 33x4 [x,y,z,vis]（归一化坐标）
    left/right_knee_angle: 可选，实时膝角标注
    """
    if landmarks is None:
        return frame
    h, w = frame.shape[:2]
    pts = {i: (int(lm[0] * w), int(lm[1] * h)) for i, lm in enumerate(landmarks)}

    # 连线
    for e in POSE_EDGES:
        p1, p2 = pts[e.a], pts[e.b]
        cv2.line(frame, p1, p2, COLOR_EDGE, 2, cv2.LINE_AA)

    # 关键点
    for i, p in pts.items():
        if i in KEY_JOINTS:
            cv2.circle(frame, p, 5, COLOR_KEY, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, p, 3, COLOR_JOINT, -1, cv2.LINE_AA)

    # 实时膝角标注（跟随膝盖位置）
    for idx, ang in ((25, left_knee_angle), (26, right_knee_angle)):
        if ang is None:
            continue
        p = pts[idx]
        cv2.putText(frame, f"{int(ang)}°", (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{int(ang)}°", (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 220), 1, cv2.LINE_AA)
    return frame


def angle_3d(a, b, c):
    """三点夹角（度）：以 b 为顶点。输入 3D 坐标 (N,3)。"""
    v1 = np.asarray(a) - np.asarray(b)
    v2 = np.asarray(c) - np.asarray(b)
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return float(np.degrees(np.clip(np.arccos(np.clip(cos, -1, 1)), 0, 180)))
