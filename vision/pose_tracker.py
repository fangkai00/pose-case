# -*- coding: utf-8 -*-
"""MediaPipe 姿态追踪封装：33 个 3D 关键点 + EMA 平滑

使用 MediaPipe 1.x tasks API（PoseLandmarker, VIDEO 模式）。
每帧输出：
  - landmarks: 33x3 归一化坐标（相对画面，含可见度）
  - world: 33x3 米制 3D 坐标（以髋部为原点，用于角度计算）
"""
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import config


class PoseTracker:
    """单帧推理 + EMA 平滑。线程安全：仅由采集线程调用。"""

    def __init__(self, model_path=None, ema_alpha=None):
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=model_path or config.POSE_MODEL_PATH
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = mp_vision.PoseLandmarker.create_from_options(opts)
        self._alpha = ema_alpha if ema_alpha is not None else config.EMA_ALPHA
        self._smoothed = None          # 平滑后的归一化关键点 33x4(x,y,z,vis)
        self._smoothed_world = None    # 平滑后的世界坐标 33x3
        self._ts_ms = 0                # 单调递增时间戳（ms）

    def process(self, frame_bgr):
        """输入 BGR ndarray，返回 (ok, landmarks 33x4, world 33x3)
        landmarks: [x, y, z, visibility]，x/y 归一化到 0-1
        world: 米制坐标（米），原点在两髋中点
        """
        rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB
        h, w = rgb.shape[:2]
        self._ts_ms += int(1000 / max(config.CAM_FPS, 1))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._detector.detect_for_video(mp_img, self._ts_ms)

        if not result.pose_landmarks:
            # 未检测到人体：清空平滑状态
            self._smoothed = None
            self._smoothed_world = None
            return False, None, None

        lm = np.array(
            [[p.x, p.y, p.z, p.visibility] for p in result.pose_landmarks[0]],
            dtype=np.float32,
        )  # 33x4
        world = np.array(
            [[p.x, p.y, p.z] for p in result.pose_world_landmarks[0]],
            dtype=np.float32,
        )  # 33x3

        # EMA 平滑
        if self._smoothed is None:
            self._smoothed = lm.copy()
            self._smoothed_world = world.copy()
        else:
            a = self._alpha
            self._smoothed = a * lm + (1 - a) * self._smoothed
            self._smoothed_world = a * world + (1 - a) * self._smoothed_world

        return True, self._smoothed.copy(), self._smoothed_world.copy()

    def close(self):
        self._detector.close()


# MediaPipe 33 关键点索引（常用子集）
class L:
    NOSE = 0
    LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
    LEFT_ELBOW, RIGHT_ELBOW = 13, 14
    LEFT_WRIST, RIGHT_WRIST = 15, 16
    LEFT_HIP, RIGHT_HIP = 23, 24
    LEFT_KNEE, RIGHT_KNEE = 25, 26
    LEFT_ANKLE, RIGHT_ANKLE = 27, 28
    LEFT_FOOT, RIGHT_FOOT = 31, 32
