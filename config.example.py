# -*- coding: utf-8 -*-
"""全局配置：阈值/模型/摄像头，均可在此调整
使用方法：复制本文件为 config.py，并在下方填入你自己的百炼 API Key"""

# ============ 摄像头 ============
CAMERA_INDEX = 0          # 摄像头索引（笔记本内置一般是 0，外接可能是 1）
CAM_WIDTH = 640
CAM_HEIGHT = 480
CAM_FPS = 30

# ============ MediaPipe ============
POSE_MODEL_PATH = "models/pose_landmarker_lite.task"
EMA_ALPHA = 0.4           # 关键点指数平滑系数，越小越平滑、越大越跟手
MIN_VISIBILITY = 0.5      # 关键点可见度低于此值视为不可信
LEG_VIS_THRESHOLD = 0.25  # 腿部 6 点可见度门限：低于此值膝角视为无效（坐姿/出画过滤）。
                          # 0.5 对暗光/宽松裤装过严，导致就绪阶段难以确认；0.35 仍可滤掉完全遮挡

# ============ 深蹲状态机阈值 ============
STAND_ANGLE = 160.0       # 膝角大于此值视为站立
DESCEND_ANGLE = 160.0     # 膝角小于此值进入下降
REBOUND_ANGLE = 15.0      # 底部回升超过此角度认为开始起身

# ============ 指标阈值（参考值，后续用教程GIF基准替换/校准） ============
DEPTH_MIN_RATIO = 0.45    # 髋部下降量/腿长 低于此为幅度不足
TORSO_LEAN_MAX = 45.0     # 躯干前倾角上限（度）
KNEE_ANGLE_FULL = 100.0   # 标准深蹲最低点膝角参考
L_R_DIFF_MAX = 15.0       # 左右膝角差上限（度）

# ============ 改善判定容差（本地指标对比，超过容差才算变化） ============
IMP_TOL_KNEE = 5.0        # 膝角改善容差（度，变小为改善）
IMP_TOL_DEPTH = 0.05      # 下蹲深度改善容差（比例，变大为改善）
IMP_TOL_LEAN = 3.0        # 躯干前倾改善容差（度，变小为改善）
IMP_TOL_SYM = 3.0         # 左右对称改善容差（度，变小为改善）

# ============ 百炼 VLM（M3 使用） ============
DASHSCOPE_API_KEY = "sk-你的百炼APIKey"   # https://bailian.console.aliyun.com/ 获取
VLM_MODEL = "qwen-vl-max"
LLM_MODEL = "qwen-plus"

# ============ 教程素材 ============
DATASET_DIR = "exercises-dataset-main"
TUTORIAL_EXERCISE_ID = "3119"   # potty squat 徒手标准深蹲
