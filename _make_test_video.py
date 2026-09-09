# -*- coding: utf-8 -*-
"""从数据集 GIF 生成测试视频（放大+循环），用于无摄像头环境下调试"""
import cv2
import numpy as np
from PIL import Image

gif_path = r"exercises-dataset-main\videos\3119-75Bgtjy.gif"
out_path = "test_squat.mp4"

im = Image.open(gif_path)
frames = []
try:
    i = 0
    while True:
        im.seek(i)
        frames.append(im.convert("RGB"))
        i += 1
except EOFError:
    pass
print(f"GIF 帧数: {len(frames)}, 尺寸: {im.size}")

# 放大到 640x480（保持正方形居中画布）—— GIF 是正方形
size = (640, 640)
vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 12, size)
for _ in range(15):  # 循环 15 次 ≈ 15s
    for f in frames:
        arr = cv2.cvtColor(np.array(f.resize(size, Image.LANCZOS)), cv2.COLOR_RGB2BGR)
        vw.write(arr)
vw.release()
print(f"测试视频已生成: {out_path} (15s, 12fps, {size})")
