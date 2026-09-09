# -*- coding: utf-8 -*-
"""教程分析：数据集 GIF → 6 帧拼图 → VLM 提取中文教学 CPs

对应论文 AgentCoach 的 CP 提取环节：教程视频 + 文字说明 → 结构化教学要点。
结果缓存 tutorial_cps.json，二次启动免调用。
"""
import json
import os

import cv2
import numpy as np
from PIL import Image

import config
from coach.vlm_reviewer import extract_tutorial_cps

DATASET = config.DATASET_DIR


def load_exercise_meta(exercise_id=None):
    """从 exercises.json 读取动作元数据（英文教程等）"""
    exercise_id = exercise_id or config.TUTORIAL_EXERCISE_ID
    path = os.path.join(DATASET, "data", "exercises.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for e in data:
        if e["id"] == exercise_id:
            return e
    raise KeyError(f"未找到动作 id={exercise_id}")


def find_gif(exercise_id=None):
    """定位该动作的 GIF 文件"""
    exercise_id = exercise_id or config.TUTORIAL_EXERCISE_ID
    vdir = os.path.join(DATASET, "videos")
    for f in os.listdir(vdir):
        if f.startswith(exercise_id + "-") and f.endswith(".gif"):
            return os.path.join(vdir, f)
    raise FileNotFoundError(f"未找到 {exercise_id} 的 GIF")


def gif_to_grid(gif_path, n_frames=6, cell=240):
    """GIF 均匀抽 n 帧 → 2x3 拼图（放大，白底）"""
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
    total = len(frames)
    idxs = [int(k * (total - 1) / (n_frames - 1)) for k in range(n_frames)]
    # 2x3 网格
    rows = []
    for r in range(2):
        cells = []
        for c in range(3):
            f = frames[idxs[r * 3 + c]].resize((cell, cell), Image.LANCZOS)
            cells.append(cv2.cvtColor(np.array(f), cv2.COLOR_RGB2BGR))
        rows.append(cv2.hconcat(cells))
    return cv2.vconcat(rows)


def extract_cps(force=False):
    """主入口：返回 (TutorialCPs, grid_image)"""
    meta = load_exercise_meta()
    gif = find_gif()
    grid = gif_to_grid(gif)
    cps = extract_tutorial_cps(grid, meta["instructions"]["en"], force=force)
    return cps, grid, meta


if __name__ == "__main__":
    print("提取教学 CPs（首次调用真实 VLM）...")
    cps, grid, meta = extract_cps()
    cv2.imwrite("tutorial_grid.jpg", grid)
    if cps is None:
        print("提取失败！")
    else:
        print(f"动作: {cps.exercise}")
        print(f"简介: {cps.intro}")
        for c in cps.cps:
            print(f"  [{c.name}] {c.requirement} | 判定: {c.how_to_judge}")
        print(f"\n缓存已写入: {os.path.abspath('tutorial_cps.json')}")
