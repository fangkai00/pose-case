# -*- coding: utf-8 -*-
"""百炼 VLM 调用封装：qwen-vl-max + Pydantic 校验 + 重试降级

调用方式：OpenAI 兼容模式（base_url=https://dashscope.aliyuncs.com/compatible-mode/v1）
- 教程 CP 提取（启动一次性，结果缓存）
- rep 评审（事件驱动：顶/底帧拼图 + 指标 JSON + 教学 CPs + 历史摘要）
失败策略：超时 20s，校验失败/异常重试 1 次，二次失败返回 None（上层降级为规则反馈）
"""
import base64
import json
import os
import time

import cv2
from openai import OpenAI
from pydantic import ValidationError

import config
from coach.schemas import SquatReview, TutorialCPs

client = OpenAI(
    api_key=config.DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# ----------------------------------------------------------------------
def _encode_image_b64(bgr_frame, max_side=640, quality=85) -> str:
    """BGR 帧 → base64 JPEG（限长边，控制 token 成本）"""
    h, w = bgr_frame.shape[:2]
    scale = max_side / max(h, w) if max(h, w) > max_side else 1.0
    if scale < 1.0:
        bgr_frame = cv2.resize(bgr_frame, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", bgr_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")
    return base64.b64encode(buf.tobytes()).decode()


def _call_vlm_with_schema(messages, schema, retries=1, timeout=25):
    """通用调用：JSON mode + Pydantic 校验 + 重试。失败返回 None"""
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
                model=config.VLM_MODEL,
                messages=messages,
                response_format={"type": "json_object"},
                timeout=timeout,
            )
            text = resp.choices[0].message.content
            data = json.loads(text)
            return schema.model_validate(data)
        except (ValidationError, json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"[VLM] 响应校验失败(第{attempt+1}次): {type(e).__name__}: {e}")
            if attempt < retries:
                continue  # 校验失败 → 重试
            return None
        except Exception as e:
            print(f"[VLM] 调用异常(第{attempt+1}次): {type(e).__name__}: {e}")
            time.sleep(1)
            if attempt < retries:
                continue
            return None
    return None


# ----------------------------------------------------------------------
# 教程 CP 提取（启动一次性，带磁盘缓存）
# ----------------------------------------------------------------------
CP_CACHE_FILE = "tutorial_cps.json"

CP_EXTRACT_PROMPT = """你是资深健身教练。请分析这个深蹲教程的连续动作帧（2x3 网格，按时间顺序）和文字教程，提取教学要点。

英文教程：
{instructions}

要求：
1. 提取 3-5 条最关键的教学要点（Coaching Points），中文输出
2. 每条要点必须附带可量化的判定建议（角度/幅度/姿态描述）
3. 重点关注：下蹲深度、膝关节角度、躯干姿态、双脚站位
4. 只输出 JSON，结构如下：
{{
  "exercise": "动作名（中文）",
  "intro": "动作简介，中文1-2句",
  "cps": [
    {{"name": "要点短名", "requirement": "具体要求一句话", "how_to_judge": "量化判定建议"}}
  ]
}}"""


def extract_tutorial_cps(grid_image_bgr, instructions_en, force=False):
    """GIF 6 帧拼图 + 英文教程 → 中文教学 CPs。缓存到 tutorial_cps.json"""
    if not force and os.path.exists(CP_CACHE_FILE):
        try:
            with open(CP_CACHE_FILE, encoding="utf-8") as f:
                return TutorialCPs.model_validate(json.load(f))
        except Exception:
            pass  # 缓存损坏则重新提取

    b64 = _encode_image_b64(grid_image_bgr, max_side=1024)
    messages = [
        {"role": "system", "content": "你是专业的运动动作分析师，输出严格 JSON。"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": CP_EXTRACT_PROMPT.format(instructions=instructions_en)},
        ]},
    ]
    result = _call_vlm_with_schema(messages, TutorialCPs)
    if result is not None:
        with open(CP_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)
    return result


# ----------------------------------------------------------------------
# rep 评审（事件驱动）
# ----------------------------------------------------------------------
REVIEW_PROMPT = """你是实时深蹲教练 AI。用户刚完成一次深蹲，请结合画面帧和量化指标给出评审。

【画面说明】拼图中左边是开始站立的顶帧，右边是最低点的底帧（均带骨架关键点叠加）。
【量化指标】（顶点站立=约170°膝角为直立；数值越低蹲得越深）
{metrics_json}

【规则引擎初判】
{rule_summary}

【教学要点（来自教程）】
{cps_text}

【用户上一次的主要问题】
{last_issue}

【本地指标改善判定】（由两次动作的量化指标按容差对比得出，以它为准，你的任务是结合画面解释它）
{improvement_text}

要求：
1. main_issue：结合指标和画面指出最主要问题（1 句，中文）
2. encouragement：针对性鼓励（1 句）
3. suggestion：一条可立即执行的纠正建议
4. severity：good（指标全部达标）/ minor（1 项问题）/ major（2 项及以上问题）
5. improved_vs_last：与本地指标改善判定保持一致（true=改善 / false=无明显变化 / null=无法判断或第一次）
6. visual_notes：从画面帧观察到的指标之外的信息（如着装、机位、遮挡）
只输出 JSON：
{{
  "main_issue": "...",
  "encouragement": "...",
  "suggestion": "...",
  "severity": "good|minor|major",
  "improved_vs_last": true/false/null,
  "visual_notes": "..."
}}"""


def review_rep(top_frame, bottom_frame, metrics, rule_summary, cps, last_issue,
               improvement=None, save_dir=None):
    """rep 结束事件评审。返回 (SquatReview|None, 存档信息 dict)
    improvement: (verdict, detail) 本地指标改善判定，告知 VLM 并让其解释"""

    improvement_text = "第一次练习，无对比" if improvement is None else {
        "improved": f"判定：相比上次有改善。依据：{improvement[1]}",
        "no_change": f"判定：相比上次无明显变化。依据：{improvement[1]}",
        "unknown": f"判定：无法判断（{improvement[1]}）",
    }.get(improvement[0], "无对比")
    # 拼 1x2 图：左顶帧 右底帧
    h = 480
    imgs = []
    for f in (top_frame, bottom_frame):
        if f is None:
            continue
        s = h / f.shape[0]
        imgs.append(cv2.resize(f, (int(f.shape[1] * s), h)))
    if not imgs:
        return None, {}
    if len(imgs) == 1:
        grid = imgs[0]
    else:
        grid = cv2.hconcat(imgs)
    b64 = _encode_image_b64(grid, max_side=1024)

    cps_text = "\n".join(f"- {c['name']}：{c['requirement']}（{c['how_to_judge']}）" for c in cps) \
        if isinstance(cps, list) else str(cps)
    prompt = REVIEW_PROMPT.format(
        metrics_json=json.dumps(metrics, ensure_ascii=False),
        rule_summary="\n".join(rule_summary) if isinstance(rule_summary, list) else str(rule_summary),
        cps_text=cps_text,
        last_issue=last_issue or "无（第一次练习）",
        improvement_text=improvement_text,
    )
    messages = [
        {"role": "system", "content": "你是专业的实时运动教练，输出严格 JSON。"},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt},
        ]},
    ]
    t0 = time.time()
    review = _call_vlm_with_schema(messages, SquatReview)
    elapsed = round(time.time() - t0, 2)

    # 可追溯存档：拼图 + prompt + 原始结果
    archive = {"elapsed_s": elapsed, "ok": review is not None}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, "keyframes.jpg"), grid)
        with open(os.path.join(save_dir, "prompt.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)
        if review is not None:
            with open(os.path.join(save_dir, "review.json"), "w", encoding="utf-8") as f:
                json.dump(review.model_dump(), f, ensure_ascii=False, indent=2)
    return review, archive
