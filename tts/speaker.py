# -*- coding: utf-8 -*-
"""edge-tts 语音播报：异步生成 mp3 → 落盘 audio/ 目录 → 前端 <audio> 播放

策略：每次反馈生成一个编号 mp3（rep_0001.mp3），前端按序播放。
同一时间只有最新一条反馈在播，新反馈打断旧的（前端控制）。
"""
import asyncio
import os

import edge_tts

TTS_DIR = "audio"
VOICE = "zh-CN-XiaoxiaoNeural"   # 晓晓：中文女声，自然度高


def _tts_to_file(text, out_path, voice=VOICE):
    async def run():
        comm = edge_tts.Communicate(text, voice)
        await comm.save(out_path)
    asyncio.run(run())


def speak(text, rep_index):
    """生成第 rep_index 次的播报音频。返回 mp3 相对路径（前端可访问）"""
    os.makedirs(TTS_DIR, exist_ok=True)
    rel = f"{TTS_DIR}/rep_{rep_index:04d}.mp3"
    out = os.path.abspath(rel)
    try:
        # 播报文本：主要问题 + 建议（控制在 60 字内保证 TTS 时长合理）
        text_short = text if len(text) <= 80 else text[:77] + "…"
        _tts_to_file(text_short, out)
        return rel
    except Exception as e:
        print(f"[TTS] 生成失败: {e}")
        return None


if __name__ == "__main__":
    p = speak("下蹲幅度不足，请蹲至大腿与地面平行。保持背部挺直，加油！", 0)
    print("生成:", p, "大小:", os.path.getsize(p) if p else None)
