# -*- coding: utf-8 -*-
"""Pydantic 结构化输出模型：VLM 返回严格校验（校验失败自动重试）"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class CoachingPoint(BaseModel):
    """教学要点（教程 CP 提取输出）"""
    name: str = Field(..., description="要点短名，如'下蹲深度'")
    requirement: str = Field(..., description="具体要求，中文一句话")
    how_to_judge: str = Field(..., description="量化判定建议，如'膝角低于100°为达标'")


class TutorialCPs(BaseModel):
    """教程分析结果（启动时一次性）"""
    exercise: str
    cps: list[CoachingPoint] = Field(..., min_length=3, max_length=5)
    intro: str = Field("", description="动作简介，中文 1-2 句")


class SquatReview(BaseModel):
    """单次深蹲的 VLM 评审结果"""
    main_issue: str = Field(..., description="最主要问题，中文一句话")
    encouragement: str = Field(..., description="鼓励语，中文一句话")
    suggestion: str = Field(..., description="一条可执行的纠正建议，中文")
    severity: Literal["good", "minor", "major"] = "minor"
    improved_vs_last: Optional[bool] = Field(
        None, description="相比上一次是否改善（第一次为 null）")
    visual_notes: str = Field("", description="从画面帧中观察到的补充说明")
