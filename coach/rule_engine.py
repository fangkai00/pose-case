# -*- coding: utf-8 -*-
"""规则引擎：CP-wise 实时/逐 rep 规则初判（本地毫秒级，零 LLM 延迟）

实时模式：逐帧检查明显违规（如下降中过度前倾），出横幅提示。
rep 模式：汇总 4 指标判定，作为 VLM 评审输入（呼应论文 CP→参数映射）。
"""
import config


class RuleEngine:
    # 实时横幅提示（下降阶段触发，冷却避免刷屏）
    REALTIME_RULES = [
        # (条件函数, 提示文本)
        (lambda rt: rt["torso_lean"] > config.TORSO_LEAN_MAX,
         "躯干过度前倾，挺胸收腹"),
        (lambda rt: rt["l_r_diff"] > config.L_R_DIFF_MAX + 5,
         "左右膝角差异较大，注意两侧均衡发力"),
    ]

    def __init__(self, cooldown_sec=2.0):
        self._last_fire = {}      # 规则索引 -> 上次触发时间
        self.cooldown_sec = cooldown_sec

    # ------------------------------------------------------------------
    def realtime_hint(self, rt_metrics, phase, now):
        """逐帧调用。rt_metrics: {torso_lean, l_r_diff, ...}
        返回横幅提示 str|None。仅下降/底部阶段触发，带冷却。"""
        if phase not in ("descend", "bottom"):
            return None
        import time
        for i, (cond, text) in enumerate(self.REALTIME_RULES):
            if cond(rt_metrics):
                t = self._last_fire.get(i, 0)
                if now - t >= self.cooldown_sec:
                    self._last_fire[i] = now
                    return text
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def rep_summary(judged):
        """rep 结束的规则初判汇总（供 VLM 输入和结果区展示）
        judged: metrics_engine.judge_metrics 的返回值
        返回 (整体判定, 中文摘要列表)"""
        issues = [(name, comment) for name, val, ok, comment in judged if not ok]
        if not issues:
            return "good", ["动作标准，继续保持！"]
        level = "minor" if len(issues) == 1 else "major"
        return level, [f"{n}：{c}" for n, c in issues]
