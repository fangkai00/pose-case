# -*- coding: utf-8 -*-
"""Flask 入口：摄像头/视频源采集线程 + MediaPipe 骨架 + MJPEG 推流 + 深蹲状态机

M2 范围：实时骨架 + 膝角 + 深蹲状态机（rep 检测）+ 4 指标 + 规则提示横幅。
运行：python app.py                          → 摄像头
      python app.py --video test_squat.mp4   → 视频文件源（无摄像头时调试/演示用）
      python app.py --camera 1               → 指定摄像头索引
浏览器打开 http://127.0.0.1:5000
"""
import collections
import queue
import sys
import threading
import time

import cv2
from flask import Flask, Response, jsonify, render_template

import config
from coach.rule_engine import RuleEngine
from coach.schemas import TutorialCPs
from coach.session_memory import (load_prev_session, save_session_summary,
                                  summarize_session)
from coach.vlm_reviewer import review_rep as vlm_review
from metrics.metrics_engine import MetricsEngine, judge_metrics, judge_improvement
from tts.speaker import speak
from vision.pose_tracker import PoseTracker, L
from vision.skeleton_draw import draw_skeleton, angle_3d
from vision.state_machine import SquatState, SquatStateMachine

app = Flask(__name__)


def _parse_args():
    """解析命令行：--video <路径> / --camera <索引>"""
    video_path, camera_index = None, None
    argv = sys.argv[1:]
    if "--video" in argv:
        video_path = argv[argv.index("--video") + 1]
    if "--camera" in argv:
        camera_index = int(argv[argv.index("--camera") + 1])
    return video_path, camera_index


class CameraStream:
    """后台采集+推理线程：MJPEG 推流 + 状态机 + 指标 + 事件队列"""

    FRAME_BUF = 60        # 原始帧缓存（VLM 拼图用），约 2 秒
    POSE_BUF = 150        # 姿态历史缓存（指标回查用），极小

    def __init__(self, source=None):
        self.source = source if source is not None else config.CAMERA_INDEX
        self._cap = None
        self._tracker = None
        self._frame_jpeg = None          # 最新一帧 JPEG
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self.fps = 0.0
        self.person_detected = False      # 是否检测到人体（前端提示用）

        # M2: 状态机 + 指标 + 规则
        self._sm = SquatStateMachine()
        self._metrics = MetricsEngine()
        self._rules = RuleEngine()
        self._frames = collections.deque(maxlen=self.FRAME_BUF)       # (idx, BGR frame)
        self._poses = collections.deque(maxlen=self.POSE_BUF)         # (idx, lm, world)
        self._frame_idx = 0

        # 供前端读取的状态（_lock 保护）
        self.rep_count = 0
        self.state = "idle"
        self.session_ready = False         # 就绪信号：全身入镜且站立确认后置 True（前端语音提示）
        self._reset_pending = True         # 启动/重置后等待首次就绪判定
        self.realtime_hint = None          # 实时横幅提示（带过期时间）
        self._hint_expire = 0.0
        self.reps = []                     # 已完成 rep 的结果摘要列表
        self.tutorial_cps = None           # 教学 CPs（启动时加载/提取）
        self._review_q = queue.Queue(maxsize=2)  # VLM 评审任务队列（满则丢弃旧任务）

        # 跨会话记忆：上一次练习摘要先验（runs/session_summary.json；重置时
        # 更新为刚完成的一轮）→ 本轮首次 rep 的对比基线 + VLM"上次主要问题"
        self.prev_session = load_prev_session()
        self._session_started_at = time.time()
        self._session_gen = 0              # 轮次代号：重置 +1，作废在途旧轮评审
        if self.prev_session:
            print(f"[会话记忆] 上次练习 {self.prev_session['rep_count']} 次，"
                  f"最常见问题：{self.prev_session.get('top_issue') or '无记录'}")

    # ---------- 生命周期 ----------
    def start(self):
        if self._running:
            return
        if isinstance(self.source, str):
            self._cap = cv2.VideoCapture(self.source)  # 视频文件源
        else:
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)  # Windows 下 DSHOW 更稳
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
        if not self._cap.isOpened():
            kind = "视频文件" if isinstance(self.source, str) else f"摄像头 index={self.source}"
            raise RuntimeError(f"无法打开{kind}: {self.source}")
        self._tracker = PoseTracker()
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        # 教学 CPs 提取（后台，带磁盘缓存；首次约 10s）
        threading.Thread(target=self._load_tutorial, daemon=True).start()
        # VLM 评审 worker（单线程顺序处理队列）
        threading.Thread(target=self._review_worker, daemon=True).start()

    def _review_worker(self):
        while self._running:
            try:
                task = self._review_q.get(timeout=1)
            except queue.Empty:
                continue
            try:
                self._vlm_review_async(*task)
            except Exception as e:
                print(f"[VLM worker] 异常: {e}")

    def _load_tutorial(self):
        try:
            from tutorial.cp_extractor import extract_cps
            cps, _, meta = extract_cps()
            if cps is not None:
                self.tutorial_cps = {"exercise": cps.exercise, "intro": cps.intro,
                                     "cps": [c.model_dump() for c in cps.cps],
                                     "instructions_en": meta["instructions"]["en"]}
                print(f"[教程] CPs 加载完成: {cps.exercise}（{len(cps.cps)} 条要点）")
            else:
                print("[教程] CPs 提取失败，VLM 评审将不带教学要点")
        except Exception as e:
            print(f"[教程] CPs 加载异常: {e}")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._tracker:
            self._tracker.close()
            self._tracker = None

    # ---------- 主循环 ----------
    def _loop(self):
        t_prev = time.time()
        fps_smooth = 0.0
        t_frame = time.time()
        self._src_fps = self._cap.get(cv2.CAP_PROP_FPS) if isinstance(self.source, str) else 0.0
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                if isinstance(self.source, str):
                    # 视频文件源：播完后循环
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.01)
                continue

            # 视频源帧率适配：按文件原始帧率节流，避免快进
            if isinstance(self.source, str) and self._src_fps > 0:
                time.sleep(max(0, 1.0 / self._src_fps - (time.time() - t_frame)))
            t_frame = time.time()

            self._frame_idx += 1
            # 缓存原始帧（rep 关键帧拼图用）
            self._frames.append((self._frame_idx, frame.copy()))

            # 姿态推理
            detected, landmarks, world = self._tracker.process(frame)
            if detected:
                self._poses.append((self._frame_idx, landmarks.copy(), world.copy()))

            # 腿部关键点可见性：坐姿/腿部出画时膝角是无效估计，
            # 不参与状态机、指标与基线（否则会产生假 rep 与错误基线）。
            # 阈值见 config.LEG_VIS_THRESHOLD（0.35：兼顾暗光/宽松裤装）
            legs_ok = False
            if detected:
                legs_ok = all(landmarks[i][3] > config.LEG_VIS_THRESHOLD for i in (
                    L.LEFT_HIP, L.RIGHT_HIP, L.LEFT_KNEE,
                    L.RIGHT_KNEE, L.LEFT_ANKLE, L.RIGHT_ANKLE))

            # 实时膝角（world 3D 坐标计算）
            lk = rk = None
            if detected and legs_ok:
                lk = angle_3d(world[L.LEFT_HIP], world[L.LEFT_KNEE], world[L.LEFT_ANKLE])
                rk = angle_3d(world[L.RIGHT_HIP], world[L.RIGHT_KNEE], world[L.RIGHT_ANKLE])
            draw_skeleton(frame, landmarks if detected else None, lk, rk)

            # ---- 状态机 + 指标 + 规则 ----
            knee_avg = (lk + rk) / 2 if lk is not None and rk is not None else None
            if detected and legs_ok:
                self._metrics.update_baseline(landmarks, world, knee_avg)
                # 传入摄像头帧号：rep 事件关键帧索引与 _poses/_frames 缓存严格对齐
                state, rep_ev = self._sm.update(knee_avg, True, frame_idx=self._frame_idx)

                # 实时规则横幅（下降/底部阶段）
                rt = {"torso_lean": 0.0, "l_r_diff": abs(lk - rk) if lk and rk else 0.0}
                if lk and rk:
                    # 逐帧快速前倾角（world）
                    import numpy as np
                    sm = (world[L.LEFT_SHOULDER] + world[L.RIGHT_SHOULDER]) / 2
                    hm = (world[L.LEFT_HIP] + world[L.RIGHT_HIP]) / 2
                    v = sm - hm
                    cos = -v[1] / (np.linalg.norm(v) + 1e-9)
                    rt["torso_lean"] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
                hint = self._rules.realtime_hint(rt, state.value if state else "idle", time.time())
                if hint:
                    self.realtime_hint = hint
                    self._hint_expire = time.time() + 2.5

                # rep 完成 → 计算指标 + 存档
                if rep_ev:
                    self._on_rep_done(rep_ev)
                with self._lock:
                    self.state = state.value
                    self.rep_count = self._sm.rep_count
                # 就绪判定：重置后首次"全身入镜 + 站立确认"（STAND 需膝角>阈值持续 stand_hold，
                # 此时站立基线必然已建立）→ 通知前端可以开始
                if self._reset_pending and state == SquatState.STAND:
                    self._reset_pending = False
                    self.session_ready = True
                    print("[会话] 已就绪：可以开始深蹲")
            else:
                state, _ = self._sm.update(None, False)
                with self._lock:
                    self.state = state.value
                if detected:
                    # 检测到人但腿部不可见：明确引导，避免"做了动作却没反应"
                    self.realtime_hint = "请后退，确保全身（含腿部）入镜"
                    self._hint_expire = time.time() + 1.0

            # 状态提示
            with self._lock:
                self.person_detected = detected
                if self.realtime_hint and time.time() > self._hint_expire:
                    self.realtime_hint = None
                hint_now = self.realtime_hint
            if not detected:
                cv2.putText(frame, "No person detected", (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
            elif not legs_ok:
                cv2.putText(frame, "Step back: legs out of frame", (12, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)

            # 画面叠加：状态 + rep 计数 + 横幅
            cv2.putText(frame, f"State: {self.state}  Reps: {self._sm.rep_count}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 200, 255), 2, cv2.LINE_AA)
            if hint_now:
                cv2.rectangle(frame, (0, frame.shape[0] - 44), (frame.shape[1], frame.shape[0]),
                              (0, 0, 160), -1)
                cv2.putText(frame, hint_now, (12, frame.shape[0] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

            # FPS
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)
            self.fps = fps_smooth
            cv2.putText(frame, f"{fps_smooth:.0f} FPS", (frame.shape[1] - 90, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # 编码 JPEG
            ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok_enc:
                with self._lock:
                    self._frame_jpeg = buf.tobytes()

    # ---------- rep 完成：指标 + 关键帧 + 摘要 + VLM 评审（异步） ----------
    def _on_rep_done(self, rep_ev):
        # 按底帧索引回查该帧姿态（指标必须用最低点姿态计算，非当前站立帧）
        bottom_lm = bottom_world = None
        for idx, lm, world in reversed(self._poses):
            if idx == rep_ev.frame_bottom:
                bottom_lm, bottom_world = lm, world
                break
        if bottom_lm is None:
            return  # 底帧已滚出缓存，跳过本次

        m = self._metrics.compute_rep_metrics(bottom_lm, bottom_world)
        judged = judge_metrics(m)
        level, summary = RuleEngine.rep_summary(judged)

        # 本地改善判定：与上一次 rep 指标按容差对比（不依赖 VLM，降级时也有效）
        # 本轮第一次且存在跨会话记忆 → 与上次练习末次指标对比
        with self._lock:
            prev_m = self.reps[-1]["metrics"] if self.reps else None
            prev_sess = self.prev_session
        improvement = None
        vs_prev_session = False
        if prev_m is None and prev_sess and prev_sess.get("last_metrics"):
            prev_m = prev_sess["last_metrics"]
            vs_prev_session = True
        if prev_m is not None:
            v, d = judge_improvement(prev_m, m)
            improvement = {"verdict": v, "detail": d}
            if vs_prev_session:
                improvement["vs"] = "prev_session"

        # 抓关键帧原始图像（顶帧/底帧，供 VLM 拼图）
        frames = {i: f for i, f in self._frames}
        top_frame = frames.get(rep_ev.frame_stand)
        bottom_frame = frames.get(rep_ev.frame_bottom)

        rec = {
            "rep": rep_ev.rep_index,
            "ts": round(rep_ev.ts, 2),
            "metrics": m,
            "judged": [{"name": n, "value": v, "ok": bool(ok), "comment": c}
                       for n, v, ok, c in judged],
            "rule_level": level,
            "rule_summary": summary,
            "improvement": improvement,   # 本地改善判定（第二次起）
            "vlm": None,      # VLM 评审结果（异步回填）
            "audio": None,    # TTS 音频路径（异步回填）
        }
        with self._lock:
            self.reps.append(rec)
            reps_snapshot = list(self.reps)
            gen = self._session_gen
        # 会话记忆：每 rep 增量落盘（直接关闭/崩溃也不丢已完成练习）
        save_session_summary(reps_snapshot, self._session_started_at)
        imp_log = f" 改善判定={improvement['verdict']}" if improvement else ""
        if improvement and improvement.get("vs") == "prev_session":
            imp_log += "（vs 上次练习）"
        print(f"[REP {rep_ev.rep_index}] 膝角={m['knee_angle']}° 深度={m.get('depth_ratio')} "
              f"前倾={m['torso_lean']}° 对称差={m['l_r_diff']}° → {level}{imp_log}")

        # VLM 评审 + TTS 交给单 worker 队列（积压时只保留最新 rep，跳过中间的）
        try:
            self._review_q.put_nowait(
                (rep_ev.rep_index, top_frame, bottom_frame, m, summary, gen))
        except Exception:
            pass  # 队列满：丢弃（旧 rep 未评审，属预期节流）

    # ---------- VLM 评审 + TTS（后台线程） ----------
    def _vlm_review_async(self, rep_index, top_frame, bottom_frame, metrics, rule_summary,
                          session_gen):
        with self._lock:
            if session_gen != self._session_gen:
                return  # 旧轮评审作废：重置后新轮 rep 编号复用，防止串写
            # 找上一次的 VLM 主要问题（本轮没有 → 跨会话记忆：上次练习的主要问题）
            last_issue = None
            for r in reversed(self.reps[:-1] if self.reps and self.reps[-1]["rep"] == rep_index else self.reps):
                if r.get("vlm"):
                    last_issue = r["vlm"]["main_issue"]
                    break
            prev_sess = self.prev_session
            # 取该 rep 的本地改善判定，告知 VLM 并让其解释（以本地判定为准）
            imp = None
            for r in self.reps:
                if r["rep"] == rep_index:
                    imp = r.get("improvement")
                    break
        if last_issue is None and prev_sess:
            last_issue = prev_sess.get("last_issue") or prev_sess.get("top_issue")
        imp_arg = (imp["verdict"], imp["detail"]) if imp else None
        cps_list = self.tutorial_cps["cps"] if self.tutorial_cps else []

        review, archive = vlm_review(
            top_frame, bottom_frame, metrics, rule_summary, cps_list, last_issue,
            improvement=imp_arg,
            save_dir=f"runs/rep_{rep_index:04d}",
        )
        with self._lock:
            if session_gen != self._session_gen:
                return  # 评审期间会话被重置：结果作废
            for r in self.reps:
                if r["rep"] == rep_index:
                    if review is not None:
                        r["vlm"] = review.model_dump()
                        # TTS 播报：问题 + 建议
                        tts_text = f"{review.main_issue}。{review.suggestion}。"
                        audio_rel = speak(tts_text, rep_index)
                        r["audio"] = "/" + audio_rel.replace("\\", "/") if audio_rel else None
                        print(f"[REP {rep_index}] VLM({archive.get('elapsed_s')}s): "
                              f"{review.main_issue} | 改善={review.improved_vs_last}")
                    else:
                        print(f"[REP {rep_index}] VLM 评审失败，降级为规则反馈")
                        r["vlm"] = {
                            "main_issue": rule_summary[0] if rule_summary else "动作待改进",
                            "encouragement": "继续保持练习！",
                            "suggestion": rule_summary[1] if len(rule_summary) > 1 else "注意动作幅度",
                            "severity": "minor", "improved_vs_last": None,
                            "visual_notes": "（VLM 不可用，规则引擎反馈）",
                            "degraded": True,
                        }
                    break
            reps_snapshot = list(self.reps)
        # 会话记忆：评审回填后更新摘要（main_issue 统计 / last_issue）
        save_session_summary(reps_snapshot, self._session_started_at)

    # ---------- 会话重置（前端"开始"按钮） ----------
    def reset_session(self):
        """清空 rep 计数/历史/站立基线/待评审队列，从零开始一轮练习。
        已完成的 rep 聚合为"上一轮"摘要 → 跨会话记忆先验（下次首次对比/评审用）。
        轮次代号 +1：在途的旧轮 VLM 评审作废（新轮 rep 编号从 1 复用，防串写）。"""
        with self._lock:
            finished = list(self.reps)
            if finished:
                self.prev_session = summarize_session(finished, self._session_started_at) \
                    or self.prev_session
            self._session_gen += 1
            self._sm = SquatStateMachine()   # 基线重新校准 + rep 计数归零
            self.reps = []
            self.rep_count = 0
            self.state = "idle"
            self.session_ready = False       # 等待重新站直确认
            self._reset_pending = True
            self.realtime_hint = None
            self._session_started_at = time.time()
        while True:                          # 丢弃队列中待评审的旧 rep
            try:
                self._review_q.get_nowait()
            except queue.Empty:
                break
        print("[会话] 已重置：计数与基线从零开始"
              + (f"；上一轮 {len(finished)} 次已存入会话记忆" if finished else ""))

    # ---------- MJPEG 帧 ----------
    def get_jpeg(self):
        with self._lock:
            return self._frame_jpeg


_video_path, _camera_index = _parse_args()
_source = _video_path if _video_path else _camera_index
stream = CameraStream(_source)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    def generate():
        while stream._running:
            jpeg = stream.get_jpeg()
            if jpeg:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n")
            time.sleep(1 / 60)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def status():
    return {"running": stream._running, "fps": round(stream.fps, 1),
            "person": stream.person_detected, "state": stream.state,
            "ready": stream.session_ready,
            "reps": stream.rep_count, "source": str(stream.source)}


@app.route("/api/reps")
def reps():
    """已完成 rep 的指标结果（前端轮询渲染指标卡）"""
    with stream._lock:
        return jsonify({"reps": stream.reps[-20:], "count": stream.rep_count})


@app.route("/api/hint")
def hint():
    """当前实时横幅提示"""
    return jsonify({"hint": stream.realtime_hint})


@app.route("/api/reset", methods=["POST"])
def reset():
    """前端"开始练习"按钮：会话归零（计数/基线/历史/评审队列）"""
    stream.reset_session()
    return jsonify({"ok": True})


@app.route("/api/tutorial")
def tutorial():
    """教学 CPs（左侧教程面板）"""
    return jsonify({"tutorial": stream.tutorial_cps})


@app.route("/api/raw/<int:rep_id>")
def raw_record(rep_id):
    """VLM 评审原始记录（可追溯）：prompt + 原始响应 + 关键帧拼图"""
    import json
    import os
    d = f"runs/rep_{rep_id:04d}"
    if not os.path.isdir(d):
        return jsonify({"error": "no archive"}), 404
    out = {"image": f"/api/raw/{rep_id}/keyframes"}
    try:
        with open(os.path.join(d, "prompt.txt"), encoding="utf-8") as f:
            out["prompt"] = f.read()
    except OSError:
        out["prompt"] = None
    try:
        with open(os.path.join(d, "review.json"), encoding="utf-8") as f:
            out["review"] = json.load(f)
    except (OSError, json.JSONDecodeError):
        out["review"] = None
    return jsonify(out)


@app.route("/api/raw/<int:rep_id>/keyframes")
def raw_keyframes(rep_id):
    """关键帧拼图图片"""
    import os
    path = os.path.join(f"runs/rep_{rep_id:04d}", "keyframes.jpg")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return Response(f.read(), mimetype="image/jpeg")
    return "not found", 404


@app.route("/audio/<path:name>")
def audio_file(name):
    """TTS 音频文件服务"""
    import os
    safe = os.path.basename(name)
    path = os.path.join("audio", safe)
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        return Response(data, mimetype="audio/mpeg")
    return "not found", 404


if __name__ == "__main__":
    stream.start()
    print(f"* 视频源: {stream.source}")
    try:
        app.run(host="127.0.0.1", port=5000, threaded=True)
    finally:
        stream.stop()
