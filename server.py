from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import base64
import numpy as np
import requests as http_requests

from voice_agent_realtime import (
    ensure_api_key,
    SAMPLE_RATE,
    SAM_INSTRUCTIONS,
    END_TOKEN,
    transcribe_audio_to_text,
    build_kb,
    build_instructions_with_kb,
    call_realtime_once,
    call_realtime_text_only,
)

app = FastAPI(title="Sophia Realtime Backend (Unity Frontend)")

# 全局初始化：API Key + 知识库
api_key = ensure_api_key()
kb = build_kb()

# ── 对话历史（全局，单会话）──────────────────────────────────────
# 每条格式：{"role": "user"/"assistant", "text": "..."}
conversation_history: list[dict] = []

# ── 故事进度追踪（全局，单会话）────────────────────────────────
# 三个核心故事要素：基本情况 / 内心矛盾 / 被开导安慰
story_progress = {
    "basic_situation": False,   # 用户已了解 Sophia 的基本情况（身份、工作、这次旅程的原因）
    "inner_conflict":  False,   # 用户已了解 Sophia 的内心矛盾（离家与责任、成长与愧疚）
    "comforted":       False,   # 用户已给予 Sophia 安慰或开导，Sophia 表达了接受/感谢
}

# 故事完成后进入"收尾倒计时"，最多再走 2 轮后强制结束
# winding_down_turns = 0 表示尚未进入收尾模式
winding_down_turns: int = 0
WINDING_DOWN_MAX: int = 2        # 收尾轮数上限

# ── 故事检测：最少对话轮数保护 ──────────────────────────────────
# 前 N 轮对话不做故事检测，避免 Sophia 刚开口就误触发
# 每次用户说一句话 = 一轮
STORY_CHECK_MIN_TURNS: int = 1   # 至少对话 1 轮后才开始检测
dialog_turn_count: int = 0       # 已完成的对话轮数计数

# ── 方案A：对话摘要缓存（减少 Evaluator Token 消耗）────────────
# 每隔 SUMMARY_UPDATE_EVERY 轮，把旧历史压缩成一段摘要
# Evaluator 只看：摘要 + 最近 RECENT_TURNS_KEPT 轮原文
conversation_summary: str = ""   # 当前维护的对话摘要
SUMMARY_UPDATE_EVERY: int = 5    # 每 5 轮更新一次摘要
RECENT_TURNS_KEPT: int = 3       # 摘要之外保留的最近原文轮数

# ── 收尾引导指令（三要素全达成后追加）──────────────────────────
# 让 Sophia 在接下来 1-2 轮自然地把话收拢，做出告别信号，最终带上 END_TOKEN
WINDING_DOWN_INSTRUCTION = """
## Gentle Story Wrap-Up (Internal Guidance — do NOT mention this to the player)
The conversation has reached a natural emotional conclusion. You've shared your story and felt genuine warmth from this person.
From now on, gradually bring the conversation to a close. Do this naturally over the next 1-2 exchanges:
- Speak in a softer, more settled tone — you feel lighter, like something has been gently untied.
- Weave in small physical cues of the journey ending: glance out the window, notice the landscape changing, mention the train will arrive soon.
- You do NOT need to end immediately. Let the goodbye feel earned and unhurried.
- When the moment feels right (within the next reply or two), say a warm and sincere farewell.
- End your very last message with the exact token: [END_CONVERSATION]
"""

# 强制最终告别指令（收尾轮数耗尽时使用）
FAREWELL_INSTRUCTION = """
## IMPORTANT — This Is Your Final Message
The journey is almost over. Say a heartfelt, natural goodbye — you feel lighter now, like something has been gently untied.
Thank this person sincerely. Keep it to 2-3 warm sentences.
End your message with the exact token: [END_CONVERSATION]
"""


def all_story_told() -> bool:
    return all(story_progress.values())


def update_summary() -> None:
    """
    方案A：将【除最近 RECENT_TURNS_KEPT 条之外】的历史对话压缩成摘要。
    每隔 SUMMARY_UPDATE_EVERY 轮触发一次，用 gpt-4o-mini 生成简短摘要。
    """
    global conversation_summary
    # 如果历史条数不够多，无需摘要
    if len(conversation_history) <= RECENT_TURNS_KEPT:
        return

    # 需要被摘要的部分（除掉最近几轮）
    to_summarize = conversation_history[:-RECENT_TURNS_KEPT]
    lines = []
    for item in to_summarize:
        role_label = "User" if item["role"] == "user" else "Sophia"
        lines.append(f'{role_label}: "{item["text"]}"')
    dialog_text = "\n".join(lines)

    try:
        resp = http_requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a concise dialogue summarizer for a narrative game. "
                            "Summarize the following conversation excerpt in plain English. "
                            "Focus on: who Sophia is, why she's on the train, what emotional conflicts she mentioned, "
                            "and any comfort/support the user offered. Keep the summary under 120 words. "
                            "Be factual and specific — do not add anything not in the conversation."
                        ),
                    },
                    {"role": "user", "content": dialog_text},
                ],
                "max_tokens": 180,
                "temperature": 0,
            },
            timeout=15,
        )
        resp.raise_for_status()
        conversation_summary = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"[Summary] ✅ 摘要已更新（覆盖 {len(to_summarize)} 条历史）:\n{conversation_summary}")
    except Exception as e:
        print(f"[Summary] 摘要生成失败（忽略）: {e}")


def check_story_progress(sophia_text: str, user_text: str = "") -> None:
    """
    方案A：基于【对话摘要 + 最近几轮原文】做累积判断，只检测还未完成的要素。
    使用 system/user 双 role 强制约束，并要求模型先输出判断理由再给结论。
    """
    pending = [k for k, v in story_progress.items() if not v]
    if not pending or (not sophia_text and not user_text):
        return

    # ── 1. 拼装上下文：摘要 + 最近 RECENT_TURNS_KEPT 轮原文 ────────
    recent_lines = []
    for item in conversation_history[-RECENT_TURNS_KEPT * 2:]:   # 每轮含 user+assistant 共 2 条
        role_label = "User" if item["role"] == "user" else "Sophia"
        recent_lines.append(f'{role_label}: "{item["text"]}"')
    recent_text = "\n".join(recent_lines) if recent_lines else "(no recent dialogue)"

    if conversation_summary:
        full_history = (
            f"[Earlier conversation summary]\n{conversation_summary}\n\n"
            f"[Recent dialogue]\n{recent_text}"
        )
        print(f"[StoryCheck] 使用摘要模式（摘要+最近{RECENT_TURNS_KEPT}轮原文）")
    else:
        # 摘要尚未生成时，退化为原始全历史（最多30条）
        raw_lines = []
        for item in conversation_history[-30:]:
            role_label = "User" if item["role"] == "user" else "Sophia"
            raw_lines.append(f'{role_label}: "{item["text"]}"')
        full_history = "\n".join(raw_lines) if raw_lines else "(conversation just started)"
        print(f"[StoryCheck] 使用完整历史模式（摘要未就绪，共{len(conversation_history)}条）")

    # ── 2. 根据 pending 动态生成判断标准 ────────────────────────────
    criteria_blocks = []

    if "basic_situation" in pending:
        criteria_blocks.append(
            "=== CRITERION A: basic_situation ===\n"
            "Mark DONE only when Sophia has stated, CLEARLY AND EXPLICITLY, ALL THREE facts:\n"
            "  [A1] She is a nurse (or works in a medical/healthcare role) in a city.\n"
            "  [A2] She is currently on a long-distance train traveling toward her hometown.\n"
            "  [A3] The specific reason for this trip is to visit a family member who is sick or hospitalized.\n"
            "STRICT rule: if ANY of A1/A2/A3 has only been hinted at, implied, or never mentioned → answer 'no'.\n"
            "Example of NOT enough: Sophia says 'I work long shifts' (A1 missing specifics), or 'I'm going home' (A3 missing).\n"
            "Example of ENOUGH: Sophia says 'I'm a nurse in the city, and I'm on this train because my grandmother is in the hospital back home.'"
        )

    if "inner_conflict" in pending:
        criteria_blocks.append(
            "=== CRITERION B: inner_conflict ===\n"
            "Mark DONE only when Sophia has EXPLICITLY verbalized a deep emotional tension. She must express at least ONE of:\n"
            "  [B1] Guilt or regret for having been physically absent from family for years (not just 'I miss home').\n"
            "  [B2] Feeling torn between her professional life in the city and her duty/love for family back home.\n"
            "  [B3] The pain of having missed important family milestones or moments because of her career.\n"
            "STRICT rule:\n"
            "  - Saying she is 'tired' or 'stressed at work' is NOT enough.\n"
            "  - Saying she 'misses home' or 'hasn't been back in a while' is NOT enough.\n"
            "  - The internal conflict must be emotionally explicit — she must name the tension, not just describe fatigue.\n"
            "Example of NOT enough: 'Work has been really exhausting lately.'\n"
            "Example of ENOUGH: 'I keep telling myself the career was worth it, but then I think about all the birthdays I missed, and I'm not sure I believe it anymore.'"
        )

    if "comforted" in pending:
        criteria_blocks.append(
            "=== CRITERION C: comforted ===\n"
            "Mark DONE only when BOTH conditions are clearly met in the conversation:\n"
            "  [C1] The USER made a genuine, specific gesture of support, comfort, or empathy toward Sophia's situation.\n"
            "       - A simple question ('really?', 'oh?') or neutral acknowledgment ('I see') does NOT count.\n"
            "       - The user must say something that shows they understand Sophia's struggle and offer emotional support.\n"
            "       - Examples that COUNT: 'You shouldn't be so hard on yourself.', 'It sounds like you really care about your family.', 'Being there now is what matters.'\n"
            "  [C2] Sophia's response shows she genuinely received and accepted that comfort.\n"
            "       - She must express relief, gratitude, or that the conversation helped (e.g. 'That actually helps.', 'Thank you, I needed to hear that.', 'I feel a little lighter now.').\n"
            "       - Sophia simply saying 'thank you' for something unrelated does NOT count.\n"
            "STRICT rule: if C1 is missing (user didn't comfort her) OR C2 is missing (Sophia didn't show she felt comforted) → answer 'no'."
        )

    criteria_text = "\n\n".join(criteria_blocks)
    pending_keys_format = "\n".join(f"{k}: yes/no  # your brief reason" for k in pending)

    system_prompt = (
        "You are a STRICT pass/fail evaluator for a narrative game. "
        "Your only job is to check whether specific story milestones have been clearly achieved in a conversation. "
        "You have a strong bias toward 'no' — you only say 'yes' when the evidence is unambiguous and explicit. "
        "Vague hints, implications, and near-misses always count as 'no'. "
        "Do not be generous. Do not give benefit of the doubt."
    )

    user_prompt = (
        f"Here is the full conversation so far:\n"
        f"---\n{full_history}\n---\n\n"
        f"Evaluate ONLY these pending criteria:\n\n"
        f"{criteria_text}\n\n"
        f"Reply in EXACTLY this format (one line per criterion, include a short reason after #):\n"
        f"{pending_keys_format}"
    )

    try:
        resp = http_requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                "max_tokens": 120,
                "temperature": 0,
            },
            timeout=15,
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"[StoryCheck] ── evaluating pending={pending}")
        print(f"[StoryCheck] GPT-4o result:\n{result}")

        for line in result.splitlines():
            # 格式：  basic_situation: yes  # because ...
            # 取冒号后、#号前的部分
            if ":" not in line:
                continue
            key_part, _, rest = line.partition(":")
            key_part = key_part.strip().lower()
            verdict = rest.split("#")[0].strip().lower()   # 去掉 # 注释部分
            if key_part in pending and "yes" in verdict:
                if not story_progress[key_part]:
                    story_progress[key_part] = True
                    print(f"[StoryCheck] ✅ '{key_part}' 已覆盖！进度: {story_progress}")

    except Exception as e:
        print(f"[StoryCheck] 检测失败（忽略）: {e}")


def extract_end_flag(transcript: str) -> tuple[str, bool]:
    """
    检查 transcript 中是否包含 END_TOKEN。
    如有，则移除该标记并返回 (cleaned_transcript, True)。
    """
    if END_TOKEN in transcript:
        cleaned = transcript.replace(END_TOKEN, "").strip()
        return cleaned, True
    return transcript, False


def reset_story_progress() -> None:
    """重置故事进度和对话历史（如需支持多次对话）。"""
    global winding_down_turns, dialog_turn_count, conversation_summary
    for key in story_progress:
        story_progress[key] = False
    conversation_history.clear()
    winding_down_turns = 0
    dialog_turn_count = 0
    conversation_summary = ""    # 方案A：同步清空摘要缓存


# ── 数据模型 ──────────────────────────────────────────────────

class SpeakRequest(BaseModel):
    audio_pcm16_b64: str


class SpeakResponse(BaseModel):
    reply_pcm16_b64: str
    transcript: str
    user_transcript: str = ""
    conversation_ended: bool = False
    # 故事进度（三个核心要素）
    progress_basic_situation: bool = False
    progress_inner_conflict: bool = False
    progress_comforted: bool = False


# ── 路由 ──────────────────────────────────────────────────────

@app.get("/sophia/greet", response_model=SpeakResponse)
async def sophia_greet():
    """让 Sophia 主动打一次招呼，重置故事进度。"""
    reset_story_progress()

    greeting_user_text = (
        "You are already on the night train. "
        "Please start the conversation with 1-2 short, natural English sentences as a greeting, "
        "then stop and wait for the other person to reply."
    )

    audio_reply, transcript = await call_realtime_text_only(
        api_key=api_key,
        instructions=SAM_INSTRUCTIONS,
        user_text=greeting_user_text,
        history=[],
    )
    transcript, ended = extract_end_flag(transcript or "")

    # 把 Sophia 的开场白存入历史
    if transcript:
        conversation_history.append({"role": "assistant", "text": transcript})

    reply_b64 = base64.b64encode(audio_reply).decode("utf-8") if audio_reply else ""
    return SpeakResponse(
        reply_pcm16_b64=reply_b64,
        transcript=transcript,
        user_transcript="",
        conversation_ended=ended,
    )


@app.post("/sophia/speak", response_model=SpeakResponse)
async def sophia_speak(req: SpeakRequest):
    """
    对话主流程：
    1. STT 识别用户语音；
    2. 若故事已全部讲完，在 instructions 末尾追加强制告别指令；
    3. 否则正常构造带 KB 的 instructions；
    4. 调用 Realtime 生成 Sophia 的语音回复；
    5. 用 GPT-4o-mini 检测本轮 Sophia 的回复覆盖了哪些故事要素；
    6. 检测 END_TOKEN 或故事全覆盖，决定是否返回 conversation_ended=True。
    """
    audio_bytes = base64.b64decode(req.audio_pcm16_b64)
    if not audio_bytes:
        return SpeakResponse(reply_pcm16_b64="", transcript="", conversation_ended=False)

    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0

    # 1) STT
    try:
        user_text = transcribe_audio_to_text(audio_np, SAMPLE_RATE, api_key)
    except Exception as e:
        print("[Backend] STT 失败：", e)
        user_text = ""

    if user_text:
        print(f"[Backend] User STT: {user_text}")
        # 把用户这句话追加进历史
        conversation_history.append({"role": "user", "text": user_text})

    # 2) 构造 instructions
    global winding_down_turns
    if winding_down_turns >= WINDING_DOWN_MAX:
        # 收尾轮数已耗尽 → 强制最终告别
        print(f"[Backend] 收尾轮数耗尽({winding_down_turns}/{WINDING_DOWN_MAX})，强制触发告别。")
        instructions = SAM_INSTRUCTIONS + FAREWELL_INSTRUCTION
    elif winding_down_turns > 0:
        # 收尾倒计时进行中 → 继续引导自然收尾
        print(f"[Backend] 收尾倒计时中({winding_down_turns}/{WINDING_DOWN_MAX})，引导自然结束。")
        instructions = SAM_INSTRUCTIONS + WINDING_DOWN_INSTRUCTION
    elif all_story_told():
        # 三要素刚全部达成，这一轮就开始收尾（不立刻结束）
        print("[Backend] 三要素首次全覆盖，进入收尾模式第 1 轮。")
        winding_down_turns = 1
        instructions = SAM_INSTRUCTIONS + WINDING_DOWN_INSTRUCTION
    else:
        instructions = build_instructions_with_kb(SAM_INSTRUCTIONS, user_text, kb)

    # 3) 调用 Realtime（带完整历史）
    try:
        audio_reply, transcript = await call_realtime_once(
            audio_pcm16=audio_bytes,
            api_key=api_key,
            instructions=instructions,
            history=list(conversation_history),
        )
    except Exception as e:
        print("[Backend] 调用 Realtime 失败：", e)
        return SpeakResponse(
            reply_pcm16_b64="",
            transcript="Sorry, something went wrong while talking to Sophia.",
            conversation_ended=False,
        )

    transcript = transcript or ""

    # 4) 把 Sophia 的回复追加进历史，再检测 END_TOKEN
    if transcript:
        conversation_history.append({"role": "assistant", "text": transcript})

    transcript, token_ended = extract_end_flag(transcript)

    # 5) 更新故事进度（仅在收尾模式之前，且达到最少轮数要求后）
    global dialog_turn_count
    if user_text:
        dialog_turn_count += 1

    # 方案A：每隔 SUMMARY_UPDATE_EVERY 轮，异步更新对话摘要
    # 在收尾模式开始前才做摘要，收尾后无需再更新
    if (
        dialog_turn_count > 0
        and dialog_turn_count % SUMMARY_UPDATE_EVERY == 0
        and winding_down_turns == 0
        and not all_story_told()
    ):
        print(f"[Summary] 第 {dialog_turn_count} 轮，触发摘要更新...")
        update_summary()

    was_complete_before = all_story_told()
    if transcript and not was_complete_before:
        if dialog_turn_count >= STORY_CHECK_MIN_TURNS:
            check_story_progress(transcript, user_text or "")
        else:
            print(f"[StoryCheck] 跳过检测（当前轮数 {dialog_turn_count} < 最低要求 {STORY_CHECK_MIN_TURNS}）")
    just_completed = (not was_complete_before) and all_story_told()

    # 6) 收尾倒计时推进
    if winding_down_turns > 0 and not token_ended:
        # 已在收尾模式但 Sophia 还没说 END_TOKEN → 推进计数
        winding_down_turns += 1
        print(f"[Backend] 收尾轮数推进至 {winding_down_turns}/{WINDING_DOWN_MAX}")
    elif just_completed and winding_down_turns == 0:
        # 三要素本轮刚达成（之前还没进收尾），从下一轮开始收尾
        # （此轮 instructions 已经是 WINDING_DOWN，所以 winding_down_turns 设为 1）
        winding_down_turns = 1
        print(f"[Backend] 三要素达成，收尾倒计时启动（turns=1）")

    # 7) 判断是否对话结束
    ended = token_ended
    if ended:
        print("[Backend] conversation_ended=True（Sophia 发出了 END_TOKEN）")

    reply_b64 = base64.b64encode(audio_reply).decode("utf-8") if audio_reply else ""
    return SpeakResponse(
        reply_pcm16_b64=reply_b64,
        transcript=transcript,
        user_transcript=user_text or "",
        conversation_ended=ended,
        progress_basic_situation=story_progress["basic_situation"],
        progress_inner_conflict=story_progress["inner_conflict"],
        progress_comforted=story_progress["comforted"],
    )


@app.post("/sophia/reset")
async def sophia_reset():
    """手动重置故事进度（调试用）。"""
    reset_story_progress()
    return {"status": "ok", "story_progress": story_progress}


@app.get("/sophia/progress")
async def sophia_progress():
    """查看当前故事进度（调试用）。"""
    return {
        "story_progress": story_progress,
        "all_told": all_story_told(),
    }