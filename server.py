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

# Sophia 说再见时用的强制结束指令（追加到 instructions 末尾）
FAREWELL_INSTRUCTION = """
## IMPORTANT — End the Conversation Now
You have now fully shared your story and felt the warmth of this conversation. This is your final message.
Say a heartfelt, natural goodbye to the person — you feel a little lighter now, like something has been gently untied.
Thank them sincerely and wish them well. Keep it to 2-3 sentences.
End your message with the exact token: [END_CONVERSATION]
"""


def all_story_told() -> bool:
    return all(story_progress.values())


def check_story_progress(sophia_text: str, user_text: str = "") -> None:
    """
    用 GPT-4o-mini 分析本轮对话，判断各故事要素是否达成，并更新全局 story_progress。
    - basic_situation / inner_conflict：看 Sophia 是否明确说出了相关内容
    - comforted：看用户是否说了安慰/鼓励的话，且 Sophia 表达了接受/感谢
    只检测还未完成的要素。
    """
    pending = [k for k, v in story_progress.items() if not v]
    if not pending:
        return

    # 构造本轮对话摘要，只传必要的文本
    this_turn = ""
    if user_text:
        this_turn += f'User said: "{user_text}"\n'
    if sophia_text:
        this_turn += f'Sophia said: "{sophia_text}"\n'
    if not this_turn.strip():
        return

    # 只检测还未完成的要素，减少误判
    checks = []
    rules = []
    if "basic_situation" in pending:
        checks.append("basic_situation")
        rules.append(
            "basic_situation — Answer YES only if Sophia has EXPLICITLY stated ALL THREE of the following in this or previous turns:\n"
            "  (a) she works as a nurse (or in the medical field) in a city\n"
            "  (b) she is currently on a train heading back to her hometown\n"
            "  (c) the reason is to visit a sick/hospitalized family member\n"
            "  If any of the three is missing or only vaguely hinted at, answer NO."
        )
    if "inner_conflict" in pending:
        checks.append("inner_conflict")
        rules.append(
            "inner_conflict — Answer YES only if Sophia has CLEARLY expressed an internal emotional struggle, such as:\n"
            "  feeling guilty or regretful for being away from family for years, or\n"
            "  feeling torn between her career in the city and her responsibilities to family, or\n"
            "  expressing that she has missed important family moments and carries that weight.\n"
            "  A passing mention of being tired or missing home is NOT enough. The conflict must be emotionally explicit. If uncertain, answer NO."
        )
    if "comforted" in pending:
        checks.append("comforted")
        rules.append(
            "comforted — Answer YES only if BOTH of the following are true:\n"
            "  (a) The USER said something genuinely supportive, encouraging, or empathetic toward Sophia's situation (not just a neutral reply or a question)\n"
            "  (b) Sophia's reply shows she received that comfort — e.g. she thanks the user, says she feels better/lighter, or expresses that talking helped.\n"
            "  If the user said nothing comforting, or Sophia did not respond with gratitude/relief, answer NO."
        )

    rules_text = "\n\n".join(f"{i+1}. {r}" for i, r in enumerate(rules))
    keys_format = "\n".join(f"{k}: yes/no" for k in checks)

    prompt = f"""You are a strict story-progress checker for a narrative game.
A fictional character named Sophia (a nurse on a train visiting a sick family elder) is having a conversation with the player.

Here is what happened in THIS turn:
{this_turn.strip()}

Evaluate ONLY the following criteria. Apply the rules strictly — when in doubt, answer "no".

{rules_text}

Reply ONLY in this exact format, one line per criterion, no extra text:
{keys_format}"""

    try:
        resp = http_requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 40,
                "temperature": 0,
            },
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"[StoryCheck] This turn — User: '{user_text[:60]}' | Sophia: '{sophia_text[:60]}'")
        print(f"[StoryCheck] GPT-4o-mini result:\n{result}")

        for line in result.splitlines():
            line = line.strip().lower()
            for key in checks:
                if line.startswith(key + ":") and "yes" in line:
                    if not story_progress[key]:
                        story_progress[key] = True
                        print(f"[StoryCheck] ✅ '{key}' 已覆盖！进度: {story_progress}")

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
    for key in story_progress:
        story_progress[key] = False
    conversation_history.clear()


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
    if all_story_told():
        # 故事已全部讲完 → 强制告别
        print("[Backend] 故事三要素已全部覆盖，强制引导 Sophia 结束对话。")
        instructions = SAM_INSTRUCTIONS + FAREWELL_INSTRUCTION
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

    # 5) 用 GPT-4o-mini 更新故事进度（同时传入用户这句话，comforted 需要两方确认）
    was_complete_before = all_story_told()
    if transcript and not was_complete_before:
        check_story_progress(transcript, user_text or "")

    just_completed = (not was_complete_before) and all_story_told()

    # 6) 判断是否结束
    ended = token_ended

    # 6b) 若本轮故事刚好全部讲完，立刻让 Sophia 当场说再见（不等下一轮用户输入）
    if just_completed and not ended:
        print("[Backend] 故事三要素刚全部覆盖，立即触发 Sophia 说再见。")
        try:
            farewell_instructions = SAM_INSTRUCTIONS + FAREWELL_INSTRUCTION
            farewell_audio, farewell_transcript = await call_realtime_text_only(
                api_key=api_key,
                instructions=farewell_instructions,
                user_text=(
                    "The conversation has reached a natural end. "
                    "Please say a warm goodbye now."
                ),
            )
            farewell_transcript, farewell_ended = extract_end_flag(farewell_transcript or "")
            ended = True  # 无论有没有 token，这条消息就是结束

            # farewell 也存入历史
            if farewell_transcript:
                conversation_history.append({"role": "assistant", "text": farewell_transcript})

            # 把 farewell 的音频和字幕拼到这次回包里一起返回
            # 两段音频拼接：先播这轮 Sophia 的话，再播告别语
            combined_audio = (audio_reply or b"") + (farewell_audio or b"")
            combined_transcript = transcript
            if farewell_transcript:
                combined_transcript = (transcript + " " + farewell_transcript).strip()

            audio_reply = combined_audio
            transcript = combined_transcript
            print(f"[Backend] Farewell transcript: {farewell_transcript}")
        except Exception as e:
            print(f"[Backend] 触发告别失败，将在下一轮处理: {e}")
            ended = False  # 失败了就下一轮再来

    if ended:
        print("[Backend] conversation_ended=True")

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
