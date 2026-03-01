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

app = FastAPI(title="Sam Realtime Backend (Unity Frontend)")

# 全局初始化：API Key + 知识库
api_key = ensure_api_key()
kb = build_kb()

# ── 故事进度追踪（全局，单会话）────────────────────────────────
# 三个核心故事要素：分手原因 / 搬家原因 / 对未来的期待
story_progress = {
    "breakup_reason": False,    # 讲过分手/结束感情的原因
    "move_reason":    False,    # 讲过为什么选择这座新城市
    "future_hope":    False,    # 表达过对未来的期待或希望
}

# Sam 说再见时用的强制结束指令（追加到 instructions 末尾）
FAREWELL_INSTRUCTION = """
## IMPORTANT — End the Conversation Now
You have now fully shared your story. This is your final message.
Say a warm, natural goodbye to the person — something like you're feeling tired and going to sleep,
or wishing them well on their journey. Keep it to 2-3 sentences.
End your message with the exact token: [END_CONVERSATION]
"""


def all_story_told() -> bool:
    return all(story_progress.values())


def check_story_progress(sam_text: str) -> None:
    """
    用 GPT-4o-mini 分析 Sam 刚说的这句话，判断覆盖了哪些故事要素，
    并更新全局 story_progress。只对还未覆盖的要素做检测。
    """
    # 已经全部讲完就不用再检测了
    pending = [k for k, v in story_progress.items() if not v]
    if not pending or not sam_text:
        return

    prompt = f"""You are analyzing a line of dialogue spoken by a fictional character named Sam.
Sam is on a night train, having just ended a long relationship and moving to a new city.

Sam just said:
"{sam_text}"

For each item below, answer only "yes" or "no" (lowercase, no punctuation):
1. breakup_reason — Did Sam mention or hint at WHY the relationship ended or why she broke up?
2. move_reason — Did Sam mention or hint at WHY she chose this specific new city to move to?
3. future_hope — Did Sam express any hope, excitement, or positive feeling about her future?

Reply in this exact format (one per line):
breakup_reason: yes/no
move_reason: yes/no
future_hope: yes/no"""

    try:
        resp = http_requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 60,
                "temperature": 0,
            },
            timeout=10,
        )
        resp.raise_for_status()
        result = resp.json()["choices"][0]["message"]["content"].strip()
        print(f"[StoryCheck] Sam said: {sam_text[:80]}...")
        print(f"[StoryCheck] GPT-4o-mini result:\n{result}")

        for line in result.splitlines():
            line = line.strip().lower()
            for key in story_progress:
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
    """重置故事进度（如需支持多次对话）。"""
    for key in story_progress:
        story_progress[key] = False


# ── 数据模型 ──────────────────────────────────────────────────

class SpeakRequest(BaseModel):
    audio_pcm16_b64: str


class SpeakResponse(BaseModel):
    reply_pcm16_b64: str
    transcript: str
    user_transcript: str = ""
    conversation_ended: bool = False


# ── 路由 ──────────────────────────────────────────────────────

@app.get("/sam/greet", response_model=SpeakResponse)
async def sam_greet():
    """让 Sam 主动打一次招呼，重置故事进度。"""
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
    )
    transcript, ended = extract_end_flag(transcript or "")
    reply_b64 = base64.b64encode(audio_reply).decode("utf-8") if audio_reply else ""
    return SpeakResponse(
        reply_pcm16_b64=reply_b64,
        transcript=transcript,
        user_transcript="",
        conversation_ended=ended,
    )


@app.post("/sam/speak", response_model=SpeakResponse)
async def sam_speak(req: SpeakRequest):
    """
    对话主流程：
    1. STT 识别用户语音；
    2. 若故事已全部讲完，在 instructions 末尾追加强制告别指令；
    3. 否则正常构造带 KB 的 instructions；
    4. 调用 Realtime 生成 Sam 的语音回复；
    5. 用 GPT-4o-mini 检测本轮 Sam 的回复覆盖了哪些故事要素；
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

    # 2) 构造 instructions
    if all_story_told():
        # 故事已全部讲完 → 强制告别
        print("[Backend] 故事三要素已全部覆盖，强制引导 Sam 结束对话。")
        instructions = SAM_INSTRUCTIONS + FAREWELL_INSTRUCTION
    else:
        instructions = build_instructions_with_kb(SAM_INSTRUCTIONS, user_text, kb)

    # 3) 调用 Realtime
    try:
        audio_reply, transcript = await call_realtime_once(
            audio_pcm16=audio_bytes,
            api_key=api_key,
            instructions=instructions,
        )
    except Exception as e:
        print("[Backend] 调用 Realtime 失败：", e)
        return SpeakResponse(
            reply_pcm16_b64="",
            transcript="Sorry, something went wrong while talking to Sam.",
            conversation_ended=False,
        )

    transcript = transcript or ""

    # 4) 检测 END_TOKEN（强制告别路径会带这个 token）
    transcript, token_ended = extract_end_flag(transcript)

    # 5) 用 GPT-4o-mini 更新故事进度（异步检测，不阻塞回包）
    if transcript and not all_story_told():
        check_story_progress(transcript)

    # 6) 判断是否结束
    #    - Sam 在回复里带了 END_TOKEN，说明她说了再见
    #    - 或者故事刚好在这轮全部讲完（下一轮会被强制告别，这里先不结束）
    ended = token_ended
    if ended:
        print("[Backend] conversation_ended=True（Sam 说了再见）")

    reply_b64 = base64.b64encode(audio_reply).decode("utf-8") if audio_reply else ""
    return SpeakResponse(
        reply_pcm16_b64=reply_b64,
        transcript=transcript,
        user_transcript=user_text or "",
        conversation_ended=ended,
    )


@app.post("/sam/reset")
async def sam_reset():
    """手动重置故事进度（调试用）。"""
    reset_story_progress()
    return {"status": "ok", "story_progress": story_progress}


@app.get("/sam/progress")
async def sam_progress():
    """查看当前故事进度（调试用）。"""
    return {
        "story_progress": story_progress,
        "all_told": all_story_told(),
    }