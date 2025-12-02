from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import base64
import numpy as np

from voice_agent_realtime import (
    ensure_api_key,
    SAMPLE_RATE,
    SAM_INSTRUCTIONS,
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


class SpeakRequest(BaseModel):
    # Unity 传来的原始 pcm16 单声道音频，采样率 SAMPLE_RATE (默认 24000)，base64 编码
    audio_pcm16_b64: str


class SpeakResponse(BaseModel):
    # Sam 回复的音频（pcm16 单声道 SAMPLE_RATE Hz，base64）
    reply_pcm16_b64: str
    # Sam 这句话的文本字幕
    transcript: str
    # 用户这一句说话的识别结果（STT）
    user_transcript: str = ""


@app.get("/sam/greet", response_model=SpeakResponse)
async def sam_greet():
    """
    让 Sam 像在你的 VS Code 脚本里一样，先用 Realtime 主动打一次招呼。
    这里直接复用 call_realtime_text_only，只用 SAM_INSTRUCTIONS，不做 KB 检索。
    """
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
    reply_b64 = base64.b64encode(audio_reply).decode("utf-8") if audio_reply else ""
    return SpeakResponse(
    reply_pcm16_b64=reply_b64,
    transcript=transcript or "",
    user_transcript=""
)


@app.post("/sam/speak", response_model=SpeakResponse)
async def sam_speak(req: SpeakRequest):
    """
    完整复刻你在 VS Code 项目里的对话逻辑：
    1. Unity 把“一句话”的 pcm16 音频发过来（base64）；
    2. 后端先用 gpt-4o-mini-transcribe 做 STT → user_text；
    3. 基于 user_text 在外接知识库 KB 里检索，构造增强后的 instructions；
    4. 再把原始 pcm16 音频交给 Realtime，生成 Sam 的语音回复和字幕。
    """
    audio_bytes = base64.b64decode(req.audio_pcm16_b64)
    if not audio_bytes:
        return SpeakResponse(reply_pcm16_b64="", transcript="")

    # 转成 float32 [-1, 1]，复用你原脚本里的 transcribe_audio_to_text
    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0

    # 1) STT 得到 user_text
    try:
        user_text = transcribe_audio_to_text(audio_np, SAMPLE_RATE, api_key)
    except Exception as e:
        print("[Backend] STT 失败：", e)
        user_text = ""

    # 2) 基于 user_text 构造带 KB 的 instructions（和你原 main 一样）
    instructions = SAM_INSTRUCTIONS
    if user_text:
        print(f"[Backend] User STT: {user_text}")
        instructions = build_instructions_with_kb(SAM_INSTRUCTIONS, user_text, kb)

    # 3) 调用 Realtime：复用你原来的 call_realtime_once（直接吃 pcm16）
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
        )

    reply_b64 = base64.b64encode(audio_reply).decode("utf-8") if audio_reply else ""
    return SpeakResponse(
        reply_pcm16_b64=reply_b64,
        transcript=transcript or "",
        user_transcript=user_text or ""
    )
