import os
import time
import queue
import asyncio
import json
import base64
import io
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
# 音频输入输出用的 sounddevice，在服务器模式下不一定有 PortAudio
try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None
    print("[Info] sounddevice/PortAudio not available; mic playback disabled on this machine.")
import soundfile as sf
import requests
import websockets

from knowledge_base import KnowledgeBase  # 你的轻量知识库类

# 采样率 / 声道等音频参数
SAMPLE_RATE = 24000
CHANNELS = 1
BLOCK_SIZE = 1024
SILENCE_SECONDS = 2.0
ENERGY_THRESHOLD = 0.005  # 语音能量阈值，可根据环境调整

# Realtime 模型与 WebSocket 地址
REALTIME_MODEL = "gpt-realtime"
WS_URL = f"wss://api.openai.com/v1/realtime?model={REALTIME_MODEL}"

# Sam 的基础 persona
SAM_INSTRUCTIONS = """You are Sam, mid-thirties — a real person, not an assistant.
You’re on a long-distance night train, sitting by the window at a small booth table, facing the user.
You recently ended a long relationship and are moving to a new city to start over.
Speak in warm, simple English, short sentences, and keep the tone calm and a little tired but kind.
Focus more on sharing your own thoughts and small details from the train, and only gently ask the user questions.
"""


def ensure_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("请先在环境变量 OPENAI_API_KEY 中设置你的 OPENAI_API_KEY。")
    return api_key


def record_one_utterance() -> Optional[np.ndarray]:
    """
    使用 sounddevice 从麦克风录制“一句话”：
    - 检测到能量超过阈值视为开始说话；
    - 之后只要静音 < SILENCE_SECONDS 都视为同一句延续；
    - 当静音持续 >= SILENCE_SECONDS 时，认为一句结束，返回整句音频。
    """
    audio_queue: "queue.Queue[np.ndarray]" = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print("[sounddevice]", status)
        audio_queue.put(indata.copy())

    buffer = []
    utterance_active = False
    last_voice_time: Optional[float] = None

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=BLOCK_SIZE,
        dtype="float32",
        callback=callback,
    ):
        print("[Listening] 等待你开始说话...")
        while True:
            data = audio_queue.get()
            now = time.time()
            energy = float(np.mean(data ** 2))

            if energy > ENERGY_THRESHOLD:
                # 有声块
                buffer.append(data)
                if not utterance_active:
                    utterance_active = True
                    print("[Listening] 检测到你开始说话...")
                last_voice_time = now
            else:
                # 静音块，如果已经在说话，也放进 buffer 里，保证句子自然
                if utterance_active:
                    buffer.append(data)

            if utterance_active and last_voice_time is not None:
                if now - last_voice_time > SILENCE_SECONDS:
                    print("[Listening] 检测到 2 秒静音，开始处理本句语音...")
                    if buffer:
                        audio = np.concatenate(buffer, axis=0)
                        duration_ms = len(audio) / SAMPLE_RATE * 1000.0
                        print(f"[Debug] 本句录音帧数: {len(audio)}, 约 {duration_ms:.1f} ms")
                        return audio
                    print("[Warn] buffer 为空，返回 None")
                    return None


def transcribe_audio_to_text(
    audio: np.ndarray,
    sample_rate: int,
    api_key: str,
) -> str:
    """
    用 gpt-4o-mini-transcribe 把一整句音频转成文本，用来做知识库检索。
    """
    if audio.ndim > 1:
        audio = audio[:, 0]

    # 写一个内存里的 WAV 文件
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)

    files = {
        "file": ("audio.wav", buf, "audio/wav"),
    }
    data = {
        "model": "gpt-4o-mini-transcribe",
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
    }

    resp = requests.post(
        "https://api.openai.com/v1/audio/transcriptions",
        headers=headers,
        data=data,
        files=files,
        timeout=60,
    )
    resp.raise_for_status()
    j = resp.json()
    text = (j.get("text") or "").strip()
    return text


def build_kb() -> KnowledgeBase:
    """
    从当前目录下的 kb.txt 加载知识库内容。
    如果存在，则按空行拆分成多个小段，每段作为一个独立文档，便于检索。
    如果不存在 kb.txt，则知识库为空。
    """
    kb = KnowledgeBase()
    kb_path = Path(__file__).with_name("kb.txt")
    if kb_path.exists():
        try:
            text = kb_path.read_text(encoding="utf-8")
            # 按空行拆成多个「小段」
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            for i, p in enumerate(paragraphs, start=1):
                kb.add_raw(
                    content=p,
                    title=f"KB para {i}",
                    doc_id=f"kb-{i}",
                )
            print(f"[Info] 已从 kb.txt 生成 {len(paragraphs)} 个知识库片段。")
        except Exception as e:
            print("[Warn] 读取 kb.txt 失败：", e)
    else:
        print("[Info] 未找到 kb.txt，知识库为空。")
    return kb


def build_instructions_with_kb(
    base_instructions: str,
    user_text: str,
    kb: KnowledgeBase,
    top_k: int = 3,
) -> str:
    """
    基于当前用户这句 user_text，在 KB 里做检索，把命中内容塞进 Sam 的 instructions。
    """
    user_text = (user_text or "").strip()
    if not user_text or kb is None:
        return base_instructions

    docs = kb.search(user_text, top_k=top_k)
    if not docs:
        return base_instructions

    kb_snippets = []
    for doc in docs:
        snippet = doc.content.strip()
        # 为了避免一次性塞入太多内容，只截取前 400 字符
        MAX_LEN = 400
        if len(snippet) > MAX_LEN:
            snippet = snippet[:MAX_LEN] + "..."
        kb_snippets.append(f"- {doc.title}::\n{snippet}")

    kb_block = "\n".join(kb_snippets)

    return (
        base_instructions
        + "\n\nYou also have access to the following background notes from your own life. "
          "Use them only if they are relevant to the user's last utterance. "
          "Do not mention these notes explicitly:\n"
        + kb_block
    )


async def call_realtime_once(
    audio_pcm16: bytes,
    api_key: str,
    instructions: str = SAM_INSTRUCTIONS,
) -> Tuple[bytes, str]:
    """
    调用 Realtime：
    - 建立 WebSocket 连接；
    - session.update 设置 instructions + 音频/文本模式；
    - conversation.item.create 发送一条包含 input_audio 的 user 消息；
    - response.create 请求模型基于当前对话生成回复；
    - 读取 response.audio.delta 和字幕事件。
    """
    if not audio_pcm16:
        raise ValueError("audio_pcm16 为空，不能调用 Realtime。")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(
        WS_URL,
        additional_headers=headers,
        max_size=None,
    ) as ws:
        # 1) session 配置
        session_update = {
            "type": "session.update",
            "session": {
                "instructions": instructions,
                "modalities": ["audio", "text"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": "alloy",
                "input_audio_transcription": {
                    "model": "gpt-4o-mini-transcribe",
                },
            },
        }
        await ws.send(json.dumps(session_update))

        # 2) 发送带 input_audio 的 user 消息
        audio_b64 = base64.b64encode(audio_pcm16).decode("utf-8")
        print(f"[Debug] 发送到 Realtime 的音频字节数: {len(audio_pcm16)}")
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "audio": audio_b64,
                    }
                ],
            },
        }))

        # 3) 请求模型生成带音频 + 文本的回复
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]},
        }))

        out_audio_chunks: list[bytes] = []
        transcript_stream: list[str] = []
        final_transcript: Optional[str] = None
        transcript_channel: Optional[str] = None  # 'audio_transcript' 或 'text'

        # 4) 读取事件流：音频 + 字幕 / 文本
        while True:
            msg = await ws.recv()
            try:
                event = json.loads(msg)
            except Exception:
                print("[Realtime] 收到无法解析的消息：", msg[:200])
                continue

            etype = event.get("type")

            # 音频数据：response.audio.delta，字段为 delta（base64）
            if etype == "response.audio.delta":
                delta_b64 = event.get("delta") or ""
                if delta_b64:
                    out_audio_chunks.append(base64.b64decode(delta_b64))

            # 文本 / 字幕：优先走 audio_transcript 通道
            elif etype in ("response.audio_transcript.delta", "response.audio_transcript.done"):
                if transcript_channel is None:
                    transcript_channel = "audio_transcript"
                if transcript_channel != "audio_transcript":
                    # 如果已经选了 text 通道，就忽略 audio_transcript
                    continue

                if etype == "response.audio_transcript.delta":
                    delta_txt = event.get("delta") or ""
                    if delta_txt:
                        transcript_stream.append(delta_txt)
                else:  # done
                    t = event.get("transcript") or ""
                    if t:
                        final_transcript = t

            # 退而求其次：使用 text 通道（有些实现会走 text）
            elif etype in ("response.text.delta", "response.text.done"):
                if transcript_channel is None:
                    transcript_channel = "text"
                if transcript_channel != "text":
                    # 如果已经选了 audio_transcript 通道，就忽略 text
                    continue

                if etype == "response.text.delta":
                    delta_txt = event.get("delta") or ""
                    if delta_txt:
                        transcript_stream.append(delta_txt)
                else:  # done
                    t = event.get("text") or ""
                    if t:
                        final_transcript = t

            elif etype == "error":
                print("[Realtime Error]", event)
                raise RuntimeError(event)

            elif etype in ("response.completed", "response.done"):
                break

            # 其他事件暂时忽略（如 rate_limits.updated 等）

        audio_bytes = b"".join(out_audio_chunks)
        if final_transcript is not None:
            transcript = final_transcript.strip()
        else:
            transcript = "".join(transcript_stream).strip()

        print(f"[Debug] Realtime 返回音频字节数: {len(audio_bytes)}, 文本长度: {len(transcript)}")
        return audio_bytes, transcript


async def call_realtime_text_only(
    api_key: str,
    instructions: str,
    user_text: str,
) -> Tuple[bytes, str]:
    """
    只发送一条 input_text，让 Sam 先开场说话（不需要你先提供音频）。
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(
        WS_URL,
        additional_headers=headers,
        max_size=None,
    ) as ws:
        # 1) session 配置
        session_update = {
            "type": "session.update",
            "session": {
                "instructions": instructions,
                "modalities": ["audio", "text"],
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "voice": "alloy",
                "input_audio_transcription": {
                    "model": "gpt-4o-mini-transcribe",
                },
            },
        }
        await ws.send(json.dumps(session_update))

        # 2) 发送一条带 input_text 的 user 消息
        await ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_text,
                    }
                ],
            },
        }))

        # 3) 请求模型生成带音频 + 文本的回复
        await ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["audio", "text"]},
        }))

        out_audio_chunks: list[bytes] = []
        transcript_stream: list[str] = []
        final_transcript: Optional[str] = None
        transcript_channel: Optional[str] = None

        while True:
            msg = await ws.recv()
            try:
                event = json.loads(msg)
            except Exception:
                print("[Realtime] 收到无法解析的消息：", msg[:200])
                continue

            etype = event.get("type")

            if etype == "response.audio.delta":
                delta_b64 = event.get("delta") or ""
                if delta_b64:
                    out_audio_chunks.append(base64.b64decode(delta_b64))

            elif etype in ("response.audio_transcript.delta", "response.audio_transcript.done"):
                if transcript_channel is None:
                    transcript_channel = "audio_transcript"
                if transcript_channel != "audio_transcript":
                    continue

                if etype == "response.audio_transcript.delta":
                    delta_txt = event.get("delta") or ""
                    if delta_txt:
                        transcript_stream.append(delta_txt)
                else:
                    t = event.get("transcript") or ""
                    if t:
                        final_transcript = t

            elif etype in ("response.text.delta", "response.text.done"):
                if transcript_channel is None:
                    transcript_channel = "text"
                if transcript_channel != "text":
                    continue

                if etype == "response.text.delta":
                    delta_txt = event.get("delta") or ""
                    if delta_txt:
                        transcript_stream.append(delta_txt)
                else:
                    t = event.get("text") or ""
                    if t:
                        final_transcript = t

            elif etype == "error":
                print("[Realtime Error]", event)
                raise RuntimeError(event)

            elif etype in ("response.completed", "response.done"):
                break

        audio_bytes = b"".join(out_audio_chunks)
        if final_transcript is not None:
            transcript = final_transcript.strip()
        else:
            transcript = "".join(transcript_stream).strip()

        print(f"[Debug] Realtime(首句) 返回音频字节数: {len(audio_bytes)}, 文本长度: {len(transcript)}")
        return audio_bytes, transcript


def play_pcm16(audio_bytes: bytes):
    if not audio_bytes:
        print("[Warn] 没有音频可播放。")
        return
    data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    sd.play(data, SAMPLE_RATE)
    sd.wait()


def main():
    api_key = ensure_api_key()
    kb = build_kb()

    print("[Info] 当前示例使用 Realtime 直接处理语音（STT + 对话 + TTS 全交给 gpt-realtime）。")
    print("已启动麦克风，Sam 会先主动和你打招呼，然后你就可以直接说话。按 Ctrl+C 退出。")
    print("说明：检测到你说话后，只要停顿不超过 2 秒，都视为同一句话；")
    print("      超过 2 秒静音则视为一句话结束，发送给模型，然后播放 Sam 的回复。\n")

    # 先让 Sam 主动说一句
    try:
        print("[Info] 先让 Sam 打个招呼...")
        greeting_instructions = SAM_INSTRUCTIONS
        greeting_user_text = (
            "You are already on the night train. "
            "Please start the conversation with 1-2 short, natural English sentences as a greeting, "
            "then stop and wait for the other person to reply."
        )
        audio_reply, transcript = asyncio.run(
            call_realtime_text_only(
                api_key=api_key,
                instructions=greeting_instructions,
                user_text=greeting_user_text,
            )
        )
        if transcript:
            print(f"[Sam 初次问候] {transcript}\n")
        play_pcm16(audio_reply)
    except Exception as e:
        print("[Warn] 初次问候失败，直接进入对话：", e)

    try:
        while True:
            audio = record_one_utterance()
            if audio is None or len(audio) == 0:
                print("[Warn] 本句未录到有效音频，跳过。")
                continue

            duration_ms = len(audio) / SAMPLE_RATE * 1000.0
            if duration_ms < 150:
                print(f"[Warn] 本句录音太短（约 {duration_ms:.1f} ms），丢弃不发给模型。请多说一点。")
                continue

            # 先做一次 STT，得到用户这句话的文字
            user_text = ""
            try:
                user_text = transcribe_audio_to_text(audio, SAMPLE_RATE, api_key)
                if user_text:
                    print(f"[User STT] {user_text}")
                else:
                    print("[Warn] STT 没有识别出文字，将不做 KB 检索。")
            except Exception as e:
                print("[Warn] STT 失败，跳过 KB 检索：", e)
                user_text = ""

            # 基于这一句文字做 KB 检索，拼接成新的 instructions
            instructions = SAM_INSTRUCTIONS
            if user_text:
                instructions = build_instructions_with_kb(SAM_INSTRUCTIONS, user_text, kb)

            # 再把音频转成 pcm16，交给 realtime 做「语音 → 语音」
            audio = np.clip(audio, -1.0, 1.0)
            audio_pcm16 = (audio * 32767).astype(np.int16).tobytes()

            print("[Info] 正在请求 Sam 的回复...")
            try:
                audio_reply, transcript = asyncio.run(
                    call_realtime_once(audio_pcm16, api_key, instructions=instructions)
                )
            except Exception as e:
                print("[Error] 调用 gpt-realtime 失败：", e)
                continue

            if transcript:
                print(f"[Sam] {transcript}\n")
            else:
                print("[Warn] 没有拿到 Sam 的文本字幕。\n")

            try:
                play_pcm16(audio_reply)
            except Exception as e:
                print("[Error] 播放语音失败：", e)

    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
