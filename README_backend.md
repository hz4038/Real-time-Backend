# Sam Realtime Backend（基于你当前 VSCode 成熟项目的后端封装）

这个 backend 完全复用你现在在 VS Code 里已经跑通的 `voice_agent_realtime.py` 逻辑，
只是在外面用 FastAPI 包了一层 HTTP 接口，方便 Unity 作为前端调用。

## 文件说明

- `voice_agent_realtime.py`
  你当前成熟项目的核心脚本：
  - 采样率 / 静音检测参数：`SAMPLE_RATE=24000`, `SILENCE_SECONDS=2.0`, `ENERGY_THRESHOLD=0.005`
  - `build_kb()`：从 `kb.txt` 读取内容，并**按空行拆分成多个知识片段**
  - `transcribe_audio_to_text()`：用 `gpt-4o-mini-transcribe` 做 STT
  - `build_instructions_with_kb()`：基于当前句子在 KB 里检索，并把片段拼进 Sam 的 instructions
  - `call_realtime_once()` / `call_realtime_text_only()`：用 WebSocket 调 OpenAI Realtime，返回音频 + 字幕

- `knowledge_base.py` / `kb.txt`
  你的轻量外挂知识库实现和文本内容，未做任何改动。

- `server.py`
  新增的 HTTP 后端：
  - `GET /sam/greet`：调用 `call_realtime_text_only()`，让 Sam 像原脚本一样先打招呼；
  - `POST /sam/speak`：
    1. 收到 Unity 传来的 pcm16 音频；
    2. 用 `transcribe_audio_to_text()` 做 STT；
    3. 用 `build_instructions_with_kb()` 做 KB 检索并增强 instructions；
    4. 用 `call_realtime_once()` 生成 Sam 的语音回复和字幕；
    5. 以 base64 形式把 pcm16 返回给 Unity。

- `requirements.txt`
  在你原有依赖基础上，补充了：
  - `websockets`（Realtime 必需）
  - `fastapi` / `uvicorn[standard]` / `pydantic`（HTTP 后端必需）

## 启动步骤

1. 安装依赖（建议虚拟环境）：

   ```bash
   pip install -r requirements.txt
   ```

2. 设置环境变量 `OPENAI_API_KEY`：

   ```bash
   # PowerShell
   $env:OPENAI_API_KEY = "你的key"
   ```

3. 启动后端：

   ```bash
   uvicorn server:app --host 127.0.0.1 --port 8000
   ```

4. 在浏览器打开 `http://127.0.0.1:8000/docs`，可以看到：

   - `GET /sam/greet`
   - `POST /sam/speak`

至此，Unity 就可以把麦克风录到的一句话音频以 pcm16 形式 POST 到 `/sam/speak`，
从后端拿到 **完全和你 VSCode 项目一致的 Sam 对话逻辑** 的回复。

