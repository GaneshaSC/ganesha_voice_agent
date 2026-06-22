import json
import os
import base64
import httpx
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response

app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is required")

SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt")
with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()


@app.post("/voice")
async def voice(request: Request):
    domain = request.url.hostname
    twiml = f"""
<Response>
  <Connect>
    <Stream url="wss://{domain}/stream" />
  </Connect>
</Response>
""".strip()
    return Response(content=twiml, media_type="text/xml")


@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 20ms of silence (320 bytes of 0x00), base64 encoded
    SILENCE_FRAME = base64.b64encode(b"\x00" * 320).decode("utf-8")

    try:
        while True:
            frame = await websocket.receive_text()
            data = json.loads(frame)

            if data.get("event") != "media":
                continue

            # No STT → placeholder text
            stt_text = "Hello"

            conversation.append({"role": "user", "content": stt_text})

            # LLM call
            llm_payload = {
                "model": "gpt-4o-mini",
                "messages": conversation
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {OPENAI_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json=llm_payload
                )
                response.raise_for_status()
                llm_response = response.json()

            reply_text = llm_response["choices"][0]["message"]["content"]
            conversation.append({"role": "assistant", "content": reply_text})

            # Send valid silence frame to keep Twilio call alive
            await websocket.send_text(json.dumps({
                "event": "media",
                "media": {"payload": SILENCE_FRAME}
            }))

    except Exception as e:
        print("Stream closed:", e)
        await websocket.close()
