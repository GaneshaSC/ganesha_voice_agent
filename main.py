import json
import os
import requests
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import PlainTextResponse

app = FastAPI()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SYSTEM_PROMPT_PATH = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt")

# Load system prompt
with open(SYSTEM_PROMPT_PATH, "r") as f:
    SYSTEM_PROMPT = f.read()


@app.post("/voice", response_class=PlainTextResponse)
async def voice(request: Request):
    """
    Twilio hits this endpoint when a call comes in.
    We return TwiML telling Twilio to open a WebSocket stream.
    """
    domain = request.url.hostname
    twiml = f"""
<Response>
  <Connect>
    <Stream url="wss://{domain}/stream" />
  </Connect>
</Response>
""".strip()
    return twiml


@app.websocket("/stream")
async def stream(websocket: WebSocket):
    """
    Twilio sends audio frames here.
    Since STT/TTS are removed, we do:
    - Receive audio frames
    - Ignore audio
    - Send a placeholder text to LLM
    - Return empty audio back to Twilio
    """
    await websocket.accept()
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        while True:
            frame = await websocket.receive_text()
            data = json.loads(frame)

            # Only process media frames
            if data.get("event") != "media":
                continue

            # Placeholder transcription (since STT removed)
            stt_text = "Hello"

            conversation.append({"role": "user", "content": stt_text})

            # LLM call
            llm_payload = {
                "model": "gpt-4o-mini",
                "messages": conversation
            }

            llm_response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json=llm_payload
            ).json()

            reply_text = llm_response["choices"][0]["message"]["content"]
            conversation.append({"role": "assistant", "content": reply_text})

            # No TTS → return empty audio
            empty_audio_b64 = ""

            await websocket.send_text(json.dumps({
                "event": "media",
                "media": {"payload": empty_audio_b64}
            }))

    except Exception as e:
        print("Stream closed:", e)
        await websocket.close()