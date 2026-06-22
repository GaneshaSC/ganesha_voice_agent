import json
import os
import base64
import httpx
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
import io

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


async def text_to_speech(text: str, client: httpx.AsyncClient) -> bytes:
    """Convert text to speech using OpenAI TTS API."""
    response = await client.post(
        "https://api.openai.com/v1/audio/speech",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        json={
            "model": "tts-1",
            "input": text,
            "voice": "alloy",
            "response_format": "pcm"  # Raw PCM audio
        }
    )
    response.raise_for_status()
    return response.content


def pcm_to_mulaw(pcm_data: bytes) -> bytes:
    """Convert 16-bit PCM to 8-bit µ-law (Twilio format)."""
    import array
    
    # Unpack PCM as 16-bit signed integers (little-endian)
    pcm_array = array.array('h')
    pcm_array.frombytes(pcm_data)
    
    mulaw_data = bytearray()
    for sample in pcm_array:
        # µ-law encoding
        sign = 0x80 if sample < 0 else 0x00
        sample = abs(sample)
        
        if sample > 32635:
            sample = 32635
        
        sample = sample + 132
        exponent = 7
        for i in range(7, 0, -1):
            if sample > (0xFF << i):
                exponent = i
                break
        
        mantissa = (sample >> (exponent + 1)) & 0x0F
        mulaw_byte = ~(sign | (exponent << 4) | mantissa) & 0xFF
        mulaw_data.append(mulaw_byte)
    
    return bytes(mulaw_data)


@app.websocket("/stream")
async def stream(websocket: WebSocket):
    await websocket.accept()
    conversation = [{"role": "system", "content": SYSTEM_PROMPT}]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
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

                # Generate speech from reply
                try:
                    pcm_audio = await text_to_speech(reply_text, client)
                    mulaw_audio = pcm_to_mulaw(pcm_audio)
                    audio_b64 = base64.b64encode(mulaw_audio).decode("utf-8")
                except Exception as tts_error:
                    print(f"TTS error: {tts_error}")
                    # Fallback to silence if TTS fails
                    audio_b64 = base64.b64encode(b"\x00" * 320).decode("utf-8")

                # Send audio frames to Twilio
                # Split audio into 20ms chunks (160 bytes at 8kHz µ-law)
                chunk_size = 160
                for i in range(0, len(mulaw_audio), chunk_size):
                    chunk = mulaw_audio[i:i + chunk_size]
                    chunk_b64 = base64.b64encode(chunk).decode("utf-8")
                    
                    await websocket.send_text(json.dumps({
                        "event": "media",
                        "media": {"payload": chunk_b64}
                    }))

    except Exception as e:
        print("Stream closed:", e)
        await websocket.close()

