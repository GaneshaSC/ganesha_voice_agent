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


def pcm_to_mulaw(pcm_data: bytes) -> bytes:
    """Convert 16-bit PCM to 8-bit µ-law (Twilio format)."""
    import array
    
    pcm_array = array.array('h')
    pcm_array.frombytes(pcm_data)
    
    mulaw_data = bytearray()
    for sample in pcm_array:
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
    
    print("WebSocket connected, waiting for media frames...")

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            while True:
                try:
                    frame = await websocket.receive_text()
                    data = json.loads(frame)

                    if data.get("event") != "media":
                        continue

                    stt_text = "Hello"
                    conversation.append({"role": "user", "content": stt_text})

                    # LLM call
                    try:
                        print(f"Calling OpenAI LLM with {len(conversation)} messages...")
                        response = await client.post(
                            "https://api.openai.com/v1/chat/completions",
                            headers={
                                "Authorization": f"Bearer {OPENAI_API_KEY}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": "gpt-4o-mini",
                                "messages": conversation
                            },
                            timeout=30.0
                        )
                        response.raise_for_status()
                        llm_response = response.json()
                        print(f"LLM response: {llm_response}")

                        if "error" in llm_response:
                            print(f"OpenAI error: {llm_response['error']}")
                            reply_text = "Error from API"
                        elif "choices" in llm_response and llm_response["choices"]:
                            reply_text = llm_response["choices"][0]["message"]["content"]
                            print(f"LLM reply: {reply_text}")
                        else:
                            print(f"Invalid response structure: {llm_response}")
                            reply_text = "Invalid response"

                    except httpx.HTTPError as e:
                        print(f"HTTP error: {e}")
                        reply_text = "Connection error"
                    except Exception as e:
                        print(f"LLM error: {e}")
                        reply_text = "Error"

                    conversation.append({"role": "assistant", "content": reply_text})

                    # TTS
                    mulaw_audio = None
                    try:
                        print(f"Calling TTS for: {reply_text[:50]}...")
                        tts_response = await client.post(
                            "https://api.openai.com/v1/audio/speech",
                            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                            json={
                                "model": "tts-1",
                                "input": reply_text,
                                "voice": "alloy",
                                "response_format": "pcm"
                            },
                            timeout=30.0
                        )
                        tts_response.raise_for_status()
                        pcm_audio = tts_response.content
                        mulaw_audio = pcm_to_mulaw(pcm_audio)
                        print(f"TTS generated {len(mulaw_audio)} bytes of audio")
                    except Exception as e:
                        print(f"TTS error: {e}")
                        mulaw_audio = b"\x00" * 320

                    # Send audio in 20ms chunks (160 bytes)
                    chunk_size = 160
                    for i in range(0, len(mulaw_audio), chunk_size):
                        chunk = mulaw_audio[i:i + chunk_size]
                        chunk_b64 = base64.b64encode(chunk).decode("utf-8")
                        
                        await websocket.send_text(json.dumps({
                            "event": "media",
                            "media": {"payload": chunk_b64}
                        }))
                    
                    print(f"Sent {len(mulaw_audio) // chunk_size} audio chunks")

                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                except Exception as e:
                    print(f"Frame processing error: {e}")

    except Exception as e:
        print(f"WebSocket closed: {e}")
        await websocket.close()

