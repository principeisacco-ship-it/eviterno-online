# ============================================================
#  EVITERNO ONLINE · un'unica app (pagina + copilota EVA)
#  Serve index.html su "/" e l'endpoint EVA su "/api/chat".
#  La chiave Groq si legge da variabile d'ambiente GROQ_API_KEY
#  (che imposterai come "secret" sul servizio di hosting).
# ============================================================

import os
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI(title="EVITERNO online")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
HERE = os.path.dirname(os.path.abspath(__file__))


@app.get("/", response_class=HTMLResponse)
async def home():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        html = f.read()
    # Niente cache: ogni aggiornamento è subito visibile, nessuna versione vecchia.
    return HTMLResponse(html, headers={"Cache-Control": "no-store, max-age=0"})


@app.post("/api/chat")
async def chat(req: Request):
    body = await req.json()
    system = body.get("system", "")
    messages = body.get("messages", [])
    message = body.get("message", "")
    context = body.get("context", {})

    if not GROQ_API_KEY:
        name = (context.get("user") or {}).get("name") or "esploratore"
        return {"reply": f"(demo) Ciao {name}! Ricevuto: '{message}'. Manca il secret GROQ_API_KEY sul server."}

    groq_messages = [{"role": "system", "content": system}] + [
        {"role": m.get("role", "user"), "content": m.get("content", "")} for m in messages
    ]
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
                json={"model": GROQ_MODEL, "messages": groq_messages, "max_tokens": 1000, "temperature": 0.7},
            )
        data = r.json()
        if "error" in data:
            return {"reply": "Errore da Groq: " + str(data["error"].get("message", data["error"]))}
        return {"reply": data["choices"][0]["message"]["content"].strip()}
    except Exception as e:
        return {"reply": "Errore lato server: " + str(e)}


if __name__ == "__main__":
    import uvicorn
    # HF Spaces usa 7860; Render (e altri) impostano PORT.
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
