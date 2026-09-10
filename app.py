# ============================================================
#  EVITERNO ONLINE · un'unica app (pagina + copilota EVA + social)
#  Serve index.html su "/", l'endpoint EVA su "/api/chat" e il
#  feed social interno su "/api/social/posts".
#  La chiave Groq si legge da variabile d'ambiente GROQ_API_KEY
#  (che imposterai come "secret" sul servizio di hosting).
#
#  Sicurezza applicata in questo file:
#  - CORS ristretto al proprio dominio (non più "*")
#  - Rate limiting per IP su /api/chat e sulla pubblicazione post
#  - Limite alla dimensione dei messaggi/post in ingresso
#  - Intestazioni di sicurezza del browser (anti-clickjacking, anti-sniffing)
#  - Errori generici verso il client, dettagli solo nei log del server
#
#  NOTA sul feed social: i post sono salvati in un file JSON sul
#  disco del server (nessun database vero, ancora). Su Render il
#  disco del piano gratuito NON è permanente: i post si azzerano
#  a ogni nuovo deploy. Per un feed che sopravvive ai deploy serve
#  un vero database (prossimo passo della roadmap).
# ============================================================

import os
import time
import json
from pathlib import Path
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI(title="EVITERNO online")

# --- CORS: solo il proprio dominio (configurabile via env, con fallback al dominio Render noto) ---
# In produzione imposta ALLOWED_ORIGIN sul valore esatto del tuo sito (es. https://eviterno-online.onrender.com).
_default_origin = "https://eviterno-online.onrender.com"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGIN", _default_origin).split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
HERE = os.path.dirname(os.path.abspath(__file__))

MAX_MESSAGE_CHARS = 4000       # limite per singolo messaggio
MAX_HISTORY_ITEMS = 40         # limite ai messaggi di contesto inviati
MAX_BODY_BYTES = 60_000        # limite dimensione totale della richiesta

# --- Rate limiting semplice per IP (in memoria: 20 richieste/minuto per IP) ---
RATE_LIMIT = 20
RATE_WINDOW = 60  # secondi
_hits = defaultdict(deque)


def _client_ip(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return req.client.host if req.client else "unknown"


def _rate_limited(ip: str) -> bool:
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > RATE_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False


# --- Feed social interno: post salvati in un file JSON sul server ---
POSTS_FILE = Path(HERE) / "social_posts.json"
MAX_POSTS = 200          # quanti post tenere in memoria/file
MAX_POST_CHARS = 500      # lunghezza massima di un post
MAX_AUTHOR_CHARS = 60


def _load_posts():
    try:
        return json.loads(POSTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save_posts(posts):
    try:
        POSTS_FILE.write_text(json.dumps(posts[-MAX_POSTS:], ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        print("Errore salvataggio post social:", repr(e))


@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    return response


@app.get("/", response_class=HTMLResponse)
async def home():
    with open(os.path.join(HERE, "index.html"), encoding="utf-8") as f:
        html = f.read()
    # Niente cache: ogni aggiornamento è subito visibile, nessuna versione vecchia.
    return HTMLResponse(html, headers={"Cache-Control": "no-store, max-age=0"})


@app.post("/api/chat")
async def chat(req: Request):
    ip = _client_ip(req)
    if _rate_limited(ip):
        return JSONResponse(
            status_code=429,
            content={"reply": "Troppe richieste in poco tempo. Aspetta un momento e riprova."},
        )

    raw = await req.body()
    if len(raw) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"reply": "Messaggio troppo grande."})

    try:
        body = await req.json()
    except Exception:
        return JSONResponse(status_code=400, content={"reply": "Richiesta non valida."})

    system = str(body.get("system", ""))[:MAX_MESSAGE_CHARS]
    message = str(body.get("message", ""))[:MAX_MESSAGE_CHARS]
    context = body.get("context", {}) if isinstance(body.get("context", {}), dict) else {}

    messages_in = body.get("messages", [])
    if not isinstance(messages_in, list):
        messages_in = []
    messages_in = messages_in[-MAX_HISTORY_ITEMS:]

    if not GROQ_API_KEY:
        name = (context.get("user") or {}).get("name") or "esploratore"
        return {"reply": f"(demo) Ciao {name}! Ricevuto: '{message}'. Manca il secret GROQ_API_KEY sul server."}

    groq_messages = [{"role": "system", "content": system}] + [
        {
            "role": m.get("role", "user") if isinstance(m, dict) else "user",
            "content": str(m.get("content", ""))[:MAX_MESSAGE_CHARS] if isinstance(m, dict) else "",
        }
        for m in messages_in
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
            # Dettaglio completo solo nei log del server, mai al client.
            print("Groq error:", data["error"])
            return {"reply": "Il motore AI non è disponibile in questo momento. Riprova tra poco."}
        return {"reply": data["choices"][0]["message"]["content"].strip()}
    except Exception as e:
        print("Chat endpoint error:", repr(e))
        return {"reply": "Errore lato server. Riprova tra poco."}


@app.get("/api/social/posts")
async def get_social_posts():
    return {"posts": _load_posts()}


@app.post("/api/social/posts")
async def create_social_post(req: Request):
    ip = _client_ip(req)
    if _rate_limited(ip):
        return JSONResponse(status_code=429, content={"error": "Troppe richieste in poco tempo. Aspetta un momento."})

    raw = await req.body()
    if len(raw) > MAX_BODY_BYTES:
        return JSONResponse(status_code=413, content={"error": "Post troppo grande."})

    try:
        body = await req.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Richiesta non valida."})

    author = str(body.get("author", "")).strip()[:MAX_AUTHOR_CHARS] or "Anonimo"
    text = str(body.get("text", "")).strip()[:MAX_POST_CHARS]
    if not text:
        return JSONResponse(status_code=400, content={"error": "Il post non può essere vuoto."})

    posts = _load_posts()
    posts.append({"author": author, "text": text, "ts": time.time()})
    posts = posts[-MAX_POSTS:]
    _save_posts(posts)
    return {"posts": posts}


if __name__ == "__main__":
    import uvicorn
    # HF Spaces usa 7860; Render (e altri) impostano PORT.
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))

