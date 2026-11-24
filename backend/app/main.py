"""
main.py
-------
Squelette d'entrée FastAPI montrant comment brancher le nouveau router
d'analyse ET charger le modèle Whisper une seule fois au démarrage.

⚠️ Ceci est un exemple minimal à FUSIONNER avec ton main.py existant
(qui a déjà /api/health, /api/sessions, /api/coaching/{session_id}, JWT...).
Ne remplace pas ton fichier — copie juste les parties `lifespan` et
`app.include_router(...)`.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.routers.analysis_router import MAX_UPLOAD_BYTES, router as analysis_router
from app.routers.auth_router import router as auth_router
from app.services.database import connect, initialize_database

ENVIRONMENT = os.environ.get("CONSULTPRO_ENVIRONMENT", "development").lower()
_CORS_ORIGINS_SETTING = os.environ.get("CONSULTPRO_CORS_ORIGINS")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in (
        _CORS_ORIGINS_SETTING
        or "http://localhost:5173,https://consult-pro-praticien.vercel.app"
    ).split(",")
    if origin.strip()
]
_ALLOWED_HOSTS_SETTING = os.environ.get("CONSULTPRO_ALLOWED_HOSTS")
ALLOWED_HOSTS = [
    host.strip()
    for host in (_ALLOWED_HOSTS_SETTING or "localhost,127.0.0.1,testserver").split(",")
    if host.strip()
]
ENABLE_HSTS = os.environ.get("CONSULTPRO_ENABLE_HSTS", "false").lower() == "true"

if ENVIRONMENT == "production":
    if not _CORS_ORIGINS_SETTING or not _ALLOWED_HOSTS_SETTING:
        raise RuntimeError("CONSULTPRO_CORS_ORIGINS et CONSULTPRO_ALLOWED_HOSTS doivent être définis en production.")
    if (
        not ALLOWED_ORIGINS
        or not ALLOWED_HOSTS
        or any("*" in origin for origin in ALLOWED_ORIGINS)
        or any("*" in host for host in ALLOWED_HOSTS)
        or any(not origin.startswith("https://") for origin in ALLOWED_ORIGINS)
    ):
        raise RuntimeError("Configure uniquement des origines HTTPS et hôtes précis en production.")
    if not ENABLE_HSTS:
        raise RuntimeError("CONSULTPRO_ENABLE_HSTS=true est requis en production derrière HTTPS.")

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    yield


app = FastAPI(title="ConsultPro API", lifespan=lifespan)

app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def security_headers_and_upload_limit(request, call_next):
    response = None
    if request.method == "POST" and request.url.path == "/api/analysis/upload":
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                body_size = int(content_length)
            except ValueError:
                response = JSONResponse(
                    status_code=400, content={"detail": "Content-Length invalide."}
                )
            else:
                max_request_size = MAX_UPLOAD_BYTES + 1024 * 1024
                if body_size < 0:
                    response = JSONResponse(
                        status_code=400, content={"detail": "Content-Length invalide."}
                    )
                elif body_size > max_request_size:
                    response = JSONResponse(
                        status_code=413, content={"detail": "Fichier trop volumineux."}
                    )

    if response is None:
        response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    if ENABLE_HSTS:
        response.headers["Strict-Transport-Security"] = "max-age=31536000"
    return response


# ── Routers existants (à garder tels quels dans ton vrai main.py) ────────
# app.include_router(sessions_router)
# app.include_router(coaching_router)
# app.include_router(auth_router)

# ── Nouveau router d'analyse audio ─────────────────────────────────────
app.include_router(analysis_router)
app.include_router(auth_router)


@app.get("/api/health")
async def health():
    try:
        with connect() as connection:
            connection.execute("SELECT 1")
    except Exception:
        return JSONResponse(status_code=503, content={"status": "unavailable"})
    return {"status": "ok"}
