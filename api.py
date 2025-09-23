from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from livekit.api.access_token import AccessToken, VideoGrants
from pathlib import Path
import os

API_KEY = os.getenv("LIVEKIT_API_KEY")
API_SECRET = os.getenv("LIVEKIT_API_SECRET")

app = FastAPI()

# Allow same-origin by default; adjust if you host separately
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_PATH = Path(__file__).resolve().parent / "frontend.html"

@app.get("/")
async def serve_frontend():
    if not FRONTEND_PATH.exists():
        raise HTTPException(status_code=404, detail="frontend.html not found")
    return FileResponse(str(FRONTEND_PATH), media_type="text/html")

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/favicon.ico")
async def favicon():
    # Return a 1x1 transparent PNG to satisfy browsers
    import base64
    png_base64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGMAAQAABQAB"
        "J6k6WQAAAABJRU5ErkJggg=="
    )
    data = base64.b64decode(png_base64)
    return Response(content=data, media_type="image/png")

@app.get("/token")
async def get_token(room: str, identity: str):
    if not API_KEY or not API_SECRET:
        raise HTTPException(status_code=500, detail="LIVEKIT credentials not configured")
    token = (
        AccessToken(API_KEY, API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )
    return {"token": token}

@app.post("/api/token")
async def post_token(payload: dict):
    room = payload.get("room")
    identity = payload.get("name") or payload.get("identity")
    if not room or not identity:
        raise HTTPException(status_code=400, detail="Missing 'room' or 'name'")
    if not API_KEY or not API_SECRET:
        raise HTTPException(status_code=500, detail="LIVEKIT credentials not configured")

    token = (
        AccessToken(API_KEY, API_SECRET)
        .with_identity(identity)
        .with_name(identity)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room,
                can_publish=True,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )
    return JSONResponse({"token": token})
