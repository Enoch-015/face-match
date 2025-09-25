from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from livekit.api.access_token import AccessToken, VideoGrants
from pathlib import Path
import os
import asyncio

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
REFERENCE_DIR = Path(__file__).resolve().parent / "reference"
REFERENCE_DIR.mkdir(exist_ok=True)
REFERENCE_IMAGE_PATH = REFERENCE_DIR / "current.jpg"
CAPTURED_DIR = Path(__file__).resolve().parent / "captured_faces"
CAPTURED_DIR.mkdir(exist_ok=True)

# Store active WebSocket connections for match events
match_ws_connections: set[WebSocket] = set()

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

@app.post("/api/reference")
async def upload_reference(file: UploadFile = File(...)):
    try:
        content = await file.read()
        REFERENCE_IMAGE_PATH.write_bytes(content)
        return {"status": "ok", "message": "Reference image updated", "path": str(REFERENCE_IMAGE_PATH)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save reference image: {e}")

@app.post("/api/match")
async def post_match(event: dict):
    # Broadcast the match event to all connected WebSocket clients
    stale: list[WebSocket] = []
    for ws in list(match_ws_connections):
        try:
            await ws.send_json({"type": "match", **event})
        except Exception:
            stale.append(ws)
    for ws in stale:
        match_ws_connections.discard(ws)
    return {"ok": True, "delivered": len(match_ws_connections) - len(stale)}

@app.websocket("/ws/matches")
async def ws_matches(ws: WebSocket):
    await ws.accept()
    match_ws_connections.add(ws)
    try:
        while True:
            # Keep the connection alive; we don't expect messages from clients
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        match_ws_connections.discard(ws)

# Serve captured images and reference via static routes
app.mount("/captures", StaticFiles(directory=str(CAPTURED_DIR)), name="captures")
app.mount("/reference", StaticFiles(directory=str(REFERENCE_DIR)), name="reference")

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
