# Face Match (LiveKit + FastAPI + face_recognition)

Deterministic, low‑duty‑cycle face recognition client that joins a LiveKit room as a subscriber and periodically inspects small batches of video frames for a single reference face. Instead of continuously running detection, it executes a simple batch / hold / wait loop for each remote video track to keep CPU and memory usage predictable.

> IMPORTANT: The process may intentionally stop (graceful exit) when RAM usage crosses configured thresholds, or it can be killed by the OS (OOM) if those thresholds are set too high. See “Memory Management & Potential Stops”.

## 1. Architecture Overview

```
┌──────────────┐     token      ┌─────────────┐
│  Browser UI  │ ─────────────▶ │  FastAPI    │
│ (frontend)   │  websocket ◀── │  api.py     │ ◀─────┐ reference upload
└─────┬────────┘    events      └─────┬───────┘       │
			│                               │               │
			│                               │match/check    │
			│                               ▼               │
			│                        WebSocket broadcast    │
			│                                               │
			│   subscribed video tracks                     │
			▼                                               │
┌──────────────┐  batches  ┌──────────────────────┐    │
│ LiveKit SFU  │ ◀──────── │ main.py (recognizer) │ ◀──┘
└──────────────┘            └──────────────────────┘
```

For each remote participant video track:
1. Subscribe & process up to `BATCH_MAX_FRAMES` frames (Probe Batch).
2. If a match occurs: emit Match event(s), then after batch emit a Check event (match_found=true), enter HOLD (`HOLD_SECONDS`) without processing.
3. After hold, run the next batch. Continue the loop while matches keep occurring.
4. If no match in a batch: emit a Check event (match_found=false), unsubscribe the track, sleep `WAIT_SECONDS`, then probe again.

No adaptive heuristics, no presence heartbeat, no frame skipping complexity—just a transparent deterministic cycle per track.

## 2. Components

| File | Purpose |
|------|---------|
| `main.py` | Connects to LiveKit, orchestrates per‑track batch cycles, performs face detection (HOG via `face_recognition`), emits events. |
| `api.py` | FastAPI app: serves frontend, issues LiveKit access tokens, accepts reference image uploads, relays Match & Check events over a WebSocket, serves saved snapshots. |
| `frontend.html` | Minimal UI: upload reference, join room, display match snapshots & last check status per participant. |
| `captured_faces/` | Folder where match snapshots are stored. |
| `reference/current.jpg` | Hot‑reloadable reference face (overwritten by uploads). |
| `run.sh` | Convenience launcher: starts API then recognition loop. |

## 3. Event Payloads

Two WebSocket‑broadcasted event categories:

1. Match (POST to `/api/match`, broadcast immediately on each accepted match – debounced by `MATCH_COOLDOWN_SECONDS`).
2. Check (POST to `/api/check` after every batch, even if no match found).

### 3.1 Match Event
```json
{
	"participant_id": "PA_xxx",
	"participant_name": "Alice42",
	"confidence": 0.63,
	"filename": "/abs/path/captured_faces/match_Alice42_20250925_140708.jpg",
	"snapshot": "match_Alice42_20250925_140708.jpg",
	"timestamp": "2025-09-25T14:07:08.123456+00:00",
	"bbox": [top, right, bottom, left],
	"frame_size": [width, height]
}
```

### 3.2 Check Event
```json
{
	"type": "check",
	"participant_id": "PA_xxx",
	"participant_name": "Alice42",
	"match_found": true,
	"timestamp": "2025-09-25T14:07:08.987654+00:00"
}
```

Frontend logic:
* Each Match adds / updates a participant card with the snapshot & confidence.
* Each Check updates a “last check” timestamp and shows whether the last batch found a match.

## 4. Batch Cycle (Deterministic Loop)

| Phase | Action | Outcome |
|-------|--------|---------|
| Probe Batch | Subscribe & process up to N frames | Stop early on first match |
| Batch End | Emit Check (match_found flag) | Decide next state |
| If Match | Hold (no processing) | Conserve CPU while subject presumed present |
| If No Match | Unsubscribe & Wait | Free bandwidth & CPU until next probe |

Reasons for this design:
* Predictable CPU cost (N frames per active batch).
* Reduced overhead when subject consistently present (HOLD avoids redundant detections).
* Transparent: no hidden timers or adaptive heuristics.

## 5. Environment Variables

Mandatory for connectivity (if your LiveKit install differs):
| Variable | Default | Purpose |
|----------|---------|---------|
| `LIVEKIT_URL` | `ws://localhost:7880` | LiveKit server WebSocket URL. |
| `LIVEKIT_API_KEY` | `demo-key` | API key for token generation. |
| `LIVEKIT_API_SECRET` | `demo-secret` | API secret for token generation. |
| `ROOM_NAME` | `face-recognition-room` | Room to join as subscriber. |

Core cycle & matching controls:
| Variable | Default | Purpose |
|----------|---------|---------|
| `BATCH_MAX_FRAMES` | `5` | Frames per batch (upper bound). |
| `HOLD_SECONDS` | `10` | Idle duration after a successful batch (track stays subscribed). |
| `WAIT_SECONDS` | `10` | Sleep after a no‑match batch (track unsubscribed). |
| `MATCH_COOLDOWN_SECONDS` | `3` | Minimum seconds between snapshots per participant. |
| `FRAME_EXECUTOR_WORKERS` | `2` | Threads for CPU face encoding (>=1). |
| `FRAME_DOWNSCALE_WIDTH` | `640` | Resize width before detection (aspect preserved). |

Filesystem & API:
| Variable | Default | Purpose |
|----------|---------|---------|
| `OUTPUT_DIR` | `captured_faces` | Snapshot save directory. |
| `REFERENCE_IMAGE` | `reference/current.jpg` | Override static reference path (hot‑reload watched). |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Where `main.py` posts `/api/match` & `/api/check`. |

Memory watchdog (graceful stop thresholds):
| Variable | Default | Purpose |
|----------|---------|---------|
| `MEMORY_WARN_RATIO` | `0.80` | Log a warning at this fraction of total RAM. |
| `MEMORY_EXIT_RATIO` | `0.92` | Trigger graceful shutdown if reached. |

Legacy (currently non‑impacting; scheduled for removal):
| Variable | Current Default | Notes |
|----------|-----------------|-------|
| `INITIAL_SYNC_FRAMES` | `5` | No effect in new deterministic loop. |
| `PROCESS_EVERY_N_FRAME` | `2` | Not used in batch model. |

## 6. Setup & Running

### 6.1 Prerequisites
* Python 3.12+ (face_recognition & dlib compiled; dev container handles build toolchain).
* Running LiveKit server (local or remote) with API key/secret.
* (Optional) `psutil` auto‑installed via `requirements.txt` for memory monitoring. If absent, memory watchdog still runs with limited info.

### 6.2 Install
```
./install_fix.sh
```

### 6.3 Launch (Convenience Script)
```
./run.sh
```
Script steps:
1. Start FastAPI (`api.py`) on `:8000`.
2. Launch `main.py` recognition loop.
3. Open `http://localhost:8000` in a browser, upload a reference image, join the room.

### 6.4 Manual (Separate Terminals)
```
uvicorn api:app --host 0.0.0.0 --port 8000
python main.py
```

## 7. Reference Image Hot Reload
Uploading a new image via the UI overwrites `reference/current.jpg`. The recognizer polls modification time (throttled to once per second) and reloads encodings atomically—no restart needed.

## 8. Memory Management & Potential Stops

The process tracks overall system memory usage (via `psutil.virtual_memory()` when available):
* When usage ≥ `MEMORY_WARN_RATIO`, a warning log is emitted.
* When usage ≥ `MEMORY_EXIT_RATIO`, the main loop exits gracefully (disconnects room, cancels tasks).

If limits are set too high (or `psutil` is missing) the OS may still kill the process (e.g. OOM killer). Symptoms:
* Graceful path: logs like `Memory usage 93.0% >= exit ratio ...` then `Graceful shutdown flag detected`.
* OOM kill: sudden termination with `Killed` / exit code 137 and no final stack logs.

Mitigation tips:
* Lower `BATCH_MAX_FRAMES`.
* Increase `HOLD_SECONDS` / `WAIT_SECONDS` to reduce detection frequency.
* Adjust `FRAME_DOWNSCALE_WIDTH` downward (e.g. 480) to shrink per‑frame memory.
* Cap snapshot retention (future enhancement – not yet implemented).

## 9. Graceful Shutdown & Signals
Signals `SIGINT` / `SIGTERM` trigger:
1. Thread stack dump (debug visibility).
2. Global shutdown flag set.
3. Main loop notices flag and disconnects from LiveKit cleanly.

You will see log lines like:
```
Received SIGTERM; initiating graceful shutdown and dumping stacks...
Graceful shutdown flag detected; breaking main loop
```

## 10. Troubleshooting
| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| No snapshots saved | Reference face not detected in uploaded image | Ensure exactly one clear frontal face in reference. |
| Repeated unsubscribes | No match in batches | Verify lighting, camera quality, tolerance (code default 0.5). |
| Immediate exit on start | Missing `reference/current.jpg` | Upload reference before running or set `REFERENCE_IMAGE`. |
| High CPU | Very frequent matches (short cycles) | Increase `HOLD_SECONDS`. |
| Process killed (no graceful logs) | OOM killer | Reduce frame size / batch size / parallel workers. |
| No memory warnings | `psutil` absent | Install `psutil` or ignore if acceptable. |

## 11. Future Enhancements (Planned / Ideas)
* Snapshot retention caps & optional disable flag.
* Multi‑face gallery support.
* Structured JSON logging & metrics endpoint.
* Per‑participant statistics endpoint.
* Remove legacy env vars once confirmed unnecessary.

## 12. License
Internal prototype / unspecified (add a proper license before external distribution).

---
Questions / improvements welcome. Adjust environment variables to tune detection duty cycle and resource usage.