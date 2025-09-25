# Face Match (LiveKit + FastAPI + face_recognition)

Real‑time face recognition subscriber that connects to a LiveKit room, matches incoming video frames against an uploaded reference face, and streams match events to a FastAPI backend + WebSocket frontend UI.

## Components

- `main.py` – Async LiveKit subscriber & face recognition pipeline (HOG dlib/face_recognition) saving snapshots of matches.
- `api.py` – FastAPI server: serves `frontend.html`, issues tokens, accepts reference image uploads, broadcasts match events over WebSockets, and serves static captures.
- `frontend.html` – Join form + dynamic “Live Matches” grid (auto creates/removes per‑participant match cards with live video, confidence %, timestamp, bounding box overlay).
- `captured_faces/` – Saved snapshot frames when a face match is detected.
- `reference/current.jpg` – Current active reference face (overwritten by uploads).

## Match Event Payload
Each `/api/match` POST & WebSocket broadcast contains:

```
{
	"participant_id": "PA_xxx",
	"participant_name": "Alice42",
	"confidence": 0.63,                 # 0..1
	"filename": "/abs/path/captured_faces/match_Alice42_20250925_140708.jpg",
	"snapshot": "match_Alice42_20250925_140708.jpg",  # basename
	"timestamp": "2025-09-25T14:07:08.123456",
	"bbox": [top, right, bottom, left],  # face_recognition format
	"frame_size": [width, height]        # of original frame
}
```

## New Logging & Control Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MATCH_COOLDOWN_SECONDS` | `3` | Minimum seconds between accepted matches per participant (debounce spam). |
| `OUTPUT_DIR` | `captured_faces` | Directory for saved match frames (created if absent). |
| `REFERENCE_IMAGE` | fallback to `reference/current.jpg` | Optional explicit reference path if not using upload workflow. |
| `API_BASE_URL` | `http://127.0.0.1:8000` | Base URL for posting match events. |
| `SUMMARY_LOG_INTERVAL_SECONDS` | `10` | Interval for periodic status summary logs. |
| `STATUS_LOG_ONLY_ON_CHANGE` | `false` | If `true`, suppress periodic summary unless totals changed. |
| `MAX_PARALLEL_FRAMES` | `4` | Max in-flight frame analysis tasks (drop excess frames to stabilize memory). |
| `FRAME_DOWNSCALE_WIDTH` | `640` | Downscale incoming frames to this width before detection (preserves aspect). |
| `INITIAL_SYNC_FRAMES` | `5` | Number of first frames per track processed synchronously for fast initial detection. |
| `PROCESS_EVERY_N_FRAME` | `2` | After warmup, process only every Nth frame (1 = process all). |
| `DETECTION_LEGACY_MODE` | `false` | If true, bypass downscaling/skipping & process every frame (original behavior). |
| `DETECTION_DEBUG` | `false` | Verbose per-frame logs: processed, skipped, dropped, faces count. |
| `SELECTIVE_SUBSCRIBE` | `false` | Enable manual subscribe/unsubscribe to remote video tracks based on recent match activity. |
| `SS_INACTIVE_SECONDS` | `15` | If no match activity for a subscribed track within this many seconds, unsubscribe it. |
| `SS_RETRY_SECONDS` | `10` | After an unsubscribe, wait this long before attempting to resubscribe a track. |
| `SS_POLL_INTERVAL_SECONDS` | `5` | Frequency of the subscription manager loop checks. |
| `SS_PROBE_DURATION_SECONDS` | `5` | While probing, stay subscribed up to this long if no match yet before giving up. |
| `SS_ACTIVE_GRACE_SECONDS` | `30` | After last match, remain subscribed this many seconds (grace period) before going inactive. |

## Recent Improvements

1. Dynamic multi‑participant match cards (frontend) replace single modal for scalable monitoring.
2. Added `snapshot` basename in match payload for simpler client thumbnail construction.
3. Guard prevents processing stale matches after participant disconnect.
4. Conditional/delta logging to reduce console spam and tunable summary interval.
5. Auto cleanup of stale match cards after 5s with no new events.
6. Frame downscaling + skipping strategy to reduce latency and CPU load.
7. Early synchronous processing of first frames for faster initial matches.
8. Selective subscription option: automatically unsubscribe inactive participant video tracks to reduce bandwidth/CPU and periodically probe them again.
9. Probe state machine (inactive → probing → active): probing unsubscribes quickly if no match; active tracks keep streaming as long as matches occur within grace window.

## Running

Typical (inside dev container / local):

```
./run.sh
```

This starts FastAPI (serving the frontend) then launches the face recognition loop. Navigate to `http://localhost:8000/` and join the LiveKit room (configure URL & API credentials as needed).

## Reference Image Workflow

1. Upload via the form before joining – it POSTs to `/api/reference`.
2. Backend writes to `reference/current.jpg`.
3. Recognizer auto hot‑reloads when file mtime changes (throttled to once per second).

## Notes & Future Ideas

- Add optional GPU/CUDA acceleration only if environment supports it (current logs show fallback on missing drivers).
- The messages about CUDA / VAAPI failing to initialize are benign on CPU-only hosts; the system falls back to pure CPU (HOG) processing.
- Persist match history to a database (SQLite/Postgres) instead of in‑memory.
- Provide REST endpoint to query historical matches or download snapshots.
- Add threshold slider in UI to filter low‑confidence matches.
- Support multiple simultaneous reference faces (labeling & selection UI).
- Adaptive dynamic frame rate: adjust PROCESS_EVERY_N_FRAME based on moving average detection time.
- Optional JSON structured logging mode.
 - Integrate subscription metrics endpoint exposing active/inactive track counts.
 - Adaptive probe scheduling (shorter probes for long-idle tracks, exponential backoff).

## License

Internal prototype / unspecified (add a license if distributing externally).