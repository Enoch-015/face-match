# main.py - LiveKit Face Recognition System
import asyncio
import cv2
import face_recognition
import numpy as np
import sys
"""Simplified batch-cycle LiveKit + face_recognition integration.

Behavior (single default model):
  For every remote video track:
    - Probe Batch: subscribe, process up to BATCH_MAX_FRAMES (default 5) frames looking for a match.
        * If a match is found: stay subscribed and enter HOLD_SECONDS (default 10s) with NO processing.
        * After hold, run a Recheck Batch (same size). If still matched (at least one match), repeat hold.
        * If recheck finds no match: unsubscribe and wait WAIT_SECONDS (default 10s) before next probe.
    - If probe batch finds no match: unsubscribe and wait WAIT_SECONDS.

No presence mode, no adaptive skipping, no frame dropping logic, no probe attempt counters.
Only these environment variables affect behavior:
  BATCH_MAX_FRAMES (int, default 5)
  HOLD_SECONDS (float, default 10)
  WAIT_SECONDS (float, default 10)
  MATCH_COOLDOWN_SECONDS (int, default 3) - suppress duplicate match events close in time.
"""

import asyncio
import os
import cv2
import face_recognition
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime, timezone
import aiohttp
import threading
from concurrent.futures import ThreadPoolExecutor
from livekit import rtc
from livekit.api.access_token import AccessToken, VideoGrants
import signal
import traceback
import os as _os
try:  # optional dependency
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None  # fallback

try:
    from livekit.rtc import VideoBufferType
except ImportError:
    try:
        from livekit.rtc.video_frame import VideoBufferType
    except ImportError:  # pragma: no cover
        from livekit.proto import video_pb2 as proto_video
        VideoBufferType = proto_video.VideoBufferType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LiveKitFaceRecognizer:
    def __init__(self, reference_image_path: str, tolerance: float = 0.5):
        self.room: Optional[rtc.Room] = None
        self.reference_encodings: Dict[str, np.ndarray] = {}
        self.participants: Dict[str, rtc.RemoteParticipant] = {}
        self.tolerance = tolerance
        max_workers = self._parse_int_env("FRAME_EXECUTOR_WORKERS", 2)
        if max_workers < 1:
            max_workers = 1
        self.frame_executor = ThreadPoolExecutor(max_workers=max_workers)
        self.match_history: Dict[str, List] = {}
        self.last_match_time: Dict[str, datetime] = {}
        self.ref_lock = threading.Lock()

        # Batch-cycle config
        self.batch_max_frames = self._parse_int_env("BATCH_MAX_FRAMES", 5)
        self.hold_seconds = self._parse_float_env("HOLD_SECONDS", 10.0)
        self.wait_seconds = self._parse_float_env("WAIT_SECONDS", 10.0)
        self.match_cooldown_seconds = self._parse_int_env("MATCH_COOLDOWN_SECONDS", 3)

        # Track + batch bookkeeping
        self.cycle_tasks: Dict[str, asyncio.Task] = {}
        self.track_tasks: Dict[str, asyncio.Task] = {}
        self.track_owner: Dict[str, str] = {}  # track_sid -> participant.sid
        self.batch_state: Dict[str, Dict] = {}  # track_sid -> {frames_remaining, match_found, done Event}

        # Reference + output
        self.load_reference_face(reference_image_path)
        self.output_dir = os.path.abspath(os.getenv("OUTPUT_DIR", "captured_faces"))
        os.makedirs(self.output_dir, exist_ok=True)
        self.reference_image_path = reference_image_path
        self._ref_mtime = os.path.getmtime(reference_image_path) if os.path.exists(reference_image_path) else 0
        self._last_reload_check = 0.0
        # Performance tuning parameters
        self.downscale_width = self._parse_int_env("FRAME_DOWNSCALE_WIDTH", 640)
        self.process_every_n = self._parse_int_env("PROCESS_EVERY_N_FRAME", 2)  # 2 = every other frame
        if self.process_every_n < 1:
            self.process_every_n = 1
        self.initial_sync_frames = self._parse_int_env("INITIAL_SYNC_FRAMES", 5)
        self.track_frame_counters: Dict[str, int] = {}
        self.track_sync_remaining: Dict[str, int] = {}

    # ---------------------- Utilities ----------------------
    def try_reload_reference(self):
        try:
            loop = asyncio.get_event_loop()
            now = loop.time() if loop.is_running() else 0.0
            if now - self._last_reload_check < 1.0:
                return
            self._last_reload_check = now
            if not os.path.exists(self.reference_image_path):
                return
            mtime = os.path.getmtime(self.reference_image_path)
            if mtime > self._ref_mtime:
                if self.load_reference_face(self.reference_image_path, is_reload=True):
                    self._ref_mtime = mtime
                    logger.info("🔄 Reference image reloaded")
        except Exception:
            pass

    def load_reference_face(self, image_path: str, is_reload: bool = False) -> bool:
        try:
            reference_image = face_recognition.load_image_file(image_path)
            encs = face_recognition.face_encodings(reference_image)
            if not encs:
                msg = "No face found in reference image"
                if is_reload:
                    logger.warning(msg + " (reload ignored)")
                    return False
                raise RuntimeError(msg)
            with self.ref_lock:
                self.reference_encodings['reference'] = encs[0]
            if not is_reload:
                logger.info("✅ Reference face loaded")
            return True
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed loading reference: {e}")
            if is_reload:
                return False
            raise

    # ---------------------- LiveKit Wiring ----------------------
    async def connect_to_room(self, url: str, token: str) -> bool:
        try:
            self.room = rtc.Room()
            self.room.on("participant_connected", self.on_participant_connected)
            self.room.on("participant_disconnected", self.on_participant_disconnected)
            self.room.on("track_published", self.on_track_published)
            self.room.on("track_subscribed", self.on_track_subscribed)
            self.room.on("track_unsubscribed", self.on_track_unsubscribed)
            self.room.on("disconnected", self.on_disconnected)
            await self.room.connect(url, token, rtc.RoomOptions(auto_subscribe=False))
            logger.info("✅ Connected to LiveKit (auto_subscribe disabled)")
            return True
        except Exception as e:  # pragma: no cover
            logger.error(f"Failed to connect: {e}")
            return False

    # Event callbacks -------------------------------------------------
    def on_track_published(self, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        if publication.kind != rtc.TrackKind.KIND_VIDEO:
            return
        sid = publication.sid
        if sid in self.cycle_tasks:
            return
        logger.info(f"[Cycle] Track published sid={sid} participant={participant.identity}")
        self.cycle_tasks[sid] = asyncio.create_task(self.track_cycle(publication, participant))

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        self.participants[participant.sid] = participant
        logger.info(f"Participant connected: {participant.identity}")

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        logger.info(f"Participant disconnected: {participant.identity}")
        self.participants.pop(participant.sid, None)
        # Cancel related cycle tasks
        for sid, owner in list(self.track_owner.items()):
            if owner == participant.sid:
                t = self.track_tasks.pop(sid, None)
                if t and not t.done():
                    t.cancel()
        for sid, task in list(self.cycle_tasks.items()):
            if sid not in self.track_owner:  # track removed
                if task and not task.done():
                    task.cancel()
                self.cycle_tasks.pop(sid, None)

    def on_track_subscribed(self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return
        sid = publication.sid
        logger.info(f"[Cycle] Subscribed sid={sid} participant={participant.identity}")
        self.track_owner[sid] = participant.sid
        if sid not in self.track_tasks:
            self.track_tasks[sid] = asyncio.create_task(self.process_video_track(track, participant, sid))

    def on_track_unsubscribed(self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        if track.kind != rtc.TrackKind.KIND_VIDEO:
            return
        sid = publication.sid
        logger.info(f"[Cycle] Unsubscribed sid={sid}")
        task = self.track_tasks.pop(sid, None)
        self.track_owner.pop(sid, None)
        if task and not task.done():
            task.cancel()

    def on_disconnected(self, reason: str):
        logger.info(f"Disconnected: {reason}")
        for d in (self.track_tasks, self.cycle_tasks):
            for sid, task in list(d.items()):
                if task and not task.done():
                    task.cancel()
            d.clear()

    # ---------------------- Frame Processing ----------------------

    def livekit_frame_to_opencv(self, video_frame: rtc.VideoFrame) -> Optional[np.ndarray]:
        try:
            target_format = getattr(VideoBufferType, 'RGB24', 1)
            if video_frame.type != target_format:
                video_frame = video_frame.convert(target_format)
            data = np.frombuffer(video_frame.data, dtype=np.uint8)
            frame = data.reshape((video_frame.height, video_frame.width, 3))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame.copy()
        except Exception:
            return None

    async def process_video_track(self, track: rtc.RemoteVideoTrack, participant: rtc.RemoteParticipant, track_sid: str):
        logger.info(f"[Cycle] Frame loop started sid={track_sid} participant={participant.identity}")
        try:
            async for event in rtc.VideoStream(track):
                if not event.frame:
                    continue
                self.try_reload_reference()
                bs = self.batch_state.get(track_sid)
                if not bs or bs.get('frames_remaining', 0) <= 0:
                    continue  # idle (not in a batch)
                frame = self.livekit_frame_to_opencv(event.frame)
                if frame is None:
                    continue
                # Simple downscale to width 640 if larger
                if frame.shape[1] > 640:
                    ratio = 640 / frame.shape[1]
                    new_h = int(frame.shape[0] * ratio)
                    proc_frame = cv2.resize(frame, (640, new_h), interpolation=cv2.INTER_LINEAR)
                else:
                    proc_frame = frame
                loop = asyncio.get_event_loop()
                try:
                    result = await loop.run_in_executor(self.frame_executor, self.detect_and_match_faces, proc_frame, participant)
                except Exception:
                    result = None
                bs['frames_remaining'] -= 1
                if result:
                    # Map boxes if scaled
                    if proc_frame.shape[:2] != frame.shape[:2]:
                        sx = frame.shape[1] / proc_frame.shape[1]
                        sy = frame.shape[0] / proc_frame.shape[0]
                        for m in result:
                            t, r, b, l = m['location']
                            m['location'] = (int(t * sy), int(r * sx), int(b * sy), int(l * sx))
                    await self.handle_face_match(result, participant, frame)
                    bs['match_found'] = True
                    bs['frames_remaining'] = 0
                if bs['frames_remaining'] <= 0 and not bs['done'].is_set():
                    bs['done'].set()
        except asyncio.CancelledError:
            logger.info(f"[Cycle] Frame loop cancelled sid={track_sid}")
        except Exception as e:  # pragma: no cover
            logger.debug(f"[Cycle] Frame loop error sid={track_sid}: {e}")

    # (Removed legacy frame wrapper methods)

    def detect_and_match_faces(self, frame: np.ndarray, participant: rtc.RemoteParticipant) -> Optional[List[Dict]]:
        try:
            rgb = frame[:, :, ::-1].copy()
            locs = face_recognition.face_locations(rgb, model="hog", number_of_times_to_upsample=0)
            if not locs:
                return None
            encs = face_recognition.face_encodings(rgb, locs)
            with self.ref_lock:
                ref = self.reference_encodings.get('reference')
            if ref is None:
                return None
            out = []
            for (top, right, bottom, left), enc in zip(locs, encs):
                match = face_recognition.compare_faces([ref], enc, tolerance=self.tolerance)[0]
                if not match:
                    continue
                dist = face_recognition.face_distance([ref], enc)[0]
                out.append({
                    'location': (top, right, bottom, left),
                    'confidence': 1 - dist,
                    'participant_id': participant.sid,
                    'participant_name': participant.identity,
                    'timestamp': datetime.now(timezone.utc),
                })
            return out or None
        except Exception:
            return None

    async def handle_face_match(self, matches: List[Dict], participant: rtc.RemoteParticipant, frame: np.ndarray):
        if participant.sid not in self.participants:
            return
        for match in matches:
            confidence = match['confidence']
            location = match['location']
            timestamp = match['timestamp']
            last_ts = self.last_match_time.get(participant.sid)
            if last_ts and (timestamp - last_ts).total_seconds() < self.match_cooldown_seconds:
                continue
            self.last_match_time[participant.sid] = timestamp
            logger.info(f"🎯 MATCH participant={participant.identity} confidence={confidence:.2f}")
            top, right, bottom, left = location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"MATCH {confidence:.2f}", (left, max(0, top-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            filename = f"{self.output_dir}/match_{participant.identity}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            self.match_history.setdefault(participant.sid, []).append({
                'timestamp': timestamp.isoformat(),
                'confidence': confidence,
                'filename': filename,
                'participant_name': participant.identity
            })
            # Emit match event
            try:
                async with aiohttp.ClientSession() as session:
                    api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
                    payload = {
                        'participant_id': participant.sid,
                        'participant_name': participant.identity,
                        'confidence': float(confidence),
                        'filename': filename,
                        'snapshot': os.path.basename(filename),
                        'timestamp': timestamp.isoformat(),
                        'bbox': [int(location[0]), int(location[1]), int(location[2]), int(location[3])],
                        'frame_size': [int(frame.shape[1]), int(frame.shape[0])]
                    }
                    await session.post(f"{api_url}/api/match", json=payload, timeout=3)
            except Exception:
                pass

    async def emit_check_event(self, participant: rtc.RemoteParticipant, match_found: bool):
        try:
            async with aiohttp.ClientSession() as session:
                api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
                payload = {
                    'type': 'check',
                    'participant_id': participant.sid,
                    'participant_name': participant.identity,
                    'match_found': match_found,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                await session.post(f"{api_url}/api/check", json=payload, timeout=3)
        except Exception:
            pass

    # ---------------------- Batch Cycle Logic ----------------------
    async def track_cycle(self, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        sid = publication.sid
        try:
            # Revised simpler logic: each loop performs one batch. If a match is found,
            # hold before running the next batch. If no match, unsubscribe and wait.
            while True:
                match_found = await self.run_batch(publication, participant)
                if match_found:
                    logger.info(f"[Cycle] Hold {self.hold_seconds}s sid={sid}")
                    await asyncio.sleep(self.hold_seconds)
                else:
                    try:
                        publication.set_subscribed(False)
                    except Exception:
                        pass
                    logger.info(f"[Cycle] No match -> unsub sid={sid}; wait {self.wait_seconds}s")
                    await asyncio.sleep(self.wait_seconds)
        except asyncio.CancelledError:
            logger.info(f"[Cycle] Cancel track cycle sid={sid}")
        except Exception as e:  # pragma: no cover
            logger.error(f"[Cycle] Error sid={sid}: {e}")

    async def run_batch(self, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant) -> bool:
        sid = publication.sid
        try:
            publication.set_subscribed(True)
        except Exception:
            pass
        done = asyncio.Event()
        self.batch_state[sid] = {
            'frames_remaining': self.batch_max_frames,
            'match_found': False,
            'done': done
        }
        logger.info(f"[Cycle] Batch start sid={sid} frames={self.batch_max_frames}")
        try:
            await asyncio.wait_for(done.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            logger.info(f"[Cycle] Batch timeout sid={sid}")
        state = self.batch_state.get(sid, {})
        match_found = bool(state.get('match_found'))
        await self.emit_check_event(participant, match_found)
        self.batch_state.pop(sid, None)
        logger.info(f"[Cycle] Batch done sid={sid} match={match_found}")
        return match_found

    def get_match_summary(self) -> Dict:
        return {
            'total_participants_matched': len(self.match_history),
            'total_matches': sum(len(v) for v in self.match_history.values()),
        }

    async def disconnect(self):
        if self.room:
            try:
                await self.room.disconnect()
            except Exception:
                pass
        for d in (self.track_tasks, self.cycle_tasks):
            for sid, task in list(d.items()):
                if task and not task.done():
                    task.cancel()
            d.clear()

    def _parse_float_env(self, key: str, default: float) -> float:
        raw = os.getenv(key, str(default)).split('#', 1)[0].strip()
        try:
            return float(raw)
        except ValueError:
            return default

    def _parse_int_env(self, key: str, default: int) -> int:
        raw = os.getenv(key, str(default)).split('#', 1)[0].strip()
        try:
            return int(raw)
        except ValueError:
            return default


# ---------------------- Diagnostics / Signal Handling ----------------------
def _dump_all_thread_stacks():
    try:
        import threading as _th
        frames = sys._current_frames()
        for thread in _th.enumerate():
            frame = frames.get(thread.ident)
            if not frame:
                continue
            stack = ''.join(traceback.format_stack(frame))
            logger.warning(f"\n--- Stack dump for thread {thread.name} (id={thread.ident}) ---\n{stack}")
    except Exception:  # pragma: no cover
        logger.exception("Failed dumping thread stacks")


def _signal_handler(signum, frame):  # pragma: no cover
    signame = {signal.SIGINT: 'SIGINT', signal.SIGTERM: 'SIGTERM'}.get(signum, str(signum))
    logger.warning(f"Received {signame}; initiating graceful shutdown and dumping stacks...")
    _dump_all_thread_stacks()
    # Mark global flag; main loop checks this to exit cleanly.
    global LiveKitFaceRecognizer_shutdown_flag
    LiveKitFaceRecognizer_shutdown_flag = True

# module-level flag inspected by main()
LiveKitFaceRecognizer_shutdown_flag = False


def install_signal_handlers():  # pragma: no cover
    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        logger.debug("Signal handlers not installed (platform limitation)")

async def generate_access_token(api_key: str, api_secret: str, room_name: str, participant_identity: str) -> str:
    return (
        AccessToken(api_key, api_secret)
        .with_identity(participant_identity)
        .with_name(participant_identity)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=False,
                can_subscribe=True,
            )
        )
        .to_jwt()
    )

# Main execution
async def main():
    LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    API_KEY = os.getenv("LIVEKIT_API_KEY", "demo-key")
    API_SECRET = os.getenv("LIVEKIT_API_SECRET", "demo-secret")
    ROOM_NAME = os.getenv("ROOM_NAME", "face-recognition-room")
    uploaded_reference = "/workspaces/face-match/reference/current.jpg"
    env_reference = os.getenv("REFERENCE_IMAGE", uploaded_reference)
    ref_path = uploaded_reference if os.path.exists(uploaded_reference) else env_reference
    if not os.path.exists(ref_path):
        logger.error(f"Reference image not found: {ref_path}")
        return
    recognizer = LiveKitFaceRecognizer(ref_path, tolerance=0.5)
    install_signal_handlers()
    token = await generate_access_token(API_KEY, API_SECRET, ROOM_NAME, "face-recognition-client")
    # Memory watchdog thresholds
    mem_warn_ratio = float(_os.getenv("MEMORY_WARN_RATIO", "0.80"))  # warn at 80% by default
    mem_exit_ratio = float(_os.getenv("MEMORY_EXIT_RATIO", "0.92"))  # attempt graceful exit at 92%
    last_mem_log = 0.0
    loop = asyncio.get_event_loop()
    try:
        if not await recognizer.connect_to_room(LIVEKIT_URL, token):
            return
        logger.info("🚀 Batch-cycle face recognition running")
        logger.info(f"Room: {ROOM_NAME}  URL: {LIVEKIT_URL}")
        while True:
            await asyncio.sleep(10)
            # Check for shutdown flag from signal handler
            global LiveKitFaceRecognizer_shutdown_flag
            if LiveKitFaceRecognizer_shutdown_flag:
                logger.warning("Graceful shutdown flag detected; breaking main loop")
                break
            # Periodic memory check
            now_m = loop.time()
            if psutil and now_m - last_mem_log >= 10:
                try:
                    vmem = psutil.virtual_memory()
                    ratio = vmem.percent / 100.0
                    if ratio >= mem_exit_ratio:
                        logger.error(f"Memory usage {vmem.percent:.1f}% >= exit ratio {mem_exit_ratio*100:.0f}%; initiating graceful shutdown")
                        break
                    elif ratio >= mem_warn_ratio:
                        logger.warning(f"High memory usage {vmem.percent:.1f}% (warn threshold {mem_warn_ratio*100:.0f}%)")
                    last_mem_log = now_m
                except Exception:
                    pass
            summary = recognizer.get_match_summary()
            if summary['total_matches'] > 0:
                logger.info(f"📊 Matches={summary['total_matches']} participants={summary['total_participants_matched']}")
    except KeyboardInterrupt:
        logger.info("Stopping (KeyboardInterrupt)...")
    finally:
        await recognizer.disconnect()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception:  # pragma: no cover
        logger.exception("Fatal unhandled exception in main entrypoint")
        # Re-raise so that non-zero exit code surfaces to caller.
        raise