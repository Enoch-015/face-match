# main.py - LiveKit Face Recognition System
import asyncio
import cv2
import face_recognition
import numpy as np
import sys
import os
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
import json
import aiohttp
import threading
import math
from concurrent.futures import ThreadPoolExecutor

# Import LiveKit components
from livekit import rtc
from livekit.api.access_token import AccessToken, VideoGrants

# Handle VideoBufferType import
try:
    from livekit.rtc import VideoBufferType
except ImportError:
    # Fallback for older versions
    try:
        from livekit.rtc.video_frame import VideoBufferType
    except ImportError:
        # Use proto directly if needed
        from livekit.proto import video_pb2 as proto_video
        VideoBufferType = proto_video.VideoBufferType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LiveKitFaceRecognizer:
    def __init__(self, reference_image_path: str, tolerance: float = 0.5):
        self.room: Optional[rtc.Room] = None
        self.reference_encodings: Dict[str, np.ndarray] = {}
        self.participants: Dict[str, rtc.RemoteParticipant] = {}
        self.video_tracks: Dict[str, rtc.RemoteVideoTrack] = {}
        self.is_processing = False
        self.tolerance = tolerance
        self.frame_executor = ThreadPoolExecutor(max_workers=4)
        self.match_history: Dict[str, List] = {}
        self.track_tasks = {}
        self.track_owner = {}  # track_sid -> participant_sid
        self.last_match_time = {}
        self.match_cooldown_seconds = self._parse_int_env("MATCH_COOLDOWN_SECONDS", 3)
        self.ref_lock = threading.Lock()
        # Selective subscription controls
        self.selective_subscribe = os.getenv("SELECTIVE_SUBSCRIBE", "false").lower() in {"1","true","yes"}
        self.pub_last_activity: Dict[str, float] = {}  # publication.sid -> last match timestamp (epoch)
        self.pub_state: Dict[str, str] = {}  # publication.sid -> 'subscribed' | 'unsubscribed' | 'pending'
        self.ss_inactive_seconds = self._parse_int_env("SS_INACTIVE_SECONDS", 15)  # after this with no match -> unsubscribe
        self.ss_retry_seconds = self._parse_int_env("SS_RETRY_SECONDS", 10)  # after unsubscribe wait before resubscribe attempt
        self.ss_poll_interval = float(os.getenv("SS_POLL_INTERVAL_SECONDS", "5"))
        self.subscription_manager_task: Optional[asyncio.Task] = None
        # Probe-mode refinements
        # Robust numeric parsing (allow trailing comments) via helper
        self.ss_probe_duration = self._parse_float_env("SS_PROBE_DURATION_SECONDS", 5.0)
        self.ss_active_grace_base = self._parse_float_env("SS_ACTIVE_GRACE_SECONDS", 30.0)  # base grace; may adapt
        self.ss_active_grace = self.ss_active_grace_base
        self.pub_subscribed_at: Dict[str, float] = {}  # publication.sid -> timestamp when (re)subscribed
        self.pub_last_probe: Dict[str, float] = {}  # publication.sid -> last time we probed (unsub time)
        self.pub_backoff_exp: Dict[str, int] = {}  # exponential backoff exponent per sid
        self.ss_backoff_max_exp = self._parse_int_env("SS_BACKOFF_MAX_EXP", 4)  # cap exponent (2^exp * retry_seconds)
        self.adaptive_grace_enabled = os.getenv("SS_ADAPTIVE_GRACE", "true").lower() in {"1","true","yes"}
        # Metrics counters
        self.metrics = {
            "probes_started": 0,
            "probes_abandoned": 0,
            "probes_success": 0,
            "subscriptions_active": 0,
            "state_transitions": 0,
        }
            
        # Load reference face
        self.load_reference_face(reference_image_path)
        
        # Create output directory for captured frames (respect env var, use absolute path)
        self.output_dir = os.path.abspath(os.getenv("OUTPUT_DIR", "captured_faces"))
        os.makedirs(self.output_dir, exist_ok=True)
        self.reference_image_path = reference_image_path
        self._ref_mtime = os.path.getmtime(reference_image_path) if os.path.exists(reference_image_path) else 0
        self._last_reload_check = 0.0
        # Concurrency limiter for frame processing
        self.max_parallel = self._parse_int_env("MAX_PARALLEL_FRAMES", 4)
        self.frame_semaphore = asyncio.Semaphore(self.max_parallel)
        # Performance tuning parameters
        self.downscale_width = self._parse_int_env("FRAME_DOWNSCALE_WIDTH", 640)
        self.process_every_n = self._parse_int_env("PROCESS_EVERY_N_FRAME", 2)  # 2 = every other frame
        if self.process_every_n < 1:
            self.process_every_n = 1
        self.initial_sync_frames = self._parse_int_env("INITIAL_SYNC_FRAMES", 5)
        self.track_frame_counters: Dict[str, int] = {}
        self.track_sync_remaining: Dict[str, int] = {}

    def try_reload_reference(self):
        try:
            now = asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0.0
            if now - self._last_reload_check < 1.0:  # check at most once per second
                return
            self._last_reload_check = now
            if os.path.exists(self.reference_image_path):
                mtime = os.path.getmtime(self.reference_image_path)
                if mtime > self._ref_mtime:
                    if self.load_reference_face(self.reference_image_path, is_reload=True):
                        self._ref_mtime = mtime
                        logger.info("🔄 Reference image reloaded")
        except Exception as e:
            logger.debug(f"Reference reload skipped: {e}")

    def load_reference_face(self, image_path: str, is_reload: bool = False) -> bool:
        """Load and encode the reference face image"""
        try:
            logger.info(f"Loading reference image: {image_path}")
            reference_image = face_recognition.load_image_file(image_path)
            reference_encodings = face_recognition.face_encodings(reference_image)
            
            if len(reference_encodings) == 0:
                logger.error("❌ No face found in reference image. Please use a clearer image.")
                if is_reload:
                    return False
                else:
                    raise RuntimeError("No face in initial reference image")
            
            with self.ref_lock:
                self.reference_encodings["reference"] = reference_encodings[0]
            logger.info("✅ Reference face encoding loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Error loading reference image: {e}")
            if is_reload:
                return False
            raise

    async def connect_to_room(self, url: str, token: str):
        """Connect to the LiveKit room"""
        try:
            logger.info(f"Connecting to room at {url}")
            self.room = rtc.Room()
            
            # Set up event handlers
            self.room.on("participant_connected", self.on_participant_connected)
            self.room.on("participant_disconnected", self.on_participant_disconnected)
            self.room.on("track_subscribed", self.on_track_subscribed)
            self.room.on("track_unsubscribed", self.on_track_unsubscribed)
            self.room.on("disconnected", self.on_disconnected)
            # For selective subscription, we also need track_published events
            if self.selective_subscribe:
                self.room.on("track_published", self.on_track_published)
            
            # Connect to the room
            if self.selective_subscribe:
                # disable auto subscribe so we control which tracks we pull
                await self.room.connect(url, token, rtc.RoomOptions(auto_subscribe=False))
            else:
                await self.room.connect(url, token)
            logger.info("✅ Successfully connected to LiveKit room")
            if self.selective_subscribe and not self.subscription_manager_task:
                self.subscription_manager_task = asyncio.create_task(self.subscription_manager_loop())
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to room: {e}")
            return False

    def on_track_published(self, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        """Handle newly published remote track when using selective subscription"""
        if publication.kind == rtc.TrackKind.KIND_VIDEO and self.selective_subscribe:
            # Initialize state but do not subscribe immediately; manager loop will decide
            now = asyncio.get_event_loop().time()
            self.pub_state[publication.sid] = 'inactive'
            self.pub_last_activity[publication.sid] = 0.0  # no match yet
            self.pub_last_probe[publication.sid] = 0.0  # allow immediate probe
            self.pub_backoff_exp[publication.sid] = 0
            logger.info(f"[SS] Video track published (sid={publication.sid}) by {participant.identity}; initialized as inactive")

    async def subscription_manager_loop(self):
        """Background loop to manage selective subscription state transitions."""
        logger.info("[SS] Subscription manager started")
        try:
            while True:
                await asyncio.sleep(self.ss_poll_interval)
                now = asyncio.get_event_loop().time()
                # Iterate over current participants and their track publications
                if not self.room:
                    continue
                for p in list(self.room.remote_participants.values()):
                    for pub in list(p.track_publications.values()):
                        if pub.kind != rtc.TrackKind.KIND_VIDEO:
                            continue
                        sid = pub.sid
                        state = self.pub_state.get(sid, 'inactive')
                        last_match = self.pub_last_activity.get(sid, 0.0)
                        subscribed_at = self.pub_subscribed_at.get(sid, 0.0)
                        last_probe = self.pub_last_probe.get(sid, 0.0)

                        # State machine: inactive -> probing -> active
                        if state == 'inactive':
                            # Time to probe? apply exponential backoff factor
                            exp = self.pub_backoff_exp.get(sid, 0)
                            wait_interval = self.ss_retry_seconds * (2 ** exp)
                            if wait_interval < self.ss_retry_seconds:
                                wait_interval = self.ss_retry_seconds
                            if (now - last_probe) >= wait_interval:
                                try:
                                    pub.set_subscribed(True)
                                    self._transition_pub_state(sid, 'probing')
                                    subscribed_at = now
                                    self.pub_subscribed_at[sid] = subscribed_at
                                    self.metrics["probes_started"] += 1
                                    logger.info(f"[SS] Probing subscribe sid={sid} participant={p.identity}")
                                except Exception as e:
                                    logger.debug(f"[SS] Probe subscribe failed sid={sid}: {e}")
                        elif state == 'probing':
                            # If we got a match, handle_face_match will promote to active.
                            # If probe duration exceeded without match, unsubscribe.
                            if self.ss_probe_duration > 0 and (now - subscribed_at) >= self.ss_probe_duration and last_match < subscribed_at:
                                try:
                                    pub.set_subscribed(False)
                                    self._transition_pub_state(sid, 'inactive')
                                    self.pub_last_probe[sid] = now
                                    # increase backoff on failed probe
                                    self.pub_backoff_exp[sid] = min(self.pub_backoff_exp.get(sid, 0) + 1, self.ss_backoff_max_exp)
                                    self.metrics["probes_abandoned"] += 1
                                    logger.info(f"[SS] Probe ended (no match) sid={sid} -> inactive")
                                except Exception as e:
                                    logger.debug(f"[SS] Probe unsubscribe failed sid={sid}: {e}")
                        elif state == 'active':
                            # Stay subscribed while matches remain within grace window.
                            if (now - last_match) >= self.ss_active_grace:
                                try:
                                    pub.set_subscribed(False)
                                    self._transition_pub_state(sid, 'inactive')
                                    self.pub_last_probe[sid] = now
                                    self.pub_backoff_exp[sid] = 0  # reset backoff after active cycle ends
                                    logger.info(f"[SS] Became inactive after grace sid={sid}")
                                except Exception as e:
                                    logger.debug(f"[SS] Active->inactive unsubscribe failed sid={sid}: {e}")
        except asyncio.CancelledError:
            logger.info("[SS] Subscription manager stopped")
        except Exception as e:
            logger.error(f"[SS] Manager loop error: {e}")

    def on_participant_connected(self, participant: rtc.RemoteParticipant):
        """Handle new participant joining"""
        logger.info(f"Participant connected: {participant.identity} (SID: {participant.sid})")
        self.participants[participant.sid] = participant

    def on_participant_disconnected(self, participant: rtc.RemoteParticipant):
        """Handle participant leaving"""
        logger.info(f"Participant disconnected: {participant.identity}")
        if participant.sid in self.participants:
            del self.participants[participant.sid]
        if participant.sid in self.video_tracks:
            del self.video_tracks[participant.sid]
        # Cancel any track tasks belonging to this participant
        to_cancel = [tsid for tsid, psid in self.track_owner.items() if psid == participant.sid]
        for tsid in to_cancel:
            task = self.track_tasks.pop(tsid, None)
            self.track_owner.pop(tsid, None)
            if task and not task.done():
                task.cancel()

    def on_track_subscribed(self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        """Handle new track subscription"""
        logger.info(f"Track subscribed: {track.kind} from {participant.identity}")
        
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            self.video_tracks[participant.sid] = track
            logger.info(f"✅ Subscribed to video track from {participant.identity}")
            if self.selective_subscribe:
                now = asyncio.get_event_loop().time()
                # If we explicitly subscribed via probe logic, state may already be 'probing'
                if self.pub_state.get(publication.sid) not in {'probing','active'}:
                    self._transition_pub_state(publication.sid, 'probing')
                self.pub_subscribed_at[publication.sid] = now
            
            # Start processing this video track
            task = asyncio.create_task(self.process_video_track(track, participant, publication.sid))
            self.track_tasks[publication.sid] = task
            self.track_owner[publication.sid] = participant.sid

    def on_track_unsubscribed(self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        """Handle track unsubscription"""
        logger.info(f"Track unsubscribed: {track.kind} from {participant.identity}")
        if participant.sid in self.video_tracks:
            del self.video_tracks[participant.sid]
        # Cancel the processing task for this track
        task = self.track_tasks.pop(publication.sid, None)
        self.track_owner.pop(publication.sid, None)
        if task and not task.done():
            task.cancel()

    def on_disconnected(self, reason: str):
        """Handle room disconnection"""
        logger.info(f"Disconnected from room: {reason}")
        # Cancel any running track tasks
        for tsid, task in list(self.track_tasks.items()):
            if task and not task.done():
                task.cancel()
        self.track_tasks.clear()
        self.track_owner.clear()

    def livekit_frame_to_opencv(self, video_frame: rtc.VideoFrame) -> Optional[np.ndarray]:
        """Convert LiveKit VideoFrame to OpenCV format"""
        try:
            # Convert to RGB24 format if needed
            current_format = video_frame.type
            target_format = getattr(VideoBufferType, 'RGB24', 1)  # RGB24 = 1
            
            if current_format != target_format:
                video_frame = video_frame.convert(target_format)
            
            # Convert to numpy array
            frame_data = np.frombuffer(video_frame.data, dtype=np.uint8)
            
            # Reshape to image dimensions (height, width, channels)
            opencv_frame = frame_data.reshape((video_frame.height, video_frame.width, 3))
            
            # Convert RGB to BGR for OpenCV
            opencv_frame = cv2.cvtColor(opencv_frame, cv2.COLOR_RGB2BGR)
            
            # Copy to detach from underlying buffer (avoid invalid memory after release)
            return opencv_frame.copy()
            
        except Exception as e:
            logger.error(f"Error converting frame: {e}")
            return None

    async def process_video_track(self, track: rtc.RemoteVideoTrack, participant: rtc.RemoteParticipant, track_sid: str):
        """Process video frames from a specific track"""
        logger.info(f"Starting video processing for {participant.identity}")
        # Initialize counters for this track
        self.track_frame_counters[track_sid] = 0
        self.track_sync_remaining[track_sid] = self.initial_sync_frames
        try:
            async for event in rtc.VideoStream(track):
                if event.frame:
                    # Hot-reload reference periodically
                    self.try_reload_reference()
                    # Convert LiveKit frame to OpenCV format
                    opencv_frame = self.livekit_frame_to_opencv(event.frame)
                    if opencv_frame is not None:
                        # Legacy mode bypasses downscale, skipping, and semaphore; processes everything (closer to original behavior)
                        if os.getenv("DETECTION_LEGACY_MODE", "false").lower() in {"1","true","yes"}:
                            # Process immediately in background (unbounded) similar to original simple version
                            asyncio.create_task(self.process_frame_async(opencv_frame, participant))
                            continue
                        # Downscale frame for faster detection
                        if self.downscale_width > 0 and opencv_frame.shape[1] > self.downscale_width:
                            ratio = self.downscale_width / opencv_frame.shape[1]
                            new_h = int(opencv_frame.shape[0] * ratio)
                            small_frame = cv2.resize(opencv_frame, (self.downscale_width, new_h), interpolation=cv2.INTER_LINEAR)
                        else:
                            small_frame = opencv_frame

                        counter = self.track_frame_counters.get(track_sid, 0)
                        sync_left = self.track_sync_remaining.get(track_sid, 0)

                        if sync_left > 0:
                            # Process synchronously (await) for first few frames to get early detection
                            self.track_sync_remaining[track_sid] = sync_left - 1
                            try:
                                loop = asyncio.get_event_loop()
                                result = await loop.run_in_executor(
                                    self.frame_executor,
                                    self.detect_and_match_faces,
                                    small_frame,
                                    participant
                                )
                                if os.getenv("DETECTION_DEBUG", "false").lower() in {"1","true","yes"}:
                                    logger.info(f"[DEBUG] sync frame processed track={track_sid} counter={counter} faces={(len(result) if result else 0)}")
                                if result:
                                    # Map bbox back if scaled
                                    if small_frame.shape[:2] != opencv_frame.shape[:2]:
                                        sx = opencv_frame.shape[1] / small_frame.shape[1]
                                        sy = opencv_frame.shape[0] / small_frame.shape[0]
                                        for m in result:
                                            t, r, b, l = m["location"]
                                            m["location"] = (int(t * sy), int(r * sx), int(b * sy), int(l * sx))
                                    await self.handle_face_match(result, participant, opencv_frame)
                            except Exception:
                                pass
                        else:
                            # Frame skipping after warmup
                            if counter % self.process_every_n != 0:
                                if os.getenv("DETECTION_DEBUG", "false").lower() in {"1","true","yes"}:
                                    logger.info(f"[DEBUG] skipped frame track={track_sid} counter={counter}")
                                self.track_frame_counters[track_sid] = counter + 1
                                continue
                            try:
                                self.frame_semaphore.acquire_nowait()
                            except Exception:
                                if os.getenv("DETECTION_DEBUG", "false").lower() in {"1","true","yes"}:
                                    logger.info(f"[DEBUG] dropped frame (semaphore full) track={track_sid} counter={counter}")
                                self.track_frame_counters[track_sid] = counter + 1
                                continue
                            else:
                                asyncio.create_task(self._bounded_process_frame_scaled(small_frame, opencv_frame, participant))
                        self.track_frame_counters[track_sid] = counter + 1
        except asyncio.CancelledError:
            logger.info(f"Stopped processing video for {participant.identity} (track {track_sid})")
            return

    async def process_frame_async(self, frame: np.ndarray, participant: rtc.RemoteParticipant):
        """Process a single frame asynchronously"""
        loop = asyncio.get_event_loop()
        
        # Run face detection in thread pool
        result = await loop.run_in_executor(
            self.frame_executor,
            self.detect_and_match_faces,
            frame,
            participant
        )
        
        if result:
            await self.handle_face_match(result, participant, frame)

    async def _bounded_process_frame(self, frame: np.ndarray, participant: rtc.RemoteParticipant):
        """Wrapper that enforces concurrency limits for frame processing."""
        try:
            try:
                await self.process_frame_async(frame, participant)
            finally:
                # Explicit release (we acquired manually in process_video_track)
                self.frame_semaphore.release()
        except Exception as e:
            logger.debug(f"Frame processing error: {e}")

    async def _bounded_process_frame_scaled(self, small_frame: np.ndarray, original_frame: np.ndarray, participant: rtc.RemoteParticipant):
        """Process a scaled frame but report matches on the original frame coordinates."""
        try:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.frame_executor,
                    self.detect_and_match_faces,
                    small_frame,
                    participant
                )
                if result:
                    if small_frame.shape[:2] != original_frame.shape[:2]:
                        sx = original_frame.shape[1] / small_frame.shape[1]
                        sy = original_frame.shape[0] / small_frame.shape[0]
                        for m in result:
                            t, r, b, l = m["location"]
                            m["location"] = (int(t * sy), int(r * sx), int(b * sy), int(l * sx))
                    if os.getenv("DETECTION_DEBUG", "false").lower() in {"1","true","yes"}:
                        logger.info(f"[DEBUG] async frame processed faces={len(result)} original_size={original_frame.shape[1]}x{original_frame.shape[0]} small_size={small_frame.shape[1]}x{small_frame.shape[0]}")
                    await self.handle_face_match(result, participant, original_frame)
                else:
                    if os.getenv("DETECTION_DEBUG", "false").lower() in {"1","true","yes"}:
                        logger.info("[DEBUG] async frame processed faces=0")
            finally:
                self.frame_semaphore.release()
        except Exception as e:
            logger.debug(f"Scaled frame processing error: {e}")

    def detect_and_match_faces(self, frame: np.ndarray, participant: rtc.RemoteParticipant) -> Optional[Dict]:
        """Detect and match faces in the frame"""
        try:
            # Convert frame from BGR (OpenCV) to RGB (face_recognition)
            # Use .copy() to ensure C-contiguous memory (required by dlib)
            rgb_frame = frame[:, :, ::-1].copy()
            
            # Detect faces and compute encodings
            face_locations = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=0)  # Faster: no upsample
            
            if not face_locations:
                return None
                
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            matches = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare detected face to reference
                with self.ref_lock:
                    ref = self.reference_encodings.get("reference")
                if ref is None:
                    continue
                face_matches = face_recognition.compare_faces([ref], face_encoding, tolerance=self.tolerance)
                
                # Calculate face distance for confidence
                face_distances = face_recognition.face_distance([ref], face_encoding)
                
                if face_matches[0]:
                    confidence = 1 - face_distances[0]
                    matches.append({
                        "location": (top, right, bottom, left),
                        "confidence": confidence,
                        "participant_id": participant.sid,
                        "participant_name": participant.identity,
                        "timestamp": datetime.now()
                    })
                    
            return matches if matches else None
            
        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return None

    async def handle_face_match(self, matches: List[Dict], participant: rtc.RemoteParticipant, frame: np.ndarray):
        """Handle detected face matches"""
        # If participant already disconnected, ignore pending matches
        if participant.sid not in self.participants:
            return
        for match in matches:
            confidence = match["confidence"]
            location = match["location"]
            timestamp = match["timestamp"]
            # Cooldown to avoid repeated matches and stale notifications
            last_ts = self.last_match_time.get(participant.sid)
            if last_ts and (timestamp - last_ts).total_seconds() < self.match_cooldown_seconds:
                # Still update activity for subscription logic even if event suppressed
                if self.selective_subscribe:
                    for tsid, owner in list(self.track_owner.items()):
                        if owner == participant.sid:
                            self.pub_last_activity[tsid] = asyncio.get_event_loop().time()
                continue
            self.last_match_time[participant.sid] = timestamp
            # Mark activity & elevate state to active
            if self.selective_subscribe:
                now = asyncio.get_event_loop().time()
                for tsid, owner in list(self.track_owner.items()):
                    if owner == participant.sid:
                        self.pub_last_activity[tsid] = now
                        if self.pub_state.get(tsid) != 'active':
                            self._transition_pub_state(tsid, 'active')
                            self.metrics["probes_success"] += 1
                            # successful detection resets backoff & may adapt grace
                            self.pub_backoff_exp[tsid] = 0
                            if self.adaptive_grace_enabled:
                                # Simple adaptation: extend grace modestly for higher confidence,
                                # clamp between base and 2x base.
                                boost = (confidence - 0.5) * 10  # scale factor around mid confidence
                                new_grace = self.ss_active_grace_base * (1 + max(0, min(0.5, boost)))
                                self.ss_active_grace = max(self.ss_active_grace_base, min(self.ss_active_grace_base * 2, new_grace))
            
            logger.info(f"🎯 FACE MATCH DETECTED! Participant: {participant.identity}, Confidence: {confidence:.2f}")
            
            # Draw bounding box on frame
            top, right, bottom, left = location
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, f"MATCH: {confidence:.2f}", (left, top - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Save the matched frame
            filename = f"{self.output_dir}/match_{participant.identity}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"💾 Saved matched frame: {filename}")
            
            # Store match in history
            if participant.sid not in self.match_history:
                self.match_history[participant.sid] = []
            
            self.match_history[participant.sid].append({
                "timestamp": timestamp.isoformat(),
                "confidence": confidence,
                "filename": filename,
                "participant_name": participant.identity
            })
            
            # Notify API server about the match
            try:
                async with aiohttp.ClientSession() as session:
                    payload = {
                        "participant_id": participant.sid,
                        "participant_name": participant.identity,
                        "confidence": float(confidence),
                        "filename": filename,
                        "snapshot": os.path.basename(filename),
                        "timestamp": timestamp.isoformat(),
                        "bbox": [int(location[0]), int(location[1]), int(location[2]), int(location[3])],
                        "frame_size": [int(frame.shape[1]), int(frame.shape[0])]
                    }
                    api_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
                    await session.post(f"{api_url}/api/match", json=payload, timeout=3)
            except Exception as e:
                logger.debug(f"Failed to notify match API: {e}")

            # Optional actions
            # - Start recording video
            # - Trigger alerts
            # - Store in database
            
    def get_match_summary(self) -> Dict:
        """Get summary of all face matches"""
        summary = {
            "total_participants_matched": len(self.match_history),
            "total_matches": sum(len(matches) for matches in self.match_history.values()),
            "matches_by_participant": {}
        }
        
        for participant_sid, matches in self.match_history.items():
            if matches:
                participant_name = matches[0]["participant_name"]
                summary["matches_by_participant"][participant_name] = {
                    "count": len(matches),
                    "latest_match": matches[-1]["timestamp"],
                    "best_confidence": max(match["confidence"] for match in matches)
                }
        
        return summary

    async def disconnect(self):
        """Disconnect from the room"""
        if self.room:
            await self.room.disconnect()
            logger.info("Disconnected from room")
        if self.subscription_manager_task and not self.subscription_manager_task.done():
            self.subscription_manager_task.cancel()
            try:
                await self.subscription_manager_task
            except Exception:
                pass

    def _parse_float_env(self, key: str, default: float) -> float:
        raw = os.getenv(key, str(default))
        # Strip inline comments and whitespace
        raw = raw.split('#', 1)[0].strip()
        try:
            return float(raw)
        except ValueError:
            logger.warning(f"Env {key} value '{raw}' invalid, using default {default}")
            return default

    def _parse_int_env(self, key: str, default: int) -> int:
        raw = os.getenv(key, str(default))
        raw = raw.split('#', 1)[0].strip()
        try:
            return int(raw)
        except ValueError:
            logger.warning(f"Env {key} value '{raw}' invalid, using default {default}")
            return default

    def _transition_pub_state(self, sid: str, new_state: str):
        old = self.pub_state.get(sid)
        if old == new_state:
            return
        self.pub_state[sid] = new_state
        self.metrics["state_transitions"] += 1
        if new_state == 'active':
            self.metrics["subscriptions_active"] += 1
        if old == 'active' and new_state != 'active':
            self.metrics["subscriptions_active"] = max(0, self.metrics["subscriptions_active"] - 1)

# Token generation utility
async def generate_access_token(api_key: str, api_secret: str, room_name: str, participant_identity: str) -> str:
    """Generate an access token for LiveKit"""
    token = (
        AccessToken(api_key, api_secret)
        .with_identity(participant_identity)
        .with_name(participant_identity)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=False,  # Face recognition client doesn't need to publish
                can_subscribe=True,
            )
        )
    )
    return token.to_jwt()

# Main execution
async def main():
    # Configuration - Get from environment variables or use defaults
    LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    API_KEY = os.getenv("LIVEKIT_API_KEY", "your-api-key") 
    API_SECRET = os.getenv("LIVEKIT_API_SECRET", "your-api-secret")
    ROOM_NAME = os.getenv("ROOM_NAME", "face-recognition-room")
    # If a reference image was uploaded via API, prefer that path
    uploaded_reference = "/workspaces/face-match/reference/current.jpg"
    env_reference = os.getenv("REFERENCE_IMAGE", uploaded_reference)
    REFERENCE_IMAGE = uploaded_reference if os.path.exists(uploaded_reference) else env_reference
    
    # Check if reference image exists
    if not os.path.exists(REFERENCE_IMAGE):
        logger.error(f"Reference image not found: {REFERENCE_IMAGE}")
        logger.info("Please check the reference image path:")
        logger.info(f"  Current path: {REFERENCE_IMAGE}")
        logger.info("  Make sure the file exists and the path is correct")
        return
    
    # Create face recognizer
    recognizer = LiveKitFaceRecognizer(REFERENCE_IMAGE, tolerance=0.5)
    
    # Generate access token
    token = await generate_access_token(
        API_KEY, 
        API_SECRET, 
        ROOM_NAME, 
        "face-recognition-client"
    )
    
    try:
        # Connect to room
        if await recognizer.connect_to_room(LIVEKIT_URL, token):
            logger.info("🚀 Face recognition system is running...")
            logger.info(f"📹 Monitoring room '{ROOM_NAME}' for face matches...")
            logger.info("💡 Join the room from your browser or LiveKit playground to test!")
            logger.info(f"   Room URL: {LIVEKIT_URL}")
            logger.info(f"   Room Name: {ROOM_NAME}")

            summary_interval = int(os.getenv("SUMMARY_LOG_INTERVAL_SECONDS", "10"))
            only_on_change = os.getenv("STATUS_LOG_ONLY_ON_CHANGE", "false").lower() in {"1","true","yes"}
            last_summary = {"total_matches": None, "total_participants_matched": None}
            last_log_time = 0.0
            
            while True:
                await asyncio.sleep(1)
                now = asyncio.get_event_loop().time()
                if now - last_log_time >= summary_interval:
                    summary = recognizer.get_match_summary()
                    changed = (summary["total_matches"] != last_summary["total_matches"] or
                               summary["total_participants_matched"] != last_summary["total_participants_matched"])
                    if summary["total_matches"] > 0 and (changed or not only_on_change):
                        logger.info(f"📊 Status: {summary['total_matches']} matches found across {summary['total_participants_matched']} participants")
                    last_summary = summary
                    last_log_time = now
                    
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down face recognition system...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await recognizer.disconnect()

if __name__ == "__main__":
    asyncio.run(main())