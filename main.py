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
import threading
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
        
        # Load reference face
        self.load_reference_face(reference_image_path)
        
        # Create output directory for captured frames (respect env var, use absolute path)
        self.output_dir = os.path.abspath(os.getenv("OUTPUT_DIR", "captured_faces"))
        os.makedirs(self.output_dir, exist_ok=True)

    def load_reference_face(self, image_path: str):
        """Load and encode the reference face image"""
        try:
            logger.info(f"Loading reference image: {image_path}")
            reference_image = face_recognition.load_image_file(image_path)
            reference_encodings = face_recognition.face_encodings(reference_image)
            
            if len(reference_encodings) == 0:
                logger.error("❌ No face found in reference image. Please use a clearer image.")
                sys.exit(1)
            
            self.reference_encodings["reference"] = reference_encodings[0]
            logger.info("✅ Reference face encoding loaded successfully.")
            
        except Exception as e:
            logger.error(f"Error loading reference image: {e}")
            sys.exit(1)

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
            
            # Connect to the room
            await self.room.connect(url, token)
            logger.info("✅ Successfully connected to LiveKit room")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to room: {e}")
            return False

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

    def on_track_subscribed(self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        """Handle new track subscription"""
        logger.info(f"Track subscribed: {track.kind} from {participant.identity}")
        
        if track.kind == rtc.TrackKind.KIND_VIDEO:
            self.video_tracks[participant.sid] = track
            logger.info(f"✅ Subscribed to video track from {participant.identity}")
            
            # Start processing this video track
            asyncio.create_task(self.process_video_track(track, participant))

    def on_track_unsubscribed(self, track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
        """Handle track unsubscription"""
        logger.info(f"Track unsubscribed: {track.kind} from {participant.identity}")
        if participant.sid in self.video_tracks:
            del self.video_tracks[participant.sid]

    def on_disconnected(self, reason: str):
        """Handle room disconnection"""
        logger.info(f"Disconnected from room: {reason}")

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
            
            return opencv_frame
            
        except Exception as e:
            logger.error(f"Error converting frame: {e}")
            return None

    async def process_video_track(self, track: rtc.RemoteVideoTrack, participant: rtc.RemoteParticipant):
        """Process video frames from a specific track"""
        logger.info(f"Starting video processing for {participant.identity}")
        
        async for event in rtc.VideoStream(track):
            if event.frame:
                # Convert LiveKit frame to OpenCV format
                opencv_frame = self.livekit_frame_to_opencv(event.frame)
                if opencv_frame is not None:
                    # Process frame in thread pool to avoid blocking
                    asyncio.create_task(
                        self.process_frame_async(opencv_frame, participant)
                    )

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

    def detect_and_match_faces(self, frame: np.ndarray, participant: rtc.RemoteParticipant) -> Optional[Dict]:
        """Detect and match faces in the frame"""
        try:
            # Convert frame from BGR (OpenCV) to RGB (face_recognition)
            # Use .copy() to ensure C-contiguous memory (required by dlib)
            rgb_frame = frame[:, :, ::-1].copy()
            
            # Detect faces and compute encodings
            face_locations = face_recognition.face_locations(rgb_frame, model="hog")  # Use HOG for speed
            
            if not face_locations:
                return None
                
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            matches = []
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Compare detected face to reference
                face_matches = face_recognition.compare_faces(
                    [self.reference_encodings["reference"]], 
                    face_encoding, 
                    tolerance=self.tolerance
                )
                
                # Calculate face distance for confidence
                face_distances = face_recognition.face_distance(
                    [self.reference_encodings["reference"]], 
                    face_encoding
                )
                
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
        for match in matches:
            confidence = match["confidence"]
            location = match["location"]
            timestamp = match["timestamp"]
            
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
            
            # Optional: You can add additional actions here
            # - Send notifications
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
    REFERENCE_IMAGE = os.getenv("REFERENCE_IMAGE", "/workspaces/face-match/WhatsApp Image 2025-08-12 at 07.38.40_dec32353.jpg")
    
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
            
            # Keep the system running
            while True:
                await asyncio.sleep(10)
                # Print periodic status
                summary = recognizer.get_match_summary()
                if summary["total_matches"] > 0:
                    logger.info(f"📊 Status: {summary['total_matches']} matches found across {summary['total_participants_matched']} participants")
                    
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down face recognition system...")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        await recognizer.disconnect()

if __name__ == "__main__":
    asyncio.run(main())