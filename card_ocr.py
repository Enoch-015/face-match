"""Card detection and OCR pipeline for extracting reference face images.

This module locates an ID card inside incoming video frames, runs OCR to extract
its textual content, saves the results, and crops the printed face image to use
as the reference for downstream face matching.
"""

from __future__ import annotations

import logging
import threading
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import face_recognition
import numpy as np
import pytesseract
from pytesseract import Output

logger = logging.getLogger(__name__)


@dataclass
class CardProcessingResult:
    """Value object describing the outcome of a card processing step."""

    status: str
    message: str
    notify: bool
    reference_ready: bool = False
    card_bbox: Optional[Tuple[int, int, int, int]] = None
    frame_size: Optional[Tuple[int, int]] = None
    text_preview: Optional[str] = None
    text_path: Optional[str] = None
    card_snapshot_path: Optional[str] = None
    reference_path: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    card_polygon: Optional[List[Tuple[int, int]]] = None


class CardFaceNotFoundError(RuntimeError):
    """Raised when no face is detected on the ID card image."""


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def _detect_card(
    frame: np.ndarray,
    *,
    min_area_ratio: float = 0.01,
    min_score_delta: float = 0.02,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    height, width = frame.shape[:2]
    min_side = min(width, height)
    if min_side < 300:
        scale = 300.0 / min_side
        resized = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
        resized = frame

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.erode(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    frame_area = resized.shape[0] * resized.shape[1]
    best: Optional[Tuple[np.ndarray, np.ndarray, float]] = None
    best_score = 0.0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < frame_area * min_area_ratio:
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            pts = np.array(box, dtype=np.float32)

        pts = pts / scale
        warp = _four_point_transform(frame, pts)

        width_warp, height_warp = warp.shape[1], warp.shape[0]
        if width_warp == 0 or height_warp == 0:
            continue
        aspect = width_warp / float(height_warp)
        aspect_score = 1.0 - min(abs(aspect - 1.58), 1.0)

        normalized_area = cv2.contourArea(contour) / frame_area
        score = aspect_score * 0.4 + normalized_area * 0.6

        if score > best_score + min_score_delta:
            best = (warp, pts, score)
            best_score = score

    return best


def _enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    if gray.shape[1] < 900:
        scale = 900.0 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    return thresh


def _filter_high_confidence_words(data: Dict[str, list], min_conf: float) -> List[str]:
    words: List[str] = []
    texts = data.get("text", [])
    confs = data.get("conf", [])
    for text, conf in zip(texts, confs):
        if not text or not text.strip():
            continue
        try:
            conf_value = float(conf)
        except (TypeError, ValueError):
            continue
        if conf_value < 0:
            continue
        if conf_value >= min_conf:
            words.append(text.strip())
    return words


class CardProcessor:
    """Stateful processor that manages the card → OCR → face pipeline."""

    def __init__(
        self,
        reference_dir: Path | str,
        reference_filename: str = "current.jpg",
        text_filename: str = "card_text.txt",
        snapshot_filename: str = "card_snapshot.jpg",
        lang: str = "eng",
        min_confidence: float = 70.0,
        min_card_area_ratio: float = 0.003,
        min_focus_score: float = 120.0,
        min_brightness: float = 60.0,
        max_brightness: float = 210.0,
        max_glare_ratio: float = 0.12,
    ) -> None:
        self.reference_dir = Path(reference_dir)
        self.reference_dir.mkdir(parents=True, exist_ok=True)
        self.reference_filename = reference_filename
        self.text_filename = text_filename
        self.snapshot_filename = snapshot_filename
        self.lang = lang
        self.min_conf = min_confidence
        self.min_card_area_ratio = max(0.0005, float(min_card_area_ratio))
        self.min_focus_score = float(max(min_focus_score, 1.0))
        self.min_brightness = float(np.clip(min_brightness, 0.0, 255.0))
        self.max_brightness = float(np.clip(max_brightness, 0.0, 255.0))
        self.max_glare_ratio = float(np.clip(max_glare_ratio, 0.0, 1.0))

        try:
            pytesseract.get_tesseract_version()
        except (FileNotFoundError, pytesseract.TesseractNotFoundError) as exc:  # pragma: no cover
            raise RuntimeError(
                "Tesseract OCR is not installed. Please install system package 'tesseract-ocr'."
            ) from exc

        self._lock = threading.RLock()
        self._reset_internal_state()
        self._frames_seen = 0
        self._frame_size: Optional[Tuple[int, int]] = None
        self._last_notified_state: Optional[str] = None
        self._hint_interval = 45
        self._card_ready_for_ocr = False
        self._card_focus_score: float = 0.0
        self._card_candidate_metrics: Dict[str, float] = {}

    @property
    def reference_path(self) -> Path:
        return self.reference_dir / self.reference_filename

    @property
    def text_path(self) -> Path:
        return self.reference_dir / self.text_filename

    @property
    def snapshot_path(self) -> Path:
        return self.reference_dir / self.snapshot_filename

    def _reset_internal_state(self) -> None:
        self._card_image: Optional[np.ndarray] = None
        self._card_bbox: Optional[Tuple[int, int, int, int]] = None
        self._card_polygon: Optional[List[Tuple[int, int]]] = None
        self._card_score: float = 0.0
        self._ocr_text: Optional[str] = None
        self._ocr_in_progress: bool = False
        self._face_in_progress: bool = False
        self._reference_ready: bool = False
        self._error: Optional[str] = None
        self._metadata: Dict[str, str] = {"ocr_language": self.lang}
        self._state: str = "searching"
        self._message: str = "Looking for ID card"
        self._card_ready_for_ocr = False
        self._card_focus_score = 0.0
        self._card_candidate_metrics = {}

    def _evaluate_clarity(self, image: np.ndarray) -> Tuple[bool, Dict[str, float], str]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        focus_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        glare_ratio = float(np.mean(gray >= 245))

        passes_focus = focus_score >= self.min_focus_score
        passes_brightness = self.min_brightness <= brightness <= self.max_brightness
        passes_glare = glare_ratio <= self.max_glare_ratio
        passes = passes_focus and passes_brightness and passes_glare

        reasons: List[str] = []
        if not passes_focus:
            reasons.append(f"focus {focus_score:.1f} < {self.min_focus_score:.1f}")
        if not passes_brightness:
            if brightness < self.min_brightness:
                reasons.append(f"brightness {brightness:.1f} < {self.min_brightness:.1f}")
            else:
                reasons.append(f"brightness {brightness:.1f} > {self.max_brightness:.1f}")
        if not passes_glare:
            reasons.append(f"glare {glare_ratio:.2f} > {self.max_glare_ratio:.2f}")

        if passes:
            message = "ID card detected. Capturing text..."
        else:
            detail = ", ".join(reasons) if reasons else "card clarity below threshold"
            message = (
                "Card detected but the image looks blurred or poorly lit. Hold it steady, "
                "move closer, and reduce glare (" + detail + ")."
            )

        metrics = {
            "card_focus": focus_score,
            "card_brightness": brightness,
            "card_glare_ratio": glare_ratio,
        }
        return passes, metrics, message

    def reset(
        self,
        *,
        clear_reference: bool,
        reason: str,
        status: str,
        message: str,
        metadata: Optional[Dict[str, str]] = None,
        notify: bool = True,
    ) -> CardProcessingResult:
        with self._lock:
            self._reset_internal_state()
            self._metadata['reset_reason'] = reason
            self._metadata['frames_seen'] = '0'
            if metadata:
                self._metadata.update(metadata)

            if clear_reference:
                for path in (self.text_path, self.reference_path):
                    try:
                        path.unlink(missing_ok=True)
                    except Exception:  # pragma: no cover
                        logger.debug("Failed removing %s during reset", path, exc_info=True)
                self._reference_ready = False
            try:
                self.snapshot_path.unlink(missing_ok=True)
            except Exception:
                logger.debug("Failed removing %s during reset", self.snapshot_path, exc_info=True)

            self._frames_seen = 0
            self._frame_size = None
            self._set_state(status, message)
            if notify:
                self._last_notified_state = status
            else:
                self._last_notified_state = None

            return CardProcessingResult(
                status=status,
                message=message,
                notify=notify,
                reference_ready=self._reference_ready,
                card_bbox=None,
                frame_size=self._frame_size,
                metadata=dict(self._metadata),
                card_polygon=None,
            )

    def has_reference(self) -> bool:
        with self._lock:
            return self._reference_ready

    def _set_state(self, status: str, message: str) -> None:
        self._state = status
        self._message = message
        self._metadata['card_stage'] = status

    def _should_notify(self) -> bool:
        if self._state != self._last_notified_state:
            self._last_notified_state = self._state
            return True
        return False

    def process_frame(self, frame: np.ndarray) -> CardProcessingResult:
        frame_h, frame_w = frame.shape[:2]

        with self._lock:
            self._frame_size = (frame_w, frame_h)
            self._frames_seen += 1
            self._metadata['frames_seen'] = str(self._frames_seen)

            if self._error:
                return CardProcessingResult(
                    status="error",
                    message=self._error,
                    notify=self._should_notify(),
                    error=self._error,
                    metadata=dict(self._metadata),
                )

            if self._reference_ready:
                preview = self._preview_text(self._ocr_text)
                return CardProcessingResult(
                    status="completed",
                    message="Reference extracted from card",
                    notify=self._should_notify(),
                    reference_ready=True,
                    text_preview=preview,
                    text_path=str(self.text_path) if self.text_path.exists() else None,
                    reference_path=str(self.reference_path) if self.reference_path.exists() else None,
                    card_snapshot_path=str(self.snapshot_path) if self.snapshot_path.exists() else None,
                    card_bbox=self._card_bbox,
                    frame_size=self._frame_size,
                    metadata=dict(self._metadata),
                    card_polygon=list(self._card_polygon) if self._card_polygon else None,
                )

            card_ready = self._card_image is not None

        if not card_ready:
            detection = _detect_card(
                frame,
                min_area_ratio=self.min_card_area_ratio,
            )
            with self._lock:
                if detection:
                    warped, pts, score = detection
                    passes_clarity, clarity_metrics, clarity_message = self._evaluate_clarity(warped)
                    self._card_candidate_metrics = clarity_metrics
                    self._metadata['last_detection_score'] = f"{score:.3f}"
                    self._metadata['card_focus_score'] = f"{clarity_metrics['card_focus']:.1f}"
                    self._metadata['card_brightness'] = f"{clarity_metrics['card_brightness']:.1f}"
                    self._metadata['card_glare_ratio'] = f"{clarity_metrics['card_glare_ratio']:.3f}"
                    self._metadata['card_focus_threshold'] = f"{self.min_focus_score:.1f}"
                    self._metadata['card_brightness_range'] = f"{self.min_brightness:.0f}-{self.max_brightness:.0f}"
                    self._metadata['card_glare_threshold'] = f"{self.max_glare_ratio:.2f}"

                    polygon = [(int(pt[0]), int(pt[1])) for pt in pts]
                    self._card_polygon = polygon
                    top = max(int(min(pt[1] for pt in pts)), 0)
                    bottom = min(int(max(pt[1] for pt in pts)), frame_h)
                    left = max(int(min(pt[0] for pt in pts)), 0)
                    right = min(int(max(pt[0] for pt in pts)), frame_w)
                    self._card_bbox = (top, right, bottom, left)

                    if passes_clarity:
                        self._card_ready_for_ocr = True
                        self._card_image = warped
                        self._card_score = score
                        self._card_focus_score = clarity_metrics['card_focus']
                        try:
                            cv2.imwrite(str(self.snapshot_path), warped)
                            self._metadata['card_snapshot'] = self.snapshot_path.name
                        except Exception:
                            logger.debug("Failed saving card snapshot", exc_info=True)
                        self._metadata['card_score'] = f"{self._card_score:.3f}"
                        self._set_state("card_detected", clarity_message)
                        notify = self._should_notify()
                        return CardProcessingResult(
                            status=self._state,
                            message=self._message,
                            notify=notify,
                            card_bbox=self._card_bbox,
                            frame_size=self._frame_size,
                            card_snapshot_path=str(self.snapshot_path) if self.snapshot_path.exists() else None,
                            metadata=dict(self._metadata),
                            card_polygon=list(self._card_polygon) if self._card_polygon else None,
                        )

                    # Clarity not sufficient yet
                    self._card_ready_for_ocr = False
                    self._card_image = None
                    self._card_score = 0.0
                    self._card_focus_score = clarity_metrics['card_focus']
                    self._metadata.pop('card_snapshot', None)
                    self._metadata.pop('card_score', None)
                    self._set_state("card_clarity", clarity_message)
                    return CardProcessingResult(
                        status=self._state,
                        message=self._message,
                        notify=self._should_notify(),
                        card_bbox=self._card_bbox,
                        frame_size=self._frame_size,
                        metadata=dict(self._metadata),
                        card_polygon=list(self._card_polygon) if self._card_polygon else None,
                    )

                # No detection found this frame
                self._metadata.pop('last_detection_score', None)
                self._metadata.pop('card_score', None)
                self._card_polygon = None
                self._card_bbox = None
                hint_message = "Looking for ID card"
                if self._frames_seen >= self._hint_interval and self._frames_seen % self._hint_interval == 0:
                    hint_message = "No card detected yet. Hold the ID closer, ensure it fills more of the frame, and avoid glare."
                self._metadata['min_card_area_ratio'] = f"{self.min_card_area_ratio:.4f}"
                self._set_state("searching", hint_message)
                return CardProcessingResult(
                    status=self._state,
                    message=self._message,
                    notify=self._should_notify(),
                    card_bbox=None,
                    frame_size=self._frame_size,
                    metadata=dict(self._metadata),
                    card_polygon=None,
                )

        with self._lock:
            if self._ocr_text is None:
                if not self._ocr_in_progress:
                    self._ocr_in_progress = True
                    card_image = self._card_image.copy() if self._card_image is not None else None
                else:
                    return CardProcessingResult(
                        status="ocr_running",
                        message="OCR is in progress...",
                        notify=False,
                        card_bbox=self._card_bbox,
                        frame_size=self._frame_size,
                        metadata=dict(self._metadata),
                        card_polygon=list(self._card_polygon) if self._card_polygon else None,
                    )
            else:
                card_image = None

        if self._ocr_text is None:
            if card_image is None:
                logger.debug("OCR requested but card image missing; resetting")
                return self.reset(
                    clear_reference=False,
                    reason="card_image_missing",
                    status="searching",
                    message="Lost ID card. Please show it again.",
                    metadata={"last_ocr_error": "card_image_missing"},
                )

            try:
                prepped = _enhance_for_ocr(card_image)
                text = pytesseract.image_to_string(prepped, lang=self.lang).strip()
                filtered_words: List[str] = []
                try:
                    data = pytesseract.image_to_data(prepped, lang=self.lang, output_type=Output.DICT)
                    filtered_words = _filter_high_confidence_words(data, self.min_conf)
                except Exception:
                    filtered_words = []
                if not text and filtered_words:
                    text = " ".join(filtered_words)
            except Exception as exc:  # pragma: no cover
                with self._lock:
                    self._ocr_in_progress = False
                    self._error = f"OCR failed: {exc}"
                    self._set_state("error", self._error)
                logger.exception("OCR failed")
                return CardProcessingResult(
                    status="error",
                    message=str(exc),
                    notify=True,
                    error=str(exc),
                    metadata=dict(self._metadata),
                )

            if not text:
                with self._lock:
                    self._ocr_in_progress = False
                    prev_retry = int(self._metadata.get('ocr_retry_count', '0'))
                return self.reset(
                    clear_reference=False,
                    reason="ocr_retry",
                    status="ocr_retry",
                    message="Card detected but text unreadable. Hold steady or adjust lighting.",
                    metadata={
                        "last_ocr_error": "no_text_detected",
                        "ocr_retry_count": str(prev_retry + 1),
                    },
                )

            try:
                self.text_path.write_text(text, encoding="utf-8")
            except Exception as exc:
                with self._lock:
                    self._ocr_in_progress = False
                    self._error = f"Failed saving OCR text: {exc}"
                    self._set_state("error", self._error)
                logger.exception("Failed saving OCR text")
                return CardProcessingResult(
                    status="error",
                    message=str(exc),
                    notify=True,
                    error=str(exc),
                    metadata=dict(self._metadata),
                )

            with self._lock:
                self._ocr_in_progress = False
                self._ocr_text = text
                self._metadata['card_text_ready'] = 'true'
                self._metadata['ocr_retry_count'] = '0'
                self._metadata.pop('last_ocr_error', None)
                preview = self._preview_text(text)
                self._set_state("ocr_complete", "OCR complete. Extracting face from card...")
                notify = self._should_notify()
                return CardProcessingResult(
                    status="ocr_complete",
                    message="OCR completed successfully",
                    notify=notify,
                    text_preview=preview,
                    text_path=str(self.text_path),
                    card_snapshot_path=str(self.snapshot_path) if self.snapshot_path.exists() else None,
                    card_bbox=self._card_bbox,
                    frame_size=self._frame_size,
                    metadata=dict(self._metadata),
                    card_polygon=list(self._card_polygon) if self._card_polygon else None,
                )

        with self._lock:
            if self._face_in_progress:
                return CardProcessingResult(
                    status="face_extraction",
                    message="Extracting face from card...",
                    notify=False,
                    card_bbox=self._card_bbox,
                    frame_size=self._frame_size,
                    metadata=dict(self._metadata),
                    card_polygon=list(self._card_polygon) if self._card_polygon else None,
                )
            self._face_in_progress = True
            card_image = self._card_image.copy() if self._card_image is not None else None

        if card_image is None:
            self._face_in_progress = False
            return self.reset(
                clear_reference=False,
                reason="card_image_missing",
                status="card_retry",
                message="Lost ID card before extracting face. Please hold it steady again.",
                metadata={"last_face_error": "card_image_missing"},
            )

        try:
            reference_path = self._extract_and_save_face(card_image)
        except CardFaceNotFoundError as exc:
            return self.reset(
                clear_reference=False,
                reason="card_face_retry",
                status="card_retry",
                message="No face detected on the ID card. Hold it steady and closer so we can retry.",
                metadata={"last_face_error": str(exc)},
            )
        except Exception as exc:  # pragma: no cover
            with self._lock:
                self._face_in_progress = False
                self._error = f"Face extraction failed: {exc}"
                self._set_state("error", self._error)
            logger.exception("Face extraction failed")
            return CardProcessingResult(
                status="error",
                message=str(exc),
                notify=True,
                error=str(exc),
                metadata=dict(self._metadata),
            )

        with self._lock:
            self._reference_ready = True
            self._face_in_progress = False
            self._metadata['card_face_ready'] = 'true'
            self._metadata.pop('last_face_error', None)
            self._set_state("completed", "Reference ready from ID card")
            preview = self._preview_text(self._ocr_text)
            notify = self._should_notify()
            return CardProcessingResult(
                status="completed",
                message="Reference extracted from ID card",
                notify=notify,
                reference_ready=True,
                reference_path=str(reference_path),
                text_preview=preview,
                text_path=str(self.text_path) if self.text_path.exists() else None,
                card_snapshot_path=str(self.snapshot_path) if self.snapshot_path.exists() else None,
                card_bbox=self._card_bbox,
                frame_size=self._frame_size,
                metadata=dict(self._metadata),
                card_polygon=list(self._card_polygon) if self._card_polygon else None,
            )

    def _extract_and_save_face(self, card_image: np.ndarray) -> Path:
        rgb_card = cv2.cvtColor(card_image, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_card, model="hog")
        if not locations:
            locations = face_recognition.face_locations(
                rgb_card, model="hog", number_of_times_to_upsample=1
            )

        if not locations:
            raise CardFaceNotFoundError(
                "No face detected on the ID card. Please hold the card closer and retry."
            )

        def _area(loc: Tuple[int, int, int, int]) -> int:
            top, right, bottom, left = loc
            return (bottom - top) * (right - left)

        best_location = max(locations, key=_area)
        top, right, bottom, left = best_location

        margin_y = int((bottom - top) * 0.15)
        margin_x = int((right - left) * 0.15)

        top = max(top - margin_y, 0)
        bottom = min(bottom + margin_y, card_image.shape[0])
        left = max(left - margin_x, 0)
        right = min(right + margin_x, card_image.shape[1])

        face_crop = card_image[top:bottom, left:right]
        if face_crop.size == 0:
            raise RuntimeError("Failed to crop face from card image")

        target_width = 512
        if face_crop.shape[1] < target_width:
            scale = target_width / face_crop.shape[1]
            face_crop = cv2.resize(face_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(str(self.reference_path), face_crop)
        return self.reference_path

    @staticmethod
    def _preview_text(text: Optional[str], max_chars: int = 240) -> Optional[str]:
        if not text:
            return None
        cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return textwrap.shorten(cleaned, width=max_chars, placeholder="…")

