"""
Body Language Analyser Agent
==============================
Computer vision signal extraction agent for interview performance analysis.

Extracts objective posture and movement metrics from video input using
OpenCV and MediaPipe (NO LLM usage for metric extraction).

Key Metrics:
- Eye Contact: Face orientation toward camera
- Posture Stability: Torso angle variance
- Facial Expressiveness: Facial landmark movement
- Distractions: Rule-based detection of concerning behaviors
"""

import logging
from typing import List, Tuple, Optional
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from pydantic import BaseModel, Field

from backend.models.state import BodyLanguageModel
from backend.utils.video_utils import (
    validate_video_file,
    load_video_frames,
    calculate_variance_normalized,
    calculate_stability_from_variance
)

logger = logging.getLogger(__name__)


# MediaPipe API compatibility layer
# MediaPipe 0.10+ changed from mp.solutions to mp.tasks
MEDIAPIPE_LEGACY = False
MEDIAPIPE_VERSION = mp.__version__

try:
    # Try legacy API (MediaPipe < 0.10)
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    MEDIAPIPE_LEGACY = True
    logger.info(f"Using MediaPipe {MEDIAPIPE_VERSION} legacy API (solutions)")
except AttributeError:
    # Use new API (MediaPipe >= 0.10) - legacy API not available
    # Set to None, will use fallback implementation
    mp_pose = None
    mp_face_mesh = None
    MEDIAPIPE_LEGACY = False
    logger.info(
        f"Using MediaPipe {MEDIAPIPE_VERSION} with fallback implementation (legacy API not available)")


class BodyLanguageAnalyser:
    """
    Deterministic body language analysis using computer vision.

    Uses MediaPipe Pose and FaceMesh for objective metric extraction.
    NO LLM involvement - pure computer vision processing.
    """

    def __init__(
        self,
        frame_sample_rate: int = 5,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ):
        """
        Initialize the body language analyser.

        Args:
            frame_sample_rate: Process every Nth frame (default: 5)
            min_detection_confidence: MediaPipe detection confidence threshold
            min_tracking_confidence: MediaPipe tracking confidence threshold
        """
        self.frame_sample_rate = frame_sample_rate
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        if MEDIAPIPE_LEGACY:
            # Initialize MediaPipe Pose (legacy API)
            self.mp_pose = mp_pose
            self.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )

            # Initialize MediaPipe Face Mesh (legacy API)
            self.mp_face_mesh = mp_face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
        else:
            # New API (MediaPipe 0.10+) - uses tasks
            # Note: tasks API requires different initialization
            # For now, use simplified approach with drawing utils
            self.pose = None
            self.face_mesh = None
            logger.warning(
                "MediaPipe 0.10+ detected. Using fallback implementation. "
                "For best results, use MediaPipe < 0.10 with Python 3.11/3.12"
            )

        logger.info(
            f"BodyLanguageAnalyser initialized (MediaPipe {MEDIAPIPE_VERSION}, legacy={MEDIAPIPE_LEGACY})")

    def analyze(self, video_file_path: str) -> BodyLanguageModel:
        """
        Analyze body language from a video file.

        Args:
            video_file_path: Path to video file (.mp4, etc.)

        Returns:
            BodyLanguageModel with extracted metrics

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video format is unsupported or cannot be processed
        """
        # Step 1: Validate video file
        validate_video_file(video_file_path)
        logger.info(f"Analyzing body language from: {video_file_path}")

        # Step 2: Load video frames
        frames, video_info = load_video_frames(
            video_file_path,
            sample_rate=self.frame_sample_rate
        )

        if not frames:
            logger.warning(
                "No frames extracted from video, returning default metrics")
            return BodyLanguageModel(
                eye_contact=0.0,
                posture_stability=0.0,
                facial_expressiveness=0.0,
                distractions=["No video frames detected"]
            )

        logger.info(f"Processing {len(frames)} frames...")

        # Check if MediaPipe is properly initialized
        if not MEDIAPIPE_LEGACY:
            logger.warning(
                "MediaPipe legacy API not available. "
                "Returning fallback metrics based on frame analysis."
            )
            return self._analyze_fallback(frames)

        # Step 3: Extract metrics from frames (legacy API)
        eye_contact = self._calculate_eye_contact(frames)
        posture_stability = self._calculate_posture_stability(frames)
        facial_expressiveness = self._calculate_facial_expressiveness(frames)

        # Step 4: Detect distractions based on metrics
        distractions = self._detect_distractions(
            eye_contact,
            posture_stability,
            facial_expressiveness,
            frames
        )

        logger.info(
            f"Analysis complete - Eye Contact: {eye_contact:.2f}, "
            f"Posture: {posture_stability:.2f}, "
            f"Expressiveness: {facial_expressiveness:.2f}"
        )

        return BodyLanguageModel(
            eye_contact=eye_contact,
            posture_stability=posture_stability,
            facial_expressiveness=facial_expressiveness,
            distractions=distractions
        )

    def _analyze_fallback(self, frames: List[np.ndarray]) -> BodyLanguageModel:
        """
        Fallback analysis when MediaPipe legacy API is not available.

        Uses basic computer vision techniques to estimate metrics.
        Results are less accurate than full MediaPipe analysis.

        Args:
            frames: List of video frames (BGR format)

        Returns:
            BodyLanguageModel with estimated metrics
        """
        logger.info("Using fallback analysis (basic computer vision)")

        # Simple frame-based analysis
        frame_differences = []
        face_detected_count = 0

        # Use OpenCV's face detection as fallback
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

        prev_frame_gray = None

        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) > 0:
                face_detected_count += 1

            # Calculate frame difference (movement)
            if prev_frame_gray is not None:
                diff = cv2.absdiff(prev_frame_gray, gray)
                frame_differences.append(np.mean(diff))

            prev_frame_gray = gray

        # Estimate metrics
        total_frames = len(frames)

        # Eye contact: based on face detection rate
        eye_contact = face_detected_count / total_frames if total_frames > 0 else 0.0

        # Posture stability: inverse of movement variance
        if frame_differences:
            movement_variance = np.var(frame_differences)
            normalized_variance = min(movement_variance / 100.0, 1.0)
            posture_stability = 1.0 - normalized_variance
        else:
            posture_stability = 0.5

        # Facial expressiveness: based on average frame difference
        if frame_differences:
            avg_movement = np.mean(frame_differences)
            facial_expressiveness = min(avg_movement / 10.0, 1.0)
        else:
            facial_expressiveness = 0.5

        # Detect distractions
        distractions = []
        if eye_contact < 0.4:
            distractions.append("Low face detection rate")
        if posture_stability < 0.4:
            distractions.append("High movement variance")

        logger.info(
            f"Fallback analysis complete - Eye Contact: {eye_contact:.2f}, "
            f"Posture: {posture_stability:.2f}, "
            f"Expressiveness: {facial_expressiveness:.2f}"
        )

        return BodyLanguageModel(
            eye_contact=eye_contact,
            posture_stability=posture_stability,
            facial_expressiveness=facial_expressiveness,
            distractions=distractions
        )

    def _calculate_eye_contact(self, frames: List[np.ndarray]) -> float:
        """
        Calculate eye contact score based on face orientation toward camera.

        High score = face frequently directed toward camera.

        Args:
            frames: List of video frames (BGR format)

        Returns:
            Eye contact score (0-1)
        """
        frames_facing_camera = 0
        total_processed = 0

        for frame in frames:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with FaceMesh
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                # Face detected
                face_landmarks = results.multi_face_landmarks[0]

                # Calculate head orientation using key landmarks
                # Nose tip (1), Left eye (33), Right eye (263), Chin (152)
                nose_tip = face_landmarks.landmark[1]
                left_eye = face_landmarks.landmark[33]
                right_eye = face_landmarks.landmark[263]

                # Calculate horizontal deviation (left/right turn)
                eye_center_x = (left_eye.x + right_eye.x) / 2
                horizontal_deviation = abs(nose_tip.x - eye_center_x)

                # Face is "looking at camera" if deviation is small
                # Threshold: < 0.08 is considered facing camera
                if horizontal_deviation < 0.08:
                    frames_facing_camera += 1

                total_processed += 1

        if total_processed == 0:
            logger.warning("No faces detected in video")
            return 0.0

        eye_contact_ratio = frames_facing_camera / total_processed
        return float(eye_contact_ratio)

    def _calculate_posture_stability(self, frames: List[np.ndarray]) -> float:
        """
        Calculate posture stability based on torso angle variance.

        High score = stable posture with minimal shifting.

        Args:
            frames: List of video frames (BGR format)

        Returns:
            Posture stability score (0-1)
        """
        torso_angles = []

        for frame in frames:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with Pose
            results = self.pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Calculate torso angle using shoulders and hips
                # Left shoulder (11), Right shoulder (12)
                # Left hip (23), Right hip (24)

                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
                left_hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP]
                right_hip = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

                # Calculate midpoints
                shoulder_midpoint_y = (left_shoulder.y + right_shoulder.y) / 2
                hip_midpoint_y = (left_hip.y + right_hip.y) / 2
                shoulder_midpoint_x = (left_shoulder.x + right_shoulder.x) / 2
                hip_midpoint_x = (left_hip.x + right_hip.x) / 2

                # Calculate torso angle (in degrees)
                delta_y = hip_midpoint_y - shoulder_midpoint_y
                delta_x = hip_midpoint_x - shoulder_midpoint_x

                if delta_y != 0:
                    angle = np.arctan2(delta_x, delta_y) * 180 / np.pi
                    torso_angles.append(abs(angle))

        if len(torso_angles) < 2:
            logger.warning("Insufficient pose data for posture stability")
            return 0.5  # Neutral score if no data

        # Calculate variance and convert to stability
        variance = np.var(torso_angles)

        # Normalize: typical variance ranges from 0 to 100 for angles
        normalized_variance = min(variance / 100.0, 1.0)
        stability = calculate_stability_from_variance(normalized_variance)

        return float(stability)

    def _calculate_facial_expressiveness(self, frames: List[np.ndarray]) -> float:
        """
        Calculate facial expressiveness based on facial landmark movement.

        Moderate variability is preferred (too little = monotone, too much = excessive).

        Args:
            frames: List of video frames (BGR format)

        Returns:
            Facial expressiveness score (0-1)
        """
        landmark_positions = []  # Store landmark positions over time

        for frame in frames:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process with FaceMesh
            results = self.face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                # Track key expressive landmarks (mouth and eyebrows)
                # Upper lip (13), Lower lip (14), Left eyebrow (70), Right eyebrow (300)
                key_landmarks = [13, 14, 70, 300]

                frame_positions = []
                for idx in key_landmarks:
                    lm = face_landmarks.landmark[idx]
                    frame_positions.extend([lm.x, lm.y, lm.z])

                landmark_positions.append(frame_positions)

        if len(landmark_positions) < 2:
            logger.warning("Insufficient facial landmark data")
            return 0.5  # Neutral score

        # Calculate movement variance across frames
        positions_array = np.array(landmark_positions)
        movement_variance = np.var(positions_array, axis=0).mean()

        # Normalize: typical variance ranges from 0 to 0.01 for normalized coordinates
        normalized_movement = min(movement_variance / 0.01, 1.0)

        # Optimal expressiveness is moderate (around 0.5)
        # Map to a score where 0.5 variance = 1.0 score
        if normalized_movement < 0.5:
            # Low movement: scale linearly
            expressiveness = normalized_movement * 2
        else:
            # High movement: penalize excess
            expressiveness = max(0.0, 2.0 - normalized_movement * 2)

        return float(np.clip(expressiveness, 0.0, 1.0))

    def _detect_distractions(
        self,
        eye_contact: float,
        posture_stability: float,
        facial_expressiveness: float,
        frames: List[np.ndarray]
    ) -> List[str]:
        """
        Detect distracting behaviors based on rule-based thresholds.

        Args:
            eye_contact: Eye contact score (0-1)
            posture_stability: Posture stability score (0-1)
            facial_expressiveness: Facial expressiveness score (0-1)
            frames: List of video frames (for additional analysis)

        Returns:
            List of distraction descriptions
        """
        distractions = []

        # Rule 1: Low eye contact
        if eye_contact < 0.4:
            distractions.append("Looking away frequently")

        # Rule 2: Unstable posture
        if posture_stability < 0.4:
            distractions.append("Frequent posture shifts")

        # Rule 3: Excessive head movement (detected via posture variance)
        # This is already captured in posture stability, but we add specific message
        if posture_stability < 0.3:
            distractions.append("Excessive head movement")

        # Rule 4: Very low expressiveness (monotone)
        if facial_expressiveness < 0.2:
            distractions.append("Limited facial expression")

        # Rule 5: Excessive expressiveness
        if facial_expressiveness > 0.85:
            distractions.append("Overly animated expressions")

        return distractions

    def __del__(self):
        """Cleanup MediaPipe resources."""
        try:
            self.pose.close()
            self.face_mesh.close()
        except:
            pass


# Convenience function for standalone usage
def analyze_body_language(video_file_path: str, frame_sample_rate: int = 5) -> BodyLanguageModel:
    """
    Convenience function to analyze body language from a video file.

    Args:
        video_file_path: Path to video file
        frame_sample_rate: Process every Nth frame (default: 5)

    Returns:
        BodyLanguageModel with extracted metrics
    """
    analyser = BodyLanguageAnalyser(frame_sample_rate=frame_sample_rate)
    return analyser.analyze(video_file_path)


# LangGraph node wrapper
def body_language_node(state: "InterviewState") -> dict:
    """
    LangGraph node wrapper for BodyLanguageAnalyser.

    Args:
        state: InterviewState object containing:
            - video_path: Path to video file for analysis

    Returns:
        Dictionary with body_language field for LangGraph state merge
    """
    from backend.models.state import InterviewState, BodyLanguageModel

    logger.info(
        f"BodyLanguageNode: Starting for interview_id={state.interview_id}")

    try:
        # Check if video path is provided
        if not state.video_path:
            logger.warning(
                "BodyLanguageNode: No video_path provided, using defaults")
            return {
                "body_language": BodyLanguageModel(
                    eye_contact=0.5,
                    posture_stability=0.5,
                    facial_expressiveness=0.5,
                    distractions=[]
                )
            }

        # Initialize analyser and run analysis
        analyser = BodyLanguageAnalyser(frame_sample_rate=5)
        result = analyser.analyze(state.video_path)

        logger.info(
            f"BodyLanguageNode: Completed - "
            f"eye_contact={result.eye_contact:.2f}, "
            f"posture_stability={result.posture_stability:.2f}, "
            f"facial_expressiveness={result.facial_expressiveness:.2f}, "
            f"distractions={result.distractions}"
        )

        return {"body_language": result}

    except Exception as e:
        logger.error(f"BodyLanguageNode: Failed - {e}", exc_info=True)
        # Return default values on error
        return {
            "body_language": BodyLanguageModel(
                eye_contact=0.5,
                posture_stability=0.5,
                facial_expressiveness=0.5,
                distractions=["analysis_error"]
            )
        }
