"""
Body Language Analyser
======================
Analyzes visual behavior from video frames.

Architecture:
- Uses MediaPipe for face and pose detection
- OpenCV for frame processing
- Deterministic algorithmic analysis
- No LLM - pure computer vision
- Outputs: eye_contact, posture_stability, facial_expressiveness
"""

import logging
from typing import Dict, Any, List, Tuple
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from backend.models.state import InterviewState, BodyLanguageModel
from backend.config import settings

logger = logging.getLogger(__name__)


# MediaPipe API compatibility layer
# MediaPipe 0.10+ changed from mp.solutions to mp.tasks
MEDIAPIPE_LEGACY = False
MEDIAPIPE_VERSION = mp.__version__

try:
    # Try legacy API (MediaPipe < 0.10)
    mp_pose = mp.solutions.pose
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    MEDIAPIPE_LEGACY = True
    logger.info(f"Using MediaPipe {MEDIAPIPE_VERSION} legacy API (solutions)")
except AttributeError:
    # Use new API (MediaPipe >= 0.10) - legacy API not available
    # For now, set to None and use fallback
    mp_pose = None
    mp_face_mesh = None
    mp_drawing = None
    MEDIAPIPE_LEGACY = False
    logger.warning(
        f"MediaPipe {MEDIAPIPE_VERSION} legacy API not available - "
        f"body language analysis will use fallback values"
    )


class BodyLanguageAnalyser:
    """
    Agent responsible for body language analysis.

    Components:
    1. Face detection and tracking (MediaPipe Face Mesh)
    2. Pose detection (MediaPipe Pose)
    3. Eye contact estimation (gaze direction)
    4. Posture stability (shoulder and head movement)
    5. Facial expressiveness (landmark movement analysis)

    All metrics are deterministic computer vision algorithms.
    """

    def __init__(self):
        """Initialize MediaPipe models."""
        if not MEDIAPIPE_LEGACY:
            logger.warning(
                "MediaPipe solutions API not available - "
                "body language analysis will return default values"
            )
            self.mp_face_mesh = None
            self.mp_pose = None
            self.mp_drawing = None
            self.face_mesh = None
            self.pose = None
            return

        self.mp_face_mesh = mp_face_mesh
        self.mp_pose = mp_pose
        self.mp_drawing = mp_drawing

        # Initialize detectors
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def _extract_frames(self, video_path: str, sample_rate: int = None) -> List[np.ndarray]:
        """
        Extract frames from video at specified sample rate.

        Args:
            video_path: Path to video file
            sample_rate: Sample 1 frame every N seconds (default from config)

        Returns:
            List of frame arrays
        """
        if sample_rate is None:
            sample_rate = settings.BODY_FRAME_SAMPLE_RATE

        logger.info(
            f"Extracting frames from {video_path} (1 frame per {sample_rate}s)")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * sample_rate)

        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                # Convert BGR to RGB for MediaPipe
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

            frame_count += 1

        cap.release()
        logger.info(f"Extracted {len(frames)} frames")
        return frames

    def _analyze_eye_contact(self, frames: List[np.ndarray]) -> float:
        """
        Estimate eye contact consistency.

        Method:
        - Detect face and iris landmarks
        - Calculate gaze direction (looking at camera vs away)
        - Return ratio of frames with good eye contact

        Returns:
            Eye contact score (0.0 to 1.0)
        """
        if not frames or not self.face_mesh:
            return 0.5  # Default fallback

        good_eye_contact_count = 0
        detected_face_count = 0

        for frame in frames:
            results = self.face_mesh.process(frame)

            if results.multi_face_landmarks:
                detected_face_count += 1
                landmarks = results.multi_face_landmarks[0]

                # Use iris landmarks to estimate gaze direction
                # Landmark indices for left and right iris centers
                left_iris = landmarks.landmark[468]  # Left iris center
                right_iris = landmarks.landmark[473]  # Right iris center

                # Get face center (nose tip)
                nose_tip = landmarks.landmark[1]

                # Calculate horizontal gaze offset
                iris_center_x = (left_iris.x + right_iris.x) / 2
                gaze_offset = abs(iris_center_x - nose_tip.x)

                # Good eye contact if gaze offset is small
                if gaze_offset < 0.08:  # Threshold tuned empirically
                    good_eye_contact_count += 1

        if detected_face_count == 0:
            logger.warning("No faces detected in video")
            return 0.5  # Default fallback

        eye_contact_score = good_eye_contact_count / detected_face_count
        return round(float(eye_contact_score), 3)

    def _analyze_posture_stability(self, frames: List[np.ndarray]) -> float:
        """
        Analyze posture stability and consistency.

        Method:
        - Track shoulder and head landmarks
        - Calculate movement variance across frames
        - Low variance = stable posture

        Returns:
            Posture stability score (0.0 to 1.0)
        """
        if not frames or not self.pose:
            return 0.5  # Default fallback

        shoulder_positions = []
        head_positions = []

        for frame in frames:
            results = self.pose.process(frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # Left and right shoulders
                left_shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]

                # Nose (head position)
                nose = landmarks[self.mp_pose.PoseLandmark.NOSE]

                # Store positions
                shoulder_center = ((left_shoulder.x + right_shoulder.x) / 2,
                                   (left_shoulder.y + right_shoulder.y) / 2)
                shoulder_positions.append(shoulder_center)

                head_positions.append((nose.x, nose.y))

        if len(shoulder_positions) < 2:
            logger.warning("Insufficient pose data")
            return 0.5  # Default moderate score

        # Calculate variance in positions
        shoulder_positions_array = np.array(shoulder_positions)
        head_positions_array = np.array(head_positions)

        shoulder_variance = np.var(shoulder_positions_array)
        head_variance = np.var(head_positions_array)

        # Normalize variance to 0-1 score
        # Lower variance = higher stability
        shoulder_stability = 1.0 - min(shoulder_variance * 50, 1.0)
        head_stability = 1.0 - min(head_variance * 50, 1.0)

        # Combined score
        stability = (shoulder_stability * 0.6) + (head_stability * 0.4)
        return round(float(stability), 3)

    def _analyze_facial_expressiveness(self, frames: List[np.ndarray]) -> float:
        """
        Analyze appropriate facial expressiveness.

        Method:
        - Track facial landmark movements
        - Measure variance in expressions
        - Moderate variance = good (not too stiff, not too animated)

        Returns:
            Expressiveness score (0.0 to 1.0)
        """
        if not frames or not self.face_mesh:
            return 0.5  # Default fallback

        mouth_movements = []
        eyebrow_movements = []

        for frame in frames:
            results = self.face_mesh.process(frame)

            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark

                # Mouth corners
                left_mouth = landmarks[61]
                right_mouth = landmarks[291]
                mouth_width = abs(left_mouth.x - right_mouth.x)
                mouth_movements.append(mouth_width)

                # Eyebrows
                left_eyebrow = landmarks[46]
                right_eyebrow = landmarks[276]
                eyebrow_height = (left_eyebrow.y + right_eyebrow.y) / 2
                eyebrow_movements.append(eyebrow_height)

        if len(mouth_movements) < 2:
            return 0.5  # Default

        # Calculate variance
        mouth_variance = np.var(mouth_movements)
        eyebrow_variance = np.var(eyebrow_movements)

        # Optimal range: moderate variance (not too stiff, not over-animated)
        # Map variance to 0-1 score with peak at moderate values
        def variance_to_score(var: float, optimal: float = 0.001) -> float:
            # Gaussian-like function centered at optimal
            distance = abs(var - optimal)
            score = np.exp(-distance * 100)
            return min(max(score, 0.0), 1.0)

        mouth_score = variance_to_score(mouth_variance)
        eyebrow_score = variance_to_score(eyebrow_variance)

        expressiveness = (mouth_score * 0.6) + (eyebrow_score * 0.4)
        return round(float(expressiveness), 3)

    def _detect_distractions(self, frames: List[np.ndarray]) -> List[str]:
        """
        Detect distracting behaviors.

        Checks:
        - Excessive head movement
        - Looking away frequently
        - Hand movements near face

        Returns:
            List of detected distraction types
        """
        distractions = []

        # This is a simplified implementation
        # In production, you'd want more sophisticated detection

        if len(frames) == 0 or not self.face_mesh:
            return distractions

        # Check for missing face in many frames (looking away)
        faces_detected = sum(
            1 for f in frames if self.face_mesh.process(f).multi_face_landmarks)
        detection_ratio = faces_detected / len(frames)

        if detection_ratio < 0.7:
            distractions.append("frequently_looking_away")

        return distractions

    def execute(self, state: InterviewState) -> InterviewState:
        """
        Execute body language analysis pipeline.

        Steps:
        1. Extract frames from video
        2. Analyze eye contact
        3. Analyze posture stability
        4. Analyze facial expressiveness
        5. Detect distractions

        Args:
            state: Must contain video_path

        Returns:
            Updated state with body_language populated
        """
        # If MediaPipe not available, use fallback values
        if not MEDIAPIPE_LEGACY or self.face_mesh is None:
            logger.warning(
                "MediaPipe not available - using default body language values"
            )
            updated_state = state.model_copy(deep=True)
            updated_state.body_language = BodyLanguageModel(
                eye_contact=0.5,
                posture_stability=0.5,
                facial_expressiveness=0.5,
                distractions=[]
            )
            return updated_state

        if not state.video_path:
            logger.warning("No video path provided - using default values")
            # Return state with default values
            updated_state = state.model_copy(deep=True)
            updated_state.body_language = BodyLanguageModel(
                eye_contact=0.5,
                posture_stability=0.5,
                facial_expressiveness=0.5,
                distractions=[]
            )
            return updated_state

        video_path = state.video_path

        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            # Use fallback instead of raising error
            updated_state = state.model_copy(deep=True)
            updated_state.body_language = BodyLanguageModel(
                eye_contact=0.5,
                posture_stability=0.5,
                facial_expressiveness=0.5,
                distractions=[]
            )
            return updated_state

        logger.info(f"Starting body language analysis for: {video_path}")

        # Extract frames
        frames = self._extract_frames(video_path)

        if not frames:
            logger.warning("No frames extracted - using default scores")
            body_language = BodyLanguageModel(
                eye_contact=0.5,
                posture_stability=0.5,
                facial_expressiveness=0.5,
                distractions=[]
            )
        else:
            # Analyze metrics
            eye_contact = self._analyze_eye_contact(frames)
            posture_stability = self._analyze_posture_stability(frames)
            facial_expressiveness = self._analyze_facial_expressiveness(frames)
            distractions = self._detect_distractions(frames)

            body_language = BodyLanguageModel(
                eye_contact=eye_contact,
                posture_stability=posture_stability,
                facial_expressiveness=facial_expressiveness,
                distractions=distractions
            )

        # Update state
        updated_state = state.model_copy(deep=True)
        updated_state.body_language = body_language

        logger.info(
            f"Body language analysis complete: eye_contact={body_language.eye_contact}, "
            f"posture={body_language.posture_stability}"
        )

        return updated_state


# LangGraph node wrapper
def body_language_node(state: InterviewState) -> InterviewState:
    """LangGraph node wrapper for BodyLanguageAnalyser."""
    agent = BodyLanguageAnalyser()
    updated_state = agent.execute(state)
    return {"body_language": updated_state.body_language}
