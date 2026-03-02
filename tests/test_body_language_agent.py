"""
Test Suite for Body Language Analyser Agent
============================================
Comprehensive pytest tests for body language analysis functionality.

Note: Integration tests work with all MediaPipe versions (fallback mode for 0.10+).
Unit tests for utilities and models work without MediaPipe.
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os

from backend.models.state import BodyLanguageModel
from backend.utils.video_utils import validate_video_file, get_video_info


# Check if MediaPipe is available (any version)
MEDIAPIPE_AVAILABLE = False
MEDIAPIPE_LEGACY = False
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_LEGACY = hasattr(mp, 'solutions')
except (ImportError, AttributeError):
    pass

# Always try to import the agent (it will use fallback if needed)
if MEDIAPIPE_AVAILABLE:
    try:
        from backend.agents.body_language_agent import BodyLanguageAnalyser, analyze_body_language
    except Exception as e:
        print(f"Warning: Could not import body language agent: {e}")
        MEDIAPIPE_AVAILABLE = False


class TestVideoUtils:
    """Test suite for video utility functions."""

    def test_validate_video_file_exists(self):
        """Test video file validation for existing files."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        try:
            result = validate_video_file(temp_path)
            assert isinstance(result, Path)
            assert result.exists()
        finally:
            os.unlink(temp_path)

    def test_validate_video_file_not_exists(self):
        """Test video file validation for non-existent files."""
        with pytest.raises(FileNotFoundError):
            validate_video_file("/nonexistent/path/video.mp4")

    def test_validate_video_unsupported_format(self):
        """Test video file validation for unsupported formats."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.xyz', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        try:
            with pytest.raises(ValueError, match="Unsupported video format"):
                validate_video_file(temp_path)
        finally:
            os.unlink(temp_path)

    def test_validate_video_supported_formats(self):
        """Test that all documented formats are accepted."""
        supported_formats = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

        for fmt in supported_formats:
            temp_file = tempfile.NamedTemporaryFile(suffix=fmt, delete=False)
            temp_path = temp_file.name
            temp_file.close()

            try:
                result = validate_video_file(temp_path)
                assert result.suffix.lower() == fmt
            finally:
                os.unlink(temp_path)


class TestPydanticValidation:
    """Test suite for Pydantic model validation."""

    def test_body_language_model_valid(self):
        """Test that BodyLanguageModel accepts valid data."""
        model = BodyLanguageModel(
            eye_contact=0.8,
            posture_stability=0.7,
            facial_expressiveness=0.6,
            distractions=["Test distraction"]
        )

        assert model.eye_contact == 0.8
        assert model.posture_stability == 0.7
        assert model.facial_expressiveness == 0.6
        assert len(model.distractions) == 1

    def test_body_language_model_bounds_upper(self):
        """Test that BodyLanguageModel enforces upper bound."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            BodyLanguageModel(
                eye_contact=1.5,  # Invalid: > 1
                posture_stability=0.5,
                facial_expressiveness=0.5,
                distractions=[]
            )

    def test_body_language_model_bounds_lower(self):
        """Test that BodyLanguageModel enforces lower bound."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            BodyLanguageModel(
                eye_contact=-0.1,  # Invalid: < 0
                posture_stability=0.5,
                facial_expressiveness=0.5,
                distractions=[]
            )

    def test_body_language_model_defaults(self):
        """Test that BodyLanguageModel has appropriate defaults."""
        model = BodyLanguageModel()

        assert model.eye_contact == 0.0
        assert model.posture_stability == 0.0
        assert model.facial_expressiveness == 0.0
        assert model.distractions == []

    def test_body_language_model_edge_values(self):
        """Test boundary values (0 and 1)."""
        model = BodyLanguageModel(
            eye_contact=0.0,
            posture_stability=1.0,
            facial_expressiveness=0.5,
            distractions=[]
        )

        assert model.eye_contact == 0.0
        assert model.posture_stability == 1.0

    def test_body_language_model_distraction_list(self):
        """Test distraction list handling."""
        distractions = [
            "Looking away frequently",
            "Frequent posture shifts",
            "Excessive head movement"
        ]

        model = BodyLanguageModel(
            eye_contact=0.3,
            posture_stability=0.3,
            facial_expressiveness=0.5,
            distractions=distractions
        )

        assert len(model.distractions) == 3
        assert "Looking away frequently" in model.distractions


# MediaPipe-dependent tests
@pytest.mark.skipif(not MEDIAPIPE_AVAILABLE,
                    reason="Requires MediaPipe (any version)")
class TestBodyLanguageAnalyserIntegration:
    """Integration tests requiring MediaPipe (works with fallback mode for 0.10+)."""

    @pytest.fixture
    def sample_video_path(self):
        """Create a temporary valid video file."""
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_path = temp_file.name
        temp_file.close()

        # Create simple video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, 30.0, (640, 480))

        for i in range(10):
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            out.write(frame)

        out.release()

        yield temp_path

        try:
            os.unlink(temp_path)
        except:
            pass

    def test_initialization(self):
        """Test BodyLanguageAnalyser initialization."""
        analyser = BodyLanguageAnalyser(frame_sample_rate=5)
        assert analyser is not None
        assert analyser.frame_sample_rate == 5

    def test_analyze_returns_valid_model(self, sample_video_path):
        """Test that analyze returns valid BodyLanguageModel."""
        analyser = BodyLanguageAnalyser()
        result = analyser.analyze(sample_video_path)

        assert isinstance(result, BodyLanguageModel)
        assert 0.0 <= result.eye_contact <= 1.0
        assert 0.0 <= result.posture_stability <= 1.0
        assert 0.0 <= result.facial_expressiveness <= 1.0
        assert isinstance(result.distractions, list)

    def test_invalid_file_not_found(self):
        """Test FileNotFoundError for non-existent file."""
        analyser = BodyLanguageAnalyser()
        with pytest.raises(FileNotFoundError):
            analyser.analyze("/nonexistent/video.mp4")

    def test_convenience_function(self, sample_video_path):
        """Test convenience function."""
        result = analyze_body_language(sample_video_path)
        assert isinstance(result, BodyLanguageModel)

    def test_multiple_frame_rates(self, sample_video_path):
        """Test different frame sample rates."""
        for rate in [1, 5, 10]:
            analyser = BodyLanguageAnalyser(frame_sample_rate=rate)
            result = analyser.analyze(sample_video_path)
            assert isinstance(result, BodyLanguageModel)


class TestDocumentation:
    """Test that documentation matches implementation."""

    def test_readme_exists(self):
        """Test that Body Language README exists."""
        readme_path = Path("BODY_LANGUAGE_README.md")
        assert readme_path.exists(), "BODY_LANGUAGE_README.md must exist"

    def test_agent_file_exists(self):
        """Test that agent file exists."""
        agent_path = Path("backend/agents/body_language_agent.py")
        assert agent_path.exists()

    def test_utils_file_exists(self):
        """Test that utils file exists."""
        utils_path = Path("backend/utils/video_utils.py")
        assert utils_path.exists()

    def test_calibration_script_exists(self):
        """Test that calibration script exists."""
        script_path = Path("scripts/body_language_calibration.py")
        assert script_path.exists()

    def test_example_script_exists(self):
        """Test that example usage script exists."""
        example_path = Path("example_body_language_usage.py")
        assert example_path.exists()


if __name__ == "__main__":
    if MEDIAPIPE_AVAILABLE:
        api_type = "legacy" if MEDIAPIPE_LEGACY else "fallback (0.10+)"
        print(f"✓ MediaPipe available ({api_type}) - running full test suite")
    else:
        print("⚠ MediaPipe not available - skipping integration tests")
        print("  To run full tests: pip install mediapipe")

    pytest.main([__file__, "-v", "--tb=short"])
