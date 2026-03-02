"""
Voice Agent Test Suite
=======================
Comprehensive pytest tests for VoiceAgent.

Tests:
- Valid audio processing
- Silent audio handling
- High filler word detection
- Speech rate calculation
- Invalid file handling
- Tone classification
- Edge cases
"""

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path
import tempfile
import os

from backend.agents.voice_agent import VoiceAgent, VoiceAnalysisInput, FILLER_WORDS
from backend.models.state import VoiceAnalysisModel, InterviewState


# Test fixtures

@pytest.fixture
def temp_audio_dir():
    """Create temporary directory for test audio files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_audio_file(temp_audio_dir):
    """Create a basic test audio file."""
    # Generate 3 seconds of sine wave (440 Hz A note)
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)

    filepath = temp_audio_dir / "test_audio.wav"
    sf.write(str(filepath), audio, sr)

    return str(filepath)


@pytest.fixture
def silent_audio_file(temp_audio_dir):
    """Create a silent audio file."""
    sr = 16000
    duration = 2.0
    audio = np.zeros(int(sr * duration))

    filepath = temp_audio_dir / "silent_audio.wav"
    sf.write(str(filepath), audio, sr)

    return str(filepath)


@pytest.fixture
def short_audio_file(temp_audio_dir):
    """Create a very short audio file (0.5 seconds)."""
    sr = 16000
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.3 * np.sin(2 * np.pi * 440 * t)

    filepath = temp_audio_dir / "short_audio.wav"
    sf.write(str(filepath), audio, sr)

    return str(filepath)


@pytest.fixture
def voice_agent():
    """Create VoiceAgent instance with tiny model for faster tests."""
    return VoiceAgent(model_size="tiny")


# Test: Valid Audio Processing

def test_voice_agent_initialization():
    """Test that VoiceAgent initializes correctly."""
    agent = VoiceAgent(model_size="tiny")

    assert agent is not None
    assert agent.model_size == "tiny"
    assert agent.whisper_model is not None


def test_analyze_returns_valid_model(voice_agent, sample_audio_file):
    """Test that analyze() returns valid VoiceAnalysisModel."""
    result = voice_agent.analyze(sample_audio_file)

    assert isinstance(result, VoiceAnalysisModel)
    assert hasattr(result, 'speech_rate_wpm')
    assert hasattr(result, 'filler_ratio')
    assert hasattr(result, 'clarity')
    assert hasattr(result, 'tone')


def test_speech_rate_wpm_in_range(voice_agent, sample_audio_file):
    """Test that speech_rate_wpm is a reasonable value."""
    result = voice_agent.analyze(sample_audio_file)

    # Should be >= 0 (could be 0 for audio with no speech)
    assert result.speech_rate_wpm >= 0.0
    # Should not be absurdly high
    assert result.speech_rate_wpm < 500.0


def test_filler_ratio_in_range(voice_agent, sample_audio_file):
    """Test that filler_ratio is between 0 and 1."""
    result = voice_agent.analyze(sample_audio_file)

    assert 0.0 <= result.filler_ratio <= 1.0


def test_clarity_in_range(voice_agent, sample_audio_file):
    """Test that clarity score is between 0 and 1."""
    result = voice_agent.analyze(sample_audio_file)

    assert 0.0 <= result.clarity <= 1.0


def test_tone_valid_category(voice_agent, sample_audio_file):
    """Test that tone is one of the valid categories."""
    result = voice_agent.analyze(sample_audio_file)

    valid_tones = {"confident", "neutral", "hesitant", "uncertain"}
    assert result.tone in valid_tones


# Test: Silent Audio Handling

def test_silent_audio_safe_handling(voice_agent, silent_audio_file):
    """Test that silent audio is handled safely without crashes."""
    result = voice_agent.analyze(silent_audio_file)

    assert isinstance(result, VoiceAnalysisModel)
    # Silent audio should have low WPM (Whisper may hallucinate some words)
    assert result.speech_rate_wpm < 50.0
    # Should have neutral or uncertain tone
    assert result.tone in {"neutral", "uncertain"}


def test_silent_audio_returns_defaults(voice_agent, silent_audio_file):
    """Test that silent audio returns reasonable values."""
    result = voice_agent.analyze(silent_audio_file)

    # Speech rate should be very low for silent audio
    assert result.speech_rate_wpm < 50.0
    # Filler ratio should be low
    assert result.filler_ratio <= 0.5
    assert 0.0 <= result.clarity <= 1.0


# Test: Speech Rate Calculation

def test_calculate_speech_rate_basic(voice_agent):
    """Test basic speech rate calculation."""
    transcript = "Hello world this is a test"  # 6 words
    duration = 60.0  # 1 minute

    wpm = voice_agent._calculate_speech_rate(transcript, duration)

    assert wpm == 6.0


def test_calculate_speech_rate_30_seconds(voice_agent):
    """Test speech rate for 30-second audio."""
    transcript = "one two three four five six seven eight nine ten"  # 10 words
    duration = 30.0  # 30 seconds

    wpm = voice_agent._calculate_speech_rate(transcript, duration)

    # 10 words in 30 seconds = 20 WPM
    assert wpm == 20.0


def test_calculate_speech_rate_zero_duration(voice_agent):
    """Test speech rate calculation with zero duration."""
    transcript = "Hello world"
    duration = 0.0

    wpm = voice_agent._calculate_speech_rate(transcript, duration)

    assert wpm == 0.0


def test_calculate_speech_rate_empty_transcript(voice_agent):
    """Test speech rate with empty transcript."""
    transcript = ""
    duration = 60.0

    wpm = voice_agent._calculate_speech_rate(transcript, duration)

    assert wpm == 0.0


# Test: Filler Word Detection

def test_calculate_filler_ratio_no_fillers(voice_agent):
    """Test filler ratio with no filler words."""
    transcript = "The quick brown fox jumps over the lazy dog"

    ratio = voice_agent._calculate_filler_ratio(transcript)

    assert ratio == 0.0


def test_calculate_filler_ratio_with_fillers(voice_agent):
    """Test filler ratio with filler words."""
    transcript = "um I think uh that like you know this is good"
    # Words: um, I, think, uh, that, like, you, know, this, is, good = 11 words
    # Fillers: um, uh, like = 3 fillers (plus "you know" = 1 more)
    # Total: 4 fillers out of 11 words = ~0.36

    ratio = voice_agent._calculate_filler_ratio(transcript)

    # Should detect multiple fillers
    assert ratio > 0.0
    assert ratio <= 1.0


def test_high_filler_audio_detection(voice_agent):
    """Test detection of high filler ratio."""
    # Create transcript with many fillers
    transcript = "um uh like um well uh you know um uh like"  # 10 words, all fillers

    ratio = voice_agent._calculate_filler_ratio(transcript)

    # High filler ratio
    assert ratio > 0.5


def test_filler_ratio_empty_transcript(voice_agent):
    """Test filler ratio with empty transcript."""
    transcript = ""

    ratio = voice_agent._calculate_filler_ratio(transcript)

    assert ratio == 0.0


def test_filler_ratio_multiword_fillers(voice_agent):
    """Test detection of multi-word fillers like 'you know'."""
    transcript = "I think you know this is sort of good you know"

    ratio = voice_agent._calculate_filler_ratio(transcript)

    # Should detect "you know" (appears twice) and "sort of"
    assert ratio > 0.0


# Test: Tone Classification

def test_tone_confident(voice_agent):
    """Test confident tone classification."""
    # Low filler ratio, good speech rate
    speech_rate = 150.0
    filler_ratio = 0.03

    tone = voice_agent._analyze_tone(speech_rate, filler_ratio)

    assert tone == "confident"


def test_tone_hesitant(voice_agent):
    """Test hesitant tone classification (high filler ratio)."""
    speech_rate = 140.0
    filler_ratio = 0.20  # > 0.15

    tone = voice_agent._analyze_tone(speech_rate, filler_ratio)

    assert tone == "hesitant"


def test_tone_uncertain(voice_agent):
    """Test uncertain tone classification (slow speech)."""
    speech_rate = 90.0  # < 110
    filler_ratio = 0.08

    tone = voice_agent._analyze_tone(speech_rate, filler_ratio)

    assert tone == "uncertain"


def test_tone_neutral(voice_agent):
    """Test neutral tone classification."""
    speech_rate = 140.0
    filler_ratio = 0.08  # Not low enough for confident

    tone = voice_agent._analyze_tone(speech_rate, filler_ratio)

    assert tone == "neutral"


def test_tone_boundary_confident(voice_agent):
    """Test confident tone at boundary conditions."""
    # Exactly at 120 WPM, very low fillers
    tone1 = voice_agent._analyze_tone(120.0, 0.04)
    assert tone1 == "confident"

    # Exactly at 180 WPM, very low fillers
    tone2 = voice_agent._analyze_tone(180.0, 0.04)
    assert tone2 == "confident"


def test_tone_boundary_hesitant(voice_agent):
    """Test hesitant tone at boundary."""
    # Exactly at 0.15 filler ratio - should be neutral
    tone1 = voice_agent._analyze_tone(140.0, 0.15)
    assert tone1 == "neutral"

    # Just above 0.15 - should be hesitant
    tone2 = voice_agent._analyze_tone(140.0, 0.16)
    assert tone2 == "hesitant"


# Test: Invalid File Handling

def test_invalid_file_path(voice_agent):
    """Test that non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        voice_agent.analyze("/nonexistent/path/to/audio.wav")


def test_unsupported_format(voice_agent, temp_audio_dir):
    """Test that unsupported format raises ValueError."""
    # Create a text file with .txt extension
    txt_file = temp_audio_dir / "not_audio.txt"
    txt_file.write_text("This is not an audio file")

    with pytest.raises(ValueError, match="Unsupported audio format"):
        voice_agent.analyze(str(txt_file))


def test_invalid_wav_file(voice_agent, temp_audio_dir):
    """Test that invalid WAV file raises ValueError."""
    # Create a file with .wav extension but invalid content
    fake_wav = temp_audio_dir / "fake.wav"
    fake_wav.write_bytes(b"Not a real WAV file")

    with pytest.raises(ValueError):
        voice_agent.analyze(str(fake_wav))


def test_directory_instead_of_file(voice_agent, temp_audio_dir):
    """Test that directory path raises ValueError."""
    with pytest.raises(ValueError, match="Path is not a file"):
        voice_agent.analyze(str(temp_audio_dir))


# Test: Input Validation

def test_voice_analysis_input_validation_success(sample_audio_file):
    """Test that valid input passes validation."""
    input_model = VoiceAnalysisInput(audio_file_path=sample_audio_file)

    assert input_model.audio_file_path == sample_audio_file


def test_voice_analysis_input_validation_empty_path():
    """Test that empty path fails validation."""
    with pytest.raises(ValueError, match="audio_file_path cannot be empty"):
        VoiceAnalysisInput(audio_file_path="")


def test_voice_analysis_input_validation_nonexistent():
    """Test that non-existent file fails validation."""
    with pytest.raises(FileNotFoundError):
        VoiceAnalysisInput(audio_file_path="/nonexistent/file.wav")


# Test: LangGraph Integration (execute method)

def test_execute_with_valid_state(voice_agent, sample_audio_file):
    """Test execute() method with valid InterviewState."""
    state = InterviewState(
        interview_id="test-123",
        role="Software Engineer",
        experience_level="Mid",
        audio_path=sample_audio_file
    )

    updated_state = voice_agent.execute(state)

    assert updated_state.transcript is not None
    assert len(updated_state.transcript) >= 0
    assert updated_state.voice_analysis is not None
    assert isinstance(updated_state.voice_analysis, VoiceAnalysisModel)


def test_execute_without_audio_path(voice_agent):
    """Test execute() raises error when audio_path is missing."""
    state = InterviewState(
        interview_id="test-123",
        role="Software Engineer",
        experience_level="Mid",
        audio_path=None
    )

    with pytest.raises(ValueError, match="audio_path is required"):
        voice_agent.execute(state)


def test_execute_with_nonexistent_file(voice_agent):
    """Test execute() raises error for non-existent file."""
    state = InterviewState(
        interview_id="test-123",
        role="Software Engineer",
        experience_level="Mid",
        audio_path="/nonexistent/audio.wav"
    )

    with pytest.raises(FileNotFoundError):
        voice_agent.execute(state)


# Test: Edge Cases

def test_very_short_audio(voice_agent, short_audio_file):
    """Test analysis of very short audio file."""
    result = voice_agent.analyze(short_audio_file)

    # Should not crash
    assert isinstance(result, VoiceAnalysisModel)
    assert result.speech_rate_wpm >= 0.0


def test_transcription_empty_result(voice_agent, silent_audio_file):
    """Test handling when transcription returns empty string."""
    transcript = voice_agent._transcribe_audio(silent_audio_file)

    # Should return empty or minimal text for silent audio
    assert isinstance(transcript, str)


def test_clarity_analysis_with_silent_audio(voice_agent, silent_audio_file):
    """Test clarity analysis doesn't crash on silent audio."""
    clarity = voice_agent._analyze_clarity(silent_audio_file)

    assert isinstance(clarity, float)
    assert 0.0 <= clarity <= 1.0


# Test: Determinism

def test_analysis_is_deterministic(voice_agent, sample_audio_file):
    """Test that running analysis twice gives same results."""
    result1 = voice_agent.analyze(sample_audio_file)
    result2 = voice_agent.analyze(sample_audio_file)

    # Speech rate and filler ratio should be identical
    assert result1.speech_rate_wpm == result2.speech_rate_wpm
    assert result1.filler_ratio == result2.filler_ratio
    assert result1.tone == result2.tone
    # Clarity might vary slightly due to floating point, but should be close
    assert abs(result1.clarity - result2.clarity) < 0.01


# Test: Model Output Validation

def test_output_model_validation():
    """Test that VoiceAnalysisModel validates constraints."""
    # Valid model
    model = VoiceAnalysisModel(
        speech_rate_wpm=150.0,
        filler_ratio=0.05,
        clarity=0.8,
        tone="confident"
    )

    assert model.speech_rate_wpm == 150.0
    assert model.filler_ratio == 0.05
    assert model.clarity == 0.8
    assert model.tone == "confident"


def test_output_model_boundary_values():
    """Test VoiceAnalysisModel with boundary values."""
    # Minimum values
    model1 = VoiceAnalysisModel(
        speech_rate_wpm=0.0,
        filler_ratio=0.0,
        clarity=0.0,
        tone="uncertain"
    )
    assert model1.speech_rate_wpm == 0.0

    # Maximum values
    model2 = VoiceAnalysisModel(
        speech_rate_wpm=300.0,
        filler_ratio=1.0,
        clarity=1.0,
        tone="confident"
    )
    assert model2.filler_ratio == 1.0
    assert model2.clarity == 1.0
