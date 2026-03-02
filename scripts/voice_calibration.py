"""
Voice Agent Calibration Script
===============================
Validates Voice Agent metrics across sample audio files.

Purpose:
- Verify metrics are within expected ranges
- Detect abnormal values
- Provide statistics on voice analysis performance

Usage:
    python scripts/voice_calibration.py

Requirements:
- Sample audio files in data/uploads/audio/ (or custom directory)
- Whisper model downloaded
"""

from backend.utils import audio_utils
from backend.models.state import VoiceAnalysisModel
from backend.agents.voice_agent import VoiceAgent
import sys
import os
from pathlib import Path
from typing import List, Dict
import statistics

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Calibration thresholds
SPEECH_RATE_WARNING_MIN = 80.0
SPEECH_RATE_WARNING_MAX = 220.0
FILLER_RATIO_WARNING = 0.4
CLARITY_WARNING_MIN = 0.2


def find_sample_audio_files(directory: str = "data/uploads/audio") -> List[str]:
    """
    Find all audio files in the specified directory.

    Args:
        directory: Directory to search for audio files

    Returns:
        List of audio file paths
    """
    audio_dir = Path(directory)

    if not audio_dir.exists():
        print(f"⚠️  Directory not found: {directory}")
        print("Creating sample directory structure...")
        audio_dir.mkdir(parents=True, exist_ok=True)
        return []

    # Supported extensions
    extensions = ['.wav', '.mp3', '.m4a', '.ogg', '.flac']

    audio_files = []
    for ext in extensions:
        audio_files.extend(audio_dir.glob(f"*{ext}"))

    return [str(f) for f in audio_files]


def analyze_audio_file(agent: VoiceAgent, audio_path: str) -> Dict:
    """
    Analyze single audio file and return metrics.

    Args:
        agent: VoiceAgent instance
        audio_path: Path to audio file

    Returns:
        Dictionary with metrics and metadata
    """
    print(f"\n📊 Analyzing: {Path(audio_path).name}")

    try:
        # Get audio info
        duration = audio_utils.get_audio_duration(audio_path)

        # Run analysis
        result = agent.analyze(audio_path)

        print(f"   Duration: {duration:.2f}s")
        print(f"   Speech Rate: {result.speech_rate_wpm:.2f} WPM")
        print(
            f"   Filler Ratio: {result.filler_ratio:.4f} ({result.filler_ratio * 100:.2f}%)")
        print(f"   Clarity: {result.clarity:.3f}")
        print(f"   Tone: {result.tone}")

        return {
            'path': audio_path,
            'duration': duration,
            'speech_rate_wpm': result.speech_rate_wpm,
            'filler_ratio': result.filler_ratio,
            'clarity': result.clarity,
            'tone': result.tone,
            'success': True,
            'error': None
        }

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return {
            'path': audio_path,
            'success': False,
            'error': str(e)
        }


def print_statistics(results: List[Dict]):
    """
    Print summary statistics and warnings.

    Args:
        results: List of analysis results
    """
    # Filter successful results
    successful = [r for r in results if r['success']]

    if not successful:
        print("\n❌ No successful analyses to report.")
        return

    # Extract metrics
    speech_rates = [r['speech_rate_wpm']
                    for r in successful if r['speech_rate_wpm'] > 0]
    filler_ratios = [r['filler_ratio'] for r in successful]
    clarity_scores = [r['clarity'] for r in successful]
    tones = [r['tone'] for r in successful]

    print("\n" + "="*60)
    print("📈 CALIBRATION SUMMARY")
    print("="*60)

    print(f"\n✅ Successfully analyzed: {len(successful)}/{len(results)} files")

    # Speech Rate Statistics
    if speech_rates:
        print(f"\n🗣️  SPEECH RATE (WPM)")
        print(f"   Mean:   {statistics.mean(speech_rates):.2f}")
        print(f"   Median: {statistics.median(speech_rates):.2f}")
        print(f"   Min:    {min(speech_rates):.2f}")
        print(f"   Max:    {max(speech_rates):.2f}")
        if len(speech_rates) > 1:
            print(f"   StdDev: {statistics.stdev(speech_rates):.2f}")

        # Warnings
        abnormal_rates = [r for r in speech_rates
                          if r < SPEECH_RATE_WARNING_MIN or r > SPEECH_RATE_WARNING_MAX]
        if abnormal_rates:
            print(
                f"   ⚠️  WARNING: {len(abnormal_rates)} file(s) with abnormal speech rate")
            print(
                f"      Expected range: {SPEECH_RATE_WARNING_MIN}-{SPEECH_RATE_WARNING_MAX} WPM")

    # Filler Ratio Statistics
    if filler_ratios:
        print(f"\n💬 FILLER RATIO")
        print(
            f"   Mean:   {statistics.mean(filler_ratios):.4f} ({statistics.mean(filler_ratios)*100:.2f}%)")
        print(
            f"   Median: {statistics.median(filler_ratios):.4f} ({statistics.median(filler_ratios)*100:.2f}%)")
        print(f"   Min:    {min(filler_ratios):.4f}")
        print(f"   Max:    {max(filler_ratios):.4f}")
        if len(filler_ratios) > 1:
            print(f"   StdDev: {statistics.stdev(filler_ratios):.4f}")

        # Warnings
        high_filler = [r for r in filler_ratios if r > FILLER_RATIO_WARNING]
        if high_filler:
            print(
                f"   ⚠️  WARNING: {len(high_filler)} file(s) with very high filler ratio (>{FILLER_RATIO_WARNING})")

    # Clarity Statistics
    if clarity_scores:
        print(f"\n🔊 CLARITY SCORE")
        print(f"   Mean:   {statistics.mean(clarity_scores):.3f}")
        print(f"   Median: {statistics.median(clarity_scores):.3f}")
        print(f"   Min:    {min(clarity_scores):.3f}")
        print(f"   Max:    {max(clarity_scores):.3f}")
        if len(clarity_scores) > 1:
            print(f"   StdDev: {statistics.stdev(clarity_scores):.3f}")

        # Warnings
        low_clarity = [r for r in clarity_scores if r < CLARITY_WARNING_MIN]
        if low_clarity:
            print(
                f"   ⚠️  WARNING: {len(low_clarity)} file(s) with low clarity (<{CLARITY_WARNING_MIN})")

    # Tone Distribution
    if tones:
        print(f"\n🎭 TONE DISTRIBUTION")
        tone_counts = {}
        for tone in tones:
            tone_counts[tone] = tone_counts.get(tone, 0) + 1

        for tone, count in sorted(tone_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(tones)) * 100
            print(f"   {tone:12s}: {count:2d} ({percentage:5.1f}%)")

    # Failed analyses
    failed = [r for r in results if not r['success']]
    if failed:
        print(f"\n❌ FAILED ANALYSES: {len(failed)}")
        for r in failed:
            print(f"   {Path(r['path']).name}: {r['error']}")

    print("\n" + "="*60)

    # Overall health check
    print("\n🏥 HEALTH CHECK")

    issues = []

    if speech_rates:
        mean_rate = statistics.mean(speech_rates)
        if mean_rate < SPEECH_RATE_WARNING_MIN or mean_rate > SPEECH_RATE_WARNING_MAX:
            issues.append(
                f"Mean speech rate ({mean_rate:.1f} WPM) outside normal range")

    if filler_ratios:
        mean_filler = statistics.mean(filler_ratios)
        if mean_filler > 0.2:
            issues.append(f"Mean filler ratio ({mean_filler:.3f}) is high")

    if clarity_scores:
        mean_clarity = statistics.mean(clarity_scores)
        if mean_clarity < 0.3:
            issues.append(f"Mean clarity ({mean_clarity:.3f}) is low")

    if issues:
        print("   ⚠️  Issues detected:")
        for issue in issues:
            print(f"      - {issue}")
    else:
        print("   ✅ All metrics within expected ranges")

    print()


def create_sample_test_audio():
    """Create sample audio files for testing if none exist."""
    import numpy as np
    import soundfile as sf

    sample_dir = Path("data/uploads/audio")
    sample_dir.mkdir(parents=True, exist_ok=True)

    print("\n📝 Creating sample audio files for calibration...")

    # Sample 1: Clean speech simulation (3 seconds)
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))

    # Simulate speech with varying frequency
    audio1 = 0.3 * np.sin(2 * np.pi * 200 * t) + 0.2 * \
        np.sin(2 * np.pi * 400 * t)
    audio1 += 0.05 * np.random.randn(len(audio1))  # Add slight noise

    file1 = sample_dir / "calibration_sample_1.wav"
    sf.write(str(file1), audio1, sr)
    print(f"   Created: {file1.name}")

    # Sample 2: Different tone (2 seconds)
    duration2 = 2.0
    t2 = np.linspace(0, duration2, int(sr * duration2))
    audio2 = 0.25 * np.sin(2 * np.pi * 300 * t2)

    file2 = sample_dir / "calibration_sample_2.wav"
    sf.write(str(file2), audio2, sr)
    print(f"   Created: {file2.name}")

    # Sample 3: Quiet sample (1.5 seconds)
    duration3 = 1.5
    audio3 = 0.1 * np.random.randn(int(sr * duration3))

    file3 = sample_dir / "calibration_sample_3.wav"
    sf.write(str(file3), audio3, sr)
    print(f"   Created: {file3.name}")

    print("   ✅ Sample files created\n")

    return [str(file1), str(file2), str(file3)]


def main():
    """Main calibration routine."""
    print("="*60)
    print("🎙️  VOICE AGENT CALIBRATION")
    print("="*60)

    # Initialize agent
    print("\n⚙️  Initializing Voice Agent...")
    agent = VoiceAgent(model_size="base")
    print("   ✅ Agent initialized")

    # Find audio files
    print("\n🔍 Searching for audio files...")
    audio_files = find_sample_audio_files()

    if not audio_files:
        print("   ⚠️  No audio files found")
        print("\n💡 Creating sample audio files for calibration...")
        audio_files = create_sample_test_audio()

    print(f"   Found {len(audio_files)} audio file(s)")

    if not audio_files:
        print("\n❌ No audio files available for calibration.")
        print("   Please add audio files to data/uploads/audio/")
        return

    # Limit to 5 files for calibration
    if len(audio_files) > 5:
        print(f"   Limiting to first 5 files for calibration")
        audio_files = audio_files[:5]

    # Analyze each file
    results = []
    for audio_path in audio_files:
        result = analyze_audio_file(agent, audio_path)
        results.append(result)

    # Print statistics
    print_statistics(results)

    print("✅ Calibration complete!\n")


if __name__ == "__main__":
    main()
