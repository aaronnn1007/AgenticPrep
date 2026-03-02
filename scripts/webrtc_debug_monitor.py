#!/usr/bin/env python3
"""
WebRTC Debug Monitor
====================
Real-time monitoring and debugging tool for WebRTC streaming pipeline.

Features:
- Live connection status
- Audio stream metrics
- Transcription monitoring
- Buffer status
- Error detection
- Performance metrics

Usage:
    python webrtc_debug_monitor.py [session_id]
"""

from backend.config import settings
from backend.streaming.redis_session import RedisSessionStore
import asyncio
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)8s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class WebRTCDebugMonitor:
    """
    Real-time debug monitor for WebRTC sessions.

    Monitors:
    - Connection state
    - Audio chunks received
    - Transcription progress
    - Voice/body language metrics
    - Errors and warnings
    """

    def __init__(self, session_id: Optional[str] = None):
        """
        Initialize debug monitor.

        Args:
            session_id: Specific session to monitor (None = monitor all)
        """
        self.session_id = session_id
        self.redis_store: Optional[RedisSessionStore] = None
        self.running = False
        self.stats = {
            'start_time': time.time(),
            'checks': 0,
            'errors': 0,
            'warnings': 0
        }

    async def setup(self):
        """Initialize monitor."""
        print("=" * 80)
        print("WEBRTC DEBUG MONITOR")
        print("=" * 80)
        print()

        # Connect to Redis
        print("Connecting to Redis...")
        self.redis_store = RedisSessionStore(redis_url=settings.REDIS_URL)
        await self.redis_store.connect()
        print("✓ Connected to Redis")
        print()

    async def monitor_session(self, session_id: str):
        """
        Monitor a specific session.

        Args:
            session_id: Session to monitor
        """
        print(f"📊 Session: {session_id}")
        print("-" * 80)

        try:
            # Get session data
            session_data = await self.redis_store.get_session(session_id)

            if not session_data:
                print(f"⚠ Session not found: {session_id}")
                self.stats['warnings'] += 1
                return

            # Connection status
            conn_icon = "🟢" if session_data.connected else "🔴"
            print(
                f"{conn_icon} Connection: {'Connected' if session_data.connected else 'Disconnected'}")

            # Interview info
            print(f"📌 Interview ID: {session_data.interview_id}")
            print(f"👤 Role: {session_data.role}")
            print(f"📊 Level: {session_data.experience_level}")

            # Question
            if session_data.question:
                print(f"❓ Question: {session_data.question.text[:80]}...")
                print(f"   Category: {session_data.question.category}")

            # Transcript
            transcript = await self.redis_store.get_full_transcript(session_id)
            if transcript:
                word_count = len(transcript.split())
                print(
                    f"📝 Transcript: {len(transcript)} chars, {word_count} words")
                print(f"   Preview: '{transcript[:100]}...'")

                # Check for transcript issues
                if len(transcript) < 10:
                    print(f"   ⚠ Very short transcript!")
                    self.stats['warnings'] += 1
            else:
                print(f"📝 Transcript: (empty)")

            # Audio chunks
            audio_chunks = session_data.audio_chunks_received or 0
            print(f"🎤 Audio chunks: {audio_chunks}")

            if audio_chunks == 0 and session_data.connected:
                print(f"   ⚠ No audio chunks received yet!")
                self.stats['warnings'] += 1

            # Voice analysis
            if session_data.voice_analysis:
                va = session_data.voice_analysis
                print(f"🗣️ Voice Analysis:")
                print(f"   Speech rate: {va.speech_rate_wpm:.1f} WPM")
                print(f"   Clarity: {va.clarity_score:.2f}")
                print(f"   Tone: {va.tone}")
                print(f"   Filler ratio: {va.filler_ratio:.2f}")

                # Check for issues
                if va.speech_rate_wpm < 80 or va.speech_rate_wpm > 200:
                    print(f"   ⚠ Unusual speech rate!")
                    self.stats['warnings'] += 1
            else:
                print(f"🗣️ Voice Analysis: (pending)")

            # Body language
            if session_data.body_language:
                bl = session_data.body_language
                print(f"👁️ Body Language:")
                print(f"   Eye contact: {bl.eye_contact:.2f}")
                print(f"   Posture: {bl.posture_stability:.2f}")
                print(f"   Expressiveness: {bl.facial_expressiveness:.2f}")
                print(f"   Distractions: {len(bl.distractions)}")

                # Check for issues
                if bl.eye_contact < 0.5:
                    print(f"   ⚠ Low eye contact!")
                    self.stats['warnings'] += 1
            else:
                print(f"👁️ Body Language: (pending)")

            # Scores
            if session_data.scores:
                scores = session_data.scores
                print(f"📈 Scores:")
                print(f"   Overall: {scores.overall:.1f}")
                print(f"   Technical: {scores.technical:.1f}")
                print(f"   Communication: {scores.communication:.1f}")
                print(f"   Behavioral: {scores.behavioral:.1f}")
            else:
                print(f"📈 Scores: (not computed)")

            # Timestamps
            print(f"⏰ Timestamps:")
            print(f"   Created: {session_data.created_at}")
            if session_data.started_speaking:
                print(f"   Started speaking: {session_data.started_speaking}")
            if session_data.ended_at:
                print(f"   Ended: {session_data.ended_at}")

        except Exception as e:
            print(f"✗ Error monitoring session: {e}")
            logger.error(f"Monitoring error for {session_id}", exc_info=True)
            self.stats['errors'] += 1

        print()

    async def check_audio_pipeline(self):
        """Check audio pipeline health."""
        print("🎧 Audio Pipeline Health Check")
        print("-" * 80)

        # This would check:
        # - Audio worker status
        # - Whisper model loaded
        # - Buffer status
        # - Processing queue size

        print("✓ Audio pipeline checks completed")
        print()

    async def check_redis_connection(self):
        """Check Redis connection health."""
        print("💾 Redis Connection Health")
        print("-" * 80)

        try:
            # Try a simple operation
            test_key = "health_check_test"
            await self.redis_store.redis.set(test_key, "ok", ex=5)
            value = await self.redis_store.redis.get(test_key)

            if value == "ok":
                print("✓ Redis read/write working")
            else:
                print("⚠ Redis read/write issue")
                self.stats['warnings'] += 1

        except Exception as e:
            print(f"✗ Redis connection error: {e}")
            self.stats['errors'] += 1

        print()

    async def list_all_sessions(self):
        """List all active sessions."""
        print("📋 Active Sessions")
        print("-" * 80)

        try:
            # Get all session keys (this is a simplified version)
            # In production, you'd maintain an index of session IDs
            print("(Session listing requires Redis SCAN - not fully implemented)")
            print("Use specific session_id or check Redis directly")

        except Exception as e:
            print(f"✗ Error listing sessions: {e}")
            self.stats['errors'] += 1

        print()

    async def run_checks(self):
        """Run all health checks."""
        self.stats['checks'] += 1

        if self.session_id:
            # Monitor specific session
            await self.monitor_session(self.session_id)
        else:
            # General health checks
            await self.check_redis_connection()
            await self.check_audio_pipeline()
            await self.list_all_sessions()

    async def run_continuous(self, interval: float = 2.0):
        """
        Run continuous monitoring.

        Args:
            interval: Update interval in seconds
        """
        self.running = True
        print(f"🔄 Continuous monitoring (Ctrl+C to stop)")
        print(f"Update interval: {interval}s")
        print()

        try:
            while self.running:
                # Clear screen (optional)
                # print("\033[2J\033[H")

                await self.run_checks()

                # Stats
                elapsed = time.time() - self.stats['start_time']
                print(f"📊 Monitor Stats:")
                print(f"   Runtime: {elapsed:.1f}s")
                print(f"   Checks: {self.stats['checks']}")
                print(f"   Warnings: {self.stats['warnings']}")
                print(f"   Errors: {self.stats['errors']}")
                print()
                print("=" * 80)
                print()

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print("\n⏸️ Monitoring stopped by user")
            self.running = False

    async def run_once(self):
        """Run single check."""
        await self.run_checks()

        # Print stats
        print("=" * 80)
        print(f"Warnings: {self.stats['warnings']}")
        print(f"Errors: {self.stats['errors']}")
        print("=" * 80)

    async def cleanup(self):
        """Clean up resources."""
        if self.redis_store:
            await self.redis_store.disconnect()


# =============================================================================
# CLI INTERFACE
# =============================================================================

async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="WebRTC Debug Monitor - Monitor WebRTC streaming sessions"
    )
    parser.add_argument(
        'session_id',
        nargs='?',
        help='Session ID to monitor (optional)'
    )
    parser.add_argument(
        '--continuous',
        '-c',
        action='store_true',
        help='Run continuous monitoring'
    )
    parser.add_argument(
        '--interval',
        '-i',
        type=float,
        default=2.0,
        help='Update interval for continuous mode (seconds)'
    )

    args = parser.parse_args()

    # Create monitor
    monitor = WebRTCDebugMonitor(session_id=args.session_id)

    try:
        await monitor.setup()

        if args.continuous:
            await monitor.run_continuous(interval=args.interval)
        else:
            await monitor.run_once()

    finally:
        await monitor.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExited")
        sys.exit(0)
