"""
Live Streaming Test Script
===========================
Comprehensive test for live interview streaming functionality.

Tests:
- Session creation
- WebSocket connection
- Audio streaming
- Transcription
- Body metrics updates
- Session ending and analysis

Usage:
    python scripts/live_test.py
"""

import asyncio
import json
import base64
import logging
from pathlib import Path
import wave
import sys

import httpx
import websockets
from websockets.exceptions import WebSocketException

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"


async def create_test_audio_chunk(duration_ms: int = 1000, sample_rate: int = 16000) -> bytes:
    """
    Create a test audio chunk (silence).

    Args:
        duration_ms: Duration in milliseconds
        sample_rate: Sample rate

    Returns:
        Audio bytes
    """
    import numpy as np

    # Generate silence (zeros)
    num_samples = int(sample_rate * duration_ms / 1000)
    audio_data = np.zeros(num_samples, dtype=np.int16)

    return audio_data.tobytes()


async def test_start_interview(client: httpx.AsyncClient) -> dict:
    """
    Test starting a live interview.

    Args:
        client: HTTP client

    Returns:
        Response data
    """
    logger.info("=" * 60)
    logger.info("TEST: Start Live Interview")
    logger.info("=" * 60)

    payload = {
        "role": "Software Engineer",
        "experience_level": "Mid"
    }

    response = await client.post(f"{API_BASE_URL}/live/start", json=payload)
    response.raise_for_status()

    data = response.json()
    logger.info(f"✓ Session created: {data['session_id']}")
    logger.info(f"✓ Interview ID: {data['interview_id']}")
    logger.info(f"✓ Question: {data['question']['text'][:100]}...")
    logger.info(f"✓ WebSocket URL: {data['websocket_url']}")

    return data


async def test_websocket_streaming(session_id: str, duration_seconds: int = 10) -> None:
    """
    Test WebSocket streaming with audio chunks.

    Args:
        session_id: Session identifier
        duration_seconds: How long to stream
    """
    logger.info("=" * 60)
    logger.info("TEST: WebSocket Streaming")
    logger.info("=" * 60)

    ws_url = f"{WS_BASE_URL}/live/stream/{session_id}"

    try:
        async with websockets.connect(ws_url) as websocket:
            logger.info(f"✓ WebSocket connected to {ws_url}")

            # Receive connected message
            message = await websocket.recv()
            data = json.loads(message)
            logger.info(f"✓ Received: {data['type']}")

            # Send speaking state
            await websocket.send(json.dumps({
                "type": "speaking_state",
                "data": {"is_speaking": True}
            }))
            logger.info("✓ Sent speaking state: true")

            # Stream audio chunks
            chunk_duration_ms = 1000  # 1 second chunks
            num_chunks = duration_seconds

            for i in range(num_chunks):
                # Create audio chunk
                audio_data = await create_test_audio_chunk(chunk_duration_ms)
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')

                # Send audio chunk
                message = {
                    "type": "audio_chunk",
                    "data": {
                        "audio": audio_b64,
                        "sample_rate": 16000,
                        "channels": 1,
                        "timestamp": "2024-01-01T00:00:00Z"
                    }
                }

                await websocket.send(json.dumps(message))
                logger.info(
                    f"✓ Sent audio chunk {i+1}/{num_chunks} ({len(audio_data)} bytes)")

                # Wait and check for transcript updates
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.5)
                    response_data = json.loads(response)

                    if response_data['type'] == 'transcript_update':
                        logger.info(
                            f"✓ Received transcript: {response_data['data']['text'][:100]}...")
                    elif response_data['type'] == 'metrics_update':
                        logger.info(f"✓ Received metrics update")

                except asyncio.TimeoutError:
                    pass  # No message received, continue

                await asyncio.sleep(0.5)  # Simulate real-time streaming

            # Send body metrics
            body_metrics = {
                "type": "body_metrics",
                "data": {
                    "eye_contact": 0.85,
                    "posture_stability": 0.90,
                    "facial_expressiveness": 0.78,
                    "distractions": []
                }
            }

            await websocket.send(json.dumps(body_metrics))
            logger.info("✓ Sent body metrics")

            # Receive acknowledgment
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=2.0)
                response_data = json.loads(response)
                logger.info(f"✓ Received: {response_data['type']}")
            except asyncio.TimeoutError:
                logger.warning("⚠ No response received for body metrics")

            # Send speaking state end
            await websocket.send(json.dumps({
                "type": "speaking_state",
                "data": {"is_speaking": False}
            }))
            logger.info("✓ Sent speaking state: false")

            logger.info("✓ Streaming test completed successfully")

    except WebSocketException as e:
        logger.error(f"✗ WebSocket error: {e}")
        raise
    except Exception as e:
        logger.error(f"✗ Streaming test failed: {e}", exc_info=True)
        raise


async def test_session_status(client: httpx.AsyncClient, session_id: str) -> dict:
    """
    Test getting session status.

    Args:
        client: HTTP client
        session_id: Session identifier

    Returns:
        Status data
    """
    logger.info("=" * 60)
    logger.info("TEST: Get Session Status")
    logger.info("=" * 60)

    response = await client.get(f"{API_BASE_URL}/live/session/{session_id}")
    response.raise_for_status()

    data = response.json()
    logger.info(f"✓ Session ID: {data['session_id']}")
    logger.info(f"✓ Interview ID: {data['interview_id']}")
    logger.info(f"✓ Is connected: {data['is_connected']}")
    logger.info(f"✓ Is speaking: {data['is_speaking']}")
    logger.info(f"✓ Transcript length: {data['transcript_length']} chars")
    logger.info(f"✓ Audio chunks: {data['audio_chunks_received']}")

    return data


async def test_end_answer(client: httpx.AsyncClient, session_id: str) -> dict:
    """
    Test ending answer and triggering analysis.

    Args:
        client: HTTP client
        session_id: Session identifier

    Returns:
        Analysis results
    """
    logger.info("=" * 60)
    logger.info("TEST: End Answer & Trigger Analysis")
    logger.info("=" * 60)

    payload = {"session_id": session_id}

    response = await client.post(f"{API_BASE_URL}/live/end-answer", json=payload)
    response.raise_for_status()

    data = response.json()
    logger.info(f"✓ Analysis completed for interview: {data['interview_id']}")
    logger.info(f"✓ Transcript length: {len(data['transcript'])} chars")
    logger.info(f"✓ Overall score: {data['scores']['overall']:.1f}/100")
    logger.info(f"✓ Technical score: {data['scores']['technical']:.1f}/100")
    logger.info(
        f"✓ Communication score: {data['scores']['communication']:.1f}/100")
    logger.info(f"✓ Behavioral score: {data['scores']['behavioral']:.1f}/100")
    logger.info(f"✓ Strengths: {len(data['recommendations']['strengths'])}")
    logger.info(f"✓ Weaknesses: {len(data['recommendations']['weaknesses'])}")
    logger.info(
        f"✓ Improvement plans: {len(data['recommendations']['improvement_plan'])}")

    return data


async def test_realtime_audio_file(
    client: httpx.AsyncClient,
    audio_file_path: str,
    chunk_size_ms: int = 1000
) -> None:
    """
    Test with real audio file.

    Args:
        client: HTTP client
        audio_file_path: Path to WAV audio file
        chunk_size_ms: Chunk size in milliseconds
    """
    logger.info("=" * 60)
    logger.info("TEST: Real Audio File Streaming")
    logger.info("=" * 60)

    # Check file exists
    audio_path = Path(audio_file_path)
    if not audio_path.exists():
        logger.warning(f"⚠ Audio file not found: {audio_file_path}")
        logger.info("Skipping real audio test")
        return

    # Start interview
    interview_data = await test_start_interview(client)
    session_id = interview_data['session_id']

    # Read audio file
    with wave.open(str(audio_path), 'rb') as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()

        logger.info(f"✓ Audio file: {audio_path.name}")
        logger.info(f"✓ Sample rate: {sample_rate} Hz")
        logger.info(f"✓ Channels: {channels}")

        # Calculate chunk size in frames
        chunk_frames = int(sample_rate * chunk_size_ms / 1000)

        # Connect WebSocket
        ws_url = f"{WS_BASE_URL}/live/stream/{session_id}"

        async with websockets.connect(ws_url) as websocket:
            # Receive connected
            await websocket.recv()

            # Start speaking
            await websocket.send(json.dumps({
                "type": "speaking_state",
                "data": {"is_speaking": True}
            }))

            # Stream audio chunks
            chunk_num = 0
            while True:
                audio_data = wav_file.readframes(chunk_frames)
                if not audio_data:
                    break

                chunk_num += 1
                audio_b64 = base64.b64encode(audio_data).decode('utf-8')

                message = {
                    "type": "audio_chunk",
                    "data": {
                        "audio": audio_b64,
                        "sample_rate": sample_rate,
                        "channels": channels,
                        "timestamp": "2024-01-01T00:00:00Z"
                    }
                }

                await websocket.send(json.dumps(message))
                logger.info(f"✓ Sent chunk {chunk_num}")

                # Check for transcript
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                    data = json.loads(response)
                    if data['type'] == 'transcript_update':
                        logger.info(f"✓ Transcript: {data['data']['text']}")
                except asyncio.TimeoutError:
                    pass

                # Real-time simulation
                await asyncio.sleep(chunk_size_ms / 1000)

            # Stop speaking
            await websocket.send(json.dumps({
                "type": "speaking_state",
                "data": {"is_speaking": False}
            }))

    # End answer
    await asyncio.sleep(2)  # Wait for final processing
    results = await test_end_answer(client, session_id)

    logger.info("✓ Real audio streaming test completed")


async def run_all_tests():
    """Run all live streaming tests."""
    logger.info("\n" + "=" * 60)
    logger.info("LIVE INTERVIEW STREAMING TESTS")
    logger.info("=" * 60 + "\n")

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            # Test 1: Start interview
            interview_data = await test_start_interview(client)
            session_id = interview_data['session_id']

            await asyncio.sleep(1)

            # Test 2: WebSocket streaming
            await test_websocket_streaming(session_id, duration_seconds=5)

            await asyncio.sleep(1)

            # Test 3: Session status
            await test_session_status(client, session_id)

            await asyncio.sleep(1)

            # Test 4: End answer
            await test_end_answer(client, session_id)

            logger.info("\n" + "=" * 60)
            logger.info("✓ ALL TESTS PASSED")
            logger.info("=" * 60)

        except httpx.HTTPError as e:
            logger.error(f"✗ HTTP Error: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            sys.exit(1)

        except Exception as e:
            logger.error(f"✗ Test failed: {e}", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    # Check if audio file provided for real test
    if len(sys.argv) > 1:
        audio_file = sys.argv[1]
        logger.info(f"Running with audio file: {audio_file}")
        asyncio.run(test_realtime_audio_file(
            httpx.AsyncClient(timeout=60.0),
            audio_file
        ))
    else:
        asyncio.run(run_all_tests())
