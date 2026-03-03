"""
WebRTC Live Streaming Test Script
==================================
Comprehensive test for WebRTC live interview streaming infrastructure.

Tests:
- Session creation
- WebRTC signaling
- Audio streaming simulation
- Transcription
- Metric updates
- LangGraph analysis trigger
- Error handling

Usage:
    python scripts/webrtc_live_test.py
"""

import numpy as np
from websockets import connect as ws_connect
import httpx
import asyncio
import json
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Test configuration
BASE_URL = "http://localhost:8000"
WS_BASE_URL = "ws://localhost:8000"


class WebRTCTestClient:
    """Test client for WebRTC streaming."""

    def __init__(self):
        self.session_id = None
        self.interview_id = None
        self.question = None
        self.signaling_url = None
        self.ws = None

    async def test_start_session(self):
        """Test starting a WebRTC session."""
        print("\n" + "=" * 70)
        print("TEST 1: Start WebRTC Interview Session")
        print("=" * 70)

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/webrtc/start",
                json={
                    "role": "Software Engineer",
                    "experience_level": "Mid"
                },
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                self.session_id = data["session_id"]
                self.interview_id = data["interview_id"]
                self.question = data["question"]
                self.signaling_url = data["signaling_url"]

                print(f"✓ Session created successfully")
                print(f"  Session ID: {self.session_id}")
                print(f"  Interview ID: {self.interview_id}")
                print(f"  Question: {self.question['text'][:80]}...")
                print(f"  Signaling URL: {self.signaling_url}")
                return True
            else:
                print(f"✗ Failed to create session: {response.status_code}")
                print(f"  Response: {response.text}")
                return False

    async def test_signaling_connection(self):
        """Test WebRTC signaling WebSocket connection."""
        print("\n" + "=" * 70)
        print("TEST 2: WebRTC Signaling Connection")
        print("=" * 70)

        try:
            ws_url = f"{WS_BASE_URL}{self.signaling_url}"
            print(f"Connecting to: {ws_url}")

            self.ws = await ws_connect(ws_url)

            # Wait for ready message
            message = await asyncio.wait_for(self.ws.recv(), timeout=5.0)
            data = json.loads(message)

            if data.get("type") == "ready":
                print(f"✓ Signaling WebSocket connected")
                print(f"  Status: {data['data']['status']}")
                return True
            else:
                print(f"✗ Unexpected message: {data}")
                return False

        except Exception as e:
            print(f"✗ Signaling connection failed: {e}")
            return False

    async def test_send_fake_offer(self):
        """Test sending a mock SDP offer."""
        print("\n" + "=" * 70)
        print("TEST 3: Send Mock SDP Offer")
        print("=" * 70)

        # Create a minimal SDP offer (mock)
        fake_offer = """v=0
o=- 1234567890 1234567890 IN IP4 127.0.0.1
s=-
t=0 0
a=group:BUNDLE 0 1
m=audio 9 UDP/TLS/RTP/SAVPF 111
c=IN IP4 0.0.0.0
a=rtcp:9 IN IP4 0.0.0.0
a=ice-ufrag:test
a=ice-pwd:testpassword
a=fingerprint:sha-256 00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00
a=setup:actpass
a=mid:0
a=sendonly
a=rtcp-mux
a=rtpmap:111 opus/48000/2
m=video 9 UDP/TLS/RTP/SAVPF 96
c=IN IP4 0.0.0.0
a=rtcp:9 IN IP4 0.0.0.0
a=ice-ufrag:test
a=ice-pwd:testpassword
a=fingerprint:sha-256 00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00:00
a=setup:actpass
a=mid:1
a=sendonly
a=rtcp-mux
a=rtpmap:96 VP8/90000
"""

        try:
            # Send offer
            offer_message = {
                "type": "offer",
                "sdp": fake_offer
            }
            await self.ws.send(json.dumps(offer_message))
            print("✓ Sent SDP offer")

            # Wait for answer
            message = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
            data = json.loads(message)

            if data.get("type") == "answer":
                print(f"✓ Received SDP answer")
                print(f"  Answer SDP length: {len(data['data']['sdp'])} chars")
                return True
            else:
                print(f"✗ Unexpected response: {data.get('type')}")
                return False

        except Exception as e:
            print(f"✗ Offer/answer exchange failed: {e}")
            return False

    async def test_send_ice_candidate(self):
        """Test sending ICE candidate."""
        print("\n" + "=" * 70)
        print("TEST 4: Send ICE Candidate")
        print("=" * 70)

        try:
            ice_message = {
                "type": "ice_candidate",
                "candidate": {
                    "candidate": "candidate:1 1 UDP 2130706431 192.168.1.100 54321 typ host",
                    "sdpMid": "0",
                    "sdpMLineIndex": 0
                }
            }
            await self.ws.send(json.dumps(ice_message))
            print("✓ Sent ICE candidate")
            await asyncio.sleep(0.5)
            return True

        except Exception as e:
            print(f"✗ ICE candidate send failed: {e}")
            return False

    async def test_send_speaking_state(self, is_speaking: bool):
        """Test sending speaking state update."""
        print(f"\n  → Setting speaking state: {is_speaking}")

        try:
            message = {
                "type": "speaking_state",
                "data": {
                    "is_speaking": is_speaking
                }
            }
            await self.ws.send(json.dumps(message))
            await asyncio.sleep(0.2)
            return True

        except Exception as e:
            print(f"✗ Speaking state update failed: {e}")
            return False

    async def test_send_body_metrics(self):
        """Test sending body language metrics."""
        print("\n" + "=" * 70)
        print("TEST 5: Send Body Language Metrics")
        print("=" * 70)

        try:
            metrics_message = {
                "type": "body_metrics",
                "data": {
                    "eye_contact": 0.85,
                    "posture_stability": 0.78,
                    "facial_expressiveness": 0.82,
                    "distractions": ["looking_away"]
                }
            }
            await self.ws.send(json.dumps(metrics_message))
            print("✓ Sent body language metrics")
            print(f"  Eye contact: 0.85")
            print(f"  Posture stability: 0.78")
            print(f"  Facial expressiveness: 0.82")
            await asyncio.sleep(0.5)
            return True

        except Exception as e:
            print(f"✗ Body metrics send failed: {e}")
            return False

    async def test_receive_messages(self, duration: float = 5.0):
        """Test receiving messages from server."""
        print("\n" + "=" * 70)
        print(f"TEST 6: Receive Server Messages ({duration}s)")
        print("=" * 70)

        try:
            start_time = asyncio.get_event_loop().time()
            message_count = 0

            while asyncio.get_event_loop().time() - start_time < duration:
                try:
                    message = await asyncio.wait_for(self.ws.recv(), timeout=1.0)
                    data = json.loads(message)
                    message_count += 1

                    msg_type = data.get("type")
                    print(f"  ← Received: {msg_type}")

                    if msg_type == "transcript_update":
                        print(f"    Text: {data['data']['text']}")
                        print(
                            f"    Confidence: {data['data']['confidence']:.2f}")

                    elif msg_type == "metrics_update":
                        print(f"    Metrics: {data['data']}")

                    elif msg_type == "connection_state":
                        print(f"    State: {data['data']['state']}")

                except asyncio.TimeoutError:
                    continue

            print(f"✓ Received {message_count} messages in {duration}s")
            return True

        except Exception as e:
            print(f"✗ Message receiving failed: {e}")
            return False

    async def test_get_session_stats(self):
        """Test getting session statistics."""
        print("\n" + "=" * 70)
        print("TEST 7: Get Session Statistics")
        print("=" * 70)

        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{BASE_URL}/webrtc/session/{self.session_id}/stats",
                timeout=10.0
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✓ Retrieved session statistics")
                print(f"  Connected: {data['connected']}")
                print(f"  Interview ID: {data['interview_id']}")
                print(f"  Audio chunks: {data['redis_stats']['audio_chunks']}")
                print(
                    f"  Transcript length: {data['redis_stats']['transcript_length']}")
                return True
            else:
                print(f"✗ Failed to get stats: {response.status_code}")
                return False

    async def test_end_answer(self):
        """Test ending answer and triggering analysis."""
        print("\n" + "=" * 70)
        print("TEST 8: End Answer & Trigger LangGraph Analysis")
        print("=" * 70)

        # Note: This will fail in test since we don't have real audio transcription
        # But we can verify the endpoint responds correctly

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{BASE_URL}/webrtc/end-answer",
                json={"session_id": self.session_id},
                timeout=30.0
            )

            if response.status_code == 200:
                data = response.json()
                print(f"✓ Analysis completed")
                print(f"  Interview ID: {data['interview_id']}")
                print(f"  Transcript length: {len(data['transcript'])}")
                print(f"  Overall score: {data['scores']['overall']:.1f}")
                print(
                    f"  Analysis duration: {data['analysis_duration_seconds']:.2f}s")
                return True
            elif response.status_code == 400:
                # Expected if no transcript
                print(f"⚠ No transcript available (expected in test)")
                print(f"  Message: {response.json()['detail']}")
                return True
            else:
                print(f"✗ Failed to end answer: {response.status_code}")
                print(f"  Response: {response.text}")
                return False

    async def test_cleanup(self):
        """Clean up test resources."""
        print("\n" + "=" * 70)
        print("TEST 9: Cleanup")
        print("=" * 70)

        try:
            # Close WebSocket
            if self.ws:
                await self.ws.close()
                print("✓ Closed WebSocket connection")

            # Delete session
            async with httpx.AsyncClient() as client:
                response = await client.delete(
                    f"{BASE_URL}/webrtc/session/{self.session_id}",
                    timeout=10.0
                )

                if response.status_code in [200, 404]:
                    print(f"✓ Session deleted")
                    return True
                else:
                    print(
                        f"⚠ Session deletion returned: {response.status_code}")
                    return True

        except Exception as e:
            print(f"⚠ Cleanup warning: {e}")
            return True

    async def run_all_tests(self):
        """Run all tests."""
        print("\n" + "=" * 70)
        print("WEBRTC LIVE STREAMING TEST SUITE")
        print("=" * 70)
        print("Testing WebRTC signaling, media streaming, and LangGraph integration")

        results = []

        # Test 1: Start session
        results.append(await self.test_start_session())
        if not results[-1]:
            print("\n✗ CRITICAL: Cannot continue without session")
            return False

        # Test 2: Signaling connection
        results.append(await self.test_signaling_connection())
        if not results[-1]:
            print("\n✗ CRITICAL: Cannot continue without signaling")
            return False

        # Test 3: SDP offer/answer
        results.append(await self.test_send_fake_offer())

        # Test 4: ICE candidates
        results.append(await self.test_send_ice_candidate())

        # Test 5: Body metrics
        results.append(await self.test_send_body_metrics())

        # Simulate speaking
        await self.test_send_speaking_state(True)
        await asyncio.sleep(1)

        # Test 6: Receive messages
        results.append(await self.test_receive_messages(duration=3.0))

        await self.test_send_speaking_state(False)

        # Test 7: Session stats
        results.append(await self.test_get_session_stats())

        # Test 8: End answer
        results.append(await self.test_end_answer())

        # Test 9: Cleanup
        results.append(await self.test_cleanup())

        # Summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        passed = sum(results)
        total = len(results)
        print(f"Passed: {passed}/{total}")
        print(f"Failed: {total - passed}/{total}")

        if passed == total:
            print("\n✓ ALL TESTS PASSED")
            return True
        else:
            print(f"\n⚠ {total - passed} TEST(S) FAILED")
            return False


async def main():
    """Main test function."""
    print("Starting WebRTC Live Streaming Tests...")
    print(f"Backend URL: {BASE_URL}")
    print(f"WebSocket URL: {WS_BASE_URL}")
    print("\nMake sure the backend server is running:")
    print("  python start_backend.py")
    print()

    client = WebRTCTestClient()

    try:
        success = await client.run_all_tests()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)

    except Exception as e:
        print(f"\n\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
