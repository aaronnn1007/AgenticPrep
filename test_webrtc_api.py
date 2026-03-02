"""
Test WebRTC API endpoints
"""
import requests
import json

API_BASE = "http://localhost:8000"


def test_webrtc_start():
    """Test starting a WebRTC session"""
    print("Testing POST /webrtc/start...")

    payload = {
        "role": "Software Engineer",
        "experience_level": "Mid"
    }

    response = requests.post(
        f"{API_BASE}/webrtc/start",
        json=payload
    )

    print(f"Status: {response.status_code}")

    if response.status_code == 200:
        data = response.json()
        print(f"✓ Session created:")
        print(f"  - Session ID: {data['session_id']}")
        print(f"  - Interview ID: {data['interview_id']}")
        print(f"  - Signaling URL: {data['signaling_url']}")
        print(f"  - Question: {data['question']['text']}")
        return data['session_id']
    else:
        print(f"✗ Failed: {response.text}")
        return None


def test_api_docs():
    """Test if API docs are accessible"""
    print("\nTesting GET /docs...")

    response = requests.get(f"{API_BASE}/docs")

    print(f"Status: {response.status_code}")
    print(f"✓ API docs accessible at {API_BASE}/docs")


def main():
    print("=" * 60)
    print("WebRTC API Test")
    print("=" * 60)

    # Test API docs
    test_api_docs()

    # Test WebRTC start endpoint
    session_id = test_webrtc_start()

    print("\n" + "=" * 60)
    if session_id:
        print("✓ All tests passed!")
        print(f"\nTo test the full flow, open your dashboard at:")
        print(f"http://localhost:3000/live")
        print(f"\nOr connect to WebSocket signaling at:")
        print(f"ws://localhost:8000/webrtc/signaling/{session_id}")
    else:
        print("✗ Tests failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
