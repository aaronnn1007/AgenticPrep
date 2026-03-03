"""
Start Backend with WebRTC Support
==================================
Launches FastAPI backend with WebRTC live streaming infrastructure.

Usage:
    python start_backend_webrtc.py
    
Options:
    --host: Host to bind to (default: 0.0.0.0)
    --port: Port to bind to (default: 8000)
    --reload: Enable auto-reload for development (default: False)
"""

import sys
import os
import logging
import asyncio
import argparse
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check all required dependencies are installed."""
    logger.info("Checking dependencies...")

    required = {
        'fastapi': 'FastAPI web framework',
        'uvicorn': 'ASGI server',
        'redis': 'Redis client',
        'faster_whisper': 'Whisper transcription',
        'aiortc': 'WebRTC implementation',
        'av': 'Audio/video processing',
        'scipy': 'Audio resampling',
        'langchain': 'LangChain framework',
        'langgraph': 'LangGraph orchestration'
    }

    missing = []
    for module, description in required.items():
        try:
            __import__(module)
            logger.info(f"  ✓ {module} ({description})")
        except ImportError:
            missing.append((module, description))
            logger.error(f"  ✗ {module} ({description}) - MISSING")

    if missing:
        logger.error("\nMissing dependencies detected!")
        logger.error("Install with:")
        logger.error("  pip install -r backend_requirements.txt")
        logger.error("  pip install -r webrtc_requirements.txt")
        return False

    logger.info("✓ All dependencies installed")
    return True


def check_redis():
    """Check Redis connection."""
    logger.info("Checking Redis connection...")

    try:
        import redis
        from backend.config import settings

        client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        client.ping()
        logger.info(f"✓ Redis connected: {settings.REDIS_URL}")
        return True

    except Exception as e:
        logger.error(f"✗ Redis connection failed: {e}")
        logger.error("\nRedis is required for WebRTC streaming!")
        logger.error("Start Redis:")
        logger.error("  Windows: See WINDOWS_REDIS_INSTALL.md")
        logger.error("  Linux: sudo systemctl start redis")
        logger.error("  macOS: brew services start redis")
        return False


def check_whisper_model():
    """Check Whisper model availability."""
    logger.info("Checking Whisper model...")

    try:
        from faster_whisper import WhisperModel
        from backend.config import settings

        logger.info(f"  Loading model: {settings.WHISPER_MODEL_SIZE}")
        model = WhisperModel(
            settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE
        )
        logger.info("✓ Whisper model loaded successfully")
        return True

    except Exception as e:
        logger.error(f"✗ Whisper model load failed: {e}")
        logger.error("\nWhisper model is required for transcription!")
        return False


def check_environment():
    """Check environment configuration."""
    logger.info("Checking environment configuration...")

    from backend.config import settings

    critical_vars = {
        'REDIS_URL': settings.REDIS_URL,
        'WHISPER_MODEL_SIZE': settings.WHISPER_MODEL_SIZE,
    }

    for var, value in critical_vars.items():
        if value:
            logger.info(f"  ✓ {var}={value}")
        else:
            logger.warning(f"  ⚠ {var} not set (using default)")

    logger.info("✓ Environment configuration OK")
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Start WebRTC streaming backend')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to bind to')
    parser.add_argument('--reload', action='store_true',
                        help='Enable auto-reload')
    parser.add_argument('--skip-checks', action='store_true',
                        help='Skip pre-flight checks')

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("WebRTC Live Streaming Backend")
    logger.info("=" * 70)

    # Pre-flight checks
    if not args.skip_checks:
        logger.info("\nRunning pre-flight checks...\n")

        if not check_dependencies():
            sys.exit(1)

        if not check_redis():
            sys.exit(1)

        if not check_whisper_model():
            logger.warning("Whisper model check failed, but continuing...")

        check_environment()

    # Import FastAPI app
    logger.info("\n" + "=" * 70)
    logger.info("Loading FastAPI application...")
    logger.info("=" * 70)

    from backend.api.main import app
    from backend.api.webrtc_live_api import (
        router as webrtc_router,
        initialize_webrtc_infrastructure,
        shutdown_webrtc_infrastructure
    )
    from backend.api.results_api import router as results_router

    # Register WebRTC router
    logger.info("Registering WebRTC routes...")
    app.include_router(webrtc_router)
    logger.info("✓ WebRTC routes registered")

    # Register Results & History router
    logger.info("Registering Results & History routes...")
    app.include_router(results_router)
    logger.info("✓ Results & History routes registered")

    # NOTE: Startup/shutdown lifecycle is handled by main.py — do NOT add it here

    # Start server
    logger.info("\n" + "=" * 70)
    logger.info("Starting Server")
    logger.info("=" * 70)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Reload: {args.reload}")
    logger.info("\nEndpoints available:")
    logger.info(f"  - REST API: http://{args.host}:{args.port}")
    logger.info(
        f"  - WebRTC Signaling: ws://{args.host}:{args.port}/webrtc/signaling/{{session_id}}")
    logger.info(f"  - API Docs: http://{args.host}:{args.port}/docs")
    logger.info("\nPress CTRL+C to stop")
    logger.info("=" * 70 + "\n")

    import uvicorn

    # IMPORTANT: Pass app object directly (not string path)
    # so that our WebRTC router and startup events are included
    uvicorn.run(
        app,  # Pass app object, not string
        host=args.host,
        port=args.port,
        reload=False,  # Reload doesn't work with app object
        log_level="info"
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nShutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
