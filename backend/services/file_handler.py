"""
File Handler Service
====================
Handles file upload, validation, and storage.
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import UploadFile, HTTPException

from backend.config import settings

logger = logging.getLogger(__name__)


def validate_file(file: UploadFile, file_type: str):
    """
    Validate uploaded file.
    
    Checks:
    - File extension
    - File size
    
    Args:
        file: Uploaded file
        file_type: "audio" or "video"
    
    Raises:
        ValueError: If validation fails
    """
    if not file or not file.filename:
        raise ValueError(f"{file_type.capitalize()} file is required")
    
    # Get file extension
    extension = Path(file.filename).suffix.lower().lstrip('.')
    
    # Check extension
    if file_type == "audio":
        if extension not in settings.ALLOWED_AUDIO_EXTENSIONS:
            raise ValueError(
                f"Invalid audio format: {extension}. "
                f"Allowed: {', '.join(settings.ALLOWED_AUDIO_EXTENSIONS)}"
            )
        max_size_mb = settings.MAX_AUDIO_SIZE_MB
    elif file_type == "video":
        if extension not in settings.ALLOWED_VIDEO_EXTENSIONS:
            raise ValueError(
                f"Invalid video format: {extension}. "
                f"Allowed: {', '.join(settings.ALLOWED_VIDEO_EXTENSIONS)}"
            )
        max_size_mb = settings.MAX_VIDEO_SIZE_MB
    else:
        raise ValueError(f"Unknown file type: {file_type}")
    
    # Check file size (file.size may not be available, validate during read)
    logger.info(f"File validated: {file.filename} ({file_type})")


async def save_upload_file(
    upload_file: UploadFile,
    interview_id: str,
    file_type: str
) -> Path:
    """
    Save uploaded file to disk.
    
    Args:
        upload_file: FastAPI UploadFile
        interview_id: Interview session ID
        file_type: "audio" or "video"
    
    Returns:
        Path to saved file
    """
    # Create directory for this interview
    interview_dir = Path(settings.UPLOAD_DIR) / interview_id
    interview_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate file path
    extension = Path(upload_file.filename).suffix
    file_path = interview_dir / f"{file_type}{extension}"
    
    # Save file
    try:
        contents = await upload_file.read()
        
        # Check size
        size_mb = len(contents) / (1024 * 1024)
        max_size = settings.MAX_AUDIO_SIZE_MB if file_type == "audio" else settings.MAX_VIDEO_SIZE_MB
        
        if size_mb > max_size:
            raise ValueError(f"File too large: {size_mb:.1f}MB (max: {max_size}MB)")
        
        # Write to disk
        with open(file_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"File saved: {file_path} ({size_mb:.1f}MB)")
        return file_path
    
    except Exception as e:
        logger.error(f"Failed to save file: {e}")
        # Cleanup on error
        if file_path.exists():
            file_path.unlink()
        raise


def cleanup_interview_files(interview_id: str):
    """
    Clean up files for a completed interview.
    
    Args:
        interview_id: Interview session ID
    """
    interview_dir = Path(settings.UPLOAD_DIR) / interview_id
    
    if interview_dir.exists():
        import shutil
        shutil.rmtree(interview_dir)
        logger.info(f"Cleaned up files for interview: {interview_id}")
