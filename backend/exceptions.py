"""
Backend custom exceptions.
"""


class TranscriptionError(Exception):
    """
    Raised when audio transcription fails.

    Carries the original cause so callers can implement retry logic
    or surface a meaningful error to the user instead of receiving an
    empty transcript silently.

    Attributes:
        audio_path: Path to the audio file that failed transcription (optional).
        cause: The underlying exception that triggered the failure.
    """

    def __init__(self, message: str, audio_path: str = "", cause: Exception = None):
        super().__init__(message)
        self.audio_path = audio_path
        self.cause = cause

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.audio_path:
            parts.append(f"audio_path={self.audio_path!r}")
        if self.cause is not None:
            parts.append(f"cause={type(self.cause).__name__}: {self.cause}")
        return " | ".join(parts)
