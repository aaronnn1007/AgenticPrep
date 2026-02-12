import json
import os
import numpy as np
import soundfile as sf
from pathlib import Path
from graph import create_speech_agent_graph


def create_sample_audio(filename: str, duration: float = 5.0, sr: int = 16000) -> str:
    """Create a simple test audio file with speech-like content."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    t = np.linspace(0, duration, int(sr * duration))
    
    frequencies = [200, 250, 300]
    audio = np.zeros_like(t)
    for freq in frequencies:
        audio += 0.3 * np.sin(2 * np.pi * freq * t)
    
    audio = audio / np.max(np.abs(audio))
    sf.write(filename, audio, sr)
    return filename


def main():
    audio_file = "data/audio/audiocheck2.m4a"
    
    if not os.path.exists(audio_file):
        create_sample_audio(audio_file)
    
    agent_graph = create_speech_agent_graph()
    
    initial_state = {
        "audio_path": audio_file,
        "question_id": "q1",
        "transcript": "",
        "speech_metrics": {}
    }
    
    output = agent_graph.invoke(initial_state)
    
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
