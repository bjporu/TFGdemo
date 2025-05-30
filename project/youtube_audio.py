import os
import uuid
import io

from pydub import AudioSegment
from yt_dlp import YoutubeDL

def extract_youtube_audio(url, output_folder):
    """
    Downloads and converts YouTube audio to WAV.
    Returns the file path to the saved .wav file.
    """
    temp_basename = f"temp_{uuid.uuid4()}"
    temp_m4a_path = f"{temp_basename}.m4a"
    output_filename = f"{uuid.uuid4()}.wav"
    output_path = os.path.join(output_folder, output_filename)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': temp_basename,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
            'preferredquality': '192',
        }],
        'quiet': True
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Read and convert 
        with open(temp_m4a_path, 'rb') as f:
            audio_bytes = io.BytesIO(f.read())

        audio = AudioSegment.from_file(audio_bytes, format="m4a")
        audio.export(output_path, format="wav")

    finally:
        # Remove temp file
        if os.path.exists(temp_m4a_path):
            os.remove(temp_m4a_path)

    return output_path
