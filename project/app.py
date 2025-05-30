from flask import Flask, request, jsonify, send_from_directory
import os
import uuid
import traceback

import engine  
import youtube_audio
from flask_cors import CORS


app = Flask(
    __name__,
    static_folder="../",  # TFG root
    static_url_path=""
)

CORS(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def serve_index():
    return send_from_directory('..', 'landing.html')

@app.route('/process', methods=['POST'])
def process_audio():
    try:
        file_path = None
        filename = f"{uuid.uuid4()}.webm"

        # Handle file upload
        if 'audio' in request.files:
            print("Processing uploaded file...")
            audio = request.files['audio']
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            audio.save(file_path)

        # Handle YouTube link
        elif 'youtube_link' in request.form:
            youtube_link = request.form['youtube_link'].strip()
            if not youtube_link:
                return jsonify({'error': 'YouTube link is empty'}), 400

            print("Processing YouTube link...")
            downloaded_path = youtube_audio.extract_youtube_audio(youtube_link, output_folder=UPLOAD_FOLDER)

            # If extract_youtube_audio returns a path, rename it to keep .webm consistency if needed
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            os.rename(downloaded_path, file_path)

        else:
            return jsonify({'error': 'No audio file or YouTube link provided'}), 400

        annotations = engine.main(audio_object=file_path, sr=44100, criteria="cosdif")

        audio_url = f"http://localhost:5000/uploads/{filename}"

        for element in annotations:
            print(element)

        return jsonify({
            "audio_url": audio_url,
            "annotations": annotations
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
