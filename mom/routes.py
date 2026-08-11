import os
from flask import render_template, request, flash, jsonify, current_app
from werkzeug.utils import secure_filename

from . import bp
from .service import generate_mom

@bp.route("/")
def index():
    return render_template("index.html")

# Testing melalui index.html
@bp.route("/audio-to-mom", methods=["POST"])
def audio_to_mom():
    audio = request.files.get("audio")
    if not audio or not audio.filename:
        flash("Audio file is required.")
        return render_template("index.html")
    filepath = save_audio(audio)

    try:
        transcript, minutes = generate_mom(filepath)
        return render_template("index.html", transcript=transcript, minutes=minutes)
    except Exception as e:
        flash(str(e))
        return render_template("index.html")
    finally:
        delete_audio(filepath)

# API untuk CRM Django
@bp.route("/api/mom", methods=["POST"])
def create_mom():
    audio = request.files.get("audio")
    if not audio or not audio.filename:
        return jsonify({
            "error": "Audio file is required."
        }), 400
    filepath = save_audio(audio)

    try:
        _, minutes = generate_mom(filepath)
        return jsonify({"minutes": minutes})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        delete_audio(filepath)

def save_audio(audio):
    upload_folder = os.path.join(current_app.root_path, "temp")
    os.makedirs(upload_folder, exist_ok=True)
    filename = secure_filename(audio.filename)
    filepath = os.path.join(upload_folder, filename)
    audio.save(filepath)
    return filepath

def delete_audio(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)