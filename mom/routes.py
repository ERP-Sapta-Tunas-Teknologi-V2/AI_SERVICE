import os
import json
from flask import render_template, request, jsonify, current_app, Response, stream_with_context
from werkzeug.utils import secure_filename

from . import bp
from .service import generate_mom

def build_context(form):
    fields = [
        ("Tanggal", "date"),
        ("Tipe meeting", "meeting_type"),
        ("Channel", "channel"),
        ("Target audience", "audience"),
        ("Peserta", "participants"),
    ]
    lines = [f"- {label}: {form.get(key, '').strip()}" for label, key in fields if form.get(key, "").strip()]
    return "\n".join(lines) or "-"

@bp.route("/")
def index():
    return render_template("index.html")

# Testing melalui index.html
@bp.route("/audio-to-mom", methods=["POST"])
def audio_to_mom():
    audio = request.files.get("audio")

    if not audio or not audio.filename:
        return jsonify({"error": "Audio file is required."}), 400

    filepath = save_audio(audio)

    def generate():
        try:
            for event in generate_mom(filepath):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            delete_audio(filepath)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

# API untuk CRM
@bp.route("/api/mom", methods=["POST"])
def create_mom():
    audio = request.files.get("audio")

    if not audio or not audio.filename:
        return jsonify({"error": "Audio file is required."}), 400

    filepath = save_audio(audio)
    context = build_context(request.form)

    try:
        events = list(generate_mom(filepath, context))

        minutes = {"abstract_summary": "", "key_points": "", "action_items": ""}
        for e in events:
            if e["type"] == "mom":
                minutes[e["key"]] += e["content"]

        formatted = (
            f"Abstract Summary:\n{minutes['abstract_summary']}\n\n"
            f"Key Points:\n{minutes['key_points']}\n\n"
            f"Action Items:\n{minutes['action_items']}"
        )

        return jsonify({"minutes": formatted})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        delete_audio(filepath)

def save_audio(audio):
    upload_folder = os.path.join(current_app.root_path, "temp_audio")
    os.makedirs(upload_folder, exist_ok=True)

    filename = secure_filename(audio.filename)
    filepath = os.path.join(upload_folder, filename)

    audio.save(filepath)
    return filepath

def delete_audio(filepath):
    if os.path.exists(filepath):
        os.remove(filepath)