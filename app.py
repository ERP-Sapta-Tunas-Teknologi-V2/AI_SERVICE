import os
import requests
import time
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from mom import bp as mom_bp

load_dotenv()

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,https://2023.smartindo.com").split(",")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
SKIP_SERVICE_CHECK = os.getenv("SKIP_SERVICE_CHECK", "false").lower() in ("true", "1", "yes")

def check_services():
    if SKIP_SERVICE_CHECK:
        print("[CHECK] Skipping services check via SKIP_SERVICE_CHECK.")
        return True

    print(f"[CHECK] Checking Ollama service at {OLLAMA_BASE_URL}...")
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        response.raise_for_status()
        print("[OK] Ollama is reachable")
        return True
    except Exception as e:
        print(f"[WARN] Ollama check failed: {e}")
        return False

def create_app():
    # Only retry up to 3 times on initial boot, then continue to allow app to start
    retry_count = 0
    max_retries = int(os.getenv("MAX_SERVICE_CHECK_RETRIES", "3"))
    while not check_services() and retry_count < max_retries:
        retry_count += 1
        print(f"[RETRY] Ollama belum siap ({retry_count}/{max_retries}). Coba lagi dalam 3 detik...", flush=True)
        time.sleep(3)

    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}, r"/audio-to-mom": {"origins": ALLOWED_ORIGINS}})
    app.register_blueprint(mom_bp)

    @app.route("/healthz")
    @app.route("/health")
    def health_check():
        return jsonify({
            "status": "healthy",
            "service": "AI_SERVICE",
            "ollama_url": OLLAMA_BASE_URL
        }), 200

    return app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)