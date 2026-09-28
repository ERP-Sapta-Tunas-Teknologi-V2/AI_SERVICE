import requests
import time
from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from mom import bp as mom_bp

ALLOWED_ORIGINS = ["http://localhost:5173", "https://2023.smartindo.com"]

load_dotenv()

def check_services():
    print("[CHECK] Checking services...")
    
    # Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        response.raise_for_status()
        print("[OK] Ollama")
    except Exception as e:
        print(f"[ERROR] Ollama: {e}")
        return False

    print("[CHECK] All services are running")
    return True

def create_app():
    while not check_services():
        print("[RETRY] Services belum siap. Coba lagi dalam 5 detik...", flush=True)
        time.sleep(5)

    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": ALLOWED_ORIGINS}})
    app.register_blueprint(mom_bp)
    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)