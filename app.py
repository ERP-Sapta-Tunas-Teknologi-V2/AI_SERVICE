from flask import Flask
from dotenv import load_dotenv

from mom import bp as mom_bp

load_dotenv()

def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "dev"
    app.register_blueprint(mom_bp)
    return app

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)