from flask import Blueprint

bp = Blueprint("mom", __name__)

from . import routes