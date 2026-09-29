import os
import uuid
from datetime import datetime
from contextvars import ContextVar

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIME_LOG_FILE = os.path.join(BASE_DIR, "timing.log")
MOM_LOG_FILE = os.path.join(BASE_DIR, "mom.log")

run_id_var = ContextVar("run_id", default="-")

def new_run_id():
    run_id = uuid.uuid4().hex[:8]
    run_id_var.set(run_id)
    return run_id

def log_space():
    with open(TIME_LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n")

def log_model(max_token, model):
    with open(TIME_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{run_id_var.get()} | max_token = {max_token} | model = {model}\n")

def log_time(label, seconds):
    with open(TIME_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{seconds:.2f}s | {label} | {datetime.now():%Y-%m-%d %H:%M:%S}\n")

def log_mom(filename, minutes):
    with open(MOM_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"===== {datetime.now():%Y-%m-%d %H:%M:%S} | {run_id_var.get()} | {filename} =====\n")
        f.write(f"Abstract Summary:\n{minutes['abstract_summary']}\n\n")
        f.write(f"Key Points:\n{minutes['key_points']}\n\n")
        f.write(f"Action Items:\n{minutes['action_items']}\n\n")