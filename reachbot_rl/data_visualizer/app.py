import os
import json
from flask import Flask, render_template, jsonify, request, send_from_directory
import re
from datetime import datetime

app = Flask(__name__)
RESULTS_DIR = "results"

def get_runs():
    """List all subfolders in the results directory, sorted by date in the folder name."""
    runs = []
    if not os.path.exists(RESULTS_DIR):
        return runs
    date_pattern = re.compile(r"(getup|joystick)_(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})")
    for folder in os.listdir(RESULTS_DIR):
        full_path = os.path.join(RESULTS_DIR, folder)
        if os.path.isdir(full_path):
            match = date_pattern.search(folder)
            if match:
                date_str = match.group(2)  # Correctly extract the date string
                date_obj = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")
                runs.append((folder, date_obj))
    runs.sort(key=lambda x: x[1], reverse=True)
    return [run[0] for run in runs]  # Correctly return the folder names

@app.route("/")
def index():
    runs = get_runs() 
    return render_template("index.html", runs=runs)

def get_run_data_normal(run):
    """Return the rewards and config data for a given run."""
    run_dir = os.path.join(RESULTS_DIR, run)
    if "getup" in run:
        task = "Getup"
    elif "joystick" in run:
        task = "Joystick"
    else:
        task = "Unknown"
    rewards_path = os.path.join(run_dir, "rewards.json")
    config_path = os.path.join(run_dir, "configs.json")
    rewards = []
    config = {}
    try:
        with open(rewards_path, "r") as f:
            rewards = json.load(f)
    except Exception as e:
        print(f"Error loading rewards for {run}: {e}")
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        print(f"Error loading config for {run}: {e}")
    return task, rewards, config

@app.route("/data/<run>")
def get_run_data(run):
    """Return the rewards and config data for a given run."""
    task, rewards, config = get_run_data_normal(run)
    return jsonify({"task": task, "rewards": rewards, "config": config})

@app.route("/compare")
def compare():
    runs = get_runs()
    return render_template("compare.html", runs=runs)

@app.route("/compare_data")
def compare_data():
    """Return rewards data for the selected runs."""
    selected_runs = request.args.getlist('runs[]')
    data = {}
    for run in selected_runs:
        task, rewards, config = get_run_data_normal(run)
        data[run] = {"task": task, "rewards": rewards, "config": config}
    return jsonify(data)

@app.route('/videos/<run>/<filename>')
def serve_video(run, filename):
    video_path = os.path.join("results", run)
    return send_from_directory(video_path, filename)

if __name__ == "__main__":
    app.run(debug=True)
