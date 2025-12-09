import streamlit as st
import subprocess
import sys
import os
import time
import platform
from pathlib import Path

st.set_page_config(page_title="Face Recognition — Launcher (Option B)", layout="wide")

ROOT = Path.cwd()

# Scripts (adjust names if yours differ)
SCRIPTS = {
    "Capture Dataset": "dataset_builder.py",
    "Train Model": "train_recognizer.py",
    "Recognize (Live)": "recognize_live.py",
}

# Utility: start a process for a script
def start_script(script_path):
    """Start the given script as a new OS process and return the Popen handle."""
    python_exe = sys.executable  # uses the same Python interpreter as Streamlit
    args = [python_exe, str(script_path)]

    # On Windows create a new console so cv2 windows and input() work as usual:
    creationflags = 0
    if platform.system() == "Windows":
        try:
            creationflags = subprocess.CREATE_NEW_CONSOLE
        except Exception:
            creationflags = 0

    # Start process without capturing stdout/stderr (they'll appear in the new console)
    proc = subprocess.Popen(
        args,
        stdout=None,
        stderr=None,
        stdin=None,
        shell=False,
        creationflags=creationflags,
        cwd=str(ROOT),
    )
    return proc

def stop_process(proc, timeout=2.0):
    """Try to terminate gracefully, then kill if necessary."""
    if proc is None:
        return None
    if proc.poll() is not None:
        return proc.returncode
    try:
        proc.terminate()
    except Exception:
        pass
    # wait a bit
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
        except Exception:
            pass
    return proc.returncode

# Store processes in session state keyed by script name
if "processes" not in st.session_state:
    st.session_state.processes = {}

st.title("Face Recognition — Launcher")
st.markdown(
    "This app launches your existing scripts as separate OS processes so their OpenCV windows and keyboard interaction work unchanged."
)
st.markdown("**Note:** When you start a script, a new console/window may open. Interact with that window (press keys like `q`, `a`, `s` there).")

cols = st.columns([1, 1, 1])

for i, (tab_label, script_name) in enumerate(SCRIPTS.items()):
    col = cols[i]
    with col:
        st.subheader(tab_label)
        script_path = ROOT / script_name
        if not script_path.exists():
            st.error(f"Script not found: {script_name}")
            continue

        # Show process status if present
        proc = st.session_state.processes.get(script_name)

        if proc is None or proc.poll() is not None:
            # not running
            if proc is None:
                st.info("Status: Not running")
            else:
                st.success(f"Last run finished (exit code {proc.returncode})" if proc.returncode is not None else "Last run finished")
            start_btn = st.button(f"Start {tab_label}", key=f"start_{script_name}")
            stop_btn = st.button(f"Stop {tab_label}", key=f"stop_{script_name}", disabled=True)
        else:
            # running
            st.success(f"Status: Running (pid={proc.pid})")
            start_btn = st.button(f"Start {tab_label}", key=f"start_{script_name}", disabled=True)
            stop_btn = st.button(f"Stop {tab_label}", key=f"stop_{script_name}")

        # Handle start request
        if start_btn:
            if proc is not None and proc.poll() is None:
                st.warning("Process already running.")
            else:
                try:
                    new_proc = start_script(script_path)
                    st.session_state.processes[script_name] = new_proc
                    st.success(f"Started {script_name} (pid={new_proc.pid})")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Failed to start: {e}")

        # Handle stop request
        if stop_btn:
            proc_to_stop = st.session_state.processes.get(script_name)
            if proc_to_stop is None:
                st.info("No process to stop.")
            else:
                ret = stop_process(proc_to_stop)
                st.session_state.processes[script_name] = proc_to_stop  # keep it; poll will show exit
                st.info(f"Stop requested. Exit code: {ret}")
                st.experimental_rerun()

st.markdown("---")
st.write("Session process table:")

# Small status table
table = []
for name, p in st.session_state.processes.items():
    table.append(
        {
            "script": name,
            "pid": p.pid if p is not None else "-",
            "running": ("Yes" if (p is not None and p.poll() is None) else "No"),
            "returncode": p.returncode if p is not None else None,
        }
    )
st.table(table)

st.markdown(
    """

1. Choose a tab's **Start** button to launch that script.  
2. A new console/window will open; interact there (for example, press `q` to quit or `a` to add a new person in `recognize_live.py`).  
3. Use **Stop** in this Streamlit app to request termination (it sends terminate, then kill if needed).  
4. When training finishes, the model files `lbph_model.yml` and `labels.pickle` are updated; you can then start the Recognize script to use the new model.

**Notes & tips**
- If the camera is in use by a running script, other scripts won't be able to access it. Stop the running script first.
- On Windows you should see a new console window for the script. If you prefer logs inside Streamlit we can extend the app to capture and stream stdout (but that requires non-blocking log handling).
"""
)
