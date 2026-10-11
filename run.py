"""Unified application runner.

Starts both the FastAPI backend and Streamlit frontend in one command:
    python run.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import requests


def wait_for_backend(url: str, timeout: int = 15) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url}/health", timeout=1)
            if r.ok:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main() -> None:
    api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
    backend_proc = None

    print("=" * 60)
    print(">> Domain-Specific RAG Application Launcher")
    print("=" * 60)

    # 1. Check if backend is already running
    try:
        r = requests.get(f"{api_url}/health", timeout=1.5)
        backend_already_up = r.ok
    except Exception:
        backend_already_up = False

    if backend_already_up:
        print(f"[OK] FastAPI backend is already running at {api_url}")
    else:
        print(f"[..] Starting FastAPI backend on {api_url}...")
        backend_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ]
        backend_proc = subprocess.Popen(backend_cmd)

        print("[..] Waiting for backend to initialize...")
        if wait_for_backend(api_url, timeout=15):
            print("[OK] FastAPI backend is healthy and ready!")
        else:
            print("[!] Backend took longer than expected; proceeding to launch UI...")

    # 2. Launch Streamlit
    streamlit_port = os.getenv("PORT", os.getenv("STREAMLIT_PORT", "8501"))
    print(f"[..] Starting Streamlit frontend on port {streamlit_port}...")
    frontend_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        "frontend/app.py",
        "--server.port",
        str(streamlit_port),
        "--server.address",
        "0.0.0.0",
    ]

    try:
        frontend_proc = subprocess.Popen(frontend_cmd)
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n[..] Stopping servers...")
    finally:
        if backend_proc and backend_proc.poll() is None:
            backend_proc.terminate()
            try:
                backend_proc.wait(timeout=5)
            except Exception:
                backend_proc.kill()
        print("[OK] All services stopped.")


if __name__ == "__main__":
    main()
