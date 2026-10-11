"""Unified application runner.

Starts both the FastAPI backend and Streamlit frontend in one command:
    python run.py

Works seamlessly both locally and when deployed to cloud platforms like Render.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from urllib.parse import urlparse

# Ensure safe UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import requests


def wait_for_backend(url: str, timeout: int = 20) -> bool:
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
    # 1. Determine frontend public port (injected by Render via $PORT or local 8501)
    streamlit_port = int(os.getenv("PORT", os.getenv("STREAMLIT_PORT", "8501")))

    # 2. Assign non-conflicting internal port for the FastAPI backend
    backend_port = 8001 if streamlit_port == 8000 else 8000

    # 3. Configure API_URL
    raw_api_url = os.getenv("API_URL")
    if not raw_api_url:
        api_url = f"http://127.0.0.1:{backend_port}"
        os.environ["API_URL"] = api_url
    else:
        api_url = raw_api_url.rstrip("/")

    parsed = urlparse(api_url)
    is_local_backend = parsed.hostname in ("127.0.0.1", "localhost", "0.0.0.0")

    backend_proc = None

    print("=" * 60)
    print(">> Domain-Specific RAG Application Launcher")
    print(f">> Frontend Port: {streamlit_port}")
    print(f">> Backend Target: {api_url}")
    print("=" * 60)

    # 4. If targeting a local backend, check if running or start uvicorn
    if is_local_backend:
        try:
            r = requests.get(f"{api_url}/health", timeout=1.5)
            backend_already_up = r.ok
        except Exception:
            backend_already_up = False

        if backend_already_up:
            print(f"[OK] FastAPI backend is already running at {api_url}")
        else:
            print(f"[..] Starting FastAPI backend internally on port {backend_port}...")
            backend_cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(backend_port),
            ]
            backend_proc = subprocess.Popen(backend_cmd)

            print("[..] Waiting for backend to initialize...")
            if wait_for_backend(api_url, timeout=20):
                print("[OK] FastAPI backend is healthy and ready!")
            else:
                print("[!] Backend initialization took longer than expected; proceeding to launch UI...")
    else:
        print(f"[OK] Connecting to external backend at {api_url}")

    # 5. Launch Streamlit
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
        "--server.headless",
        "true",
        "--server.enableCORS",
        "false",
        "--server.enableXsrfProtection",
        "false",
        "--browser.gatherUsageStats",
        "false",
    ]

    try:
        frontend_proc = subprocess.Popen(frontend_cmd)
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n[..] Stopping application...")
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
