"""Supervise the internal API and public dashboard for a single-node free demo."""

import os
import signal
import subprocess
import sys
import time
import urllib.request


def main():
    # Public demo administration stays disabled even if a secret was supplied accidentally.
    api_port = int(os.getenv("DEMO_API_PORT", "8000"))
    env = {
        **os.environ,
        "SCHEDULER_ENABLED": "false",
        "DEMO_MODE": "true",
        "FORECASTOPS_API_KEY": "",
        "API_URL": f"http://127.0.0.1:{api_port}",
        "OMP_NUM_THREADS": "1",
    }
    children = []
    stopping = False

    def shutdown(signum, frame):
        nonlocal stopping
        stopping = True
        for child in children:
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    try:
        children.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "app.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(api_port),
                ],
                env=env,
            )
        )
        deadline = time.monotonic() + 120
        while not stopping and time.monotonic() < deadline:
            if children[0].poll() is not None:
                raise RuntimeError("Internal API exited before readiness")
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{api_port}/ready", timeout=5
                ) as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("Internal API did not become ready")
        if stopping:
            return
        children.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "frontend/streamlit_app.py",
                    "--server.address=0.0.0.0",
                    f"--server.port={int(env.get('PORT', '10000'))}",
                    "--server.headless=true",
                    "--browser.gatherUsageStats=false",
                    "--server.fileWatcherType=none",
                ],
                env=env,
            )
        )
        while not stopping:
            if any(child.poll() is not None for child in children):
                raise RuntimeError("Demo process exited; stopping both services")
            time.sleep(0.5)
    finally:
        shutdown(None, None)
        for child in children:
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
