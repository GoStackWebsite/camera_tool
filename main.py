"""Entry point: python main.py  (or run.bat; hidden through pythonw.exe when started by tools\\stack.ps1;
CameraEngine.exe when packaged by tools\\build_exe.ps1)."""
import logging
import logging.handlers
import multiprocessing
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

FROZEN = getattr(sys, "frozen", False)      # running as the packaged exe
ROOT_DIR = Path(sys.executable).resolve().parent if FROZEN else Path(__file__).resolve().parent
LOG_DIR = ROOT_DIR / "logs"
LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"
_fault_log = None   # kept open for faulthandler for the life of the process


def setup_logging() -> None:
    """Always log to logs/cam.log (rotating); also to the console when there is one.
    Under pythonw.exe there is no console at all (stdout/stderr are None), so prints from libraries
    go to logs/cam.stdout.log instead of crashing the process."""
    LOG_DIR.mkdir(exist_ok=True)
    had_console = sys.stderr is not None
    if sys.stdout is None or sys.stderr is None:
        out = open(LOG_DIR / "cam.stdout.log", "a", buffering=1, encoding="utf-8", errors="replace")
        sys.stdout = sys.stderr = out
    handlers: list[logging.Handler] = [
        logging.handlers.RotatingFileHandler(LOG_DIR / "cam.log", maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8"),
    ]
    if had_console:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=handlers)
    # a crash inside a native library (CUDA, OpenCV, FFmpeg) kills the process without a Python traceback; under
    # pythonw.exe nothing would be left behind, so the interpreter writes the thread stacks to logs/cam.fault.log
    import faulthandler
    global _fault_log
    _fault_log = open(LOG_DIR / "cam.fault.log", "a", buffering=1, encoding="utf-8", errors="replace")
    faulthandler.enable(file=_fault_log, all_threads=True)


def open_dashboard(port: int) -> None:
    """Packaged exe: a double-click ends on the dashboard, once the server answers (OPEN_BROWSER=0 in .env turns it off)."""
    for _ in range(300):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
        except OSError:
            time.sleep(1)
            continue
        webbrowser.open(f"http://127.0.0.1:{port}")
        return


if __name__ == "__main__":
    multiprocessing.freeze_support()    # a child process of the exe must not start a second server
    if FROZEN:
        os.chdir(ROOT_DIR)              # the YOLO weights are named relative to the working directory, as with run.bat
        os.environ.setdefault("YOLO_AUTOINSTALL", "False")     # ultralytics must not try pip inside the exe
    setup_logging()
    import uvicorn

    from app.config import settings

    if FROZEN and os.getenv("OPEN_BROWSER", "1") != "0":
        threading.Thread(target=open_dashboard, args=(settings.port,), daemon=True).start()
    try:
        # log_config=None: uvicorn's loggers propagate to the root handlers above (one file, one format)
        uvicorn.run("app.server:app", host=settings.host, port=settings.port, log_level="info", log_config=None, access_log=False)
    finally:
        if FROZEN:                      # like the pause in run.bat: a startup error stays on screen
            input("Engine stopped. Press Enter to close this window.")
