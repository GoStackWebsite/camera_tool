"""Entry point: python main.py  (or run.bat; hidden through pythonw.exe when started by tools\\stack.ps1)."""
import logging
import logging.handlers
import sys
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_FORMAT = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"


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


if __name__ == "__main__":
    setup_logging()
    import uvicorn

    from app.config import settings

    # log_config=None: uvicorn's loggers propagate to the root handlers above (one file, one format)
    uvicorn.run("app.server:app", host=settings.host, port=settings.port, log_level="info", log_config=None, access_log=False)
