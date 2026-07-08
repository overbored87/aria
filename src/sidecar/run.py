"""Start the sidecar's uvicorn server in a daemon thread.

Signal handlers are disabled because uvicorn is NOT on the main thread —
Aria's app.run_polling() owns the main thread and its signals. The thread is
daemon so it dies with the process.
"""

import threading

import uvicorn

from src.config import cfg
from src.sidecar.server import app
from src.utils.logger import log


def start_sidecar_in_thread() -> None:
    config = uvicorn.Config(app, host="0.0.0.0", port=cfg.sidecar_port, log_level="warning")
    server = uvicorn.Server(config)
    server.install_signal_handlers = lambda: None  # not the main thread
    threading.Thread(target=server.run, daemon=True, name="aria-sidecar").start()
    log.info(f"Sidecar starting on :{cfg.sidecar_port}")
