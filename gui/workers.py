"""
gui/workers.py
--------------
Background jobs that report progress and can be stopped.

A sweep over k, a DBSCAN eps scan or a day-by-day classification can run for
minutes.  Each one is a loop, so it can check between iterations whether the
user has asked it to stop, and say how far it has got.
"""
from __future__ import annotations

from PyQt6.QtCore import QThread, pyqtSignal


class Cancelled(Exception):
    """Raised inside a worker when the user has asked it to stop."""


class CancellableWorker(QThread):
    """A QThread whose work reports progress and honours a cancel request.

    Subclasses implement ``work()`` and call ``self.tick(done, total, message)``
    inside their loops.  ``tick`` raises :class:`Cancelled` when a stop has been
    requested, which unwinds the job at a safe point.
    """
    progress = pyqtSignal(int, int, str)                 # done, total, message
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._stop = False

    def cancel(self) -> None:
        self._stop = True

    def tick(self, done: int, total: int, message: str = "") -> None:
        if self._stop:
            raise Cancelled
        self.progress.emit(int(done), int(total), message)

    def work(self):
        raise NotImplementedError

    def run(self):
        try:
            self.finished.emit(self.work())
        except Cancelled:
            self.cancelled.emit()
        except Exception as exc:                          # reported, never silent
            self.error.emit(str(exc))
