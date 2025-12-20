"""Journal log writer for command-style logging."""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import TextIO, Optional


class JournalWriter:
    """Writes all events and print statements to a journal.log file.

    This mimics commercial FEA software journal logs that record all
    commands, messages, and key events in chronological order.
    """

    def __init__(self, path: Path):
        """Initialize journal writer.

        Parameters
        ----------
        path : Path
            Path to journal.log file
        """
        self.path = Path(path)
        self._file: Optional[TextIO] = None

    def __enter__(self) -> JournalWriter:
        """Open journal file for writing."""
        self._file = self.path.open("w", encoding="utf-8", buffering=1)
        self._write_header()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close journal file."""
        if self._file is not None:
            if exc_type is not None:
                self.write(f"\n!!! EXCEPTION: {exc_type.__name__}: {exc_val}")
            self._write_footer()
            self._file.close()
            self._file = None

    def _write_header(self) -> None:
        """Write journal header."""
        if self._file is not None:
            self._file.write("=" * 80 + "\n")
            self._file.write("dc_solver JOURNAL LOG\n")
            self._file.write(f"Started: {datetime.now().isoformat()}\n")
            self._file.write("=" * 80 + "\n\n")
            self._file.flush()

    def _write_footer(self) -> None:
        """Write journal footer."""
        if self._file is not None:
            self._file.write("\n" + "=" * 80 + "\n")
            self._file.write(f"Ended: {datetime.now().isoformat()}\n")
            self._file.write("=" * 80 + "\n")
            self._file.flush()

    def write(self, message: str) -> None:
        """Write a message to the journal.

        Parameters
        ----------
        message : str
            Message to write
        """
        if self._file is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._file.write(f"[{timestamp}] {message}\n")
            self._file.flush()

    def write_raw(self, message: str) -> None:
        """Write a raw message without timestamp.

        Parameters
        ----------
        message : str
            Message to write
        """
        if self._file is not None:
            self._file.write(message)
            if not message.endswith("\n"):
                self._file.write("\n")
            self._file.flush()

    def write_separator(self, char: str = "-", length: int = 80) -> None:
        """Write a separator line.

        Parameters
        ----------
        char : str, default='-'
            Character to use for separator
        length : int, default=80
            Length of separator line
        """
        self.write_raw(char * length)
