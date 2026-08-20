"""
Terminal Control Module
=======================

Low-level terminal control using ANSI escape sequences.
Provides cursor manipulation, screen control, and keyboard input handling.
"""

import os
import signal
import sys
from contextlib import contextmanager
from typing import Optional, Tuple


class Terminal:
    """
    Low-level terminal control for the console UI.

    Handles:
    - Terminal size detection and resize events
    - Cursor positioning and visibility
    - Screen clearing and buffer management
    - Non-blocking keyboard input
    - Raw mode for keyboard capture

    This class uses ANSI escape sequences and is compatible with
    most Unix terminals and Windows Terminal (Windows 10+).
    """

    def __init__(self):
        """Initialize terminal controller."""
        self._original_settings = None
        self._is_tty = sys.stdout.isatty()
        self._in_raw_mode = False
        self._resize_pending = False
        self._width, self._height = self.get_size()

    @property
    def is_interactive(self) -> bool:
        """
        Check if running in an interactive terminal.

        Returns:
            True if stdout is connected to a TTY
        """
        return self._is_tty

    def get_size(self) -> Tuple[int, int]:
        """
        Get terminal dimensions.

        Returns:
            Tuple of (columns, rows)
        """
        try:
            size = os.get_terminal_size()
            return size.columns, size.lines
        except OSError:
            # Fallback for non-TTY or when size detection fails
            return 80, 24

    def setup_resize_handler(self, enabled: bool = True) -> None:
        """
        Enable or disable the terminal resize (SIGWINCH) handler.

        When enabled, SIGWINCH signals will set a flag that can be
        checked via check_resize().

        Args:
            enabled: Whether to enable the handler.
        """
        # SIGWINCH is Unix-only
        if hasattr(signal, "SIGWINCH"):
            if enabled:
                signal.signal(signal.SIGWINCH, self._handle_resize)
            else:
                signal.signal(signal.SIGWINCH, signal.SIG_DFL)

    def _handle_resize(self, signum: int, frame) -> None:
        """Handle SIGWINCH resize signal. Sets flag for thread-safe processing."""
        self._resize_pending = True
        # Update cached size immediately (safe from signal handler)
        try:
            size = os.get_terminal_size()
            self._width, self._height = size.columns, size.lines
        except OSError:
            pass

    def check_resize(self) -> Optional[Tuple[int, int]]:
        """
        Check if a resize event is pending.

        Returns:
            Tuple of (cols, rows) if resized, None otherwise.
        """
        if self._resize_pending:
            self._resize_pending = False
            return self._width, self._height
        return None

    # =========================================================================
    # Cursor Control
    # =========================================================================

    @staticmethod
    def move_cursor(row: int, col: int) -> str:
        """
        Get ANSI sequence to move cursor to position.

        Args:
            row: Target row (1-indexed)
            col: Target column (1-indexed)

        Returns:
            ANSI escape sequence
        """
        return f"\033[{row};{col}H"

    @staticmethod
    def cursor_up(n: int = 1) -> str:
        """Move cursor up n lines."""
        return f"\033[{n}A"

    @staticmethod
    def cursor_down(n: int = 1) -> str:
        """Move cursor down n lines."""
        return f"\033[{n}B"

    @staticmethod
    def cursor_forward(n: int = 1) -> str:
        """Move cursor forward (right) n columns."""
        return f"\033[{n}C"

    @staticmethod
    def cursor_back(n: int = 1) -> str:
        """Move cursor back (left) n columns."""
        return f"\033[{n}D"

    @staticmethod
    def cursor_to_column(col: int) -> str:
        """Move cursor to column n (1-indexed)."""
        return f"\033[{col}G"

    @staticmethod
    def hide_cursor() -> str:
        """Get ANSI sequence to hide cursor."""
        return "\033[?25l"

    @staticmethod
    def show_cursor() -> str:
        """Get ANSI sequence to show cursor."""
        return "\033[?25h"

    @staticmethod
    def save_cursor() -> str:
        """Get ANSI sequence to save cursor position."""
        return "\033[s"

    @staticmethod
    def restore_cursor() -> str:
        """Get ANSI sequence to restore cursor position."""
        return "\033[u"

    # =========================================================================
    # Screen Control
    # =========================================================================

    @staticmethod
    def clear_screen() -> str:
        """Get ANSI sequence to clear entire screen."""
        return "\033[2J"

    @staticmethod
    def clear_line() -> str:
        """Get ANSI sequence to clear current line."""
        return "\033[2K"

    @staticmethod
    def clear_to_end_of_line() -> str:
        """Get ANSI sequence to clear from cursor to end of line."""
        return "\033[K"

    @staticmethod
    def clear_to_end_of_screen() -> str:
        """Get ANSI sequence to clear from cursor to end of screen."""
        return "\033[J"

    @staticmethod
    def clear_to_start_of_screen() -> str:
        """Get ANSI sequence to clear from cursor to start of screen."""
        return "\033[1J"

    @staticmethod
    def scroll_up(n: int = 1) -> str:
        """Scroll screen up n lines."""
        return f"\033[{n}S"

    @staticmethod
    def scroll_down(n: int = 1) -> str:
        """Scroll screen down n lines."""
        return f"\033[{n}T"

    # =========================================================================
    # Alternate Buffer
    # =========================================================================

    @contextmanager
    def alternate_buffer(self):
        """
        Context manager for using the alternate screen buffer.

        The alternate buffer preserves the original terminal content
        and restores it when the context exits.

        Usage:
            with terminal.alternate_buffer():
                # Draw UI in alternate buffer
                pass
            # Original content is restored
        """
        if not self._is_tty:
            yield
            return

        # Enter alternate buffer
        sys.stdout.write("\033[?1049h")
        sys.stdout.flush()
        try:
            yield
        finally:
            # Exit alternate buffer
            sys.stdout.write("\033[?1049l")
            sys.stdout.flush()

    # =========================================================================
    # Keyboard Input
    # =========================================================================

    def get_keypress(self, timeout: float = 0.1) -> Optional[str]:
        """
        Non-blocking keypress detection.

        Args:
            timeout: Maximum time to wait for input in seconds

        Returns:
            Key character if pressed, None if no input
        """
        if sys.platform == "win32":
            return self._get_keypress_windows(timeout)
        else:
            return self._get_keypress_unix(timeout)

    def _get_keypress_unix(self, timeout: float) -> Optional[str]:
        """Unix implementation of non-blocking keypress."""
        import select
        import termios
        import tty

        if not self._is_tty:
            return None

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                ch = sys.stdin.read(1)
                # Handle escape sequences (arrow keys, etc.)
                if ch == "\x1b":
                    # Check for more characters (extended escape sequences)
                    # We use a short timeout and read hasta 5 more chars
                    for _ in range(5):
                        rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                        if rlist:
                            ch += sys.stdin.read(1)
                        else:
                            break
                return ch
            return None
        except Exception:
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def _get_keypress_windows(self, timeout: float) -> Optional[str]:
        """Windows implementation of non-blocking keypress."""
        try:
            import msvcrt
            import time

            start = time.time()
            while (time.time() - start) < timeout:
                if msvcrt.kbhit():
                    ch = msvcrt.getch()
                    # Handle extended keys
                    if ch in (b"\x00", b"\xe0"):
                        ch = msvcrt.getch()
                        return f'\x1b[{ch.decode("utf-8", errors="ignore")}'
                    return ch.decode("utf-8", errors="ignore")
                time.sleep(0.01)
            return None
        except ImportError:
            return None

    @contextmanager
    def raw_mode(self):
        """
        Context manager for raw terminal mode (Unix only).

        In raw mode, input is not line-buffered and special
        characters are not processed.
        """
        if sys.platform == "win32" or not self._is_tty:
            yield
            return

        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        self._original_settings = old_settings

        try:
            tty.setraw(fd)
            self._in_raw_mode = True
            yield
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            self._in_raw_mode = False

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def write(self, text: str) -> None:
        """
        Write text to terminal and flush.

        Args:
            text: Text to write (may include ANSI sequences)
        """
        sys.stdout.write(text)
        sys.stdout.flush()

    def write_at(self, row: int, col: int, text: str) -> None:
        """
        Write text at specific position.

        Args:
            row: Target row (1-indexed)
            col: Target column (1-indexed)
            text: Text to write
        """
        self.write(f"{self.move_cursor(row, col)}{text}")

    def clear_and_home(self) -> None:
        """Clear screen and move cursor to top-left."""
        self.write(f"{self.clear_screen()}{self.move_cursor(1, 1)}")

    def _save_terminal_state(self) -> None:
        """Save current terminal state (Unix only)."""
        if sys.platform != "win32" and self._is_tty:
            try:
                import termios

                self._original_settings = termios.tcgetattr(sys.stdin.fileno())
            except Exception:
                pass

    def _restore_terminal_state(self) -> None:
        """Restore saved terminal state (Unix only)."""
        if sys.platform != "win32" and self._is_tty and self._original_settings:
            try:
                import termios

                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._original_settings)
            except Exception:
                pass

    def setup(self) -> None:
        """Setup terminal for UI rendering."""
        if not self._is_tty:
            return

        # Save current terminal state
        self._save_terminal_state()

        # Enter alternate screen buffer
        self.write("\033[?1049h")

        # Hide cursor
        self.write(self.hide_cursor())

        # Move to top-left
        self.write(self.move_cursor(1, 1))

    def cleanup(self) -> None:
        """Restore terminal to original state."""
        if not self._is_tty:
            return

        # Show cursor
        self.write(self.show_cursor())

        # Exit alternate screen buffer
        self.write("\033[?1049l")

        # Restore terminal state
        self._restore_terminal_state()

    def bell(self) -> None:
        """Sound terminal bell."""
        self.write("\a")

    def set_title(self, title: str) -> None:
        """
        Set terminal window title.

        Args:
            title: New window title
        """
        # Works on xterm, GNOME Terminal, Windows Terminal, etc.
        self.write(f"\033]0;{title}\a")

    # =========================================================================
    # Line Drawing Helpers
    # =========================================================================

    def draw_horizontal_line(self, row: int, col: int, width: int, char: str = "\u2500") -> None:
        """
        Draw a horizontal line.

        Args:
            row: Starting row (1-indexed)
            col: Starting column (1-indexed)
            width: Line width in characters
            char: Character to use for the line
        """
        self.write_at(row, col, char * width)

    def draw_vertical_line(self, row: int, col: int, height: int, char: str = "\u2502") -> None:
        """
        Draw a vertical line.

        Args:
            row: Starting row (1-indexed)
            col: Column (1-indexed)
            height: Line height in characters
            char: Character to use for the line
        """
        for i in range(height):
            self.write_at(row + i, col, char)


# Convenience function for quick terminal operations
def get_terminal() -> Terminal:
    """
    Get a Terminal instance.

    Returns:
        Terminal controller instance
    """
    return Terminal()
