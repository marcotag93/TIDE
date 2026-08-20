"""
Console UI Styles and Visual Constants
======================================

ANSI color codes and Unicode characters for the terminal console UI.
Colors are matched to the GUI theme (Cyan primary, Green success).
"""

from typing import List


class Colors:
    """
    ANSI escape codes for terminal colors.

    Color scheme matches the GUI theme from theme.py:
    - Primary: Cyan (#00BCD4)
    - Success: Green (#16a34a)
    - Warning: Amber (#d97706)
    - Error: Red (#dc2626)
    """

    # Reset
    RESET = "\033[0m"

    # Text styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Primary colors (256-color mode for better matching)
    CYAN = "\033[38;5;44m"  # Primary accent (#00BCD4)
    CYAN_BRIGHT = "\033[38;5;51m"  # Hover state
    GREEN = "\033[38;5;34m"  # Success (#16a34a)
    GREEN_BRIGHT = "\033[38;5;40m"  # Success bright
    YELLOW = "\033[38;5;178m"  # Warning (#d97706)
    RED = "\033[38;5;160m"  # Error (#dc2626)

    # Neutral colors
    WHITE = "\033[38;5;255m"  # Text primary
    GRAY = "\033[38;5;245m"  # Text secondary
    DARK_GRAY = "\033[38;5;240m"  # Borders, muted
    LIGHT_GRAY = "\033[38;5;250m"  # Subtle text

    # Background colors
    BG_DARK = "\033[48;5;235m"  # Dark background
    BG_HIGHLIGHT = "\033[48;5;238m"  # Row highlight
    BG_CYAN = "\033[48;5;23m"  # Cyan tint background
    BG_GREEN = "\033[48;5;22m"  # Success background
    BG_RED = "\033[48;5;52m"  # Error background

    @classmethod
    def colorize(cls, text: str, *styles: str) -> str:
        """
        Apply multiple color/style codes to text.

        Args:
            text: Text to colorize
            *styles: Color/style constants to apply

        Returns:
            Styled text with reset at end
        """
        prefix = "".join(styles)
        return f"{prefix}{text}{cls.RESET}"


class BoxChars:
    """
    Unicode box-drawing characters.

    Uses rounded corners for a modern look matching the GUI border-radius.
    """

    # Single-line rounded corners
    TOP_LEFT = "\u256d"  # ╭
    TOP_RIGHT = "\u256e"  # ╮
    BOTTOM_LEFT = "\u2570"  # ╰
    BOTTOM_RIGHT = "\u256f"  # ╯

    # Lines
    HORIZONTAL = "\u2500"  # ─
    VERTICAL = "\u2502"  # │

    # T-junctions
    T_DOWN = "\u252c"  # ┬
    T_UP = "\u2534"  # ┴
    T_RIGHT = "\u251c"  # ├
    T_LEFT = "\u2524"  # ┤

    # Cross
    CROSS = "\u253c"  # ┼

    # Double line (for emphasis)
    DOUBLE_HORIZONTAL = "\u2550"  # ═
    DOUBLE_VERTICAL = "\u2551"  # ║

    # Heavy lines (for section separators)
    HEAVY_HORIZONTAL = "\u2501"  # ━
    HEAVY_VERTICAL = "\u2503"  # ┃


class ProgressChars:
    """
    Unicode characters for progress bars.
    """

    # Main progress characters
    FILLED = "\u2588"  # █ Full block
    EMPTY = "\u2591"  # ░ Light shade

    # Partial fills (1/8 to 7/8)
    PARTIAL: List[str] = [
        "\u258f",  # ▏ 1/8
        "\u258e",  # ▎ 2/8
        "\u258d",  # ▍ 3/8
        "\u258c",  # ▌ 4/8
        "\u258b",  # ▋ 5/8
        "\u258a",  # ▊ 6/8
        "\u2589",  # ▉ 7/8
    ]

    # Alternative style (thinner)
    BAR_START = "\u2595"  # ▕
    BAR_END = "\u258f"  # ▏


class StatusIcons:
    """
    Unicode icons for status indication.
    """

    # Status indicators
    RUNNING = "\u25cf"  # ● Filled circle
    COMPLETE = "\u2713"  # ✓ Check mark
    FAILED = "\u2717"  # ✗ X mark
    PENDING = "\u25cb"  # ○ Empty circle
    WARNING = "\u26a0"  # ⚠ Warning
    INFO = "\u2139"  # ℹ Info

    # Arrows
    ARROW_RIGHT = "\u2192"  # →
    ARROW_LEFT = "\u2190"  # ←
    ARROW_UP = "\u2191"  # ↑
    ARROW_DOWN = "\u2193"  # ↓

    # Bullets
    BULLET = "\u2022"  # •
    DIAMOND = "\u25c6"  # ◆
    TRIANGLE = "\u25b6"  # ▶

    # Spinner frames (for animation)
    SPINNER: List[str] = [
        "\u280b",  # ⠋
        "\u2819",  # ⠙
        "\u2839",  # ⠹
        "\u2838",  # ⠸
        "\u283c",  # ⠼
        "\u2834",  # ⠴
        "\u2826",  # ⠦
        "\u2827",  # ⠧
        "\u2807",  # ⠇
        "\u280f",  # ⠏
    ]

    # Alternative spinner (simpler)
    SPINNER_SIMPLE: List[str] = ["|", "/", "-", "\\"]


class Symbols:
    """
    Additional Unicode symbols for UI elements.
    """

    # Section markers
    SECTION_START = "\u2500\u2500"  # ──
    SECTION_END = "\u2500\u2500"  # ──

    # List markers
    LIST_ITEM = "\u2023"  # ‣

    # Time/clock
    CLOCK = "\u23f1"  # ⏱
    HOURGLASS = "\u23f3"  # ⏳

    # File/folder
    FOLDER = "\u1f4c1"  # 📁 (may not render in all terminals)
    FILE = "\u1f4c4"  # 📄

    # Simple alternatives for better compatibility
    FOLDER_ASCII = ">"
    FILE_ASCII = "-"


def supports_unicode() -> bool:
    """
    Check if the terminal likely supports Unicode.

    Returns:
        True if Unicode is likely supported
    """
    import os
    import sys

    # Check encoding
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if "utf" in encoding.lower():
        return True

    # Check LANG environment variable
    lang = os.environ.get("LANG", "").lower()
    if "utf" in lang:
        return True

    # Check terminal type
    term = os.environ.get("TERM", "").lower()
    if any(t in term for t in ["xterm", "vt100", "screen", "tmux", "linux"]):
        return True

    # Windows Terminal and modern Windows support Unicode
    if sys.platform == "win32":
        # Windows 10 1903+ supports UTF-8 by default
        try:
            import ctypes

            _ = ctypes.windll.kernel32
            # Check if running in Windows Terminal or modern console
            return True
        except Exception:
            pass

    return False


def supports_colors() -> bool:
    """
    Check if the terminal supports ANSI colors.

    Returns:
        True if colors are likely supported
    """
    import os
    import sys

    # Check if stdout is a TTY
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False

    # Check TERM environment variable
    term = os.environ.get("TERM", "").lower()
    if term in ("dumb", ""):
        return False

    # Check NO_COLOR environment variable (standard)
    if os.environ.get("NO_COLOR"):
        return False

    # Check FORCE_COLOR environment variable
    if os.environ.get("FORCE_COLOR"):
        return True

    # Windows check
    if sys.platform == "win32":
        # Enable ANSI on Windows 10+
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            # Enable virtual terminal processing
            kernel32.SetConsoleMode(
                kernel32.GetStdHandle(-11),  # STD_OUTPUT_HANDLE
                7,  # ENABLE_PROCESSED_OUTPUT | ENABLE_WRAP_AT_EOL_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            )
            return True
        except Exception:
            return False

    return True


# Convenience functions for common patterns


def status_icon(status: str) -> str:
    """
    Get colored status icon for a status string.

    Args:
        status: One of 'running', 'complete', 'failed', 'pending', 'warning'

    Returns:
        Colored status icon string
    """
    icons = {
        "running": Colors.colorize(StatusIcons.RUNNING, Colors.CYAN),
        "complete": Colors.colorize(StatusIcons.COMPLETE, Colors.GREEN),
        "failed": Colors.colorize(StatusIcons.FAILED, Colors.RED),
        "pending": Colors.colorize(StatusIcons.PENDING, Colors.GRAY),
        "warning": Colors.colorize(StatusIcons.WARNING, Colors.YELLOW),
        "idle": Colors.colorize(StatusIcons.PENDING, Colors.DARK_GRAY),
    }
    return icons.get(status.lower(), StatusIcons.PENDING)


def phase_color(phase: str) -> str:
    """
    Get appropriate color for a workflow phase.

    Args:
        phase: Phase name (e.g., 'Optimization', 'FEM Simulation')

    Returns:
        ANSI color code
    """
    phase_colors = {
        "optimization": Colors.YELLOW,
        "fem simulation": Colors.CYAN,
        "e-field sampling": Colors.CYAN_BRIGHT,
        "activating function": Colors.GREEN,
        "bundle analysis": Colors.GREEN_BRIGHT,
        "saving results": Colors.GRAY,
        "complete": Colors.GREEN,
        "failed": Colors.RED,
        "idle": Colors.DARK_GRAY,
        "starting": Colors.LIGHT_GRAY,
    }
    return phase_colors.get(phase.lower(), Colors.WHITE)
