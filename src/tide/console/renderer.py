"""
UI Renderer Module
==================

High-level rendering primitives for the console UI.
Renders boxes, tables, progress bars, and other visual components.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .styles import BoxChars, Colors, ProgressChars, StatusIcons, phase_color


@dataclass
class TableColumn:
    """
    Table column definition.

    Attributes:
        header: Column header text
        width: Column width in characters
        align: Text alignment ('left', 'center', 'right')
        key: Key to use when extracting data from row dicts
    """

    header: str
    width: int
    align: str = "left"
    key: Optional[str] = None

    def __post_init__(self):
        if self.key is None:
            # Convert header to snake_case key
            self.key = self.header.lower().replace(" ", "_")


class Renderer:
    """
    High-level UI rendering engine.

    Renders visual components for the console UI:
    - Header boxes with subject ID
    - Step progress indicators
    - Progress bars with percentage
    - Worker status tables
    - Completed results lists
    - Summary sections
    """

    def __init__(self, terminal_width: int = 80):
        """
        Initialize renderer.

        Args:
            terminal_width: Available terminal width for rendering
        """
        self.width = min(terminal_width, 100)  # Cap at reasonable width
        self._min_width = 60  # Minimum width for proper rendering

    def set_width(self, width: int) -> None:
        """Update terminal width for rendering."""
        self.width = max(self._min_width, min(width, 100))

    # =========================================================================
    # Header Components
    # =========================================================================

    def render_header_box(
        self, title: str, subtitle: str = "", border_color: str = Colors.CYAN
    ) -> str:
        """
        Render a boxed header.

        Example output:
        ╭──────────────────────────────────────────────────────╮
        │        TIDE Grid Search Pipeline - sub-01           │
        ╰──────────────────────────────────────────────────────╯

        Args:
            title: Main title text
            subtitle: Optional subtitle below title
            border_color: ANSI color for border

        Returns:
            Rendered header string with newlines
        """
        lines = []
        inner_width = self.width - 2  # Account for side borders

        # Top border
        top = f"{border_color}{BoxChars.TOP_LEFT}{BoxChars.HORIZONTAL * inner_width}{BoxChars.TOP_RIGHT}{Colors.RESET}"
        lines.append(top)

        # Title line (centered)
        title_text = title[: inner_width - 2]  # Truncate if needed
        title_padded = title_text.center(inner_width)
        title_line = f"{border_color}{BoxChars.VERTICAL}{Colors.RESET}{Colors.BOLD}{Colors.WHITE}{title_padded}{Colors.RESET}{border_color}{BoxChars.VERTICAL}{Colors.RESET}"
        lines.append(title_line)

        # Subtitle line (if provided)
        if subtitle:
            sub_text = subtitle[: inner_width - 2]
            sub_padded = sub_text.center(inner_width)
            sub_line = f"{border_color}{BoxChars.VERTICAL}{Colors.RESET}{Colors.GRAY}{sub_padded}{Colors.RESET}{border_color}{BoxChars.VERTICAL}{Colors.RESET}"
            lines.append(sub_line)

        # Bottom border
        bottom = f"{border_color}{BoxChars.BOTTOM_LEFT}{BoxChars.HORIZONTAL * inner_width}{BoxChars.BOTTOM_RIGHT}{Colors.RESET}"
        lines.append(bottom)

        return "\n".join(lines)

    def render_section_header(self, text: str, color: str = Colors.CYAN) -> str:
        """
        Render a section header with decorative lines.

        Example: ── Step 6/7: Grid Search ──────────────────

        Args:
            text: Header text
            color: ANSI color for decorative elements

        Returns:
            Rendered section header string
        """
        prefix = f"{color}{BoxChars.HORIZONTAL * 2}{Colors.RESET} "
        text_part = f"{Colors.BOLD}{text}{Colors.RESET} "
        prefix_len = 3  # "── "
        text_len = len(text) + 1  # text + space

        remaining = self.width - prefix_len - text_len
        suffix = f"{color}{BoxChars.HORIZONTAL * max(0, remaining)}{Colors.RESET}"

        return f"{prefix}{text_part}{suffix}"

    # =========================================================================
    # Progress Components
    # =========================================================================

    def render_step_indicator(self, current: int, total: int, label: str, workers: int = 1) -> str:
        """
        Render step progress indicator.

        Example: Step 6/7: Grid Search (4 workers)

        Args:
            current: Current step number
            total: Total number of steps
            label: Step label
            workers: Number of workers (shown if > 1)

        Returns:
            Rendered step indicator string
        """
        worker_info = f" ({workers} worker{'s' if workers > 1 else ''})" if workers > 1 else ""
        step_part = f"{Colors.CYAN}Step {current}/{total}:{Colors.RESET} "
        label_part = f"{Colors.BOLD}{Colors.WHITE}{label}{Colors.RESET}"
        worker_part = f"{Colors.GRAY}{worker_info}{Colors.RESET}"

        return f"{step_part}{label_part}{worker_part}"

    def render_sequential_step(
        self,
        step: int,
        total_steps: int,
        step_name: str,
        status: str,
        elapsed_seconds: float,
        spinner_frame: int = 0,
        detail: str = "",
        warnings: Optional[List[str]] = None,
    ) -> str:
        """
        Render a sequential step with spinner animation.

        Example output:
            Step 2/7: CST Analysis
              Processing... (0:42 elapsed)
                └─ Sampling E-field on tractogram...
                ! Warning: Some issue occurred

        Args:
            step: Current step number
            total_steps: Total number of steps
            step_name: Name of the current step
            status: 'pending', 'running', or 'complete'
            elapsed_seconds: Time elapsed for this step
            spinner_frame: Animation frame index
            detail: Optional sub-step detail text
            warnings: Optional list of warning messages

        Returns:
            Rendered step display string
        """
        lines = []

        # Step indicator line
        step_line = self.render_step_indicator(step, total_steps, step_name, workers=1)
        lines.append(step_line)
        lines.append("")

        # Status with spinner or checkmark
        if status == "running":
            spinner = StatusIcons.SPINNER[spinner_frame % len(StatusIcons.SPINNER)]
            elapsed_str = self._format_elapsed(elapsed_seconds)
            status_line = (
                f"  {Colors.CYAN}{spinner}{Colors.RESET} " f"Processing... ({elapsed_str} elapsed)"
            )
        elif status == "complete":
            status_line = f"  {Colors.GREEN}{StatusIcons.COMPLETE}{Colors.RESET} Complete"
        else:  # pending
            status_line = f"  {Colors.GRAY}{StatusIcons.PENDING}{Colors.RESET} Pending"

        lines.append(status_line)

        # Add detail line if provided and status is running
        if detail and status == "running":
            detail_line = f"    {Colors.GRAY}\u2514\u2500 {detail}{Colors.RESET}"
            lines.append(detail_line)

        # Add warnings if any
        if warnings:
            for w in warnings:
                # Indent warning with tree connector
                lines.append(
                    f"      {Colors.GRAY}\u2514\u2500{Colors.RESET} {Colors.YELLOW}! {w}{Colors.RESET}"
                )

        return "\n".join(lines)

    def render_completed_steps(
        self,
        completed_steps: List[int],
        step_names: Dict[int, str],
        step_warnings: Optional[Dict[int, List[str]]] = None,
    ) -> str:
        """
        Render list of completed sequential steps.

        Args:
            completed_steps: List of completed step numbers
            step_names: Mapping of step numbers to names
            step_warnings: Optional mapping of step numbers to warning messages

        Returns:
            Rendered completed steps string
        """
        if not completed_steps:
            return ""

        lines = [f"{Colors.GRAY}Completed:{Colors.RESET}"]

        for step_num in sorted(completed_steps):
            step_name = step_names.get(step_num, f"Step {step_num}")
            lines.append(
                f"  {Colors.GREEN}{StatusIcons.COMPLETE}{Colors.RESET} "
                f"Step {step_num}: {step_name}"
            )

            # Check for warnings for this step
            if step_warnings and step_num in step_warnings:
                for w in step_warnings[step_num]:
                    lines.append(
                        f"      {Colors.GRAY}\u2514\u2500{Colors.RESET} {Colors.YELLOW}! {w}{Colors.RESET}"
                    )

        return "\n".join(lines)

    def _format_elapsed(self, seconds: float) -> str:
        """Format elapsed time as M:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}:{secs:02d}"

    def render_progress_bar(
        self,
        completed: int,
        total: int,
        width: int = 40,
        label: str = "Progress",
        show_percentage: bool = True,
        show_count: bool = True,
        filled_color: str = Colors.GREEN,
        empty_color: str = Colors.DARK_GRAY,
    ) -> str:
        """
        Render a Unicode progress bar.

        Example: Progress: [████████████░░░░░░░░] 8/25 (32%)

        Args:
            completed: Number of completed items
            total: Total number of items
            width: Bar width in characters
            label: Label before the bar
            show_percentage: Show percentage after bar
            show_count: Show count (completed/total)
            filled_color: Color for filled portion
            empty_color: Color for empty portion

        Returns:
            Rendered progress bar string
        """
        if total == 0:
            percent = 0
            filled_width = 0
        else:
            percent = int((completed / total) * 100)
            filled_width = int((completed / total) * width)

        # Build bar characters
        filled = ProgressChars.FILLED * filled_width
        empty = ProgressChars.EMPTY * (width - filled_width)
        bar = f"{filled_color}{filled}{Colors.RESET}{empty_color}{empty}{Colors.RESET}"

        # Build info parts
        parts = []
        if label:
            parts.append(f"{Colors.GRAY}{label}:{Colors.RESET}")

        parts.append(f"[{bar}]")

        if show_count:
            parts.append(f"{completed}/{total}")

        if show_percentage:
            parts.append(f"({Colors.CYAN}{percent:3d}%{Colors.RESET})")

        return " ".join(parts)

    # =========================================================================
    # Table Components
    # =========================================================================

    def render_worker_table(self, workers: List[Dict[str, Any]], columns: List[TableColumn]) -> str:
        """
        Render a worker status table with borders.

        Example:
        ┌──────────┬─────────────┬──────────────────┬──────────┐
        │  Worker  │ Grid Point  │  Current Phase   │   Time   │
        ├──────────┼─────────────┼──────────────────┼──────────┤
        │ Worker 1 │ grid_P07    │ FEM Simulation   │   00:42  │
        │ Worker 2 │ grid_P08    │ E-field Sampling │   00:38  │
        └──────────┴─────────────┴──────────────────┴──────────┘

        Args:
            workers: List of worker state dicts
            columns: List of TableColumn definitions

        Returns:
            Rendered table string with newlines
        """
        lines = []
        border_color = Colors.DARK_GRAY

        # Calculate column widths
        col_widths = [c.width for c in columns]

        # Top border
        top_border = self._make_table_border(col_widths, "top", border_color)
        lines.append(top_border)

        # Header row
        header_cells = []
        for col in columns:
            cell = self._align_text(col.header, col.width, "center")
            header_cells.append(f"{Colors.BOLD}{Colors.WHITE}{cell}{Colors.RESET}")

        header_line = self._make_table_row(header_cells, col_widths, border_color)
        lines.append(header_line)

        # Header separator
        sep = self._make_table_border(col_widths, "middle", border_color)
        lines.append(sep)

        # Data rows
        for i, worker in enumerate(workers):
            row_cells = []
            status = worker.get("status", "idle")
            row_color = self._get_row_color(status)

            for col in columns:
                value = str(worker.get(col.key, "-"))
                cell = self._align_text(value, col.width, col.align)

                # Apply special coloring for phase column
                if col.key == "current_phase":
                    phase = worker.get("current_phase", "")
                    cell_color = phase_color(phase)
                    row_cells.append(f"{cell_color}{cell}{Colors.RESET}")
                else:
                    row_cells.append(f"{row_color}{cell}{Colors.RESET}")

            row_line = self._make_table_row(row_cells, col_widths, border_color)
            lines.append(row_line)

            # Add separator between workers
            if i < len(workers) - 1:
                lines.append(self._make_table_border(col_widths, "middle", border_color))

        # Bottom border
        bottom_border = self._make_table_border(col_widths, "bottom", border_color)
        lines.append(bottom_border)

        return "\n".join(lines)

    def _make_table_border(self, col_widths: List[int], position: str, color: str) -> str:
        """Create a table border line."""
        if position == "top":
            left, mid, right = BoxChars.TOP_LEFT, BoxChars.T_DOWN, BoxChars.TOP_RIGHT
        elif position == "middle":
            left, mid, right = BoxChars.T_RIGHT, BoxChars.CROSS, BoxChars.T_LEFT
        else:  # bottom
            left, mid, right = BoxChars.BOTTOM_LEFT, BoxChars.T_UP, BoxChars.BOTTOM_RIGHT

        parts = [left]
        for i, width in enumerate(col_widths):
            parts.append(BoxChars.HORIZONTAL * width)
            parts.append(mid if i < len(col_widths) - 1 else right)

        return f"{color}{''.join(parts)}{Colors.RESET}"

    def _make_table_row(self, cells: List[str], col_widths: List[int], border_color: str) -> str:
        """Create a table data row."""
        sep = f"{border_color}{BoxChars.VERTICAL}{Colors.RESET}"
        return sep + sep.join(cells) + sep

    @staticmethod
    def _align_text(text: str, width: int, align: str) -> str:
        """Align text within specified width, truncating if needed."""
        # Remove ANSI codes for length calculation
        ansi_pattern = re.compile(r"\x1b\[[0-9;]*m")
        clean_text = ansi_pattern.sub("", text)

        if len(clean_text) > width:
            text = text[: width - 1] + "\u2026"  # Truncate with ellipsis

        clean_text = ansi_pattern.sub("", text)
        padding = width - len(clean_text)

        if align == "center":
            left_pad = padding // 2
            right_pad = padding - left_pad
            return " " * left_pad + text + " " * right_pad
        elif align == "right":
            return " " * padding + text
        else:  # left
            return text + " " * padding

    @staticmethod
    def _get_row_color(status: str) -> str:
        """Get color for a worker row based on status."""
        colors = {
            "running": Colors.WHITE,
            "complete": Colors.GREEN,
            "failed": Colors.RED,
            "idle": Colors.GRAY,
        }
        return colors.get(status.lower(), Colors.WHITE)

    # =========================================================================
    # Results Components
    # =========================================================================

    def render_completed_list(
        self, results: List[Dict[str, Any]], max_items: Optional[int] = None
    ) -> str:
        """
        Render list of completed results.

        Args:
            results: List of result dicts
            max_items: Optional maximum items (if None, shows all)

        Returns:
            Rendered list string with newlines
        """
        if not results:
            return ""

        lines = [f"{Colors.GRAY}Completed:{Colors.RESET}"]

        # Show results (either all or limited by max_items)
        displayed = results[-max_items:] if max_items and len(results) > max_items else results

        for r in displayed:
            success = r.get("success", True)
            icon = StatusIcons.COMPLETE if success else StatusIcons.FAILED
            icon_color = Colors.GREEN if success else Colors.RED
            label = r.get("label", "?")

            if success:
                w_mso = r.get("weighted_mso", 0)
                u_mso = r.get("unweighted_mso", 0)
                intensity_text = (
                    f"{u_mso:.1f}% max output (Unweighted I) - "
                    f"{w_mso:.1f}% max output (Weighted I)"
                )
                lines.append(
                    f"  {icon_color}{icon}{Colors.RESET} {label}: "
                    f"{Colors.CYAN}{intensity_text}{Colors.RESET}"
                )

                # Multiplier (M_CST/M_target) — surfaced inline so single-run
                # workflows (estimation) can read it without a stats panel.
                mult_w = r.get("multiplier_weighted")
                mult_u = r.get("multiplier_unweighted")
                if mult_w is not None or mult_u is not None:
                    mult_text = (
                        f"Multiplier: "
                        f"{(mult_u if mult_u is not None else 0):.4f} (Unweighted) - "
                        f"{(mult_w if mult_w is not None else 0):.4f} (Weighted)"
                    )
                    lines.append(f"    {Colors.GRAY}{mult_text}{Colors.RESET}")
            else:
                error = r.get("error", "Failed")
                lines.append(
                    f"  {icon_color}{icon}{Colors.RESET} {label}: "
                    f"{Colors.RED}{error}{Colors.RESET}"
                )

        return "\n".join(lines)

    def render_results_table(self, results: List[Dict[str, Any]], max_rows: int = 10) -> str:
        """
        Render results as a formatted table.

        Args:
            results: List of result dicts
            max_rows: Maximum rows to display

        Returns:
            Rendered table string
        """
        columns = [
            TableColumn("Grid Point", 12, "left", "label"),
            TableColumn("Cortex Coords", 18, "center", "cortex_coords"),
            TableColumn("Unweighted I", 14, "right", "unweighted_mso"),
            TableColumn("Weighted I", 14, "right", "weighted_mso"),
        ]

        # Format results for table
        table_data = []
        for r in results[:max_rows]:
            coords = r.get("cortex_coord", [0, 0, 0])
            if isinstance(coords, list) and len(coords) >= 3:
                coords_str = f"[{coords[0]:.1f}, {coords[1]:.1f}, ...]"
            else:
                coords_str = str(coords)[:16]

            table_data.append(
                {
                    "label": r.get("label", "?"),
                    "cortex_coords": coords_str,
                    "unweighted_mso": f"{r.get('unweighted_mso', 0):.1f}%",
                    "weighted_mso": f"{r.get('weighted_mso', 0):.1f}%",
                    "status": "complete" if r.get("success", True) else "failed",
                }
            )

        return self.render_worker_table(table_data, columns)

    # =========================================================================
    # Summary Components
    # =========================================================================

    def render_summary_box(
        self, title: str, rows: List[Tuple[str, str]], border_color: str = Colors.GREEN
    ) -> str:
        """
        Render a summary box with key-value pairs.

        Example:
        ╭──────────────────────────────────────────────────────╮
        │                  PIPELINE COMPLETE                   │
        ├──────────────────────────────────────────────────────┤
        │  Total time: 15m 32.4s                              │
        │  Grid points: 25/25 successful                       │
        ╰──────────────────────────────────────────────────────╯

        Args:
            title: Box title
            rows: List of (label, value) tuples
            border_color: ANSI color for border

        Returns:
            Rendered summary box string
        """
        lines = []
        inner_width = self.width - 2

        # Top border
        lines.append(
            f"{border_color}{BoxChars.TOP_LEFT}"
            f"{BoxChars.HORIZONTAL * inner_width}"
            f"{BoxChars.TOP_RIGHT}{Colors.RESET}"
        )

        # Title
        title_padded = title.center(inner_width)
        lines.append(
            f"{border_color}{BoxChars.VERTICAL}{Colors.RESET}"
            f"{Colors.BOLD}{Colors.WHITE}{title_padded}{Colors.RESET}"
            f"{border_color}{BoxChars.VERTICAL}{Colors.RESET}"
        )

        # Title separator
        lines.append(
            f"{border_color}{BoxChars.T_RIGHT}"
            f"{BoxChars.HORIZONTAL * inner_width}"
            f"{BoxChars.T_LEFT}{Colors.RESET}"
        )

        # Content rows
        for label, value in rows:
            content = f"  {label}: {Colors.CYAN}{value}{Colors.RESET}"
            # Calculate padding (accounting for ANSI codes)
            visible_len = len(label) + len(str(value)) + 4  # "  " + ": " + value
            padding = inner_width - visible_len
            padded_content = content + " " * max(0, padding)

            lines.append(
                f"{border_color}{BoxChars.VERTICAL}{Colors.RESET}"
                f"{padded_content}"
                f"{border_color}{BoxChars.VERTICAL}{Colors.RESET}"
            )

        # Bottom border
        lines.append(
            f"{border_color}{BoxChars.BOTTOM_LEFT}"
            f"{BoxChars.HORIZONTAL * inner_width}"
            f"{BoxChars.BOTTOM_RIGHT}{Colors.RESET}"
        )

        return "\n".join(lines)

    def render_output_files_list(self, files: List[Tuple[str, str]]) -> str:
        """
        Render list of output files using absolute, copy-pasteable paths.

        Example:
        Output files:
          → Results CSV: /full/abs/path/TIDE_grid_results.csv
          → Summary: /full/abs/path/TIDE_Grid_Summary_target.txt

        Args:
            files: List of (label, path) tuples. Paths are resolved to
                their absolute form so the user can copy them straight
                from the terminal without truncation.

        Returns:
            Rendered file list string
        """
        from pathlib import Path

        lines = [f"{Colors.GRAY}Output files:{Colors.RESET}"]

        for label, path in files:
            try:
                path_str = str(Path(path).expanduser().resolve())
            except (OSError, RuntimeError):
                path_str = str(path)

            lines.append(
                f"  {Colors.CYAN}{StatusIcons.ARROW_RIGHT}{Colors.RESET} "
                f"{label}: {Colors.WHITE}{path_str}{Colors.RESET}"
            )

        return "\n".join(lines)

    # =========================================================================
    # Status Line Components
    # =========================================================================

    def render_status_line(
        self, elapsed_seconds: float, hint: str = "Press 'f' for focus mode"
    ) -> str:
        """
        Render bottom status line with elapsed time and hints.

        Args:
            elapsed_seconds: Elapsed time in seconds
            hint: Keyboard hint text

        Returns:
            Rendered status line string
        """
        minutes = int(elapsed_seconds // 60)
        seconds = int(elapsed_seconds % 60)
        time_str = f"{minutes}m {seconds}s"

        return f"{Colors.GRAY}Elapsed: {time_str} | " f"{hint}{Colors.RESET}"

    def render_focus_header(self, worker_id: int, point_label: str, num_workers: int) -> str:
        """
        Render header for focus mode.

        Args:
            worker_id: Currently focused worker (0-indexed)
            point_label: Grid point being processed
            num_workers: Total number of workers

        Returns:
            Rendered focus header string
        """
        sep_line = f"{Colors.CYAN}{BoxChars.DOUBLE_HORIZONTAL * self.width}{Colors.RESET}"
        title = f"Focus Mode: Worker {worker_id + 1} - {point_label or 'Idle'}"
        hint = f"Press 'q' to exit | 0-{num_workers - 1} to switch worker"

        lines = [
            sep_line,
            f"{Colors.BOLD}{Colors.WHITE}{title}{Colors.RESET}",
            f"{Colors.GRAY}{hint}{Colors.RESET}",
            sep_line,
        ]

        return "\n".join(lines)
