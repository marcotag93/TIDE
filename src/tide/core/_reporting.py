from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional


def _parse_report_sections(lines: List[str]) -> List[Dict[str, Any]]:
    sections = []
    current = {"title": "Preamble", "lines": []}

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("--- ") and stripped.endswith(" ---"):
            if current["lines"] or current["title"] != "Preamble":
                sections.append(current)
            current = {"title": stripped.strip("- ").strip(), "lines": []}
        elif stripped and set(stripped) <= {"="}:
            continue
        else:
            current["lines"].append(line)

    if current["lines"] or current["title"] != "Preamble":
        sections.append(current)

    for section in sections:
        fields = {}
        for line in section["lines"]:
            stripped = line.strip()
            if ":" in stripped and "|" not in stripped:
                key, val = stripped.split(":", 1)
                fields[key.strip()] = val.strip()
        if fields:
            section["fields"] = fields

    return sections


def _collect_report_fields(sections: List[Dict[str, Any]]) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for section in sections:
        for key, value in section.get("fields", {}).items():
            fields.setdefault(key, value)
    return fields


def _discover_report_images(base_dir: Path, limit: int = 12) -> List[Path]:
    candidates: List[Path] = []
    for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        candidates.extend(base_dir.glob(pattern))
    for folder_name in ("visualizations", "visualization"):
        folder = base_dir / folder_name
        if folder.exists():
            for pattern in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
                candidates.extend(folder.glob(pattern))

    def rank(path: Path) -> tuple:
        name = path.name.lower()
        priority = 5
        for idx, token in enumerate(
            ("composite", "depth_analysis", "roi_side", "roi_front", "roi_top", "oblique")
        ):
            if token in name:
                priority = idx
                break
        return (priority, name)

    unique = sorted({path.resolve(): path for path in candidates}.values(), key=rank)
    return unique[:limit]


def _relative_path(path: Optional[Path], start: Path) -> str:
    if path is None:
        return "N/A"
    try:
        return path.resolve().relative_to(start.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _human_report_type(report_type: str) -> str:
    return report_type.replace("_", " ").strip().title()


def _report_overview(report_type: str, fields: Dict[str, str], data: Dict[str, Any]) -> str:
    subject = fields.get("Subject") or data.get("subject_id") or "N/A"
    target = data.get("target_label") or fields.get("Target Tractogram") or fields.get("Prefix")
    workflow = data.get("workflow") or _human_report_type(report_type)
    if target:
        return (
            f"{workflow} report for subject {subject}, focused on {target}. "
            "This HTML sidecar summarizes the existing TXT and JSON outputs and links "
            "available visualization files without embedding large binary data."
        )
    return (
        f"{workflow} report for subject {subject}. This HTML sidecar summarizes "
        "the existing TXT and JSON outputs and links available visualization files "
        "without embedding large binary data."
    )


_STATUS_OK = {"PASS", "WITHIN_RANGE", "OK", "VALID"}
_STATUS_WARN = {
    "WARN",
    "WARNING",
    "CLAMPED",
    "CLAMPED_LOW",
    "CLAMPED_HIGH",
    "DEVICE_LIMITED",
}
_STATUS_BAD = {"FAIL", "FAILED", "ERROR", "ESTIMATION_FAILED", "INVALID"}


def _status_class(token: str) -> Optional[str]:
    t = token.strip().upper()
    if t in _STATUS_OK:
        return "ok"
    if t in _STATUS_WARN:
        return "warn"
    if t in _STATUS_BAD:
        return "bad"
    return None


def _decorate_cell(escaped: str) -> str:
    """Wrap a leading QC/flag status token in a coloured badge (input pre-escaped)."""
    if not escaped:
        return escaped
    head = escaped.split(" ", 1)[0]
    cls = _status_class(head)
    if cls is None:
        return escaped
    return f'<span class="badge badge-{cls}">{head}</span>{escaped[len(head):]}'


def _slug(text: str) -> str:
    keep = [c.lower() if (c.isalnum()) else "-" for c in str(text)]
    slug = "".join(keep).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "section"


def _render_kv_table(items: List[tuple]) -> str:
    rows = []
    for key, value in items:
        if value in (None, "", []):
            continue
        val = _decorate_cell(escape(str(value)))
        rows.append("<tr>" f"<th>{escape(str(key))}</th>" f"<td>{val}</td>" "</tr>")
    if not rows:
        return ""
    return '<table class="kv-table"><tbody>' + "".join(rows) + "</tbody></table>"


def _is_separator_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= {"-", "|", " "}


def _render_pipe_table(lines: List[str]) -> str:
    rows = []
    for line in lines:
        if _is_separator_line(line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if any(cells):
            rows.append(cells)

    if not rows:
        return ""

    header, *body = rows
    thead = "<thead><tr>" + "".join(f"<th>{escape(cell)}</th>" for cell in header) + "</tr></thead>"
    tbody_rows = []
    for row in body:
        tbody_rows.append(
            "<tr>" + "".join(f"<td>{_decorate_cell(escape(cell))}</td>" for cell in row) + "</tr>"
        )
    tbody = "<tbody>" + "".join(tbody_rows) + "</tbody>"
    return f'<div class="table-wrap"><table>{thead}{tbody}</table></div>'


def _render_section(section: Dict[str, Any]) -> str:
    title = escape(str(section["title"]))
    fields = section.get("fields", {})
    field_lines = {f"{key}: {value}" for key, value in fields.items()}
    blocks = []
    pending_text = []
    pending_table = []

    def flush_text() -> None:
        if pending_text:
            text = "\n".join(pending_text).strip()
            if text:
                blocks.append(f"<pre>{escape(text)}</pre>")
            pending_text.clear()

    def flush_table() -> None:
        if pending_table:
            table = _render_pipe_table(pending_table)
            if table:
                blocks.append(table)
            pending_table.clear()

    if fields:
        blocks.append(_render_kv_table(list(fields.items())))

    for line in section.get("lines", []):
        stripped = line.strip()
        if not stripped or stripped in field_lines:
            continue
        if "|" in line or (pending_table and _is_separator_line(line)):
            flush_text()
            pending_table.append(line)
        else:
            flush_table()
            pending_text.append(line)

    flush_table()
    flush_text()

    if not blocks:
        return ""
    sec_id = _slug(section["title"])
    return f'<section id="sec-{sec_id}"><h2>{title}</h2>{"".join(blocks)}</section>'


_ICON_EXTERNAL = (
    '<svg class="ext" viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">'
    '<path fill="currentColor" d="M6.5 2H14v7.5h-2V5.41l-6.3 6.3-1.41-1.42L10.59 4H6.5V2z"/>'
    '<path fill="currentColor" d="M2 5h4v2H4v7h7v-2h2v4H2V5z"/></svg>'
)

_REPORT_CSS = """
:root{
  --bg:#f5f6f8; --panel:#ffffff; --ink:#161b22; --muted:#5c6773;
  --line:#e3e7ec; --line-2:#cdd4dd; --accent:#245c73; --accent-2:#2f7c99;
  --accent-soft:#e9f1f4; --band:#eef4f6;
  --mono:"SFMono-Regular",Consolas,"Liberation Mono",Menlo,monospace;
  --ok-bg:#e6f4ec; --ok-fg:#1c7a49; --warn-bg:#fbf0d9; --warn-fg:#8a5a00;
  --bad-bg:#fbe7e6; --bad-fg:#a4302c;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--bg);color:var(--ink);line-height:1.55;font-size:15px;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
main{max-width:1120px;margin:0 auto;padding:34px 22px 24px}
a{color:var(--accent-2);text-decoration:none}
a:hover{text-decoration:underline}
header.rep{border-left:4px solid var(--accent);padding:2px 0 16px 18px}
header.rep h1{margin:0 0 6px;font-size:27px;line-height:1.2;letter-spacing:-.01em;font-weight:700}
header.rep .lead{color:var(--muted);max-width:82ch;margin:0 0 15px;font-size:14px}
.chips{display:flex;flex-wrap:wrap;gap:8px}
.chip{font-size:12.5px;color:var(--ink);background:var(--panel);border:1px solid var(--line);
  border-radius:999px;padding:4px 12px}
.chip b{color:var(--muted);font-weight:600;margin-right:5px;font-size:11px;
  text-transform:uppercase;letter-spacing:.05em}
nav.toc{position:sticky;top:0;z-index:5;display:flex;flex-wrap:wrap;gap:2px;margin:10px 0 2px;
  background:rgba(245,246,248,.93);backdrop-filter:blur(6px);
  border-bottom:1px solid var(--line);padding:8px 0}
nav.toc a{font-size:12.5px;color:var(--muted);padding:5px 11px;border-radius:6px}
nav.toc a:hover{color:var(--ink);background:var(--accent-soft);text-decoration:none}
section{background:var(--panel);border:1px solid var(--line);border-radius:10px;
  padding:18px 20px;margin:14px 0}
section.band{background:var(--band);border-color:#cfe0e6}
h2{margin:0 0 13px;font-size:12.5px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
  color:var(--accent);border-bottom:1px solid var(--line);padding-bottom:9px}
section.band h2{border-bottom-color:#cfe0e6}
h3{margin:16px 0 9px;font-size:13px;font-weight:600;color:var(--muted)}
h3:first-of-type{margin-top:2px}
p{color:var(--muted);max-width:84ch;margin:0 0 12px;font-size:13.5px}
.grid-2{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.grid-2 section{margin:0}
table{width:100%;border-collapse:collapse}
th,td{border-bottom:1px solid var(--line);padding:7px 10px;text-align:left;
  vertical-align:top;font-size:13.5px}
tbody tr:last-child th,tbody tr:last-child td{border-bottom:0}
thead th{color:var(--muted);font-weight:700;font-size:11.5px;text-transform:uppercase;
  letter-spacing:.04em;border-bottom:1.5px solid var(--line-2)}
.kv-table th{width:36%;color:var(--muted);font-weight:600}
.kv-table td{font-variant-numeric:tabular-nums}
.kv-table td a{word-break:break-all}
.table-wrap{overflow-x:auto}
.table-wrap tbody tr:hover{background:#fafbfc}
pre{white-space:pre-wrap;word-break:break-word;overflow-x:auto;margin:0;color:var(--ink);
  font-family:var(--mono);font-size:12.5px;background:#f8f9fb;border:1px solid var(--line);
  border-radius:8px;padding:11px 13px}
.badge{display:inline-block;font-family:var(--mono);font-size:11px;font-weight:600;
  letter-spacing:.02em;padding:2px 8px;border-radius:5px;line-height:1.5}
.badge-ok{background:var(--ok-bg);color:var(--ok-fg)}
.badge-warn{background:var(--warn-bg);color:var(--warn-fg)}
.badge-bad{background:var(--bad-bg);color:var(--bad-fg)}
.stat-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.stat{background:var(--panel);border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:8px;padding:13px 15px}
.stat-label{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
  font-weight:600}
.stat-value{font-size:25px;font-weight:700;letter-spacing:-.01em;margin:5px 0 2px;
  font-variant-numeric:tabular-nums}
.stat-unit{font-size:12px;color:var(--muted);display:flex;align-items:center;gap:8px;flex-wrap:wrap}
.btn-row{display:flex;flex-wrap:wrap;gap:10px}
.btn{display:inline-flex;align-items:center;gap:8px;font-size:13.5px;font-weight:600;color:#fff;
  background:var(--accent);border:1px solid var(--accent);border-radius:8px;padding:9px 15px}
.btn:hover{background:#1d4d60;text-decoration:none}
.btn .ext{opacity:.85}
.shot-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px}
.shot{display:block;border:1px solid var(--line);border-radius:9px;overflow:hidden;background:#fff}
.shot:hover{border-color:var(--line-2);text-decoration:none}
.shot img{display:block;width:100%;height:auto}
.shot-cap{display:block;color:var(--muted);font-size:12px;padding:8px 11px;
  border-top:1px solid var(--line)}
footer{display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;color:var(--muted);
  font-size:12px;padding:16px 4px 6px;margin-top:6px;border-top:1px solid var(--line)}
@media (max-width:640px){main{padding:22px 14px}header.rep h1{font-size:23px}nav.toc{display:none}}
@media print{
  html{scroll-behavior:auto}
  body{background:#fff}
  main{max-width:none;padding:0}
  nav.toc,.btn-row{display:none}
  section{break-inside:avoid;border:1px solid #ccc}
  .shot{break-inside:avoid}
}
"""


def _humanize_view(stem: str) -> str:
    low = stem.lower()
    if "cst" in low:
        side = "CST / M1"
    elif "optimized" in low or "target" in low:
        side = "Target"
    else:
        side = ""
    views = [
        ("depth_analysis", "Depth analysis"),
        ("composite", "Composite (multi-view)"),
        ("roi_side", "ROI (sagittal)"),
        ("roi_front", "ROI (coronal)"),
        ("roi_top", "ROI (axial)"),
        ("lateral_left", "Lateral (left)"),
        ("lateral_right", "Lateral (right)"),
        ("anterior", "Anterior view"),
        ("posterior", "Posterior view"),
        ("superior", "Superior view"),
        ("oblique", "Oblique view"),
    ]
    view = next((label for token, label in views if token in low), stem.replace("_", " "))
    return f"{side} · {view}" if side else view


def _image_group(name: str) -> str:
    low = name.lower()
    if "cst" in low:
        return "CST / M1"
    if "optimized" in low or "target" in low:
        return "Target"
    return "Other"


def _discover_report_renders(base_dir: Path) -> List[tuple]:
    """Locate interactive 3D HTML viewers (bundle previews + grid map) near the report."""
    found: List[Path] = []
    seen = set()
    for folder in (base_dir, base_dir / "visualizations", base_dir / "visualization"):
        if not folder.exists():
            continue
        for pattern in ("*_interactive.html", "grid_interactive.html"):
            for path in sorted(folder.glob(pattern)):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                found.append(path)

    def classify(path: Path) -> tuple:
        name = path.name.lower()
        if name == "grid_interactive.html":
            return (2, "Interactive grid map (3D)")
        if "cst" in name:
            return (0, "CST / M1 bundle (3D)")
        if "optimized" in name or "target" in name:
            return (1, "Target bundle (3D)")
        return (1, "Bundle preview (3D)")

    items = []
    for path in found:
        order, label = classify(path)
        items.append((order, label, _relative_path(path, base_dir)))
    items.sort(key=lambda item: (item[0], item[2]))
    return [(label, src) for _, label, src in items]


def _render_link(value: Any, base_dir: Path) -> Optional[str]:
    if value in (None, "", []):
        return None
    text = str(value)
    if isinstance(value, Path) or ("/" in text or "\\" in text):
        path = Path(text)
        rel = _relative_path(path, base_dir)
        if rel not in ("N/A", ".", "") and not rel.startswith("/") and not rel.startswith(".."):
            return f'<a href="{escape(rel)}" title="{escape(text)}">{escape(path.name)}</a>'
        return escape(text)
    return _decorate_cell(escape(text))


def _render_paths_table(paths: List[tuple], base_dir: Path) -> str:
    rows = []
    for key, value in paths:
        cell = _render_link(value, base_dir)
        if cell is None:
            continue
        rows.append(f"<tr><th>{escape(str(key))}</th><td>{cell}</td></tr>")
    if not rows:
        return ""
    return '<table class="kv-table"><tbody>' + "".join(rows) + "</tbody></table>"


def _parse_results_table(sections: List[Dict[str, Any]]) -> tuple:
    for section in sections:
        rows = []
        for line in section.get("lines", []):
            if "|" in line and not _is_separator_line(line):
                cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
                if any(cells):
                    rows.append(cells)
        if len(rows) >= 2 and rows[0] and rows[0][0].strip().lower() == "metric":
            header = [cell.strip().lower() for cell in rows[0]]
            table = {row[0]: row for row in rows[1:] if row and row[0]}
            return header, table
    return None, None


def _weighted_index(header: List[str]) -> int:
    for idx, name in enumerate(header):
        if "weighted" in name and "un" not in name:
            return idx
    return len(header) - 1


def _unweighted_index(header: List[str]) -> int:
    for idx, name in enumerate(header):
        if "unweighted" in name:
            return idx
    return 1 if len(header) > 1 else 0


def _key_result_tiles(
    sections: List[Dict[str, Any]], fields: Dict[str, str]
) -> List[Dict[str, str]]:
    tiles: List[Dict[str, str]] = []
    header, table = _parse_results_table(sections)
    if table:
        widx = _weighted_index(header)
        uidx = _unweighted_index(header)

        def cell(idx: int, *needles: str) -> Optional[str]:
            for name, row in table.items():
                low = name.lower()
                if all(needle in low for needle in needles) and 0 <= idx < len(row):
                    return row[idx]
            return None

        est_w = cell(widx, "estimated", "raw")
        if est_w is not None:
            tiles.append(
                {
                    "label": "Estimated intensity (weighted)",
                    "value": est_w,
                    "unit": "% max output",
                    "badge": cell(widx, "flag") or "",
                }
            )
        est_u = cell(uidx, "estimated", "raw")
        if est_u is not None:
            tiles.append(
                {
                    "label": "Estimated intensity (unweighted)",
                    "value": est_u,
                    "unit": "% max output",
                    "badge": cell(uidx, "flag") or "",
                }
            )
    return tiles


def _build_report_html(
    *,
    txt_path: Path,
    html_path: Path,
    json_path: Optional[Path],
    report_type: str,
    generated_at: str,
    fields: Dict[str, str],
    data: Dict[str, Any],
    sections: List[Dict[str, Any]],
    images: List[Path],
) -> str:
    try:
        from tide import __version__ as version
    except Exception:
        version = ""

    title = f"TIDE Report - {_human_report_type(report_type)}"
    overview = _report_overview(report_type, fields, data)
    base_dir = html_path.parent
    subject = fields.get("Subject") or data.get("subject_id")
    workflow = data.get("workflow") or _human_report_type(report_type)
    target = data.get("target_label") or fields.get("Prefix")
    date = fields.get("Date") or generated_at

    paths = [
        ("Generated at", generated_at),
        ("Source TXT", txt_path),
        ("Source JSON", json_path or txt_path.with_suffix(".json")),
        ("Report HTML", html_path),
        ("Output folder", fields.get("Output Folder") or data.get("output_dir") or base_dir),
    ]
    key_metrics = [
        ("Subject", subject),
        ("Date", date),
        ("Workflow", workflow),
        ("Target", target),
        ("Spatial mode", fields.get("Spatial Mode")),
        ("Weight source", fields.get("Weight Source")),
        ("ROI size", fields.get("ROI Size")),
        ("Activation length", fields.get("Activation Length")),
    ]

    # Header meta chips
    chips = []
    for label, value in (
        ("Subject", subject),
        ("Workflow", workflow),
        ("Date", date),
        ("Target", target),
    ):
        if value:
            chips.append(f'<span class="chip"><b>{escape(label)}</b>{escape(str(value))}</span>')
    chips_html = f'<div class="chips">{"".join(chips)}</div>' if chips else ""

    # Key-results highlight band
    tiles = _key_result_tiles(sections, fields)
    band_html = ""
    if tiles:
        stat_cards = []
        for tile in tiles:
            badge = ""
            if tile.get("badge"):
                cls = _status_class(tile["badge"]) or "warn"
                badge = f'<span class="badge badge-{cls}">{escape(tile["badge"])}</span>'
            stat_cards.append(
                '<div class="stat">'
                f'<div class="stat-label">{escape(tile["label"])}</div>'
                f'<div class="stat-value">{escape(str(tile["value"]))}</div>'
                f'<div class="stat-unit">{escape(tile["unit"])}{badge}</div>'
                "</div>"
            )
        band_html = (
            '<section id="results-highlight" class="band"><h2>Estimated intensity (recommended dose)</h2>'
            f'<div class="stat-grid">{"".join(stat_cards)}</div></section>'
        )

    # Interactive 3D viewers
    renders = _discover_report_renders(base_dir)
    render_html = ""
    if renders:
        buttons = "".join(
            f'<a class="btn" href="{escape(src)}" target="_blank" rel="noopener">'
            f"<span>{escape(label)}</span>{_ICON_EXTERNAL}</a>"
            for label, src in renders
        )
        render_html = (
            '<section id="interactive-3d"><h2>Interactive 3D views</h2>'
            "<p>Open the WebGL bundle and grid viewers in a new browser tab. "
            "Full-resolution geometry stays in the TRK, NIfTI and CSV outputs.</p>"
            f'<div class="btn-row">{buttons}</div></section>'
        )

    # Visualization gallery (clickable to full resolution), grouped by bundle
    groups: Dict[str, List[str]] = {"CST / M1": [], "Target": [], "Other": []}
    for image in images:
        src = _relative_path(image, base_dir)
        caption = _humanize_view(image.stem)
        card = (
            f'<a class="shot" href="{escape(src)}" target="_blank" rel="noopener">'
            f'<img src="{escape(src)}" alt="{escape(caption)}" loading="lazy"/>'
            f'<span class="shot-cap">{escape(caption)}</span></a>'
        )
        groups[_image_group(image.name)].append(card)
    gallery = ""
    for group_name in ("CST / M1", "Target", "Other"):
        cards = groups[group_name]
        if not cards:
            continue
        gallery += f"<h3>{escape(group_name)}</h3>" f'<div class="shot-grid">{"".join(cards)}</div>'
    image_html = ""
    if gallery:
        image_html = (
            '<section id="visualizations"><h2>Visualizations</h2>'
            "<p>Click any panel to open the full-resolution image in a new tab.</p>"
            f"{gallery}</section>"
        )

    # Detailed parsed sections + table of contents
    toc = []
    if renders:
        toc.append(("interactive-3d", "3D views"))
    if tiles:
        toc.append(("results-highlight", "Key results"))
    if gallery:
        toc.append(("visualizations", "Visualizations"))
    rendered_sections = []
    for section in sections:
        section_html = _render_section(section)
        if not section_html:
            continue
        rendered_sections.append(section_html)
        toc.append(("sec-" + _slug(section["title"]), section["title"]))
    sections_html = "".join(rendered_sections)
    toc_html = ""
    if toc:
        links = "".join(f'<a href="#{escape(anchor)}">{escape(label)}</a>' for anchor, label in toc)
        toc_html = f'<nav class="toc">{links}</nav>'

    footer = (
        f"<footer><span>Generated {escape(generated_at)}</span>"
        f"<span>TIDE {escape(version)} reporting-only sidecar; "
        "numerics live in the TXT, JSON, CSV, TRK and NIfTI outputs</span></footer>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{escape(title)}</title>
<style>{_REPORT_CSS}</style>
</head>
<body>
<main>
  <header class="rep">
    <h1>{escape(title)}</h1>
    <p class="lead">{escape(overview)}</p>
    {chips_html}
  </header>
  {toc_html}
  {render_html}
  {band_html}
  <div class="grid-2">
    <section><h2>Run summary</h2>{_render_kv_table(key_metrics)}</section>
    <section><h2>Files and paths</h2>{_render_paths_table(paths, base_dir)}</section>
  </div>
  {image_html}
  {sections_html}
  {footer}
</main>
</body>
</html>
"""
