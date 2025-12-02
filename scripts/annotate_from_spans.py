#!/usr/bin/env python3
"""
Generate annotated HTML from span annotations in a Hugging Face dataset.

This script expects a dataset where each example contains:
- Either:
  - a `conversation` field: a list of dicts with {"role": "user"|"assistant", "content": "..."}
    In this case, the script will take the last assistant message as the text to annotate.
- Or:
  - a custom field specified via --text-field containing the text to annotate directly.

And an `annotations` field: a list of dicts like:
  {
    "index": int | None,       # character start index in the text where the span begins
    "label": str,              # e.g. "Supported" | "Not Supported" | "Insufficient Information"
    "span": str,               # the exact text span
    "verification_note": str | None
  }

The script produces one HTML per row (plus an index page) with highlighted spans color-coded by label.
It is robust to missing indices and best-effort matches them in the text if possible.
"""

from __future__ import annotations

import argparse
import dataclasses
import html
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import load_dataset

# ---- Data model (loose-typed to work with dict-based HF datasets) ----


@dataclasses.dataclass
class SpanAnnotation:
    index: Optional[int]
    label: Optional[str]
    span: str
    verification_note: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict) -> "SpanAnnotation":
        return SpanAnnotation(
            index=d.get("index", None),
            label=d.get("label", None),
            span=d.get("span", "") or "",
            verification_note=d.get("verification_note", None),
        )


# ---- Label styling and HTML helpers ----

LABEL_STYLES: Dict[str, Dict[str, str]] = {
    "Supported": {
        "bg": "#e6ffed",
        "border": "#34d058",
        "text": "#22863a",
    },
    "Not Supported": {
        "bg": "#ffeef0",
        "border": "#f85149",
        "text": "#b31d28",
    },
    "Insufficient Information": {
        "bg": "#fff5e5",
        "border": "#f2a654",
        "text": "#9a6700",
    },
}

DEFAULT_STYLE = {
    "bg": "#eef2ff",
    "border": "#6366f1",
    "text": "#3730a3",
}


def normalize_label(label: Optional[str]) -> str:
    if not label:
        return "Unknown"
    label = label.strip()
    # Normalize common variants
    mapping = {
        "supported": "Supported",
        "not supported": "Not Supported",
        "insufficient information": "Insufficient Information",
        "insufficient_info": "Insufficient Information",
        "insufficient": "Insufficient Information",
        "unknown": "Unknown",
    }
    return mapping.get(label.lower(), label)


def label_style(label: str) -> Dict[str, str]:
    return LABEL_STYLES.get(label, DEFAULT_STYLE)


def css_block() -> str:
    return """
<style>
body {
  font-family: ui-sans-serif, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
  margin: 24px;
  color: #111827;
  background: #ffffff;
}
.container {
  max-width: 920px;
  margin: 0 auto;
}
.header {
  margin-bottom: 20px;
}
.subtle {
  color: #6b7280;
  font-size: 14px;
}
.message {
  white-space: pre-wrap;
  line-height: 1.6;
  font-size: 16px;
  background: #fafafa;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 16px;
}
.legend {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  margin: 12px 0 20px 0;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
}
.badge {
  display: inline-block;
  border-radius: 6px;
  padding: 4px 8px;
  border: 1px solid #d1d5db;
  background: #f3f4f6;
  color: #374151;
  font-weight: 600;
}
.ann {
  border-radius: 4px;
  padding: 0 2px;
  border-width: 1px;
  border-style: solid;
}
.sidebar {
  margin-top: 24px;
  border-top: 1px solid #e5e7eb;
  padding-top: 12px;
}
.sidebar h3 {
  margin: 8px 0;
  font-size: 16px;
}
.sidebar .ann-item {
  margin-bottom: 10px;
  font-size: 14px;
}
blockquote {
  margin: 0 0 12px 0;
  border-left: 3px solid #d1d5db;
  padding-left: 12px;
  color: #4b5563;
}
a {
  color: #2563eb;
  text-decoration: none;
}
a:hover {
  text-decoration: underline;
}
.footer {
  margin-top: 24px;
  font-size: 12px;
  color: #6b7280;
}
</style>
    """.strip()


def build_legend_html() -> str:
    parts: List[str] = ['<div class="legend">']
    for label, st in LABEL_STYLES.items():
        parts.append(
            f'<span class="legend-item"><span class="badge" '
            f'style="background:{st["bg"]};border-color:{st["border"]};color:{st["text"]}">{html.escape(label)}</span></span>'
        )
    parts.append("</div>")
    return "".join(parts)


# ---- Span placement helpers ----


def find_non_overlapping_occurrence(
    text: str,
    span: str,
    used_ranges: List[Tuple[int, int]],
    start_search_at: int = 0,
    case_insensitive: bool = False,
) -> Optional[int]:
    """
    Find an occurrence of 'span' in 'text' that does not overlap with any used range.
    Returns the start index or None.
    """
    haystack = text if not case_insensitive else text.lower()
    needle = span if not case_insensitive else span.lower()

    cur = start_search_at
    while True:
        i = haystack.find(needle, cur)
        if i == -1:
            return None
        end = i + len(span)
        if not any(not (end <= s or i >= e) for (s, e) in used_ranges):
            return i
        cur = i + 1


def intervals_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def build_intervals(
    text: str, annotations: List[SpanAnnotation]
) -> Tuple[List[Tuple[int, int, SpanAnnotation]], List[SpanAnnotation]]:
    """
    Build non-overlapping intervals from annotations.

    Returns:
      - intervals: list of (start, end, ann) sorted by start
      - unplaced: annotations that could not be placed
    """
    intervals: List[Tuple[int, int, SpanAnnotation]] = []
    used: List[Tuple[int, int]] = []

    # Sort annotations to try deterministic placement:
    # 1) those with explicit index come first (ascending)
    # 2) then those without index (stable order)
    def sort_key(a: SpanAnnotation):
        return (0 if a.index is not None else 1, a.index if a.index is not None else 0)

    anns_sorted = sorted(annotations, key=sort_key)
    unplaced: List[SpanAnnotation] = []

    for ann in anns_sorted:
        span = ann.span or ""
        if not span:
            unplaced.append(ann)
            continue

        placed = False

        # Attempt to use provided index if valid
        if isinstance(ann.index, int) and 0 <= ann.index < len(text):
            start = ann.index
            end = start + len(span)
            # If exact match at provided index, and no overlap, accept
            if (
                end <= len(text)
                and text[start:end] == span
                and not any(intervals_overlap((start, end), r) for r in used)
            ):
                intervals.append((start, end, ann))
                used.append((start, end))
                placed = True
            else:
                # Try to relocate nearby (exact match) without overlap
                found = find_non_overlapping_occurrence(
                    text, span, used, start_search_at=max(0, start - 5)
                )
                if found is not None:
                    s, e = found, found + len(span)
                    intervals.append((s, e, ann))
                    used.append((s, e))
                    placed = True

        # If not placed, try first exact non-overlapping occurrence anywhere
        if not placed:
            found = find_non_overlapping_occurrence(text, span, used, start_search_at=0)
            if found is not None:
                s, e = found, found + len(span)
                intervals.append((s, e, ann))
                used.append((s, e))
                placed = True

        # If still not placed, try case-insensitive search
        if not placed:
            found = find_non_overlapping_occurrence(
                text, span, used, start_search_at=0, case_insensitive=True
            )
            if found is not None:
                s, e = found, found + len(span)
                intervals.append((s, e, ann))
                used.append((s, e))
                placed = True

        if not placed:
            unplaced.append(ann)

    intervals.sort(key=lambda t: t[0])
    return intervals, unplaced


def annotate_text_to_html(
    text: str, annotations: List[SpanAnnotation]
) -> Tuple[str, List[SpanAnnotation]]:
    """
    Produce HTML for the given text with spans highlighted.
    Returns: (html, unplaced_annotations)
    """
    intervals, unplaced = build_intervals(text, annotations)

    parts: List[str] = []
    pos = 0
    for start, end, ann in intervals:
        # add plain text up to start
        parts.append(html.escape(text[pos:start]))
        # add highlighted span
        label = normalize_label(ann.label)
        st = label_style(label)
        tooltip = f"{label}"
        if ann.verification_note:
            tooltip += f": {ann.verification_note}"
        # Escape tooltip for attribute context
        title_attr = html.escape(tooltip, quote=True)
        span_html = (
            f'<span class="ann" title="{title_attr}" '
            f'style="background:{st["bg"]};border-color:{st["border"]};color:{st["text"]};">'
            f"{html.escape(text[start:end])}"
            f"</span>"
        )
        parts.append(span_html)
        pos = end

    # remainder
    parts.append(html.escape(text[pos:]))

    return "".join(parts), unplaced


# ---- Conversation extraction ----


def get_assistant_text_from_conversation(conv: List[Dict]) -> Optional[str]:
    """Return the last assistant message content from a conversation list."""
    if not isinstance(conv, list):
        return None
    for msg in reversed(conv):
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return msg.get("content")
    # fallback: last message content if present
    if conv and isinstance(conv[-1], dict):
        return conv[-1].get("content")
    return None


# ---- HTML page rendering ----


def render_page_html(
    text_html: str,
    meta: Dict,
    annotations: List[SpanAnnotation],
    unplaced: List[SpanAnnotation],
) -> str:
    # Build annotation sidebar items
    def ann_item(a: SpanAnnotation) -> str:
        label = normalize_label(a.label)
        st = label_style(label)
        badge = (
            f'<span class="badge" style="background:{st["bg"]};border-color:{st["border"]};color:{st["text"]}">'
            f"{html.escape(label)}</span>"
        )
        note = (
            f" — {html.escape(a.verification_note or '')}"
            if a.verification_note
            else ""
        )
        span_txt = html.escape(a.span or "")
        idx_txt = "" if a.index is None else f" (idx {a.index})"
        return f'<div class="ann-item">{badge} <strong>{span_txt}</strong>{idx_txt}{note}</div>'

    meta_lines = []
    for k, v in meta.items():
        if v is None:
            continue
        meta_lines.append(
            f'<div class="subtle"><strong>{html.escape(str(k))}:</strong> {html.escape(str(v))}</div>'
        )
    meta_html = "\n".join(meta_lines)

    unplaced_html = ""
    if unplaced:
        unplaced_html = (
            '<div class="sidebar"><h3>Unplaced annotations</h3>'
            + "".join(ann_item(a) for a in unplaced)
            + "</div>"
        )

    placed_html = (
        '<div class="sidebar"><h3>Placed annotations</h3>'
        + "".join(ann_item(a) for a in annotations if a not in unplaced)
        + "</div>"
    )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Annotated example</title>
{css_block()}
</head>
<body>
<div class="container">
  <div class="header">
    <h2>Annotated Assistant Response</h2>
    {build_legend_html()}
    {meta_html}
  </div>
  <div class="message">{text_html}</div>
  {placed_html}
  {unplaced_html}
  <div class="footer">Generated by annotate_from_spans.py</div>
</div>
</body>
</html>"""


def render_index_html(title: str, links: List[Tuple[str, str]]) -> str:
    items = "\n".join(
        [
            f'<li><a href="{html.escape(href)}">{html.escape(label)}</a></li>'
            for label, href in links
        ]
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
{css_block()}
</head>
<body>
<div class="container">
  <div class="header">
    <h2>{html.escape(title)}</h2>
    <div class="subtle">Click an item to view its annotated HTML</div>
  </div>
  <ol>
    {items}
  </ol>
</div>
</body>
</html>"""


# ---- Main CLI ----


def main():
    p = argparse.ArgumentParser(
        description="Generate annotated HTML from span annotations in a Hugging Face dataset."
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset path on HF hub, e.g. MikiV/pubmedqa-meditron-conversations-annotated2",
    )
    p.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to load (default: test)",
    )
    p.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional dataset subset/config name (e.g., 'Meta-Llama-3.1-8B-Instruct')",
    )
    p.add_argument(
        "--annotations-field",
        type=str,
        default="annotations",
        help="Field name containing span annotations (default: annotations)",
    )
    p.add_argument(
        "--text-field",
        type=str,
        default=None,
        help="Optional field name with text to annotate. If omitted, uses the last assistant message from 'conversation'.",
    )
    p.add_argument(
        "--id-field",
        type=str,
        default=None,
        help="Optional field to use as filename/id (e.g., pubmedqa_id). Fallback is row index.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory for HTML files. Defaults to annotation_pipeline_results/annotated_html/<dataset_name>/<split>",
    )
    p.add_argument(
        "--max-items", type=int, default=None, help="Limit number of items to render"
    )
    args = p.parse_args()

    dataset_name = args.dataset
    split = args.split
    subset = args.subset
    ann_field = args.annotations_field
    text_field = args.text_field
    id_field = args.id_field

    # Output directory
    safe_dataset_dir = dataset_name.replace("/", "__")
    default_out = (
        Path("annotation_pipeline_results")
        / "annotated_html"
        / safe_dataset_dir
        / split
    )
    out_dir = Path(args.out_dir) if args.out_dir else default_out
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    ds = load_dataset(dataset_name, name=subset, split=split)

    links: List[Tuple[str, str]] = []
    total = len(ds) if args.max_items is None else min(args.max_items, len(ds))

    for i in range(total):
        ex = ds[i]

        # Extract text
        if text_field:
            text = ex.get(text_field)
            if text is None:
                # Fallback to conversation if available
                text = get_assistant_text_from_conversation(ex.get("conversation", []))
        else:
            text = get_assistant_text_from_conversation(ex.get("conversation", []))

        if not isinstance(text, str):
            text = str(text or "")

        # Extract annotations
        raw_anns = ex.get(ann_field, []) or []
        anns = [SpanAnnotation.from_dict(a) for a in raw_anns]

        # Build annotated HTML
        text_html, unplaced = annotate_text_to_html(text, anns)

        # Build metadata
        meta: Dict = {}
        # useful metadata if present
        for key in ["pubmedqa_id", "original_answer", "context", "source"]:
            if key in ex:
                meta[key] = ex[key]
        # Show user prompt if present
        if isinstance(ex.get("conversation"), list) and ex["conversation"]:
            user_text = None
            # Take the last user message if present
            for msg in reversed(ex["conversation"]):
                if msg.get("role") == "user":
                    user_text = msg.get("content")
                    break
            if user_text:
                meta["user_prompt"] = user_text

        # Render page
        page_html = render_page_html(text_html, meta, anns, unplaced)

        # Determine filename
        if id_field and id_field in ex:
            file_id = str(ex[id_field])
        else:
            # try a few fallbacks
            file_id = str(ex.get("id") or ex.get("uid") or ex.get("uuid") or i)
        filename = f"{file_id}.html"
        (out_dir / filename).write_text(page_html, encoding="utf-8")

        links.append((file_id, filename))

    # Write index
    index_title = f"Annotated HTML for {dataset_name} [{split}]"
    (out_dir / "index.html").write_text(
        render_index_html(index_title, links), encoding="utf-8"
    )

    print(f"Wrote {len(links)} items to {out_dir}")
    print(f"Open: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
