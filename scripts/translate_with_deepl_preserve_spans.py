#!/usr/bin/env python3
"""
Translate a Hugging Face dataset's assistant messages with DeepL while preserving span annotations.

Overview:
- For each example, find the text that annotations refer to (by default: last assistant message).
- Wrap each annotated span in a unique XML tag before translation.
- Send the tagged text to DeepL with XML tag handling so tags are preserved.
- Parse the translated result to:
  - remove the tags,
  - compute new indices of the translated spans,
  - update the annotations accordingly.
- Only assistant messages are translated. User messages and other fields remain unchanged.
- Push the translated dataset to the Hugging Face Hub under "<original_repo>-lang" by default.

Requirements:
- pip install deepl datasets tqdm
- Provide a DeepL API key via:
  - environment variable DEEPL_API_KEY, or
  - --deepl-key argument.

Examples:
  Translate default dataset to Czech (cs) and push:
    python translate_with_deepl_preserve_spans.py

  Translate a specific dataset and split to German without pushing:
    python translate_with_deepl_preserve_spans.py \
      --dataset MikiV/pubmedqa-meditron-conversations-annotated-claude \
      --split test \
      --target-lang de \
      --no-push

Notes:
- The default target language is "cz" which is normalized to "CS" for DeepL.
- If annotations cannot be placed in the original text (e.g., cannot match span),
  they are left unchanged in the output.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import re
from typing import Dict, List, Optional, Tuple

import deepl
from datasets import (
    Dataset,
    DatasetDict,
    get_dataset_split_names,  # type: ignore
    load_dataset,
)
from tqdm import tqdm


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

    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "label": self.label,
            "span": self.span,
            "verification_note": self.verification_note,
        }


def get_last_assistant_index_and_text(
    conv: List[Dict],
) -> Tuple[Optional[int], Optional[str]]:
    """Return index and content of the last assistant message from a conversation list."""
    if not isinstance(conv, list):
        return None, None
    for i in range(len(conv) - 1, -1, -1):
        msg = conv[i]
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            return i, msg.get("content")
    # fallback: last message content if present
    if conv and isinstance(conv[-1], dict):
        return len(conv) - 1, conv[-1].get("content")
    return None, None


def intervals_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def find_non_overlapping_occurrence(
    text: str,
    span: str,
    used_ranges: List[Tuple[int, int]],
    start_search_at: int = 0,
    case_insensitive: bool = False,
) -> Optional[int]:
    """Find an occurrence of 'span' in 'text' that does not overlap with any used range."""
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


def wrap_text_with_xml_tags(
    text: str, intervals: List[Tuple[int, int, SpanAnnotation]]
) -> Tuple[str, List[SpanAnnotation]]:
    """
    Insert XML tags around each annotated span in the provided text.

    We use <x data-i="k"> ... </x> to mark the k-th placed annotation in order.
    """
    parts: List[str] = []
    pos = 0
    placed_ordered_anns: List[SpanAnnotation] = []
    for k, (start, end, ann) in enumerate(intervals):
        parts.append(text[pos:start])
        parts.append(f'<x data-i="{k}">')
        parts.append(text[start:end])
        parts.append("</x>")
        placed_ordered_anns.append(ann)
        pos = end
    parts.append(text[pos:])
    tagged = "".join(parts)
    return tagged, placed_ordered_anns


def translate_text_with_deepl(
    translator: deepl.Translator,
    text: str,
    target_lang: str,
) -> str:
    """
    Translate a string with DeepL while preserving XML tags.
    We set:
      - tag_handling="xml": treat custom tags as XML tags
      - non_splitting_tags=["x"]: avoid sentence splits across our tag boundaries
      - preserve_formatting=True: reduce reflow changes
    """
    result = translator.translate_text(
        text,
        target_lang=target_lang,
        tag_handling="xml",
        non_splitting_tags=["x"],
        preserve_formatting=True,
        # Don't specify source_lang: let DeepL auto-detect
    )
    return result.text  # type: ignore


def strip_tags_and_collect_spans(
    tagged_text: str,
) -> Tuple[str, Dict[int, Tuple[int, str]]]:
    """
    Remove <x data-i="k">...</x> tags and return:
      - untagged text
      - mapping: k -> (start_index_in_untagged, inner_text)

    Implementation parses the text sequentially to compute start indices robustly.
    """
    # Regex to match <x ...> ... </x> with data-i="k"
    tag_re = re.compile(
        r"<x\b([^>]*)>(.*?)</x>",
        flags=re.DOTALL | re.IGNORECASE,
    )

    # To preserve indices, rebuild string while tracking offsets
    out_parts: List[str] = []
    last_end = 0
    id_to_span: Dict[int, Tuple[int, str]] = {}

    for m in tag_re.finditer(tagged_text):
        start, end = m.span()
        attrs = m.group(1) or ""
        inner = m.group(2) or ""

        # Append preceding plain text
        out_parts.append(tagged_text[last_end:start])
        current_index = sum(len(p) for p in out_parts)

        # Extract data-i
        id_match = re.search(r'\bdata-i\s*=\s*"(\d+)"', attrs)
        if not id_match:
            # If somehow our attribute is missing, just inline the inner text
            out_parts.append(inner)
        else:
            ann_id = int(id_match.group(1))
            # Record mapping and insert inner text
            id_to_span[ann_id] = (current_index, inner)
            out_parts.append(inner)

        last_end = end

    # Remainder after the last tag
    out_parts.append(tagged_text[last_end:])
    untagged_text = "".join(out_parts)
    return untagged_text, id_to_span


def normalize_target_lang(lang: str) -> str:
    """
    Normalize user-provided target language string to DeepL codes.
    Examples:
      cz, cs -> CS (Czech)
      en, en-us -> EN-US
      en-gb -> EN-GB
      de -> DE
      fr -> FR
    """
    l = (lang or "").strip().lower()
    if l in {"cz", "cs"}:
        return "CS"
    if l in {"en", "en-us", "en_us"}:
        return "EN-US"
    if l in {"en-gb", "en_gb"}:
        return "EN-GB"
    if l == "de":
        return "DE"
    if l == "fr":
        return "FR"
    if l == "es":
        return "ES"
    if l == "it":
        return "IT"
    if l == "nl":
        return "NL"
    if l == "pl":
        return "PL"
    if l == "pt":
        return "PT-PT"
    if l in {"pt-br", "pt_br"}:
        return "PT-BR"
    if l == "bg":
        return "BG"
    if l == "da":
        return "DA"
    if l == "el":
        return "EL"
    if l == "et":
        return "ET"
    if l == "fi":
        return "FI"
    if l == "hu":
        return "HU"
    if l == "ja":
        return "JA"
    if l == "lt":
        return "LT"
    if l == "lv":
        return "LV"
    if l == "ro":
        return "RO"
    if l == "ru":
        return "RU"
    if l == "sk":
        return "SK"
    if l == "sl":
        return "SL"
    if l == "sv":
        return "SV"
    if l == "zh":
        return "ZH"
    # Fallback: uppercase
    return l.upper()


def translate_conversation_assistants(
    translator: deepl.Translator,
    conversation: List[Dict],
    target_lang: str,
) -> List[Dict]:
    """Translate all assistant messages in a conversation (plain translation, no tags)."""
    new_conv: List[Dict] = []
    for msg in conversation:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content") or ""
            content = content if isinstance(content, str) else str(content)
            translated = (
                translate_text_with_deepl(translator, content, target_lang)
                if content
                else content
            )
            new_conv.append({"role": "assistant", "content": translated})
        else:
            new_conv.append(msg)
    return new_conv


def process_example(
    ex: Dict,
    translator: deepl.Translator,
    ann_field: str,
    text_field: Optional[str],
    target_lang: str,
) -> Dict:
    """
    Translate one example, preserving annotations.

    If text_field is provided, annotations are assumed to apply to ex[text_field].
    Otherwise, annotations are assumed to apply to the last assistant message of ex['conversation'].
    """
    # Extract annotations
    raw_anns = ex.get(ann_field, []) or []
    anns = [SpanAnnotation.from_dict(a) for a in raw_anns]

    # Determine the text to annotate
    if text_field:
        base_text = ex.get(text_field)
        base_text = base_text if isinstance(base_text, str) else str(base_text or "")
        conv = ex.get("conversation", None)
        # We will translate the field only, conversation stays unchanged if present
    else:
        conversation = ex.get("conversation", []) or []
        idx, base_text = get_last_assistant_index_and_text(conversation)
        base_text = base_text if isinstance(base_text, str) else str(base_text or "")
        if idx is None:
            # No assistant content found; return unchanged
            return ex

    # Build intervals for wrapping
    intervals, unplaced = build_intervals(base_text, anns)

    # Wrap spans with XML tags
    tagged_text, placed_order = wrap_text_with_xml_tags(base_text, intervals)

    # Translate the tagged text
    translated_tagged = translate_text_with_deepl(translator, tagged_text, target_lang)

    # Strip tags and collect translated span positions
    translated_text, id_to_span = strip_tags_and_collect_spans(translated_tagged)

    # Rebuild annotations: placed ones updated, unplaced left as-is
    # placed_order[k] corresponds to id_to_span[k]
    placed_updated: List[Dict] = []
    for k, ann in enumerate(placed_order):
        if k in id_to_span:
            start_idx, inner_text = id_to_span[k]
            placed_updated.append(
                SpanAnnotation(
                    index=start_idx,
                    label=ann.label,
                    span=inner_text,
                    verification_note=ann.verification_note,
                ).to_dict()
            )
        else:
            # Fallback: keep original (shouldn't happen)
            placed_updated.append(ann.to_dict())

    unplaced_dicts = [a.to_dict() for a in unplaced]
    new_annotations = placed_updated + unplaced_dicts

    # Update the example
    new_ex = dict(ex)  # shallow copy

    if text_field:
        new_ex[text_field] = translated_text
        new_ex[ann_field] = new_annotations
        # Optionally translate assistant messages too (without tags), but requirement says only model answers:
        # If conversation exists, translate assistant messages as well.
        if isinstance(ex.get("conversation"), list):
            new_ex["conversation"] = translate_conversation_assistants(
                translator, ex["conversation"], target_lang
            )
    else:
        # Update conversation: translate all assistant messages,
        # but replace the last assistant message with our tag-aware translated_text
        conversation = ex.get("conversation", []) or []
        new_conv = []
        last_assistant_idx, _ = get_last_assistant_index_and_text(conversation)

        for i, msg in enumerate(conversation):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                if i == last_assistant_idx:
                    # Use tag-aware translated text for the annotated message
                    new_conv.append({"role": "assistant", "content": translated_text})
                else:
                    # Translate other assistant messages plainly
                    content = msg.get("content") or ""
                    content = content if isinstance(content, str) else str(content)
                    translated = (
                        translate_text_with_deepl(translator, content, target_lang)
                        if content
                        else content
                    )
                    new_conv.append({"role": "assistant", "content": translated})
            else:
                new_conv.append(msg)

        new_ex["conversation"] = new_conv
        new_ex[ann_field] = new_annotations

    return new_ex


def push_datasetdict_to_hub(
    dd: DatasetDict, repo_id: str, config_name: Optional[str]
) -> None:
    """
    Push a DatasetDict to the Hub under a single repo id.
    If config_name is provided, it will be used as the dataset config name.
    """
    # datasets>=2.14 supports DatasetDict.push_to_hub(repo_id, config_name=..., private=...)
    # To be conservative, push split-by-split in case of older versions.
    # Ensure a valid string config name for Hub metadata
    cfg_name = (
        config_name
        if isinstance(config_name, str) and config_name.strip() != ""
        else "default"
    )
    for split, ds in dd.items():
        ds.push_to_hub(repo_id=repo_id, config_name=cfg_name, split=split)


def main():
    parser = argparse.ArgumentParser(
        description="Translate a HF dataset with DeepL, preserving span annotations."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MikiV/pubmedqa-meditron-conversations-annotated-claude",
        help="HF dataset repo id (e.g., owner/name). Default: MikiV/pubmedqa-meditron-conversations-annotated-claude",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Optional dataset subset/config name (e.g., 'Meta-Llama-3.1-8B-Instruct')",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Split to translate (can be provided multiple times). If omitted, will try to detect available splits.",
    )
    parser.add_argument(
        "--annotations-field",
        type=str,
        default="annotations",
        help="Field name containing span annotations (default: annotations)",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default=None,
        help="Optional field with text to annotate/translate. If omitted, uses the last assistant message from 'conversation'.",
    )
    parser.add_argument(
        "--target-lang",
        type=str,
        default="cz",
        help="Target language. Examples: cz, cs, de, fr, en, en-gb, en-us. Default: cz (Czech).",
    )
    parser.add_argument(
        "--deepl-key",
        type=str,
        default=None,
        help="DeepL API key. If omitted, reads from DEEPL_API_KEY env var.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="If provided, do not push to the Hub. Just run locally.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optionally limit number of items per split (useful for testing).",
    )
    args = parser.parse_args()

    dataset = args.dataset
    subset = args.subset
    ann_field = args.annotations_field
    text_field = args.text_field
    splits = args.split
    target_lang = normalize_target_lang(args.target_lang)
    deepl_key = args.deepl_key or os.environ.get("DEEPL_API_KEY")
    if not deepl_key:
        raise SystemExit(
            "DeepL API key not provided. Set DEEPL_API_KEY env var or use --deepl-key."
        )

    # Determine splits if not provided
    if not splits:
        try:
            splits = get_dataset_split_names(dataset, config_name=subset)
        except Exception:
            # Fallback to common splits
            splits = ["test"]

    # Prepare translator
    translator = deepl.Translator(deepl_key)

    # Process each split and collect into DatasetDict
    out_splits: Dict[str, Dataset] = {}
    for split in splits:
        ds = load_dataset(dataset, name=subset, split=split)
        if args.max_items is not None:
            ds = ds.select(range(min(args.max_items, len(ds))))
        print(
            f"Translating split '{split}' with {len(ds)} examples -> target_lang={target_lang}"
        )

        def _map_fn(ex: Dict) -> Dict:
            return process_example(
                ex=ex,
                translator=translator,
                ann_field=ann_field,
                text_field=text_field,
                target_lang=target_lang,
            )

        new_ds = ds.map(
            _map_fn,
            desc=f"Translating/preserving spans [{split}]",
            load_from_cache_file=False,
        )
        out_splits[split] = new_ds

    out_dd = DatasetDict(out_splits)

    # Push to Hub
    if not args.no_push:
        # Build output repo id: "<owner>/<name>-lang" if input includes owner
        if "/" in dataset:
            owner, name = dataset.split("/", 1)
            out_repo = f"{owner}/{name}-{args.target_lang}"
        else:
            out_repo = f"{dataset}-{args.target_lang}"
        print(
            f"Pushing translated dataset to: {out_repo} (config: {subset or 'default'})"
        )
        push_datasetdict_to_hub(out_dd, repo_id=out_repo, config_name=subset)
        print("Push complete.")
    else:
        print("Skipping push to Hub (--no-push provided).")

    print("Done.")


if __name__ == "__main__":
    main()
