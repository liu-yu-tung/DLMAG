import csv
import json
from pathlib import Path

PROMPT_STYLES = ("caption", "tags", "tags_caption", "caption_tags")


def load_captions(path: str | Path) -> dict[str, str]:
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and {"id", "caption"} <= set(rows[0]), f"{path} needs id,caption columns"
    return {row["id"]: row["caption"] for row in rows}


def flatten_tags(tags: list | dict) -> list[str]:
    # MQAScore tags: {attribute: [str or {"entity": ...}]}; "others" holds free-form phrases, not tags.
    if isinstance(tags, dict):
        values = [tag for attribute, group in tags.items() if attribute != "others" for tag in group]
    else:
        values = tags
    assert isinstance(values, list), type(values)
    names = (tag["entity"] if isinstance(tag, dict) else tag for tag in values)
    return list(dict.fromkeys(str(name).strip() for name in names if str(name).strip()))


def load_tags(path: str | Path) -> dict[str, list[str]]:
    tags = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                tags[str(item["id"])] = flatten_tags(item["tags"])
    return tags


def build_prompt(style: str, caption: str, tags: list[str] | None) -> str:
    assert style in PROMPT_STYLES, style
    if style == "caption":
        return caption
    assert tags, "tag prompt styles need tags"
    tag_text = ", ".join(tags)
    # Stable Audio Open was trained on comma-joined tags and metadata (Evans et al., ICASSP 2025), not sentences.
    if style == "tags":
        return tag_text
    # T5 keeps 128 tokens, so the part placed last is the one cut on long prompts.
    return f"{tag_text}. {caption}" if style == "tags_caption" else f"{caption} {tag_text}"
