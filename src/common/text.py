from __future__ import annotations

import hashlib
import html
import re
import unicodedata
from typing import Any


NULL_LIKE = {
    "",
    "na",
    "n/a",
    "none",
    "null",
    "nan",
    "-",
    "--",
}


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")


def strip_html(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    return text


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if text.lower().strip() in NULL_LIKE:
        return ""
    text = normalize_unicode(text)
    text = strip_html(text)
    text = text.replace("\u200b", " ").replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in NULL_LIKE:
        return ""
    return text


def normalize_for_match(value: Any) -> str:
    text = clean_text(value).lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def split_multi_values(value: Any) -> list[str]:
    text = clean_text(value)
    if not text:
        return []
    parts = re.split(r"\s*(?:\||;|,|\n|•|·|\u2022)\s*", text)
    cleaned: list[str] = []
    for part in parts:
        item = clean_text(part)
        if item:
            cleaned.append(item)
    return cleaned


def stable_hash(*parts: Any, length: int = 16) -> str:
    payload = "||".join(normalize_for_match(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]

