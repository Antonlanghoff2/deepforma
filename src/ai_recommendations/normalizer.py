from __future__ import annotations

import re
import unicodedata
from typing import Any

from common.text import clean_text, normalize_for_match


IA_VARIANTS = {
    'intelligence artificielle': 'ia',
    'intelligence artificiel': 'ia',
    'ia generative': 'ia generative',
    'generative ai': 'ia generative',
    'llm': 'llm',
    'prompt engineering': 'prompt engineering',
}


def _strip_accents(text: str) -> str:
    return ''.join(ch for ch in unicodedata.normalize('NFKD', text) if not unicodedata.combining(ch))


def normalize_ai_keyword(text: Any) -> str:
    value = clean_text(text)
    if not value:
        return ''
    value = value.replace('’', "'").replace('`', "'")
    value = value.replace('-', ' ')
    value = value.replace('"', ' ').replace('«', ' ').replace('»', ' ')
    value = re.sub(r'^[\s\-•\*\d\.:;]+', '', value)
    value = re.sub(r'[\s\-•\*\d\.:;]+$', '', value)
    value = re.sub(r'\s+', ' ', value)
    lowered = _strip_accents(value).lower()
    for src, dst in IA_VARIANTS.items():
        lowered = lowered.replace(src, dst)
    lowered = re.sub(r'[^a-z0-9\s/]+', ' ', lowered)
    lowered = re.sub(r'(s|es|x)', ' ', lowered)
    lowered = re.sub(r'(actions?|processus|solutions?|outils?|methodes?)', lambda m: m.group(0), lowered)
    lowered = re.sub(r'\s+', ' ', lowered).strip()
    return normalize_for_match(lowered)
