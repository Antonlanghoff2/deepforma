from __future__ import annotations

from typing import Any

from src.common.text import clean_text, split_multi_values


def _extract_label(item: Any) -> str:
    if isinstance(item, str):
        return clean_text(item)
    if isinstance(item, dict):
        for key in ("label", "libelle", "intitule", "name"):
            if clean_text(item.get(key, "")):
                return clean_text(item.get(key, ""))
    return ""


def extract_structured_skills(raw_offer: dict[str, Any]) -> list[dict[str, str | None]]:
    competences = raw_offer.get("competences")
    if not competences:
        return []

    items: list[dict[str, str | None]] = []
    iterable = competences if isinstance(competences, list) else [competences]
    for entry in iterable:
        if isinstance(entry, dict):
            label = _extract_label(entry)
            if not label:
                continue
            code = clean_text(entry.get("code") or entry.get("id") or "")
            requirement = clean_text(entry.get("requirement") or entry.get("niveau") or entry.get("type") or "")
            items.append(
                {
                    "code": code or None,
                    "label": label,
                    "requirement": requirement or None,
                    "source": "france_travail_structured",
                }
            )
        else:
            label = _extract_label(entry)
            if not label:
                for part in split_multi_values(entry):
                    if clean_text(part):
                        items.append(
                            {
                                "code": None,
                                "label": clean_text(part),
                                "requirement": None,
                                "source": "france_travail_structured",
                            }
                        )
            else:
                items.append(
                    {
                        "code": None,
                        "label": label,
                        "requirement": None,
                        "source": "france_travail_structured",
                    }
                )
    return items

