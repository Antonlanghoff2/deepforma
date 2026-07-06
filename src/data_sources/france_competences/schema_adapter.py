from __future__ import annotations

import csv
import io
import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from common.text import clean_text, normalize_for_match, split_multi_values
from deepforma.cpf.io import detect_text_format


CANONICAL_COLUMNS = {
    'certification_id',
    'repository_type',
    'certification_code',
    'title',
    'intitule_court',
    'status',
    'date_publication',
    'date_debut_validite',
    'date_fin_validite',
    'qualification_level',
    'level',
    'european_level',
    'certificateur',
    'partenaires',
    'codes_nsf',
    'codes_rome',
    'secteurs_activite',
    'activites_visees',
    'competences_visees',
    'blocs_competences',
    'voies_acces',
    'modalites_evaluation',
    'criteres_evaluation',
    'url_source',
    'source_updated_at',
    'source_file',
    'source_kind',
}

FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    'certification_id': ('fiche_id', 'id_fiche', 'id', 'certificate_id'),
    'repository_type': ('type_repertoire', 'type', 'repository_type', 'referential_type'),
    'certification_code': ('code_certification', 'numero_fiche', 'rncp_code', 'rs_code', 'code', 'reference'),
    'title': ('intitule', 'title', 'libelle', 'label', 'nom'),
    'intitule_court': ('intitule_court', 'short_title', 'short_label', 'label_court'),
    'status': ('etat_fiche', 'status', 'state', 'statut'),
    'date_publication': ('date_publication', 'date_publication_jo', 'publication_jo', 'date_de_publication'),
    'date_debut_validite': ('date_debut_validite', 'date_effet', 'date_actif'),
    'date_fin_validite': ('date_fin_validite', 'date_fin_enregistrement', 'date_limite_delivrance'),
    'qualification_level': ('niveau_qualification', 'niveau', 'niveau_europeen', 'exit_level'),
    'level': ('niveau', 'level'),
    'european_level': ('niveau_europeen', 'nomenclature_europe', 'european_level'),
    'certificateur': ('certificateur', 'certificateurs', 'nom_certificateur'),
    'partenaires': ('partenaires', 'partner', 'partners'),
    'codes_nsf': ('codes_nsf', 'nsf', 'nsf_codes'),
    'codes_rome': ('codes_rome', 'rome', 'rome_codes'),
    'secteurs_activite': ('secteurs_activite', 'secteurs', 'sector', 'sectors'),
    'activites_visees': ('activites_visees', 'activity', 'activities', 'objectif', 'objectifs_contexte'),
    'competences_visees': ('competences_visees', 'capacites_atteintes', 'capacites_attestees', 'liste_competences', 'skills'),
    'blocs_competences': ('blocs_competences', 'blocks', 'block', 'bloc_competences'),
    'voies_acces': ('voies_acces', 'access', 'access_paths', 'conditions_acces'),
    'modalites_evaluation': ('modalites_evaluation', 'evaluation', 'modalites', 'modes_evaluation'),
    'criteres_evaluation': ('criteres_evaluation', 'criteria', 'criteres', 'critere_evaluation'),
    'url_source': ('url_source', 'source_url', 'link', 'uri'),
    'source_updated_at': ('source_updated_at', 'updated_at', 'last_modified', 'date_mise_a_jour'),
    'source_file': ('source_file', 'file_name', 'filename', 'path'),
    'source_kind': ('source_kind', 'kind'),
}


def _normalize_header(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ''
    return normalize_for_match(re.sub(r'[_/]+', ' ', text))


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return clean_text(value)
    if isinstance(value, (int, float)):
        return clean_text(value)
    if isinstance(value, list):
        parts = [clean_text(item) for item in value if clean_text(item)]
        return clean_text(' | '.join(parts))
    if isinstance(value, dict):
        parts = [clean_text(item) for item in value.values() if clean_text(item)]
        return clean_text(' | '.join(parts))
    return clean_text(value)


def _flatten_xml_element(element: ET.Element) -> Any:
    children = list(element)
    if not children:
        return clean_text(element.text)
    grouped: dict[str, list[Any]] = {}
    for child in children:
        grouped.setdefault(child.tag, []).append(_flatten_xml_element(child))
    result: dict[str, Any] = {}
    for key, values in grouped.items():
        result[key] = values[0] if len(values) == 1 else values
    text = clean_text(element.text)
    if text and not result:
        return text
    if text:
        result['text'] = text
    return result


def _csv_reader(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    encoding = detect_text_format(path).encoding
    raw = path.read_text(encoding=encoding, errors='ignore')
    sample = io.StringIO(raw[:20000])
    try:
        dialect = csv.Sniffer().sniff(sample.read(5000), delimiters=[',', ';', '\t', '|'])
        delimiter = dialect.delimiter
    except Exception:
        delimiter = ';' if raw.count(';') >= raw.count(',') else ','
    reader = csv.DictReader(io.StringIO(raw), delimiter=delimiter)
    return list(reader), {'encoding': encoding, 'delimiter': delimiter}


@dataclass(slots=True)
class SchemaMapping:
    canonical: dict[str, Any]
    unknown: dict[str, Any]


class FranceCompetencesSchemaAdapter:
    def __init__(self, field_aliases: dict[str, tuple[str, ...]] | None = None) -> None:
        self.field_aliases = field_aliases or FIELD_ALIASES

    def normalize_field_name(self, name: str) -> str:
        return _normalize_header(name)

    def build_alias_lookup(self) -> dict[str, str]:
        lookup: dict[str, str] = {}
        for canonical, aliases in self.field_aliases.items():
            lookup[self.normalize_field_name(canonical)] = canonical
            for alias in aliases:
                lookup[self.normalize_field_name(alias)] = canonical
        return lookup

    def map_row(self, row: dict[str, Any]) -> SchemaMapping:
        lookup = self.build_alias_lookup()
        canonical: dict[str, Any] = {}
        unknown: dict[str, Any] = {}
        for key, value in row.items():
            canonical_key = lookup.get(self.normalize_field_name(key))
            if canonical_key:
                canonical[canonical_key] = value
            else:
                unknown[key] = value
        return SchemaMapping(canonical=canonical, unknown=unknown)

    def map_xml_element(self, element: ET.Element) -> SchemaMapping:
        flattened = _flatten_xml_element(element)
        if isinstance(flattened, dict):
            return self.map_row(flattened)
        return SchemaMapping(canonical={'text': flattened}, unknown={})

    @staticmethod
    def as_list(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            items: list[str] = []
            for item in value:
                items.extend(FranceCompetencesSchemaAdapter.as_list(item))
            return [clean_text(item) for item in items if clean_text(item)]
        text = clean_text(value)
        if not text:
            return []
        parts = split_multi_values(text)
        if len(parts) > 1:
            return [item for item in parts if clean_text(item)]
        return [text]

    @staticmethod
    def first_non_empty(*values: Any) -> str:
        for value in values:
            text = clean_text(value)
            if text:
                return text
        return ''

