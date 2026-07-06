from __future__ import annotations

import logging
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.text import clean_text, normalize_for_match

from .schema_adapter import FranceCompetencesSchemaAdapter
from .skill_extractor import FranceCompetencesSkillExtractor


LOGGER = logging.getLogger(__name__)


def _local_tag(tag: str) -> str:
    if '}' in tag:
        return tag.rsplit('}', 1)[1]
    return tag


def _find_text(element: ET.Element, *paths: str) -> str:
    for path in paths:
        target = element.find(path)
        if target is not None:
            text = clean_text(target.text)
            if text:
                return text
    return ''


def _find_raw_text(element: ET.Element, *paths: str) -> str:
    for path in paths:
        target = element.find(path)
        if target is not None:
            text = ''.join(target.itertext()) if list(target) else (target.text or '')
            text = text.strip()
            if text:
                return text
    return ''


def _collect_text(element: ET.Element) -> str:
    parts: list[str] = []
    for child in element.iter():
        text = clean_text(child.text)
        if text:
            parts.append(text)
    return clean_text(' '.join(parts))


def _parse_level(text: str) -> int | None:
    match = re.search(r'(\d+)', clean_text(text))
    if match:
        try:
            return int(match.group(1))
        except Exception:
            return None
    return None


def _split_codes(text: Any) -> list[str]:
    if text is None:
        return []
    raw = clean_text(text)
    if not raw:
        return []
    parts = re.split(r'[\n;,|]+', raw)
    return [clean_text(part) for part in parts if clean_text(part)]


@dataclass(slots=True)
class ParsedFranceCompetences:
    certifications: list[dict[str, Any]]
    blocks: list[dict[str, Any]]
    skills: list[dict[str, Any]]
    activities: list[dict[str, Any]]
    negatives: list[dict[str, Any]]
    metadata: dict[str, Any]


class FranceCompetencesRncpParser:
    repository_type = 'RNCP'

    def __init__(self, *, schema_adapter: FranceCompetencesSchemaAdapter | None = None, skill_extractor: FranceCompetencesSkillExtractor | None = None) -> None:
        self.schema_adapter = schema_adapter or FranceCompetencesSchemaAdapter()
        self.skill_extractor = skill_extractor or FranceCompetencesSkillExtractor()

    def _parse_fiche(self, fiche: ET.Element, *, source_file: str, source_url: str | None = None, source_updated_at: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        cert_code = _find_text(fiche, 'NUMERO_FICHE', 'CODE', 'CERTIFICATION_CODE')
        cert_id = _find_text(fiche, 'ID_FICHE') or cert_code
        title = _find_text(fiche, 'INTITULE', 'TITRE', 'LIBELLE')
        status = _find_text(fiche, 'ETAT_FICHE', 'ACTIF') or 'Publiée'
        level = _parse_level(_find_text(fiche, 'NOMENCLATURE_EUROPE/NIVEAU', 'NIVEAU', 'LEVEL'))
        short_title = _find_text(fiche, 'INTITULE_COURT', 'TITRE_COURT')
        date_publication = _find_text(fiche, 'DATE_PUBLICATION_JO', 'DATE_DE_PUBLICATION', 'PUBLICATION_JO', 'DATE_DERNIER_JO')
        valid_from = _find_text(fiche, 'DATE_EFFET', 'DATE_ACTIF', 'DATE_DE_PUBLICATION')
        valid_to = _find_text(fiche, 'DATE_FIN_ENREGISTREMENT', 'DATE_LIMITE_DELIVRANCE')
        certificateurs = [clean_text(text) for text in (_collect_text(el) for el in fiche.findall('CERTIFICATEURS/CERTIFICATEUR')) if clean_text(text)]
        partenaires = [clean_text(text) for text in (_collect_text(el) for el in fiche.findall('PARTENAIRES/PARTENAIRE')) if clean_text(text)]
        rome_codes = [code for code in (_find_text(item, 'CODE') for item in fiche.findall('CODES_ROME/ROME')) if code]
        nsf_codes = [code for code in (_find_text(item, 'CODE') for item in fiche.findall('CODES_NSF/NSF')) if code]
        sectors = [clean_text(text) for text in (_find_text(item, 'LIBELLE') for item in fiche.findall('SECTEURS_ACTIVITE/SECTEUR')) if clean_text(text)]
        activities_visees = _split_codes(_find_text(fiche, 'ACTIVITES_VISEES'))
        blocs: list[dict[str, Any]] = []
        skills: list[dict[str, Any]] = []
        activities: list[dict[str, Any]] = []
        negatives: list[dict[str, Any]] = []
        bloc_nodes = fiche.findall('BLOCS_COMPETENCES/BLOC_COMPETENCES')
        for index, block in enumerate(bloc_nodes, start=1):
            block_code = _find_text(block, 'CODE') or f'{cert_code}BC{index:02d}'
            block_title = _find_text(block, 'LIBELLE') or title
            raw_skill_text = _find_raw_text(block, 'LISTE_COMPETENCES')
            block_id = block_code if block_code else f'{cert_code}BC{index:02d}'
            blocs.append(
                {
                    'certification_id': cert_id,
                    'certification_code': cert_code,
                    'block_id': block_id,
                    'block_code': block_code,
                    'block_title': block_title,
                    'block_original': raw_skill_text,
                    'source_order': index,
                    'source_file': source_file,
                    'source_url': source_url,
                    'source_updated_at': source_updated_at,
                }
            )
            block_activities, block_skills, block_negatives = self.skill_extractor.extract_block_skills(
                block_code=block_id,
                block_name=block_title,
                text=raw_skill_text,
                source_page=index,
                origin_document=source_file,
            )
            activities.extend(block_activities)
            skills.extend(
                {
                    **item,
                    'certification_id': cert_id,
                    'certification_code': cert_code,
                    'certification_title': title,
                    'block_id': block_id,
                    'block_code': block_code,
                    'block_title': block_title,
                    'rome_codes': rome_codes,
                    'nsf_codes': nsf_codes,
                    'source_url': source_url,
                    'is_active': normalize_for_match(status) in {'publiee', 'publie', 'active'},
                    'confidence': float(item.get('confidence', 0.0)),
                    'normalization_method': item.get('match_type', 'exact'),
                    'source_order': index,
                    'source_file': source_file,
                }
                for item in block_skills
            )
            negatives.extend(block_negatives)

        certification = {
            'certification_id': cert_id,
            'repository_type': self.repository_type,
            'certification_code': cert_code,
            'title': title,
            'intitule_court': short_title,
            'status': 'active' if normalize_for_match(status) in {'publiee', 'publie', 'active'} else 'inactive',
            'qualification_level': level,
            'level': level,
            'european_level': _find_text(fiche, 'NOMENCLATURE_EUROPE/LIBELLE'),
            'certificateur': certificateurs,
            'partenaires': partenaires,
            'codes_nsf': nsf_codes,
            'codes_rome': rome_codes,
            'secteurs_activite': sectors,
            'activites_visees': activities_visees,
            'competences_visees': [item.get('libelle_officiel') for item in skills if item.get('libelle_officiel')],
            'blocs_competences': [item['block_id'] for item in blocs],
            'voies_acces': [clean_text(text) for text in _split_codes(_find_text(fiche, 'TYPE_ENREGISTREMENT')) if clean_text(text)],
            'modalites_evaluation': [clean_text(text) for text in _split_codes(_find_text(fiche, 'MODALITES_EVALUATION')) if clean_text(text)],
            'criteres_evaluation': [clean_text(text) for text in _split_codes(_find_text(fiche, 'CRITERES_EVALUATION')) if clean_text(text)],
            'url_source': source_url,
            'source_updated_at': source_updated_at or date_publication or _find_text(fiche, 'DATE_DERNIERE_MODIFICATION', 'DATE_DERNIERE_MODIFICATION_ETAT'),
            'valid_from': valid_from or date_publication,
            'valid_to': valid_to,
        }
        return certification, blocs, skills, activities, negatives

    def parse_archive(self, path: str | Path, *, source_url: str | None = None) -> ParsedFranceCompetences:
        archive_path = Path(path)
        if not archive_path.exists():
            raise FileNotFoundError(f'Archive France Compétences introuvable: {archive_path}')
        certifications: list[dict[str, Any]] = []
        blocks: list[dict[str, Any]] = []
        skills: list[dict[str, Any]] = []
        activities: list[dict[str, Any]] = []
        negatives: list[dict[str, Any]] = []
        metadata: dict[str, Any] = {'source_file': str(archive_path), 'repository_type': self.repository_type}
        if archive_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive_path) as zf:
                members = [item for item in zf.infolist() if not item.is_dir()]
                metadata['archive_members'] = [item.filename for item in members]
                for member in members:
                    lower = member.filename.lower()
                    if lower.endswith('.xml'):
                        with zf.open(member) as fh:
                            context = ET.iterparse(fh, events=('start', 'end'))
                            root = None
                            version_flux = None
                            for event, elem in context:
                                if root is None and event == 'start':
                                    root = elem
                                    continue
                                if event == 'end' and _local_tag(elem.tag) == 'VERSION_FLUX' and not version_flux:
                                    version_flux = clean_text(elem.text)
                                if event == 'end' and _local_tag(elem.tag) == 'FICHE':
                                    cert, cert_blocks, cert_skills, cert_activities, cert_negatives = self._parse_fiche(
                                        elem,
                                        source_file=str(archive_path),
                                        source_url=source_url,
                                        source_updated_at=version_flux,
                                    )
                                    certifications.append(cert)
                                    blocks.extend(cert_blocks)
                                    skills.extend(cert_skills)
                                    activities.extend(cert_activities)
                                    negatives.extend(cert_negatives)
                                    elem.clear()
                            metadata['xml_root'] = _local_tag(root.tag) if root is not None else 'FICHES'
                            metadata['version_flux'] = version_flux
                    elif lower.endswith('.csv'):
                        with zf.open(member) as fh:
                            payload = fh.read().decode('utf-8', errors='ignore')
                        metadata['csv_preview'] = payload[:2000]
        elif archive_path.suffix.lower() == '.xml':
            context = ET.iterparse(archive_path, events=('start', 'end'))
            root = None
            version_flux = None
            for event, elem in context:
                if root is None and event == 'start':
                    root = elem
                    continue
                if event == 'end' and _local_tag(elem.tag) == 'VERSION_FLUX' and not version_flux:
                    version_flux = clean_text(elem.text)
                if event == 'end' and _local_tag(elem.tag) == 'FICHE':
                    cert, cert_blocks, cert_skills, cert_activities, cert_negatives = self._parse_fiche(
                        elem,
                        source_file=str(archive_path),
                        source_url=source_url,
                        source_updated_at=version_flux,
                    )
                    certifications.append(cert)
                    blocks.extend(cert_blocks)
                    skills.extend(cert_skills)
                    activities.extend(cert_activities)
                    negatives.extend(cert_negatives)
                    elem.clear()
            metadata['xml_root'] = _local_tag(root.tag) if root is not None else 'FICHES'
            metadata['version_flux'] = version_flux
        else:
            raise ValueError(f'Format non supporté pour RNCP: {archive_path.suffix}')
        return ParsedFranceCompetences(certifications=certifications, blocks=blocks, skills=skills, activities=activities, negatives=negatives, metadata=metadata)
