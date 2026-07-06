from __future__ import annotations

from pathlib import Path
from typing import Any

from .rncp_parser import FranceCompetencesRncpParser, ParsedFranceCompetences


class FranceCompetencesRsParser(FranceCompetencesRncpParser):
    repository_type = 'RS'

    def parse_archive(self, path: str | Path, *, source_url: str | None = None) -> ParsedFranceCompetences:
        parsed = super().parse_archive(path, source_url=source_url)
        for certification in parsed.certifications:
            certification['repository_type'] = self.repository_type
        return parsed

