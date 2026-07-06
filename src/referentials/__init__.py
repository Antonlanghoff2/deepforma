from __future__ import annotations

try:
    from .ai_certification_referential import AICertificationReferential
except Exception:  # pragma: no cover - optional dependency chain
    AICertificationReferential = None  # type: ignore[assignment]
from .france_competences import (
    FranceCompetenceBlock,
    FranceCompetenceCertification,
    FranceCompetenceSkill,
    FranceCompetencesApiClient,
    FranceCompetencesOpenDataImporter,
    FranceCompetencesSource,
)
from .offer_skill_enricher import RNCPROMEOfferEnricher
from .rncp_rome_mapper import RNCPRomeMapper, RNCPRomeMatch
from .rome_referential import RomeJob, RomeReferentialImporter, RomeService, RomeSkill, get_default_rome_service, validate_rome_code
from .unified_skill_referential import (
    UnifiedSkill,
    UnifiedSkillReferential,
    UnifiedSkillSourceLink,
    build_unified_skill_referential,
    canonical_skill_id,
    write_unified_skill_referential,
)

__all__ = [
    'AICertificationReferential',
    'FranceCompetenceBlock',
    'FranceCompetenceCertification',
    'FranceCompetenceSkill',
    'FranceCompetencesApiClient',
    'FranceCompetencesOpenDataImporter',
    'FranceCompetencesSource',
    'RNCPROMEOfferEnricher',
    'RNCPRomeMapper',
    'RNCPRomeMatch',
    'RomeJob',
    'RomeReferentialImporter',
    'RomeService',
    'RomeSkill',
    'get_default_rome_service',
    'validate_rome_code',
    'UnifiedSkill',
    'UnifiedSkillReferential',
    'UnifiedSkillSourceLink',
    'build_unified_skill_referential',
    'canonical_skill_id',
    'write_unified_skill_referential',
]
