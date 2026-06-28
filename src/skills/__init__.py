from src.skills.skill_normalizer import SkillNormalizer
from src.skills.merge_offer_skills import extract_skills_from_text, merge_offer_skills
from src.skills.open_extractor import (
    extract_skills,
    tag_with_ia_categories,
    ExtractedSkill,
)
from src.skills.referential_manager import BUILTIN_REFERENTIAL, match_referential
