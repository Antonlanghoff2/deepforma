from __future__ import annotations


class DeepformaError(Exception):
    """Base exception for Deepforma domain and pipeline errors."""


class DataSourceError(DeepformaError):
    """Raised when a data source cannot be read or normalized."""


class ParsingError(DeepformaError):
    """Raised when an input payload or document cannot be parsed."""


class SkillExtractionError(DeepformaError):
    """Raised when skill extraction fails."""


class MatchingError(DeepformaError):
    """Raised when matching or scoring fails."""


class ConfigurationError(DeepformaError):
    """Raised when configuration is incomplete or invalid."""
