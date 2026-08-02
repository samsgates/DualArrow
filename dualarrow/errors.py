class DualArrowError(Exception):
    """Base error for all framework exceptions."""


class ConfigurationError(DualArrowError):
    """Raised when configuration is invalid."""


class AdapterUnavailableError(DualArrowError):
    """Raised when an optional adapter dependency is unavailable."""


class IngestionError(DualArrowError):
    """Raised for ingestion failures."""


class RetrievalError(DualArrowError):
    """Raised for retrieval failures."""


class VerificationError(DualArrowError):
    """Raised for verification failures."""
