"""Domain specific document overlays."""
from .finance import FinancialDocument, FinancialFact, XBRLContext
from .legal import LegalClause, LegalDocument, LegalReference
from .medical import EvidenceAssessment, MedicalDocument, ResearchStudy

__all__ = [
    "EvidenceAssessment",
    "FinancialDocument",
    "FinancialFact",
    "LegalClause",
    "LegalDocument",
    "LegalReference",
    "MedicalDocument",
    "ResearchStudy",
    "XBRLContext",
]
