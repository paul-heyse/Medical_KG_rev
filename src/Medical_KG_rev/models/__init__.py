"""Data models exposed by the foundation infrastructure layer."""
from .entities import Claim, Entity, Evidence, ExtractionActivity
from .ir import Block, BlockType, Document, Section, Span, Table, TableCell
from .organization import Organization, TenantContext
from .overlays.finance import FinancialDocument, FinancialFact, XBRLContext
from .overlays.legal import LegalClause, LegalDocument, LegalReference
from .overlays.medical import EvidenceAssessment, MedicalDocument, ResearchStudy
from .provenance import DataSource

__all__ = [
    "Block",
    "BlockType",
    "Claim",
    "DataSource",
    "Document",
    "Entity",
    "Evidence",
    "EvidenceAssessment",
    "FinancialDocument",
    "FinancialFact",
    "LegalClause",
    "LegalDocument",
    "LegalReference",
    "MedicalDocument",
    "Organization",
    "ResearchStudy",
    "Section",
    "Span",
    "Table",
    "TableCell",
    "TenantContext",
    "XBRLContext",
]
