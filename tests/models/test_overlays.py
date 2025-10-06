import pytest

from Medical_KG_rev.models.overlays.finance import FinancialDocument, FinancialFact, XBRLContext
from Medical_KG_rev.models.overlays.legal import LegalDocument, LegalClause, LegalReference
from Medical_KG_rev.models.overlays.medical import EvidenceAssessment, MedicalDocument


def test_medical_document_limit_assessments():
    assessments = [EvidenceAssessment() for _ in range(3)]
    document = MedicalDocument(id="d1", source="clinicaltrials", sections=[], evidence_assessments=assessments)
    assert len(document.evidence_assessments) == 3

    with pytest.raises(ValueError):
        MedicalDocument(
            id="d2",
            source="clinicaltrials",
            sections=[],
            evidence_assessments=[EvidenceAssessment() for _ in range(11)],
        )


def test_financial_document_context_validation():
    contexts = [XBRLContext(identifier="c1", entity_scheme="lei", entity_identifier="123")]
    facts = [FinancialFact(concept="Revenue", value="10", context_ref="c1")]
    document = FinancialDocument(id="f1", source="sec", sections=[], contexts=contexts, facts=facts)
    assert document.facts[0].concept == "Revenue"

    with pytest.raises(ValueError):
        FinancialDocument(id="f2", source="sec", sections=[], contexts=contexts, facts=[FinancialFact(concept="R", value="1", context_ref="missing")])


def test_legal_document_clause_deduplication():
    clause = LegalClause(id="c1", references=[LegalReference(target="CaseA"), LegalReference(target="CaseA")])
    document = LegalDocument(id="l1", source="lex", sections=[], clauses=[clause], jurisdiction="us-ca")
    assert document.clauses[0].references[0].target == "CaseA"
    assert document.jurisdiction == "US-CA"
