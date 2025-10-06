import pytest

from Medical_KG_rev.models.organization import Organization, TenantContext


def test_organization_domain_validation():
    org = Organization(id="org1", name="Org", domain="example.com")
    assert org.domain == "example.com"

    with pytest.raises(ValueError):
        Organization(id="org2", name="Org", domain="invalid")


def test_tenant_context_feature_flags_normalized():
    org = Organization(id="org1", name="Org")
    ctx = TenantContext(
        tenant_id="tenant",
        organization=org,
        correlation_id="corr",
        feature_flags={"FeatureA": True},
    )
    assert ctx.feature_flags["featurea"] is True
