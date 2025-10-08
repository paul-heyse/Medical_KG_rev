import pytest

from Medical_KG_rev.services.chunking.profiles.loader import (
    ProfileNotFoundError,
    ProfileRepository,
)


def test_profile_repository_caches_profiles(tmp_path):
    profile_path = tmp_path / "example.yaml"
    profile_path.write_text(
        """
name: example
domain: test
chunker_type: simple
target_tokens: 100
        """.strip()
    )

    repo = ProfileRepository(directory=tmp_path)
    first = repo.get("example")
    second = repo.get("example")

    assert first is second
    assert first.target_tokens == 100


def test_profile_repository_missing(tmp_path):
    repo = ProfileRepository(directory=tmp_path)
    with pytest.raises(ProfileNotFoundError):
        repo.get("missing")
