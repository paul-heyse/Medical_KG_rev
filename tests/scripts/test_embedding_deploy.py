import pytest

from scripts.embedding.deploy import build_kubectl_command, deploy


def test_build_kubectl_command_includes_overlay() -> None:
    command = build_kubectl_command("staging", dry_run=True)
    assert command[:3] == ["kubectl", "apply", "--dry-run=server"]
    assert command[-2:] == ["-k", "ops/k8s/overlays/staging"]


def test_deploy_invokes_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    executed: dict[str, list[str]] = {}

    monkeypatch.setattr("scripts.embedding.deploy.shutil.which", lambda _: ".kubectl")

    def fake_run(command, check):  # noqa: D401 - mimic subprocess.run
        executed["command"] = command
        executed["check"] = check

    monkeypatch.setattr("scripts.embedding.deploy.subprocess.run", fake_run)

    deploy("production", dry_run=False)
    assert executed["command"] == ["kubectl", "apply", "-k", "ops/k8s/overlays/production"]
    assert executed["check"] is True


def test_deploy_requires_kubectl(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("scripts.embedding.deploy.shutil.which", lambda _: None)
    with pytest.raises(RuntimeError):
        deploy("staging", dry_run=False)
