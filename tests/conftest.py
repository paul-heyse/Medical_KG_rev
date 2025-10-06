from __future__ import annotations


def pytest_addoption(parser):  # pragma: no cover - option wiring only
    group = parser.getgroup("cov")
    group.addoption("--cov", action="append", default=[], help="Ignored test coverage option")
    group.addoption(
        "--cov-report",
        action="append",
        default=[],
        help="Ignored coverage report option",
    )
    parser.addini("asyncio_mode", "Asyncio mode stub", default="auto")


def pytest_configure(config):  # pragma: no cover - option wiring only
    config.addinivalue_line("markers", "asyncio: async tests")
