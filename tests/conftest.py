"""Pytest configuration for claudetube tests."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-deep-integration",
        action="store_true",
        default=False,
        help="Run the deep integration test (slow, requires network)",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests against real provider APIs (requires API keys)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-deep-integration"):
        skip = pytest.mark.skip(reason="needs --run-deep-integration flag")
        for item in items:
            if "deep_integration" in item.keywords:
                item.add_marker(skip)

    if not config.getoption("--run-integration"):
        skip = pytest.mark.skip(reason="needs --run-integration flag")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)
