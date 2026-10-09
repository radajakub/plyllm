from __future__ import annotations

import pytest

# what the live tests spent, summed into the terminal summary
_LIVE_COSTS = pytest.StashKey[list[float]]()

# tests that leave the machine run only when asked for: marker -> the flag that enables it
_OPT_IN = {
    "online": ("--online", "needs network and the API keys in .env (free); run with --online (make test-models)"),
    "live": ("--live", "real, paid API calls; run with --live (make test-live)"),
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--online", action="store_true", default=False, help="run tests marked `online`: free calls that need network and the API keys")
    parser.addoption("--live", action="store_true", default=False, help="run tests marked `live`: real, paid API calls (use `make test-live`, which asks first)")


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_LIVE_COSTS] = []


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    # a plain `pytest` or `make test` stays offline and free: these are skipped unless their flag is passed
    for marker, (flag, reason) in _OPT_IN.items():
        if config.getoption(flag):
            continue
        for item in items:
            if marker in item.keywords:
                item.add_marker(pytest.mark.skip(reason=reason))


@pytest.fixture
def live_costs(request: pytest.FixtureRequest) -> list[float]:
    return request.config.stash[_LIVE_COSTS]


def pytest_terminal_summary(terminalreporter, exitstatus: int, config: pytest.Config) -> None:
    costs = config.stash[_LIVE_COSTS]
    if costs:
        terminalreporter.write_line(f"live LLM calls: {len(costs)}, estimated cost ${sum(costs):.4f} (from the registry prices)")
