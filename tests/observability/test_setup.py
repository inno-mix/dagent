"""Turning the no-op providers into real ones, and the scrape endpoint FR-9 wants."""

import pytest

from dagent.errors import DagentError
from dagent.observability import setup


def test_prometheus_text_renders_the_exposition_format() -> None:
    # "Metrics are scrapeable": what a /metrics endpoint would return, rendered from the
    # same registry anything already scraping this process reads.
    text = setup.prometheus_text()

    assert isinstance(text, str)
    assert "# HELP" in text or text == ""


def test_configure_survives_being_called_twice() -> None:
    # An application may configure once at startup and again after reading its own config.
    # Traces are left to `test_tracing`, which owns this process's tracer provider:
    # OpenTelemetry allows exactly one, and two modules installing one is a race.
    setup.configure(traces=False, metrics=True, logs=False)
    setup.configure(traces=False, metrics=True, logs=False)


def test_configure_can_be_asked_for_nothing_at_all() -> None:
    # Signals are opt-out as well as opt-in; asking for none must not blow up or reach
    # for the SDK.
    setup.configure(traces=False, metrics=False, logs=False)


def test_configure_reports_a_missing_sdk_rather_than_an_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The SDK is an extra. Someone who never installed it should get a sentence naming the
    # extra, not a traceback from three libraries down.
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(DagentError, match="dagent\\[otel\\]"):
        setup.configure(logs=False)


def test_metrics_recorded_after_configuring_reach_the_prometheus_registry() -> None:
    # End to end: configure, record, scrape. The instrument has to show up in the text a
    # Prometheus server would pull.
    from dagent.observability import metrics as obs_metrics

    setup.configure(traces=False, logs=False)
    obs_metrics.metrics_for().tokens.add(7, {obs_metrics.PROVIDER: "stub"})

    text = setup.prometheus_text()

    assert "dagent_model_tokens" in text
    assert 'provider="stub"' in text


def test_the_console_exporter_branch_builds_a_provider() -> None:
    # Only reachable when someone asks for it, and only useful once during development —
    # but an untested branch in the wiring is one that breaks the day it is needed.
    setup.configure(traces=True, metrics=False, logs=False, console_traces=True)


def test_configuring_only_logs_never_reaches_for_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Someone with no extra installed can still have structured logs. Pretending the SDK
    # is absent proves the early return happens before anything looks for it.
    monkeypatch.setattr(setup.importlib.util, "find_spec", lambda name: None)

    setup.configure(traces=False, metrics=False, logs=True, log_level="WARNING")
