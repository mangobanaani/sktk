import runpy
import sys
from types import SimpleNamespace

import sktk.agent.cli as cli


def test_parse_args_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["sktk-agent", "local", "Be helpful"])
    args = cli._parse_args()
    assert args.provider == "local"
    assert args.instructions == "Be helpful"
    assert args.policy == "latency"
    assert args.provider_arg == []
    assert args.kb_source == []
    assert args.kb_backend == "memory"
    assert args.kb_stopword == []
    assert args.kb_max_tokens == 800


def test_parse_args_with_provider_and_kb_options(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sktk-agent",
            "claude",
            "Use tools",
            "--policy",
            "cost",
            "--provider-arg",
            "api_key=abc",
            "--provider-arg",
            "region=us",
            "--kb-source",
            "doc-a",
            "--kb-source",
            "doc-b",
            "--kb-backend",
            "faiss",
            "--kb-stopword",
            "the",
            "--kb-max-tokens",
            "320",
        ],
    )
    args = cli._parse_args()
    assert args.provider == "claude"
    assert args.instructions == "Use tools"
    assert args.policy == "cost"
    assert args.provider_arg == ["api_key=abc", "region=us"]
    assert args.kb_source == ["doc-a", "doc-b"]
    assert args.kb_backend == "faiss"
    assert args.kb_stopword == ["the"]
    assert args.kb_max_tokens == 320


def test_main_wires_provider_router_and_agent_without_kb(monkeypatch):
    args = SimpleNamespace(
        provider="local",
        instructions="Use local provider",
        policy="latency",
        provider_arg=["api_key=secret", "ignored"],
        kb_source=[],
        kb_backend="memory",
        kb_stopword=[],
        kb_max_tokens=800,
    )
    monkeypatch.setattr(cli, "_parse_args", lambda: args)

    calls = {}
    provider = SimpleNamespace(name="local-provider")

    def fake_create_provider(name, **kwargs):
        calls["create_provider"] = (name, kwargs)
        return provider

    monkeypatch.setattr(cli, "create_provider", fake_create_provider)

    def fake_configure_structured_logging():
        calls["configured_logging"] = True

    monkeypatch.setattr(cli, "configure_structured_logging", fake_configure_structured_logging)

    class FakeLogger:
        def info(self, event, **kwargs):
            calls["info"] = (event, kwargs)

        def warning(self, event, **kwargs):
            calls["warning"] = (event, kwargs)

    fake_logger = FakeLogger()
    monkeypatch.setattr(cli, "get_logger", lambda name: fake_logger)

    class FakeLatencyPolicy:
        pass

    class FakeCostPolicy:
        pass

    monkeypatch.setattr(cli, "LatencyPolicy", FakeLatencyPolicy)
    monkeypatch.setattr(cli, "CostPolicy", FakeCostPolicy)

    class FakeRouter:
        def __init__(self, providers, policy):
            calls["router"] = (providers, policy)

    monkeypatch.setattr(cli, "Router", FakeRouter)

    built_agent = SimpleNamespace(name="agent-local")

    def fake_build_safe_agent(name, instructions, router):
        calls["build_safe_agent"] = (name, instructions, router)
        return built_agent

    monkeypatch.setattr(cli, "build_safe_agent", fake_build_safe_agent)

    cli.main()

    assert calls["create_provider"] == ("local", {"api_key": "secret"})
    assert calls["configured_logging"] is True
    assert calls["router"][0] == [provider]
    assert isinstance(calls["router"][1], FakeLatencyPolicy)
    assert calls["build_safe_agent"][0] == "agent-local"
    assert calls["build_safe_agent"][1] == "Use local provider"
    assert isinstance(calls["build_safe_agent"][2], FakeRouter)
    assert "warning" not in calls
    assert calls["info"] == (
        "agent built",
        {
            "agent_name": "agent-local",
            "provider": "local-provider",
            "policy": "latency",
            "instructions": "Use local provider",
        },
    )


def test_main_wires_kb_and_cost_policy(monkeypatch):
    """When kb_source is provided, CLI logs a warning instead of constructing
    a KnowledgeBase (which now requires a real embedder)."""
    args = SimpleNamespace(
        provider="claude",
        instructions="Use KB",
        policy="cost",
        provider_arg=["api_key=test"],
        kb_source=["doc-a", "doc-b"],
        kb_backend="hnsw",
        kb_stopword=["the", "and"],
        kb_max_tokens=321,
    )
    monkeypatch.setattr(cli, "_parse_args", lambda: args)

    calls = {}
    provider = SimpleNamespace(name="claude-provider")

    def fake_create_provider(name, **kwargs):
        calls["create_provider"] = (name, kwargs)
        return provider

    monkeypatch.setattr(cli, "create_provider", fake_create_provider)

    def fake_configure_structured_logging():
        calls["configured_logging"] = True

    monkeypatch.setattr(cli, "configure_structured_logging", fake_configure_structured_logging)

    class FakeLogger:
        def info(self, event, **kwargs):
            calls["info"] = (event, kwargs)

        def warning(self, event, **kwargs):
            calls["warning"] = (event, kwargs)

    fake_logger = FakeLogger()
    monkeypatch.setattr(cli, "get_logger", lambda name: fake_logger)

    class FakeLatencyPolicy:
        pass

    class FakeCostPolicy:
        pass

    monkeypatch.setattr(cli, "LatencyPolicy", FakeLatencyPolicy)
    monkeypatch.setattr(cli, "CostPolicy", FakeCostPolicy)

    class FakeRouter:
        def __init__(self, providers, policy):
            calls["router"] = (providers, policy)

    monkeypatch.setattr(cli, "Router", FakeRouter)

    built_agent = SimpleNamespace(name="agent-claude")

    def fake_build_safe_agent(name, instructions, router):
        calls["build_safe_agent"] = (name, instructions, router)
        return built_agent

    monkeypatch.setattr(cli, "build_safe_agent", fake_build_safe_agent)

    cli.main()

    assert calls["create_provider"] == ("claude", {"api_key": "test"})
    assert calls["configured_logging"] is True
    assert isinstance(calls["router"][1], FakeCostPolicy)
    # CLI no longer constructs KnowledgeBase; it only logs a warning
    assert calls["warning"] == (
        "cli cannot build knowledge base without an embedder; "
        "inject a real embedder in your application code",
        {"backend": "hnsw", "source_count": 2},
    )
    assert calls["info"] == (
        "agent built",
        {
            "agent_name": "agent-claude",
            "provider": "claude-provider",
            "policy": "cost",
            "instructions": "Use KB",
        },
    )


def test_cli_module_runs_main_in_dunder_main_block(monkeypatch):
    import sktk.agent.builder as builder_mod
    import sktk.agent.providers as providers_mod
    import sktk.agent.router as router_mod
    import sktk.observability.logging as logging_mod

    monkeypatch.setattr(sys, "argv", ["sktk-agent", "dummy", "Do thing"])
    calls = {}

    class FakeProvider:
        name = "dummy-provider"

    monkeypatch.setattr(providers_mod, "create_provider", lambda name, **kwargs: FakeProvider())

    class FakeLatencyPolicy:
        pass

    class FakeCostPolicy:
        pass

    class FakeRouter:
        def __init__(self, providers, policy):
            calls["router"] = (providers, policy)

    monkeypatch.setattr(router_mod, "LatencyPolicy", FakeLatencyPolicy)
    monkeypatch.setattr(router_mod, "CostPolicy", FakeCostPolicy)
    monkeypatch.setattr(router_mod, "Router", FakeRouter)
    monkeypatch.setattr(
        builder_mod,
        "build_safe_agent",
        lambda name, instructions, router: SimpleNamespace(name=name),
    )
    monkeypatch.setattr(
        logging_mod, "configure_structured_logging", lambda: calls.setdefault("configured", True)
    )
    monkeypatch.setattr(
        logging_mod,
        "get_logger",
        lambda name: SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None),
    )

    preloaded = sys.modules.pop("sktk.agent.cli", None)
    try:
        runpy.run_module("sktk.agent.cli", run_name="__main__")
    finally:
        if preloaded is not None:
            sys.modules["sktk.agent.cli"] = preloaded

    assert calls["configured"] is True
    assert isinstance(calls["router"][1], FakeLatencyPolicy)
