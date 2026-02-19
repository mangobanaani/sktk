"""Minimal CLI helpers for common agent/router setup."""

from __future__ import annotations

import argparse

from sktk.agent.builder import build_safe_agent
from sktk.agent.providers import create_provider
from sktk.agent.router import CostPolicy, LatencyPolicy, Router
from sktk.observability.logging import configure_structured_logging, get_logger


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for provider, routing, and optional KB wiring."""
    parser = argparse.ArgumentParser(description="SKTK agent helper")
    parser.add_argument(
        "provider", help="Provider name (azure-openai, claude, gemini, local, etc.)"
    )
    parser.add_argument("instructions", help="System instructions for the agent")
    parser.add_argument("--policy", choices=["latency", "cost"], default="latency")
    parser.add_argument(
        "--provider-arg", action="append", default=[], help="key=value pairs for provider init"
    )
    parser.add_argument(
        "--kb-source", action="append", default=[], help="Knowledge base sources (files or strings)"
    )
    parser.add_argument("--kb-backend", choices=["memory", "faiss", "hnsw"], default="memory")
    parser.add_argument("--kb-stopword", action="append", default=[], help="Stopwords to drop")
    parser.add_argument(
        "--kb-max-tokens", type=int, default=800, help="Approx token budget for chunking"
    )
    return parser.parse_args()


def main() -> None:
    """Build an agent/router from CLI args and print a quick setup summary."""
    configure_structured_logging()
    logger = get_logger("sktk.cli")
    args = _parse_args()
    kwargs = {}
    for item in args.provider_arg:
        if "=" in item:
            k, v = item.split("=", 1)
            kwargs[k] = v

    provider = create_provider(args.provider, **kwargs)
    policy = LatencyPolicy() if args.policy == "latency" else CostPolicy()
    router = Router([provider], policy=policy)
    agent = build_safe_agent(
        name=f"agent-{args.provider}", instructions=args.instructions, router=router
    )

    if args.kb_source:
        # KnowledgeBase now requires a real embedder at construction time.
        # The CLI cannot instantiate one without provider credentials, so we
        # only log a reminder.  Application code should construct the KB with
        # a concrete Embedder implementation.
        logger.warning(
            "cli cannot build knowledge base without an embedder; "
            "inject a real embedder in your application code",
            backend=args.kb_backend,
            source_count=len(args.kb_source),
        )

    logger.info(
        "agent built",
        agent_name=agent.name,
        provider=provider.name,
        policy=args.policy,
        instructions=args.instructions,
    )


if __name__ == "__main__":
    main()
