"""SKTK agent -- SKTKAgent, contracts, filters, capabilities, providers, loader, permissions, hooks, tools."""

import contextlib

from sktk.agent.agent import SKTKAgent
from sktk.agent.approval import ApprovalGate, ApprovalRequest, AutoApprovalFilter
from sktk.agent.capabilities import Capability, match_capabilities
from sktk.agent.contracts import parse_output, serialize_input
from sktk.agent.fallback import FallbackChain
from sktk.agent.filters import (
    AgentFilter,
    ContentSafetyFilter,
    FilterAdapter,
    FilterContext,
    PIIFilter,
    PromptInjectionFilter,
    TokenBudgetFilter,
    run_filter_pipeline,
)
from sktk.agent.hooks import LifecycleHooks
from sktk.agent.loader import (
    load_agent_from_dict,
    load_agent_from_json,
    load_agent_from_yaml,
    register_filter,
)
from sktk.agent.middleware import MiddlewareStack
from sktk.agent.openapi import tools_from_openapi, tools_from_openapi_file
from sktk.agent.permissions import PermissionPolicy, RateLimitPolicy
from sktk.agent.planner import Plan, PlanStep, StepStatus, TaskPlanner
from sktk.agent.providers import (
    CompletionResult,
    LLMProvider,
    ProviderRegistry,
    create_provider,
    get_registry,
    register_provider,
)
from sktk.agent.templates import PromptTemplate, load_prompt, load_prompts
from sktk.agent.tools import Tool, tool

# Optional deps: MCP support
with contextlib.suppress(ImportError):
    from sktk.agent.mcp import MCPToolProvider
    from sktk.agent.mcp_server import expose_as_mcp_server

# Optional deps: A2A support
with contextlib.suppress(ImportError):
    from sktk.agent.a2a import A2AClient, A2AServer, AgentCard

# Optional deps: prompt optimizer
with contextlib.suppress(ImportError):
    from sktk.agent.optimizer import OptimizationResult, PromptOptimizer

__all__ = [
    "A2AClient",
    "A2AServer",
    "AgentCard",
    "ApprovalGate",
    "ApprovalRequest",
    "AutoApprovalFilter",
    "AgentFilter",
    "Capability",
    "ContentSafetyFilter",
    "FallbackChain",
    "FilterAdapter",
    "FilterContext",
    "LifecycleHooks",
    "CompletionResult",
    "LLMProvider",
    "MCPToolProvider",
    "MiddlewareStack",
    "OptimizationResult",
    "PIIFilter",
    "PermissionPolicy",
    "Plan",
    "PlanStep",
    "PromptInjectionFilter",
    "PromptOptimizer",
    "PromptTemplate",
    "ProviderRegistry",
    "RateLimitPolicy",
    "SKTKAgent",
    "StepStatus",
    "TaskPlanner",
    "TokenBudgetFilter",
    "Tool",
    "create_provider",
    "expose_as_mcp_server",
    "get_registry",
    "load_agent_from_dict",
    "load_agent_from_json",
    "load_agent_from_yaml",
    "load_prompt",
    "load_prompts",
    "match_capabilities",
    "parse_output",
    "register_filter",
    "register_provider",
    "run_filter_pipeline",
    "serialize_input",
    "tool",
    "tools_from_openapi",
    "tools_from_openapi_file",
]
