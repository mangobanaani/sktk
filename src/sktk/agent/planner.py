"""Goal-oriented task decomposition and planning.

Breaks complex goals into ordered sub-tasks that can be executed
by agents or tools. Supports dependency tracking between steps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepStatus(Enum):
    """Lifecycle states for an individual plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PlanStep:
    """A single step in an execution plan."""

    id: str
    description: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str | None = None

    @property
    def is_ready(self) -> bool:
        """Check if step is pending with no declared dependencies.

        Note: This does not account for dependency completion state.
        Use Plan.ready_steps() for dependency-aware readiness checks.
        """
        return self.status == StepStatus.PENDING and not self.depends_on


@dataclass
class Plan:
    """An ordered execution plan with dependency tracking.

    Usage:
        plan = Plan(goal="Analyze quarterly sales data")
        plan.add_step(PlanStep(id="1", description="Fetch data", tool_name="query_db"))
        plan.add_step(PlanStep(id="2", description="Analyze", depends_on=["1"]))

        for step in plan.ready_steps():
            # execute step
            plan.complete_step(step.id, result="done")
    """

    goal: str
    steps: list[PlanStep] = field(default_factory=list)

    def add_step(self, step: PlanStep) -> None:
        self.steps.append(step)

    def get_step(self, step_id: str) -> PlanStep | None:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def ready_steps(self) -> list[PlanStep]:
        """Return steps whose dependencies are all completed.

        Steps with any failed or skipped dependency are excluded -- they
        cannot execute meaningfully and will be marked SKIPPED by
        ``execute_plan``.
        """
        completed_ids = {s.id for s in self.steps if s.status == StepStatus.COMPLETED}
        failed_ids = {
            s.id for s in self.steps if s.status in (StepStatus.FAILED, StepStatus.SKIPPED)
        }
        return [
            s
            for s in self.steps
            if s.status == StepStatus.PENDING
            and all(d in completed_ids for d in s.depends_on)
            and not any(d in failed_ids for d in s.depends_on)
        ]

    def complete_step(self, step_id: str, result: Any = None) -> None:
        step = self.get_step(step_id)
        if step is None:
            raise KeyError(f"Step '{step_id}' not found")
        step.status = StepStatus.COMPLETED
        step.result = result

    def fail_step(self, step_id: str, error: str) -> None:
        step = self.get_step(step_id)
        if step is None:
            raise KeyError(f"Step '{step_id}' not found")
        step.status = StepStatus.FAILED
        step.error = error

    @property
    def is_complete(self) -> bool:
        return all(
            s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED)
            for s in self.steps
        )

    @property
    def progress(self) -> float:
        if not self.steps:
            return 1.0
        done = sum(
            1
            for s in self.steps
            if s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED, StepStatus.FAILED)
        )
        return done / len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "steps": [
                {
                    "id": s.id,
                    "description": s.description,
                    "tool_name": s.tool_name,
                    "tool_args": s.tool_args,
                    "depends_on": s.depends_on,
                    "status": s.status.value,
                    "result": s.result,
                    "error": s.error,
                }
                for s in self.steps
            ],
            "progress": self.progress,
        }


class TaskPlanner:
    """Creates execution plans from a goal and available tools.

    In production, this would call the LLM to decompose goals.
    For now, provides manual plan construction and execution tracking.

    Usage:
        planner = TaskPlanner()
        plan = planner.create_plan("Summarize the Q1 report", available_tools=["fetch", "summarize"])
    """

    def create_plan(self, goal: str, steps: list[dict[str, Any]] | None = None) -> Plan:
        """Create a plan from explicit steps."""
        plan = Plan(goal=goal)
        if steps:
            for s in steps:
                plan.add_step(
                    PlanStep(
                        id=s.get("id", str(len(plan.steps) + 1)),
                        description=s["description"],
                        tool_name=s.get("tool_name"),
                        tool_args=s.get("tool_args", {}),
                        depends_on=s.get("depends_on", []),
                    )
                )
        return plan

    async def execute_plan(self, plan: Plan, executor: Any) -> Plan:
        """Execute a plan step by step using the provided executor.

        The executor should be callable with (step: PlanStep) -> Any.
        When a step fails, all transitively downstream steps that can no
        longer execute are marked SKIPPED.
        """
        failed_ids: set[str] = set()
        while not plan.is_complete:
            ready = plan.ready_steps()
            if not ready:
                break
            for step in ready:
                step.status = StepStatus.IN_PROGRESS
                try:
                    result = await executor(step)
                    plan.complete_step(step.id, result=result)
                except Exception as e:
                    plan.fail_step(step.id, error=str(e))
                    failed_ids.add(step.id)

        # Mark any remaining PENDING steps as SKIPPED (unreachable due to failures)
        for step in plan.steps:
            if step.status == StepStatus.PENDING:
                step.status = StepStatus.SKIPPED
        return plan
