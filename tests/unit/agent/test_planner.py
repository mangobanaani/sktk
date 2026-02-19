# tests/unit/agent/test_planner.py
import pytest

from sktk.agent.planner import Plan, PlanStep, StepStatus, TaskPlanner


def test_plan_add_and_get():
    plan = Plan(goal="test")
    step = PlanStep(id="1", description="first step")
    plan.add_step(step)
    assert plan.get_step("1") is step
    assert plan.get_step("999") is None


def test_ready_steps_no_deps():
    plan = Plan(goal="test")
    plan.add_step(PlanStep(id="1", description="first"))
    plan.add_step(PlanStep(id="2", description="second"))
    assert len(plan.ready_steps()) == 2


def test_ready_steps_with_deps():
    plan = Plan(goal="test")
    plan.add_step(PlanStep(id="1", description="first"))
    plan.add_step(PlanStep(id="2", description="second", depends_on=["1"]))
    ready = plan.ready_steps()
    assert len(ready) == 1
    assert ready[0].id == "1"


def test_complete_unlocks_dependents():
    plan = Plan(goal="test")
    plan.add_step(PlanStep(id="1", description="first"))
    plan.add_step(PlanStep(id="2", description="second", depends_on=["1"]))
    plan.complete_step("1", result="done")
    ready = plan.ready_steps()
    assert len(ready) == 1
    assert ready[0].id == "2"


def test_fail_step():
    plan = Plan(goal="test")
    plan.add_step(PlanStep(id="1", description="first"))
    plan.fail_step("1", error="boom")
    assert plan.get_step("1").status == StepStatus.FAILED
    assert plan.get_step("1").error == "boom"


def test_is_complete():
    plan = Plan(goal="test")
    plan.add_step(PlanStep(id="1", description="first"))
    assert not plan.is_complete
    plan.complete_step("1")
    assert plan.is_complete


def test_progress():
    plan = Plan(goal="test")
    plan.add_step(PlanStep(id="1", description="first"))
    plan.add_step(PlanStep(id="2", description="second"))
    assert plan.progress == 0.0
    plan.complete_step("1")
    assert plan.progress == 0.5
    plan.complete_step("2")
    assert plan.progress == 1.0


def test_progress_empty():
    plan = Plan(goal="test")
    assert plan.progress == 1.0


def test_to_dict():
    plan = Plan(goal="test goal")
    plan.add_step(PlanStep(id="1", description="step one", tool_name="search"))
    d = plan.to_dict()
    assert d["goal"] == "test goal"
    assert len(d["steps"]) == 1
    assert d["steps"][0]["tool_name"] == "search"


def test_task_planner_create():
    planner = TaskPlanner()
    plan = planner.create_plan(
        "Analyze data",
        steps=[
            {"id": "1", "description": "Fetch", "tool_name": "fetch_data"},
            {"id": "2", "description": "Analyze", "depends_on": ["1"]},
        ],
    )
    assert plan.goal == "Analyze data"
    assert len(plan.steps) == 2
    assert plan.steps[1].depends_on == ["1"]


@pytest.mark.asyncio
async def test_task_planner_execute():
    planner = TaskPlanner()
    plan = planner.create_plan(
        "test",
        steps=[
            {"id": "1", "description": "step 1"},
            {"id": "2", "description": "step 2", "depends_on": ["1"]},
        ],
    )

    async def executor(step):
        return f"done:{step.id}"

    result = await planner.execute_plan(plan, executor)
    assert result.is_complete
    assert result.get_step("1").result == "done:1"
    assert result.get_step("2").result == "done:2"


@pytest.mark.asyncio
async def test_task_planner_execute_with_failure():
    planner = TaskPlanner()
    plan = planner.create_plan(
        "test",
        steps=[{"id": "1", "description": "will fail"}],
    )

    async def executor(step):
        raise ValueError("boom")

    result = await planner.execute_plan(plan, executor)
    assert result.get_step("1").status == StepStatus.FAILED


def test_complete_step_not_found():
    plan = Plan(goal="test")
    with pytest.raises(KeyError):
        plan.complete_step("missing")


def test_fail_step_not_found():
    plan = Plan(goal="test")
    with pytest.raises(KeyError):
        plan.fail_step("missing", "err")


def test_plan_step_is_ready_respects_dependencies():
    step = PlanStep(id="2", description="second", depends_on=["1"])
    assert step.is_ready is False
