from __future__ import annotations

import pytest

from alexandros.evaluation import (
    CodeBenchmarkNotReadyError,
    CodeBenchmarkPrerequisites,
    CodeBenchmarkSuite,
    HumanEvalStyleTask,
    build_humaneval_style_prompt,
    code_benchmark_plan,
    ensure_code_benchmark_ready,
    humaneval_style_harness_report,
    validate_humaneval_style_tasks,
)


def test_code_benchmark_plan_reports_missing_requirements() -> None:
    plan = code_benchmark_plan(CodeBenchmarkSuite.HUMANEVAL_STYLE)
    assert not plan.runnable
    assert plan.missing_requirements == (
        "tokenizer",
        "trained_checkpoint",
        "execution_sandbox",
        "release_policy",
        "dataset_license_review",
    )
    with pytest.raises(CodeBenchmarkNotReadyError, match="humaneval_style"):
        ensure_code_benchmark_ready(plan)


def test_code_benchmark_plan_is_ready_when_gates_are_satisfied() -> None:
    prerequisites = CodeBenchmarkPrerequisites(
        tokenizer=True,
        trained_checkpoint=True,
        execution_sandbox=True,
        release_policy=True,
        dataset_license_review=True,
    )
    plan = code_benchmark_plan("humaneval_style", prerequisites)
    assert plan.runnable
    assert plan.missing_requirements == ()
    ensure_code_benchmark_ready(plan)


def test_agentic_benchmark_plans_require_tool_environment() -> None:
    prerequisites = CodeBenchmarkPrerequisites(
        tokenizer=True,
        trained_checkpoint=True,
        execution_sandbox=True,
        release_policy=True,
        dataset_license_review=True,
    )
    for suite in (CodeBenchmarkSuite.SWE_BENCH, CodeBenchmarkSuite.TERMINAL_BENCH):
        plan = code_benchmark_plan(suite, prerequisites)
        assert not plan.runnable
        assert plan.missing_requirements == ("tool_environment",)


def test_unknown_code_benchmark_suite_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown code benchmark suite"):
        code_benchmark_plan("not-a-suite")


def test_humaneval_style_task_validation_and_report() -> None:
    task = HumanEvalStyleTask(
        task_id="toy/add",
        prompt="def add(a, b):\n    ",
        entry_point="add",
        tests=("assert add(1, 2) == 3",),
    )
    assert build_humaneval_style_prompt(task) == "def add(a, b):\n"
    assert validate_humaneval_style_tasks([task]) == (task,)
    report = humaneval_style_harness_report([task])
    assert report.suite is CodeBenchmarkSuite.HUMANEVAL_STYLE
    assert report.task_count == 1
    assert not report.runnable
    assert "execution_sandbox" in report.missing_requirements


def test_humaneval_style_task_validation_rejects_bad_metadata() -> None:
    with pytest.raises(ValueError, match="task_id"):
        HumanEvalStyleTask("", "def f(): pass", "f", ("assert True",))
    with pytest.raises(ValueError, match="tests"):
        HumanEvalStyleTask("toy/empty", "def f(): pass", "f", ())
    task = HumanEvalStyleTask("toy/dup", "def f():\n    ", "f", ("assert f() is None",))
    with pytest.raises(ValueError, match="duplicate"):
        validate_humaneval_style_tasks([task, task])
