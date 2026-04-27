from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class CodeBenchmarkSuite(str, Enum):
    HUMANEVAL_STYLE = "humaneval_style"
    SWE_BENCH = "swe_bench"
    TERMINAL_BENCH = "terminal_bench"


class CodeBenchmarkNotReadyError(RuntimeError):
    """Raised when a benchmark is requested before its safety gates are met."""


@dataclass(frozen=True)
class CodeBenchmarkPrerequisites:
    tokenizer: bool = False
    trained_checkpoint: bool = False
    execution_sandbox: bool = False
    release_policy: bool = False
    dataset_license_review: bool = False
    tool_environment: bool = False


@dataclass(frozen=True)
class CodeBenchmarkPlan:
    suite: CodeBenchmarkSuite
    runnable: bool
    missing_requirements: tuple[str, ...]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class HumanEvalStyleTask:
    task_id: str
    prompt: str
    entry_point: str
    tests: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id.strip():
            raise ValueError("task_id must be a non-empty string")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if not isinstance(self.entry_point, str) or not self.entry_point.strip():
            raise ValueError("entry_point must be a non-empty string")
        if not isinstance(self.tests, tuple):
            raise ValueError("tests must be a tuple of strings")
        if not self.tests:
            raise ValueError("tests must contain at least one assertion")
        for test in self.tests:
            if not isinstance(test, str) or not test.strip():
                raise ValueError("tests must contain non-empty strings")


@dataclass(frozen=True)
class HumanEvalStyleHarnessReport:
    suite: CodeBenchmarkSuite
    task_count: int
    runnable: bool
    missing_requirements: tuple[str, ...]
    notes: tuple[str, ...]


def _normalize_suite(suite: CodeBenchmarkSuite | str) -> CodeBenchmarkSuite:
    if isinstance(suite, CodeBenchmarkSuite):
        return suite
    try:
        return CodeBenchmarkSuite(str(suite))
    except ValueError as exc:
        choices = ", ".join(item.value for item in CodeBenchmarkSuite)
        raise ValueError(
            f"unknown code benchmark suite; expected one of {choices}"
        ) from exc


def code_benchmark_plan(
    suite: CodeBenchmarkSuite | str,
    prerequisites: CodeBenchmarkPrerequisites | None = None,
) -> CodeBenchmarkPlan:
    suite = _normalize_suite(suite)
    prerequisites = prerequisites or CodeBenchmarkPrerequisites()
    required = {
        "tokenizer": prerequisites.tokenizer,
        "trained_checkpoint": prerequisites.trained_checkpoint,
        "execution_sandbox": prerequisites.execution_sandbox,
        "release_policy": prerequisites.release_policy,
        "dataset_license_review": prerequisites.dataset_license_review,
    }
    notes = [
        "Code benchmarks are not quality claims unless run on a trained checkpoint "
        "with a documented tokenizer and evaluation report."
    ]
    if suite in {CodeBenchmarkSuite.SWE_BENCH, CodeBenchmarkSuite.TERMINAL_BENCH}:
        required["tool_environment"] = prerequisites.tool_environment
        notes.append(
            "Agentic benchmarks require isolated repositories, controlled terminal "
            "execution, and task-level environment capture."
        )
    else:
        notes.append(
            "HumanEval-style evaluation requires generated code to run only inside "
            "a reviewed sandbox; this module does not execute code."
        )
    missing = tuple(name for name, ready in required.items() if not ready)
    return CodeBenchmarkPlan(
        suite=suite,
        runnable=not missing,
        missing_requirements=missing,
        notes=tuple(notes),
    )


def ensure_code_benchmark_ready(plan: CodeBenchmarkPlan) -> None:
    if plan.runnable:
        return
    missing = ", ".join(plan.missing_requirements)
    raise CodeBenchmarkNotReadyError(
        f"{plan.suite.value} is not ready to run; missing: {missing}"
    )


def validate_humaneval_style_tasks(
    tasks: Iterable[HumanEvalStyleTask],
) -> tuple[HumanEvalStyleTask, ...]:
    normalized = tuple(tasks)
    if not normalized:
        raise ValueError("at least one HumanEval-style task is required")
    seen: set[str] = set()
    for task in normalized:
        if not isinstance(task, HumanEvalStyleTask):
            raise ValueError("tasks must contain HumanEvalStyleTask instances")
        if task.task_id in seen:
            raise ValueError(f"duplicate HumanEval-style task_id: {task.task_id}")
        seen.add(task.task_id)
    return normalized


def build_humaneval_style_prompt(task: HumanEvalStyleTask) -> str:
    return task.prompt.rstrip() + "\n"


def humaneval_style_harness_report(
    tasks: Iterable[HumanEvalStyleTask],
    prerequisites: CodeBenchmarkPrerequisites | None = None,
) -> HumanEvalStyleHarnessReport:
    normalized = validate_humaneval_style_tasks(tasks)
    plan = code_benchmark_plan(CodeBenchmarkSuite.HUMANEVAL_STYLE, prerequisites)
    return HumanEvalStyleHarnessReport(
        suite=plan.suite,
        task_count=len(normalized),
        runnable=plan.runnable,
        missing_requirements=plan.missing_requirements,
        notes=plan.notes,
    )
