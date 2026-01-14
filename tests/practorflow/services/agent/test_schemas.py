import pytest
from pydantic import ValidationError

from practorflow.services.agent.schemas import (
    StepStatus,
    VerificationStatus,
    IssueType,
    Plan,
    PlanStep,
    RetryPolicy,
    StepResult,
    ExecutionResult,
    VerificationIssue,
    VerificationResult,
    AgentTaskResult,
)


def test_enum_values():
    assert StepStatus.SUCCESS == "success"
    assert VerificationStatus.PASSED == "passed"
    assert IssueType.TOOL_FAILURE == "tool_failure"


def test_plan_step_validation_success():
    step = PlanStep(
        step_id="s1",
        description="desc",
        tool=None,
        tool_args=None,
        expected_output="out",
    )
    assert step.step_id == "s1"


def test_plan_requires_steps():
    with pytest.raises(ValidationError):
        Plan(
            plan_id="p1",
            task="t",
            steps=[],
            success_criteria=["c"],
            retry_policy=RetryPolicy(max_retries=1),
        )


def test_retry_policy_bounds():
    with pytest.raises(ValidationError):
        RetryPolicy(max_retries=10)

    RetryPolicy(max_retries=0)
    RetryPolicy(max_retries=5)


def test_step_result_failure_without_output():
    result = StepResult(
        step_id="s1",
        status=StepStatus.FAILURE,
        error="err",
    )

    assert result.output is None
    assert result.error == "err"


def test_execution_result_required_fields():
    with pytest.raises(ValidationError):
        ExecutionResult(plan_id="p", step_results=[])


def test_verification_issue_fields():
    issue = VerificationIssue(
        issue_type=IssueType.INCONSISTENCY,
        description="bad",
        step_id="s1",
    )
    assert issue.issue_type == IssueType.INCONSISTENCY


def test_verification_result_defaults():
    result = VerificationResult(
        verification_status=VerificationStatus.PASSED,
    )

    assert result.failed_criteria == []
    assert result.issues == []
    assert result.retry_recommended is False


def test_agent_task_result_success_and_failure():
    success = AgentTaskResult(success=True, output="ok")
    assert success.output == "ok"
    assert success.error is None

    failure = AgentTaskResult(success=False, error="bad")
    assert failure.success is False
    assert failure.error == "bad"
