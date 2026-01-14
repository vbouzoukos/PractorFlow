from practorflow.services.agent.verification import heuristic_verification
from practorflow.services.agent.schemas import PlanStep, StepStatus, VerificationStatus, IssueType

from tests.practorflow.services.agent.common_agent_deps import (
    make_plan,
    make_execution_result,
    make_step_result,
)


def test_verification_all_steps_success_passed():
    plan = make_plan()
    execution = make_execution_result(plan)

    result = heuristic_verification(plan, execution)

    assert result.verification_status == VerificationStatus.PASSED
    assert result.issues == []
    assert result.failed_criteria == []
    assert result.retry_recommended is False


def test_verification_partial_when_some_steps_fail():
    plan = make_plan()
    step_result = make_step_result(
        step_id=plan.steps[0].step_id,
        status=StepStatus.FAILURE,
        error="tool error",
    )
    execution = make_execution_result(plan, [step_result])

    result = heuristic_verification(plan, execution)

    assert result.verification_status == VerificationStatus.FAILED
    assert result.failed_criteria == plan.success_criteria
    assert result.retry_recommended is True
    assert any(i.issue_type == IssueType.TOOL_FAILURE for i in result.issues)



def test_verification_failed_when_all_steps_fail():
    plan = make_plan()
    step_result = make_step_result(
        step_id=plan.steps[0].step_id,
        status=StepStatus.FAILURE,
        error="fatal",
    )
    execution = make_execution_result(plan, [step_result])

    result = heuristic_verification(plan, execution)

    assert result.verification_status == VerificationStatus.FAILED
    assert result.failed_criteria == plan.success_criteria
    assert result.retry_recommended is True


def test_verification_missing_step_detected():
    plan = make_plan()
    execution = make_execution_result(plan, step_results=[])

    result = heuristic_verification(plan, execution)

    assert any(i.issue_type == IssueType.INCOMPLETE_EXECUTION for i in result.issues)
    assert result.verification_status in {
        VerificationStatus.PARTIAL,
        VerificationStatus.FAILED,
    }

def test_verification_partial_when_some_steps_succeed_and_some_fail():
    plan = make_plan(
        steps=[
            PlanStep(
                step_id="step_1",
                description="s1",
                tool=None,
                tool_args=None,
                expected_output="ok",
            ),
            PlanStep(
                step_id="step_2",
                description="s2",
                tool=None,
                tool_args=None,
                expected_output="ok",
            ),
        ]
    )

    execution = make_execution_result(
        plan,
        step_results=[
            make_step_result(step_id="step_1", status=StepStatus.SUCCESS, output="ok"),
            make_step_result(step_id="step_2", status=StepStatus.FAILURE, error="boom"),
        ],
    )

    result = heuristic_verification(plan, execution)

    assert result.verification_status == VerificationStatus.PARTIAL
    assert result.failed_criteria == ["Some steps had issues but response was generated"]
    assert result.retry_recommended is True
    assert any(i.issue_type == IssueType.TOOL_FAILURE for i in result.issues)
