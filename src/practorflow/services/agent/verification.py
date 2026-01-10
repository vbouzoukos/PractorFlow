"""
Agent verification utilities.

Provides fallback heuristic verification when LLM verification fails.
"""

from practorflow.services.agent.schemas import (
    Plan,
    ExecutionResult,
    StepStatus,
    VerificationResult,
    VerificationStatus,
    VerificationIssue,
    IssueType,
)


def heuristic_verification(
    plan: Plan,
    execution_result: ExecutionResult,
) -> VerificationResult:
    """
    Fallback verification using heuristics when LLM verification fails.

    Args:
        plan: The original plan.
        execution_result: Results from execution.

    Returns:
        VerificationResult based on step outcomes.
    """
    issues = []
    failed_criteria = []

    executed_step_ids = {r.step_id for r in execution_result.step_results}
    for step in plan.steps:
        if step.step_id not in executed_step_ids:
            issues.append(VerificationIssue(
                issue_type=IssueType.INCOMPLETE_EXECUTION,
                description=f"Step {step.step_id} was not executed",
                step_id=step.step_id,
            ))

    failed_steps = [r for r in execution_result.step_results if r.status == StepStatus.FAILURE]
    for result in failed_steps:
        issues.append(VerificationIssue(
            issue_type=IssueType.TOOL_FAILURE,
            description=result.error or "Step failed",
            step_id=result.step_id,
        ))

    success_count = sum(1 for r in execution_result.step_results if r.status == StepStatus.SUCCESS)
    total_steps = len(plan.steps)

    if not issues and success_count == total_steps:
        status = VerificationStatus.PASSED
    elif success_count > 0:
        status = VerificationStatus.PARTIAL
        failed_criteria = ["Some steps failed or were not executed"]
    else:
        status = VerificationStatus.FAILED
        failed_criteria = plan.success_criteria

    retry_recommended = any(
        issue.issue_type == IssueType.TOOL_FAILURE for issue in issues
    )

    return VerificationResult(
        verification_status=status,
        failed_criteria=failed_criteria,
        issues=issues,
        retry_recommended=retry_recommended,
    )