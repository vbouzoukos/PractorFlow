"""
Agent verification utilities.

Provides fallback heuristic verification when LLM verification fails.
Uses lenient verification for conversational tasks - if there's a
synthesized output, the task is considered at least partially successful.
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

    Uses lenient verification for conversational tasks:
    - If there's a synthesized output, returns PARTIAL at minimum
    - Never returns FAILED if we have a usable response for the user

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

    # Determine if we have a usable response
    has_synthesized_output = bool(
        execution_result.synthesized_output 
        and execution_result.synthesized_output.strip()
    )

    # Lenient verification logic:
    # - All steps succeeded with no issues -> PASSED
    # - Has synthesized output OR at least one success -> PARTIAL (never block the user)
    # - No successes AND no synthesized output -> FAILED (but this is rare)
    if not issues and success_count == total_steps:
        status = VerificationStatus.PASSED
    elif has_synthesized_output or success_count > 0:
        # Lenient: if we have ANY usable output, treat as PARTIAL
        status = VerificationStatus.PARTIAL
        if issues:
            failed_criteria = ["Some steps had issues but response was generated"]
    else:
        # Only fail if we truly have nothing to show the user
        status = VerificationStatus.FAILED
        failed_criteria = plan.success_criteria

    # Only recommend retry for tool failures when we have no output
    retry_recommended = (
        not has_synthesized_output
        and any(issue.issue_type == IssueType.TOOL_FAILURE for issue in issues)
    )

    return VerificationResult(
        verification_status=status,
        failed_criteria=failed_criteria,
        issues=issues,
        retry_recommended=retry_recommended,
    )