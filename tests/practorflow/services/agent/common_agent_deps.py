from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
from datetime import datetime

from practorflow.services.agent.schemas import (
    Plan,
    PlanStep,
    RetryPolicy,
    ExecutionResult,
    StepResult,
    StepStatus,
    VerificationResult,
    VerificationStatus,
    VerificationIssue,
    IssueType,
)
from practorflow.services.agent.deps import AgentDeps
from practorflow.llm.base.session import Session


DEFAULT_PLAN_ID = "plan-123"
DEFAULT_TASK = "test task"
DEFAULT_STEP_ID = "step-1"
DEFAULT_USER = "test-user"


def make_plan(
    plan_id: str = DEFAULT_PLAN_ID,
    task: str = DEFAULT_TASK,
    steps: Optional[List[PlanStep]] = None,
    success_criteria: Optional[List[str]] = None,
    max_retries: int = 1,
) -> Plan:
    if steps is None:
        steps = [
            PlanStep(
                step_id=DEFAULT_STEP_ID,
                description="do something",
                tool=None,
                tool_args=None,
                expected_output="something done",
            )
        ]
    if success_criteria is None:
        success_criteria = ["something done"]
    return Plan(
        plan_id=plan_id,
        task=task,
        steps=steps,
        success_criteria=success_criteria,
        retry_policy=RetryPolicy(max_retries=max_retries),
    )


def make_step_result(
    step_id: str = DEFAULT_STEP_ID,
    status: StepStatus = StepStatus.SUCCESS,
    output: Any = "ok",
    error: Optional[str] = None,
    evidence: Optional[List[str]] = None,
) -> StepResult:
    return StepResult(
        step_id=step_id,
        status=status,
        output=output if status == StepStatus.SUCCESS else None,
        error=error,
        evidence=evidence or [],
    )


def make_execution_result(
    plan: Plan,
    step_results: Optional[List[StepResult]] = None,
    execution_log: str = "log",
) -> ExecutionResult:
    if step_results is None:
        step_results = [make_step_result(step_id=s.step_id) for s in plan.steps]
    return ExecutionResult(
        plan_id=plan.plan_id,
        step_results=step_results,
        execution_log=execution_log,
    )


def make_verification_result(
    status: VerificationStatus = VerificationStatus.PASSED,
    failed_criteria: Optional[List[str]] = None,
    issues: Optional[List[VerificationIssue]] = None,
    retry_recommended: bool = False,
) -> VerificationResult:
    return VerificationResult(
        verification_status=status,
        failed_criteria=failed_criteria or [],
        issues=issues or [],
        retry_recommended=retry_recommended,
    )


def make_tool_failure_issue(
    step_id: str = DEFAULT_STEP_ID,
    description: str = "tool failed",
) -> VerificationIssue:
    return VerificationIssue(
        issue_type=IssueType.TOOL_FAILURE,
        description=description,
        step_id=step_id,
    )


def make_session(
    session_id: str = "session-123",
    user: str = DEFAULT_USER,
) -> Session:
    session = Session(
        session_id=session_id,
        instructions="test",
        user=user,
        metadata={"type": "agent"},
    )
    session.created_at = datetime.now()
    session.updated_at = session.created_at
    return session


def make_agent_deps(
    knowledge_store: Any,
    tool_registry: Any,
    document_scope: Optional[set] = None,
) -> AgentDeps:
    return AgentDeps(
        knowledge_store=knowledge_store,
        tool_registry=tool_registry,
        document_scope=document_scope,
    )


def make_mock_tool_result(
    success: bool = True,
    data: Any = "result",
    error: Optional[str] = None,
) -> MagicMock:
    result = MagicMock()
    result.success = success
    result.data = data
    result.error = error
    return result
