"""
Pydantic schemas for multi-agent task execution system.

Defines data contracts for:
- Plan: Structured task decomposition from Planner agent
- ExecutionResult: Step-by-step execution output from Executor agent
- VerificationResult: Validation outcome from Verifier agent
- AgentTaskResult: Final service response
"""

from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class StepStatus(StrEnum):
    """Status of an executed step."""

    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


class VerificationStatus(StrEnum):
    """Overall verification outcome."""

    PASSED = "passed"
    FAILED = "failed"
    PARTIAL = "partial"


class IssueType(StrEnum):
    """Category of verification issue."""

    MISSING_EVIDENCE = "missing_evidence"
    INCONSISTENCY = "inconsistency"
    INCOMPLETE_EXECUTION = "incomplete_execution"
    TOOL_FAILURE = "tool_failure"


class PlanStep(BaseModel):
    """Single step in an execution plan."""

    step_id: str = Field(
        ...,
        description="Unique identifier for this step",
    )
    description: str = Field(
        ...,
        description="Human-readable description of what this step does",
    )
    tool: Optional[str] = Field(
        default=None,
        description="Tool name to invoke, or null for reasoning steps",
    )
    tool_args: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arguments to pass to the tool",
    )
    expected_output: str = Field(
        ...,
        description="Description of what successful output looks like",
    )


class RetryPolicy(BaseModel):
    """Configuration for retry behavior."""

    max_retries: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Maximum number of retry attempts",
    )


class Plan(BaseModel):
    """
    Structured task decomposition from Planner agent.

    Contains ordered steps, success criteria, and retry configuration.
    This is the contract between Planner and Executor.
    """

    plan_id: str = Field(
        ...,
        description="Unique identifier for this plan",
    )
    task: str = Field(
        ...,
        description="Original user task that this plan addresses",
    )
    steps: List[PlanStep] = Field(
        ...,
        min_length=1,
        description="Ordered list of steps to execute",
    )
    success_criteria: List[str] = Field(
        ...,
        min_length=1,
        description="Machine-checkable criteria that define task completion",
    )
    retry_policy: RetryPolicy = Field(
        default_factory=RetryPolicy,
        description="Configuration for retry behavior on failure",
    )


class StepResult(BaseModel):
    """Result of executing a single plan step."""

    step_id: str = Field(
        ...,
        description="Identifier matching the plan step",
    )
    status: StepStatus = Field(
        ...,
        description="Execution outcome for this step",
    )
    output: Optional[Any] = Field(
        default=None,
        description="Output produced by this step",
    )
    evidence: List[str] = Field(
        default_factory=list,
        description="Reference IDs supporting the output",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if step failed",
    )


class ExecutionResult(BaseModel):
    """
    Complete execution output from Executor agent.

    Contains results for all plan steps and an audit log.
    This is the contract between Executor and Verifier.
    """

    plan_id: str = Field(
        ...,
        description="Reference to the executed plan",
    )
    step_results: List[StepResult] = Field(
        ...,
        description="Results for each step in execution order",
    )
    execution_log: str = Field(
        ...,
        description="Human-readable, immutable audit log",
    )


class VerificationIssue(BaseModel):
    """Specific issue found during verification."""

    issue_type: IssueType = Field(
        ...,
        description="Category of the issue",
    )
    description: str = Field(
        ...,
        description="Human-readable explanation of the issue",
    )
    step_id: Optional[str] = Field(
        default=None,
        description="Related step identifier if applicable",
    )


class VerificationResult(BaseModel):
    """
    Validation outcome from Verifier agent.

    Contains overall status, failed criteria, and detailed issues.
    """

    verification_status: VerificationStatus = Field(
        ...,
        description="Overall verification outcome",
    )
    failed_criteria: List[str] = Field(
        default_factory=list,
        description="Success criteria that were not satisfied",
    )
    issues: List[VerificationIssue] = Field(
        default_factory=list,
        description="Detailed issues found during verification",
    )
    retry_recommended: bool = Field(
        default=False,
        description="Whether retrying execution might resolve issues",
    )


class AgentTaskResult(BaseModel):
    """
    Final response from AgentService.

    Contains the verified output if successful, or error details if failed.
    Internal artifacts (plan, execution, verification) are included for
    transparency and debugging.
    """

    success: bool = Field(
        ...,
        description="Whether the task completed successfully with verified output",
    )
    output: Optional[str] = Field(
        default=None,
        description="Final verified output, only present if success is True",
    )
    plan: Optional[Plan] = Field(
        default=None,
        description="The execution plan that was created",
    )
    execution_result: Optional[ExecutionResult] = Field(
        default=None,
        description="Results from executing the plan",
    )
    verification_result: Optional[VerificationResult] = Field(
        default=None,
        description="Results from verifying the execution",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if task failed",
    )
    usage: Optional[Dict[str, int]] = Field(
        default=None,
        description="Token usage statistics",
    )