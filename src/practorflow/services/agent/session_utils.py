"""
Agent session utility functions.

Provides helpers for document context, session persistence,
execution logging, and output extraction.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from practorflow.llm.base.session import Session
from practorflow.llm.base.session_store import SessionStore

from practorflow.services.agent.schemas import (
    Plan,
    ExecutionResult,
    StepResult,
    StepStatus,
    VerificationResult,
)


def get_document_context(
    session: Session,
    document_ids: Optional[Set[str]] = None,
) -> Optional[str]:
    """
    Build document context description for planner.

    Args:
        session: Current session with documents.
        document_ids: Additional document IDs to include.

    Returns:
        Description string or None if no documents.
    """
    all_docs = []

    for doc in session.documents:
        filename = doc.get("filename", "unknown")
        doc_id = doc.get("id", "")
        all_docs.append(f"- {filename} (id: {doc_id})")

    if document_ids:
        for doc_id in document_ids:
            if not any(d.get("id") == doc_id for d in session.documents):
                all_docs.append(f"- Document ID: {doc_id}")

    if not all_docs:
        return None

    return f"The following documents are available for search:\n" + "\n".join(all_docs)


def get_document_scope(
    session: Session,
    document_ids: Optional[Set[str]] = None,
) -> Optional[Set[str]]:
    """
    Get combined document scope from session and provided IDs.

    Args:
        session: Current session.
        document_ids: Additional document IDs.

    Returns:
        Set of document IDs or None if no scope.
    """
    scope = set()

    for doc in session.documents:
        if "id" in doc:
            scope.add(doc["id"])

    if document_ids:
        scope.update(document_ids)

    return scope if scope else None


def persist_to_session(
    session: Session,
    session_store: SessionStore,
    plan: Optional[Plan],
    execution_result: Optional[ExecutionResult],
    verification_result: Optional[VerificationResult],
) -> None:
    """
    Persist agent artifacts to session metadata.

    Args:
        session: Session to update.
        session_store: Store to save session to.
        plan: The execution plan.
        execution_result: Results from execution.
        verification_result: Results from verification.
    """
    session.metadata["type"] = "agent"

    if plan:
        session.metadata["plan"] = plan.model_dump()

    if execution_result:
        session.metadata["execution"] = execution_result.model_dump()

    if verification_result:
        session.metadata["verification"] = verification_result.model_dump()

    session.updated_at = datetime.now()
    session_store.save(session)


def build_execution_log(
    step_results: List[StepResult],
    plan: Plan,
) -> str:
    """
    Build human-readable execution log.

    Args:
        step_results: Results from execution.
        plan: The executed plan.

    Returns:
        Formatted log string.
    """
    lines = [
        f"=== Execution Log ===",
        f"Plan ID: {plan.plan_id}",
        f"Task: {plan.task}",
        f"Timestamp: {datetime.now().isoformat()}",
        f"",
    ]

    for result in step_results:
        step = next((s for s in plan.steps if s.step_id == result.step_id), None)
        step_desc = step.description if step else "Unknown step"

        lines.append(f"--- {result.step_id} ---")
        lines.append(f"Description: {step_desc}")
        lines.append(f"Status: {result.status}")

        if result.output is not None:
            output_str = str(result.output)
            if len(output_str) > 200:
                output_str = output_str[:200] + "..."
            lines.append(f"Output: {output_str}")

        if result.error:
            lines.append(f"Error: {result.error}")

        lines.append("")

    return "\n".join(lines)


def extract_final_output(
    plan: Plan,
    execution_result: ExecutionResult,
) -> str:
    """
    Extract the final output from execution results.

    Args:
        plan: The executed plan.
        execution_result: Results from execution.

    Returns:
        Synthesized output string.
    """
    outputs = []

    for result in execution_result.step_results:
        if result.status == StepStatus.SUCCESS and result.output:
            step = next((s for s in plan.steps if s.step_id == result.step_id), None)
            if step:
                outputs.append(f"{step.description}: {result.output}")

    if outputs:
        return "\n".join(outputs)

    return "Task completed successfully."


def build_failure_message(verification_result: VerificationResult) -> str:
    """
    Build user-facing failure message from verification result.

    Args:
        verification_result: The verification result.

    Returns:
        Formatted failure message string.
    """
    parts = [f"Task verification {verification_result.verification_status}."]

    if verification_result.failed_criteria:
        parts.append("Failed criteria:")
        for criterion in verification_result.failed_criteria:
            parts.append(f"  - {criterion}")

    if verification_result.issues:
        parts.append("Issues found:")
        for issue in verification_result.issues:
            parts.append(f"  - [{issue.issue_type}] {issue.description}")

    return "\n".join(parts)