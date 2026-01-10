# tests/practorflow/services/agent/test_session_utils.py
from unittest.mock import MagicMock

from practorflow.services.agent.session_utils import (
    get_document_context,
    get_document_scope,
    persist_to_session,
    build_execution_log,
    extract_final_output,
    build_failure_message,
)
from practorflow.services.agent.schemas import (
    StepStatus,
    VerificationStatus,
    VerificationIssue,
    IssueType,
)

from tests.practorflow.services.agent.common_agent_deps import (
    make_plan,
    make_execution_result,
    make_step_result,
    make_session,
)


def test_get_document_context_none_when_no_docs():
    session = make_session()
    session.documents = []

    assert get_document_context(session) is None


def test_get_document_context_with_session_docs_and_extra_ids():
    session = make_session()
    session.documents = [{"id": "doc1", "filename": "a.txt"}]

    context = get_document_context(session, document_ids={"doc2"})

    assert "a.txt" in context
    assert "doc1" in context
    assert "doc2" in context


def test_get_document_scope_none_when_empty():
    session = make_session()
    session.documents = []

    assert get_document_scope(session) is None


def test_get_document_scope_combines_ids():
    session = make_session()
    session.documents = [{"id": "doc1"}]

    scope = get_document_scope(session, {"doc2"})

    assert scope == {"doc1", "doc2"}


def test_persist_to_session_updates_metadata_and_saves():
    session = make_session()
    store = MagicMock()

    plan = make_plan()
    execution = make_execution_result(plan)

    persist_to_session(session, store, plan, execution, None)

    assert session.metadata["plan"]["plan_id"] == plan.plan_id
    assert session.metadata["execution"]["plan_id"] == plan.plan_id
    store.save.assert_called_once_with(session)


def test_build_execution_log_includes_output_and_error():
    plan = make_plan()
    execution = make_execution_result(
        plan,
        step_results=[
            make_step_result(step_id=plan.steps[0].step_id, output="x" * 300),
            make_step_result(
                step_id=plan.steps[0].step_id,
                status=StepStatus.FAILURE,
                error="boom",
            ),
        ],
    )

    log = build_execution_log(execution.step_results, plan)

    assert "Execution Log" in log
    assert "..." in log
    assert "boom" in log


def test_extract_final_output_with_outputs():
    plan = make_plan()
    execution = make_execution_result(plan)

    output = extract_final_output(plan, execution)

    assert plan.steps[0].description in output


def test_extract_final_output_fallback_message():
    plan = make_plan()
    execution = make_execution_result(
        plan,
        [make_step_result(step_id=plan.steps[0].step_id, output=None)],
    )

    output = extract_final_output(plan, execution)

    assert output == "Task completed successfully."


def test_build_failure_message_full():
    result = MagicMock()
    result.verification_status = VerificationStatus.FAILED
    result.failed_criteria = ["c1"]
    result.issues = [
        VerificationIssue(
            issue_type=IssueType.TOOL_FAILURE,
            description="bad",
            step_id="s1",
        )
    ]

    msg = build_failure_message(result)

    assert "FAILED" in msg.upper()
    assert "c1" in msg
    assert "bad" in msg

def test_persist_to_session_with_verification():
    session = make_session()
    store = MagicMock()

    plan = make_plan()
    execution = make_execution_result(plan)

    verification = MagicMock()
    verification.model_dump.return_value = {"verification_status": "passed"}

    persist_to_session(session, store, plan, execution, verification)

    assert session.metadata["verification"] == {"verification_status": "passed"}
    store.save.assert_called_once_with(session)
