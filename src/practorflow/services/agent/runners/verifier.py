"""
Verifier agent runner.

- run_verifier: Validates execution results against success criteria
"""

from pydantic import ValidationError
from pydantic_ai import Agent

from practorflow.llm.pool.model_pool import ModelPool
from practorflow.llm.factory import create_runner
from practorflow.llm.llm_config import LLMConfig
from practorflow.llm.knowledge.knowledge_store import KnowledgeStore
from practorflow.llm.pyai.model import LocalLLMModel
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.agent.schemas import (
    Plan,
    ExecutionResult,
    VerificationResult,
)
from practorflow.services.agent.prompts import (
    VERIFIER_SYSTEM_PROMPT,
    build_verifier_prompt,
)
from practorflow.services.agent.deps import AgentDeps
from practorflow.services.agent.parser import parse_json_from_response
from practorflow.services.agent.verification import heuristic_verification

logger = get_logger(
    "agent_verifier", level=appConfiguration.LoggerConfiguration.AgentLevel
)


async def run_verifier(
    plan: Plan,
    execution_result: ExecutionResult,
    model_pool: ModelPool,
    model_config: LLMConfig,
    knowledge_store: KnowledgeStore,
) -> VerificationResult:
    """
    Run the Verifier agent to validate execution results.

    Args:
        plan: The original plan.
        execution_result: Results from execution.
        model_pool: Pool for acquiring LLM handles.
        model_config: Configuration for the LLM model.
        knowledge_store: Store for document search.

    Returns:
        VerificationResult with validation outcome.
    """
    prompt = build_verifier_prompt(plan, execution_result)

    logger.debug(f"[Verifier] Running verifier for plan: {plan.plan_id}")

    async with model_pool.acquire_context(model_config) as handle:
        runner = create_runner(handle, knowledge_store=knowledge_store)
        model = LocalLLMModel(runner)

        agent = Agent(
            model=model,
            deps_type=AgentDeps,
            system_prompt=VERIFIER_SYSTEM_PROMPT,
        )

        result = await agent.run(prompt)
        response_text = (
            result.output if isinstance(result.output, str) else str(result.output)
        )

    parsed = parse_json_from_response(response_text)
    if not parsed:
        logger.warning("[Verifier] Verifier did not return valid JSON, using heuristic")
        return heuristic_verification(plan, execution_result)

    try:
        verification = VerificationResult.model_validate(parsed)
    except ValidationError as e:
        logger.warning(
            f"[Verifier] Invalid verification structure: {e}, using heuristic"
        )
        return heuristic_verification(plan, execution_result)

    logger.info(f"[Verifier] Verification complete: {verification.verification_status}")
    return verification