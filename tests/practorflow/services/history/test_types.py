from practorflow.services.history.types import (
    TokenEstimate,
    HistoryConfig,
    PreparedHistory,
)


def test_token_estimate_non_history_tokens():
    estimate = TokenEstimate(
        system_tokens=100,
        prompt_tokens=200,
        history_tokens=300,
        total_tokens=600,
    )

    assert estimate.non_history_tokens == 300


def test_history_config_available_context():
    config = HistoryConfig(
        n_ctx=4096,
        reserve_for_response=512,
    )

    assert config.available_context == 3584


def test_prepared_history_has_memory_true():
    history = PreparedHistory(
        extracted_memory="Important context",
    )

    assert history.has_memory is True


def test_prepared_history_has_memory_false_when_none():
    history = PreparedHistory(
        extracted_memory=None,
    )

    assert history.has_memory is False


def test_prepared_history_has_memory_false_when_empty_string():
    history = PreparedHistory(
        extracted_memory="",
    )

    assert history.has_memory is False


def test_prepared_history_reduction_ratio_no_original_messages():
    history = PreparedHistory(
        original_count=0,
        included_count=0,
    )

    assert history.reduction_ratio == 1.0


def test_prepared_history_reduction_ratio_with_reduction():
    history = PreparedHistory(
        original_count=10,
        included_count=4,
    )

    assert history.reduction_ratio == 0.4
