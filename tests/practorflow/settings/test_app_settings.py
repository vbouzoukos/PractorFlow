import os
from unittest.mock import patch, MagicMock
import pytest

from practorflow.llm.knowledge.chroma_knowledge_config import ChromaKnowledgeStoreConfig


@pytest.fixture(autouse=True)
def reset_app_configuration():
    """Reset appConfiguration global before and after each test"""
    import practorflow.settings.app_settings as app_settings_module

    # Store original value
    original_config = app_settings_module.appConfiguration

    # Reset to None before test
    app_settings_module.appConfiguration = None

    yield

    # Restore original after test
    app_settings_module.appConfiguration = original_config


@pytest.fixture(autouse=True)
def clear_environment():
    """Clear all configuration-related environment variables before each test"""
    env_vars_to_clear = [
        # Logger vars
        "LOG_RUNNER_LEVEL",
        "LOG_DOC_LEVEL",
        "LOG_KNOWLEDGE_LEVEL",
        "LOG_MODEL_POOL_LEVEL",
        "LOG_TOOL_LEVEL",
        "LOG_AGENT_LEVEL",
        # LLM vars
        "LLM_MODEL",
        "LLM_DEVICE",
        "LLM_DTYPE",
        "LLM_MAX_NEW_TOKENS",
        "LLM_TEMPERATURE",
        "LLM_TOP_P",
        "LLM_QUANTIZATION",
        "LLM_MODELS_DIR",
        "LLM_GPU_LAYERS",
        "LLM_N_CTX",
        "LLM_N_BATCH",
        "LLM_BACKEND",
        "LLM_STOP_TOKENS",
        "LLM_MAX_SEARCH_RESULTS",
        "LLM_COMPILE_MODE",
        "LLM_USE_TORCH_COMPILE",
        "LLM_WARMUP_ON_LOAD",
        # Knowledge vars
        "KB_TYPE",
        "KB_CHROMA_PERSIST_DIRECTORY",
        "KB_CHROMA_RETRIEVE_COLLECTION",
        "KB_CHROMA_CONTEXT_COLLECTION",
        "KB_CHROMA_DOCUMENT_COLLECTION",
        "KB_CHROMA_BATCH_SIZE",
        "KB_CHROMA_EMBEDDING_MODEL",
        "KB_CHROMA_EMBEDDING_MODEL_DIR",
        "KB_CHROMA_RETRIEVAL_CHUNK_SIZE",
        "KB_CHROMA_RETRIEVAL_CHUNK_OVERLAP",
        "KB_CHROMA_CONTEXT_CHUNK_SIZE",
        "KB_CHROMA_CONTEXT_CHUNK_OVERLAP",
    ]

    for var in env_vars_to_clear:
        os.environ.pop(var, None)

    yield


def test_load_configuration_with_default_structure():
    """Test that load_configuration creates proper nested structure"""
    import practorflow.settings.app_settings as app_settings

    with patch("practorflow.settings.app_settings.load_dotenv"):
        # Load configuration with mocked dotenv
        app_settings.load_configuration(config_path="/fake/path")

    # Verify configuration object exists
    assert app_settings.appConfiguration is not None

    # Verify nested structure
    assert hasattr(app_settings.appConfiguration, "LoggerConfiguration")
    assert hasattr(app_settings.appConfiguration, "ModelConfiguration")
    assert hasattr(app_settings.appConfiguration, "KnowledgeChromaConfiguration")

    # Verify default values in LoggerConfiguration
    assert app_settings.appConfiguration.LoggerConfiguration.RunnerLevel == "INFO"
    assert app_settings.appConfiguration.LoggerConfiguration.DocumentLevel == "INFO"
    assert app_settings.appConfiguration.LoggerConfiguration.KnowledgeLevel == "INFO"
    assert app_settings.appConfiguration.LoggerConfiguration.ModelPoolLevel == "INFO"
    assert app_settings.appConfiguration.LoggerConfiguration.ToolLevel == "INFO"
    assert app_settings.appConfiguration.LoggerConfiguration.AgentLevel == "INFO"


def test_load_configuration_with_logger_env_vars():
    """Test loading logger configuration from environment variables"""
    # Set environment variables BEFORE importing
    os.environ["LOG_RUNNER_LEVEL"] = "DEBUG"
    os.environ["LOG_DOC_LEVEL"] = "WARNING"
    os.environ["LOG_KNOWLEDGE_LEVEL"] = "ERROR"

    # Need to reload the module to pick up new env vars in dataclass defaults
    import importlib
    import practorflow.settings.app_settings as app_settings

    importlib.reload(app_settings)

    with patch("practorflow.settings.app_settings.load_dotenv"):
        # Load configuration
        app_settings.load_configuration(config_path="/fake/path")

    # Verify logger configuration was loaded from env vars
    assert app_settings.appConfiguration.LoggerConfiguration.RunnerLevel == "DEBUG"
    assert app_settings.appConfiguration.LoggerConfiguration.DocumentLevel == "WARNING"
    assert app_settings.appConfiguration.LoggerConfiguration.KnowledgeLevel == "ERROR"
    # Unset vars should use defaults
    assert app_settings.appConfiguration.LoggerConfiguration.ModelPoolLevel == "INFO"


def test_load_configuration_with_model_env_vars():
    """Test loading model configuration from environment variables"""
    import practorflow.settings.app_settings as app_settings

    # Set environment variables
    os.environ["LLM_MODEL"] = "custom/model"
    os.environ["LLM_DEVICE"] = "cpu"
    os.environ["LLM_BACKEND"] = "transformers"
    os.environ["LLM_MAX_NEW_TOKENS"] = "1024"
    os.environ["LLM_TEMPERATURE"] = "0.5"

    with patch("practorflow.settings.app_settings.load_dotenv"):
        # Load configuration
        app_settings.load_configuration(config_path="/fake/path")

    # Verify model configuration was loaded
    assert app_settings.appConfiguration.ModelConfiguration.model_name == "custom/model"
    assert app_settings.appConfiguration.ModelConfiguration.device == "cpu"
    assert app_settings.appConfiguration.ModelConfiguration.backend == "transformers"
    assert app_settings.appConfiguration.ModelConfiguration.max_new_tokens == 1024
    assert app_settings.appConfiguration.ModelConfiguration.temperature == 0.5


def test_load_configuration_with_knowledge_env_vars():
    """Test loading knowledge configuration from environment variables"""
    import practorflow.settings.app_settings as app_settings

    # Set environment variables
    os.environ["KB_TYPE"] = "chromadb"
    os.environ["KB_CHROMA_PERSIST_DIRECTORY"] = "/custom/path"
    os.environ["KB_CHROMA_BATCH_SIZE"] = "200"
    os.environ["KB_CHROMA_EMBEDDING_MODEL"] = "custom-model"

    with patch("practorflow.settings.app_settings.load_dotenv"):
        # Load configuration
        app_settings.load_configuration(config_path="/fake/path")

    # Verify knowledge configuration was loaded
    assert app_settings.appConfiguration.KnowledgeChromaConfiguration is not None
    assert (
        app_settings.appConfiguration.KnowledgeChromaConfiguration.persist_directory
        == "/custom/path"
    )
    assert app_settings.appConfiguration.KnowledgeChromaConfiguration.batch_size == 200
    assert (
        app_settings.appConfiguration.KnowledgeChromaConfiguration.embedding_model_name
        == "custom-model"
    )


def test_load_configuration_kb_type_not_chromadb():
    """Test that KnowledgeChromaConfiguration is None when KB_TYPE is not chromadb"""
    import practorflow.settings.app_settings as app_settings

    # Set KB_TYPE to something other than chromadb
    os.environ["KB_TYPE"] = "other"

    with patch("practorflow.settings.app_settings.load_dotenv"):
        # Load configuration
        app_settings.load_configuration(config_path="/fake/path")

    # Verify knowledge configuration is None
    assert app_settings.appConfiguration.KnowledgeChromaConfiguration is None


def test_load_configuration_multiple_calls():
    """Test that calling load_configuration multiple times updates the singleton"""
    # Set first value BEFORE importing
    os.environ["LOG_RUNNER_LEVEL"] = "INFO"

    import importlib
    import practorflow.settings.app_settings as app_settings

    importlib.reload(app_settings)

    with patch("practorflow.settings.app_settings.load_dotenv"):
        app_settings.load_configuration(config_path="/fake/path")

    assert app_settings.appConfiguration.LoggerConfiguration.RunnerLevel == "INFO"

    # Change env var and reload module to pick up new value
    os.environ["LOG_RUNNER_LEVEL"] = "DEBUG"
    importlib.reload(app_settings)

    with patch("practorflow.settings.app_settings.load_dotenv"):
        app_settings.load_configuration(config_path="/fake/path")

    assert app_settings.appConfiguration.LoggerConfiguration.RunnerLevel == "DEBUG"


def test_parse_list_string():
    """Test the parse_list_string utility function"""
    from practorflow.settings.app_settings import parse_list_string

    # Test None
    assert parse_list_string(None) is None

    # Test empty string
    assert parse_list_string("") is None

    # Test single item
    assert parse_list_string("item1") == ["item1"]

    # Test multiple items
    assert parse_list_string("item1,item2,item3") == ["item1", "item2", "item3"]

    # Test with spaces
    assert parse_list_string("item1 , item2 , item3") == ["item1", "item2", "item3"]

    # Test with empty items
    assert parse_list_string("item1,,item2") == ["item1", "item2"]


def test_load_configuration_with_list_values():
    """Test loading configuration with comma-separated list values"""
    import practorflow.settings.app_settings as app_settings

    # Set stop tokens as comma-separated list
    os.environ["LLM_STOP_TOKENS"] = "<|endoftext|>,<|im_end|>,</s>"

    with patch("practorflow.settings.app_settings.load_dotenv"):
        # Load configuration
        app_settings.load_configuration(config_path="/fake/path")

    # Verify list was parsed correctly
    assert app_settings.appConfiguration.ModelConfiguration.stop_tokens == [
        "<|endoftext|>",
        "<|im_end|>",
        "</s>",
    ]


def test_load_configuration_with_boolean_values():
    """Test loading configuration with boolean string values"""
    import practorflow.settings.app_settings as app_settings

    # Set boolean values
    os.environ["LLM_USE_TORCH_COMPILE"] = "false"
    os.environ["LLM_WARMUP_ON_LOAD"] = "true"

    with patch("practorflow.settings.app_settings.load_dotenv"):
        # Load configuration
        app_settings.load_configuration(config_path="/fake/path")

    # Verify boolean parsing
    assert app_settings.appConfiguration.ModelConfiguration.use_torch_compile is False
    assert app_settings.appConfiguration.ModelConfiguration.warmup_on_load is True


def test_appconfig_dataclass_structure():
    """Test that AppConfig is properly structured as a dataclass"""
    from practorflow.settings.app_settings import AppConfig, LoggerConfig
    from practorflow.llm.llm_config import LLMConfig

    # Create instance with defaults
    config = AppConfig(
        LoggerConfiguration=LoggerConfig(),
        ModelConfiguration=LLMConfig(),
        KnowledgeChromaConfiguration=ChromaKnowledgeStoreConfig(),
    )

    # Verify it's a dataclass with correct fields
    assert hasattr(config, "LoggerConfiguration")
    assert hasattr(config, "ModelConfiguration")
    assert hasattr(config, "KnowledgeChromaConfiguration")

    # Verify types
    assert isinstance(config.LoggerConfiguration, LoggerConfig)
    assert isinstance(config.ModelConfiguration, LLMConfig)


def test_logger_config_defaults():
    """Test LoggerConfig default values"""
    # Reload module to ensure clean defaults after any previous test modifications
    import importlib
    import practorflow.settings.app_settings as app_settings_module

    importlib.reload(app_settings_module)

    from practorflow.settings.app_settings import LoggerConfig

    # Create with no env vars set
    config = LoggerConfig()

    # All should default to INFO when no env vars are set
    assert config.RunnerLevel == "INFO"
    assert config.DocumentLevel == "INFO"
    assert config.KnowledgeLevel == "INFO"
    assert config.ModelPoolLevel == "INFO"
    assert config.ToolLevel == "INFO"
    assert config.AgentLevel == "INFO"


def test_load_knowledge_chroma_config_function():
    """Test the load_knowledge_chroma_config function directly"""
    from practorflow.settings.app_settings import load_knowledge_chroma_config

    # Test with KB_TYPE=chromadb (default)
    os.environ["KB_TYPE"] = "chromadb"
    os.environ["KB_CHROMA_PERSIST_DIRECTORY"] = "/test/path"

    config = load_knowledge_chroma_config()

    assert config is not None
    assert config.persist_directory == "/test/path"

    # Test with KB_TYPE=other
    os.environ["KB_TYPE"] = "other"
    config = load_knowledge_chroma_config()

    assert config is None


def test_load_dotenv_called_correctly():
    """Test that load_dotenv is called with correct paths"""
    import practorflow.settings.app_settings as app_settings

    with patch("practorflow.settings.app_settings.load_dotenv") as mock_load_dotenv:
        with patch("practorflow.settings.app_settings.os.path.join") as mock_join:
            # Setup path.join to return predictable paths
            mock_join.side_effect = lambda base, file: f"{base}/{file}"

            # Call load_configuration
            app_settings.load_configuration(config_path="/test/config")

            # Verify load_dotenv was called 3 times (logger, model, knowledge)
            assert mock_load_dotenv.call_count == 3

            # Verify it was called with override=True
            for call in mock_load_dotenv.call_args_list:
                assert call[1].get("override") is True


def test_configuration_isolation_between_tests():
    """Test that configuration is properly isolated between tests"""
    # Set value BEFORE importing
    os.environ["LOG_RUNNER_LEVEL"] = "CRITICAL"

    import importlib
    import practorflow.settings.app_settings as app_settings

    importlib.reload(app_settings)

    with patch("practorflow.settings.app_settings.load_dotenv"):
        app_settings.load_configuration(config_path="/fake/path")

    assert app_settings.appConfiguration.LoggerConfiguration.RunnerLevel == "CRITICAL"

    # This configuration will be reset by the fixture for the next test


def test_model_configuration_integer_parsing():
    """Test that integer values are properly parsed from env vars"""
    import practorflow.settings.app_settings as app_settings

    os.environ["LLM_MAX_NEW_TOKENS"] = "4096"
    os.environ["LLM_GPU_LAYERS"] = "32"
    os.environ["LLM_N_CTX"] = "8192"
    os.environ["LLM_N_BATCH"] = "512"

    with patch("practorflow.settings.app_settings.load_dotenv"):
        app_settings.load_configuration(config_path="/fake/path")

    assert app_settings.appConfiguration.ModelConfiguration.max_new_tokens == 4096
    assert app_settings.appConfiguration.ModelConfiguration.n_gpu_layers == 32
    assert app_settings.appConfiguration.ModelConfiguration.n_ctx == 8192
    assert app_settings.appConfiguration.ModelConfiguration.n_batch == 512


def test_model_configuration_float_parsing():
    """Test that float values are properly parsed from env vars"""
    import practorflow.settings.app_settings as app_settings

    os.environ["LLM_TEMPERATURE"] = "0.8"
    os.environ["LLM_TOP_P"] = "0.95"

    with patch("practorflow.settings.app_settings.load_dotenv"):
        app_settings.load_configuration(config_path="/fake/path")

    assert app_settings.appConfiguration.ModelConfiguration.temperature == 0.8
    assert app_settings.appConfiguration.ModelConfiguration.top_p == 0.95


def test_knowledge_configuration_integer_parsing():
    """Test that knowledge config integer values are properly parsed"""
    import practorflow.settings.app_settings as app_settings

    os.environ["KB_TYPE"] = "chromadb"
    os.environ["KB_CHROMA_BATCH_SIZE"] = "250"
    os.environ["KB_CHROMA_RETRIEVAL_CHUNK_SIZE"] = "256"
    os.environ["KB_CHROMA_RETRIEVAL_CHUNK_OVERLAP"] = "50"
    os.environ["KB_CHROMA_CONTEXT_CHUNK_SIZE"] = "2048"
    os.environ["KB_CHROMA_CONTEXT_CHUNK_OVERLAP"] = "200"

    with patch("practorflow.settings.app_settings.load_dotenv"):
        app_settings.load_configuration(config_path="/fake/path")

    config = app_settings.appConfiguration.KnowledgeChromaConfiguration
    assert config.batch_size == 250
    assert config.retrieval_chunk_size == 256
    assert config.retrieval_chunk_overlap == 50
    assert config.context_chunk_size == 2048
    assert config.context_chunk_overlap == 200
