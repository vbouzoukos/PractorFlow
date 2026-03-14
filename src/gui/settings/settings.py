"""
Settings management for PractorFlow GUI.

Provides settings data model and JSON persistence.
"""

import json
import os
from dataclasses import dataclass, asdict

from platformdirs import user_config_dir

from gui.logger import get_logger

logger = get_logger("practorflow-client", level="INFO", log_file="logs/practorflow-client.log")


# App identifiers for platformdirs
APP_NAME = "PractorFlow"
APP_AUTHOR = "PractorFlow"

# Settings directory and file path
SETTINGS_DIR = user_config_dir(APP_NAME, APP_AUTHOR)
DEFAULT_SETTINGS_PATH = os.path.join(SETTINGS_DIR, "settings.json")

# Hardcoded defaults from current implementation
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_USERNAME = "practorFlowClient"


@dataclass
class AppSettings:
    """Application settings data model."""

    api_url: str = DEFAULT_API_URL
    username: str = DEFAULT_USERNAME
    admin_secret: str = ""
    instructions: str = ""
    theme: str = "auto"  # "auto", "dark", "light"
    auto_connect: bool = False

    def to_dict(self) -> dict:
        """Convert settings to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AppSettings":
        """Create settings from dictionary."""
        return cls(
            api_url=data.get("api_url", DEFAULT_API_URL),
            username=data.get("username", DEFAULT_USERNAME),
            admin_secret=data.get("admin_secret", ""),
            instructions=data.get("instructions", ""),
            theme=data.get("theme", "auto"),
            auto_connect=data.get("auto_connect", False),
        )


def settings_exist(path: str = DEFAULT_SETTINGS_PATH) -> bool:
    """Check if settings file exists."""
    return os.path.isfile(path)


def load_settings(path: str = DEFAULT_SETTINGS_PATH) -> AppSettings:
    """
    Load settings from JSON file.
    
    Returns default settings if file does not exist or is invalid.
    """
    if not os.path.isfile(path):
        logger.debug(f"Settings file not found: {path}, using defaults")
        return AppSettings()
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Settings loaded from: {path}")
        return AppSettings.from_dict(data)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse settings JSON: {e}")
        return AppSettings()
    except IOError as e:
        logger.error(f"Failed to read settings file: {e}")
        return AppSettings()
    except Exception as e:
        logger.error(f"Unexpected error loading settings: {e}")
        return AppSettings()


def save_settings(settings: AppSettings, path: str = DEFAULT_SETTINGS_PATH) -> bool:
    """
    Save settings to JSON file.
    
    Creates the settings directory if it does not exist.
    Returns True on success, False on failure.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings.to_dict(), f, indent=2)
        logger.info(f"Settings saved to: {path}")
        return True
    except IOError as e:
        logger.error(f"Failed to write settings file: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving settings: {e}")
        return False