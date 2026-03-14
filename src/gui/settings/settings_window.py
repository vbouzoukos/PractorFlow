"""
Settings window for PractorFlow GUI.

Provides UI for editing application settings.
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLineEdit,
    QTextEdit,
    QComboBox,
    QCheckBox,
    QPushButton,
    QLabel,
    QMessageBox,
    QGroupBox,
)
from PySide6.QtCore import Signal

from gui.logger import get_logger

logger = get_logger("practorflow-client", level="INFO", log_file="logs/practorflow-client.log")

from gui.settings.settings import (
    AppSettings,
    load_settings,
    save_settings,
    DEFAULT_API_URL,
    DEFAULT_USERNAME,
)


class SettingsWindow(QWidget):
    """
    Settings window widget.
    
    Allows user to edit and save application settings.
    Emits settings_saved signal when settings are successfully saved.
    """
    
    settings_saved = Signal(AppSettings)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        try:
            self._settings = load_settings()
            self._setup_ui()
            self._load_values()
            self._connect_signals()
            logger.info("SettingsWindow initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SettingsWindow: {e}")
            raise
    
    def _setup_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        
        # Title
        title_label = QLabel("Settings")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title_label)
        
        # Connection settings group
        connection_group = QGroupBox("Connection")
        connection_layout = QFormLayout(connection_group)
        
        self._api_url_edit = QLineEdit()
        self._api_url_edit.setPlaceholderText(DEFAULT_API_URL)
        connection_layout.addRow("API URL:", self._api_url_edit)
        
        self._username_edit = QLineEdit()
        self._username_edit.setPlaceholderText(DEFAULT_USERNAME)
        connection_layout.addRow("Username:", self._username_edit)

        self._admin_secret_edit = QLineEdit()
        self._admin_secret_edit.setEchoMode(QLineEdit.Password)
        self._admin_secret_edit.setPlaceholderText("Required for MCP Settings (llm_admin)")
        connection_layout.addRow("Admin Secret:", self._admin_secret_edit)

        self._auto_connect_checkbox = QCheckBox("Connect automatically on startup")
        connection_layout.addRow("", self._auto_connect_checkbox)
        
        layout.addWidget(connection_group)
        
        # Instructions group
        instructions_group = QGroupBox("Instructions (Optional)")
        instructions_layout = QVBoxLayout(instructions_group)
        
        self._instructions_edit = QTextEdit()
        self._instructions_edit.setPlaceholderText(
            "Custom instructions to include with each message..."
        )
        self._instructions_edit.setMaximumHeight(100)
        instructions_layout.addWidget(self._instructions_edit)
        
        layout.addWidget(instructions_group)
        
        # Appearance group
        appearance_group = QGroupBox("Appearance")
        appearance_layout = QFormLayout(appearance_group)
        
        self._theme_combo = QComboBox()
        self._theme_combo.addItems(["auto", "dark", "light"])
        appearance_layout.addRow("Theme:", self._theme_combo)
        
        layout.addWidget(appearance_group)
        
        # Spacer
        layout.addStretch()
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self._reset_btn = QPushButton("Reset to Defaults")
        button_layout.addWidget(self._reset_btn)
        
        self._save_btn = QPushButton("Save")
        self._save_btn.setDefault(True)
        button_layout.addWidget(self._save_btn)
        
        layout.addLayout(button_layout)
    
    def _connect_signals(self):
        """Connect widget signals to slots."""
        self._save_btn.clicked.connect(self._on_save_clicked)
        self._reset_btn.clicked.connect(self._on_reset_clicked)
    
    def _load_values(self):
        """Load current settings into UI fields."""
        self._api_url_edit.setText(self._settings.api_url)
        self._username_edit.setText(self._settings.username)
        self._admin_secret_edit.setText(self._settings.admin_secret)
        self._instructions_edit.setPlainText(self._settings.instructions)
        self._auto_connect_checkbox.setChecked(self._settings.auto_connect)
        
        theme_index = self._theme_combo.findText(self._settings.theme)
        if theme_index >= 0:
            self._theme_combo.setCurrentIndex(theme_index)
    
    def _collect_values(self) -> AppSettings:
        """Collect values from UI fields into AppSettings."""
        return AppSettings(
            api_url=self._api_url_edit.text().strip() or DEFAULT_API_URL,
            username=self._username_edit.text().strip() or DEFAULT_USERNAME,
            admin_secret=self._admin_secret_edit.text().strip(),
            instructions=self._instructions_edit.toPlainText().strip(),
            theme=self._theme_combo.currentText(),
            auto_connect=self._auto_connect_checkbox.isChecked(),
        )
    
    def _on_save_clicked(self):
        """Handle save button click."""
        try:
            settings = self._collect_values()
            
            if save_settings(settings):
                self._settings = settings
                self.settings_saved.emit(settings)
                logger.info("Settings saved successfully")
                QMessageBox.information(self, "Settings", "Settings saved successfully.")
            else:
                logger.warning("Failed to save settings")
                QMessageBox.warning(self, "Settings", "Failed to save settings.")
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save settings:\n{e}")
    
    def _on_reset_clicked(self):
        """Handle reset button click."""
        try:
            reply = QMessageBox.question(
                self,
                "Reset Settings",
                "Reset all settings to default values?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            
            if reply == QMessageBox.Yes:
                self._settings = AppSettings()
                self._load_values()
                logger.info("Settings reset to defaults")
        except Exception as e:
            logger.error(f"Error resetting settings: {e}")
            QMessageBox.critical(self, "Error", f"Failed to reset settings:\n{e}")
    
    def get_settings(self) -> AppSettings:
        """Return current settings."""
        return self._settings