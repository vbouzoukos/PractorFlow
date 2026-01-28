"""
API Tool implementation.

Executable tool that extends AsyncBaseTool for HTTP API calls.
"""

from datetime import datetime
from typing import List, Optional

import httpx

from practorflow.llm.tools import AsyncBaseTool, ToolParameter, ToolResult
from practorflow.llm.tools.api.encryption import get_encryption_service
from practorflow.llm.tools.api.models import ApiToolConfig, AuthType, ParamType
from practorflow.llm.tools.api.request import RequestBuilder, ResponseParser
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger("tool", level=appConfiguration.LoggerConfiguration.ToolLevel)


class ApiTool(AsyncBaseTool):
    """
    API Tool that executes HTTP requests based on configuration.
    
    Extends AsyncBaseTool to integrate with the existing tool registry.
    Decrypts secrets once at load time, not on every execution.
    """
    
    def __init__(self, config: ApiToolConfig):
        """
        Initialize API tool from configuration.
        
        Decrypts secrets once at construction time.
        Sets expired flag if secret has expired.
        
        Args:
            config: Tool configuration.
        """
        self._config = config
        self.user_id = config.user_id
        self.system = config.system
        self.expired = False
        
        self._decrypted_secret: Optional[str] = None
        self._decrypted_username: Optional[str] = None
        self._decrypted_password: Optional[str] = None
        
        self._builder = RequestBuilder()
        self._parser = ResponseParser()
        
        if config.auth_secret_expires_at and config.auth_secret_expires_at < datetime.now():
            self.expired = True
            logger.warning(f"[ApiTool] Tool '{config.name}' has expired secret")
            return
        
        if config.auth_type != AuthType.NONE:
            try:
                encryption = get_encryption_service()
                
                if config.auth_secret:
                    self._decrypted_secret = encryption.decrypt(config.auth_secret)
                if config.auth_username:
                    self._decrypted_username = encryption.decrypt(config.auth_username)
                if config.auth_password:
                    self._decrypted_password = encryption.decrypt(config.auth_password)
            except Exception as e:
                logger.error(f"[ApiTool] Failed to decrypt secrets for '{config.name}': {e}")
                self.expired = True
    
    @property
    def name(self) -> str:
        """Unique tool name identifier."""
        return self._config.name
    
    @property
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        return self._config.description
    
    @property
    def parameters(self) -> List[ToolParameter]:
        """List of parameters the tool accepts."""
        return self._convert_parameters()
    
    def _convert_parameters(self) -> List[ToolParameter]:
        """
        Convert config parameters to base ToolParameter format.
        
        Returns:
            List of ToolParameter for AsyncBaseTool interface.
        """
        result = []
        
        type_mapping = {
            ParamType.STRING: "string",
            ParamType.INTEGER: "integer",
            ParamType.NUMBER: "number",
            ParamType.BOOLEAN: "boolean",
            ParamType.ARRAY: "array",
        }
        
        for param in self._config.parameters:
            result.append(
                ToolParameter(
                    name=param.name,
                    type=type_mapping.get(param.type, "string"),
                    description=param.description,
                    required=param.required,
                    default=param.default_value,
                    enum=param.enum_values,
                )
            )
        
        return result
    
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool asynchronously.
        
        Args:
            **kwargs: Tool parameters from LLM.
        
        Returns:
            ToolResult with success status and data or error.
        """
        logger.debug(f"[ApiTool] Executing '{self.name}' with params: {list(kwargs.keys())}")
        
        try:
            request_data = self._builder.build(
                config=self._config,
                decrypted_secret=self._decrypted_secret,
                decrypted_username=self._decrypted_username,
                decrypted_password=self._decrypted_password,
                **kwargs,
            )
            
            async with httpx.AsyncClient(timeout=self._config.timeout_seconds) as client:
                response = await client.request(
                    method=request_data.method,
                    url=request_data.url,
                    headers=request_data.headers,
                    params=request_data.query_params,
                    json=request_data.body if request_data.content_type == "application/json" else None,
                    data=request_data.body if request_data.content_type in ("application/x-www-form-urlencoded", "multipart/form-data") else None,
                )
            
            parsed = self._parser.parse(response, self._config)
            
            if parsed.success:
                logger.debug(f"[ApiTool] '{self.name}' succeeded with status {parsed.status_code}")
                return ToolResult(
                    success=True,
                    data=parsed.data,
                    metadata={
                        "status_code": parsed.status_code,
                        "tool_id": self._config.tool_id,
                    },
                )
            else:
                logger.warning(f"[ApiTool] '{self.name}' failed: {parsed.error}")
                return ToolResult(
                    success=False,
                    error=parsed.error,
                    metadata={
                        "status_code": parsed.status_code,
                        "category": "http_error",
                        "tool_id": self._config.tool_id,
                    },
                )
        
        except httpx.TimeoutException as e:
            logger.error(f"[ApiTool] '{self.name}' timeout: {e}")
            return ToolResult(
                success=False,
                error=f"Request timeout after {self._config.timeout_seconds}s",
                metadata={"category": "timeout_error", "tool_id": self._config.tool_id},
            )
        
        except httpx.ConnectError as e:
            logger.error(f"[ApiTool] '{self.name}' connection error: {e}")
            return ToolResult(
                success=False,
                error=f"Connection failed: {e}",
                metadata={"category": "connection_error", "tool_id": self._config.tool_id},
            )
        
        except Exception as e:
            logger.error(f"[ApiTool] '{self.name}' unexpected error: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                metadata={"category": "unexpected_error", "tool_id": self._config.tool_id},
            )