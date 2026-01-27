"""
HTTP request builder for API Tools.

Transforms ApiToolConfig and runtime parameters into HTTP request components.
"""

import base64
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

from practorflow.llm.tools.api.models import (
    ApiToolConfig,
    AuthKeyLocation,
    AuthType,
    BodyContentType,
    ParamLocation,
)


@dataclass
class RequestData:
    """HTTP request components ready for execution."""
    url: str
    method: str
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, Any] = field(default_factory=dict)
    body: Optional[Any] = None
    content_type: Optional[str] = None


class RequestBuilder:
    """
    Builds HTTP request components from tool configuration and runtime parameters.
    
    Handles:
    - URL construction with path parameter substitution
    - Header building including authentication
    - Query parameter assembly
    - Request body formatting (JSON, form, multipart)
    """
    
    def build(
        self,
        config: ApiToolConfig,
        decrypted_secret: Optional[str] = None,
        decrypted_username: Optional[str] = None,
        decrypted_password: Optional[str] = None,
        **kwargs,
    ) -> RequestData:
        """
        Build complete HTTP request data from configuration and parameters.
        
        Args:
            config: Tool configuration.
            decrypted_secret: Decrypted API key or bearer token.
            decrypted_username: Decrypted username for Basic auth.
            decrypted_password: Decrypted password for Basic auth.
            **kwargs: Runtime parameter values from LLM.
        
        Returns:
            RequestData ready for HTTP execution.
        """
        params_by_location = self._separate_params_by_location(config, **kwargs)
        
        path_params = params_by_location.get(ParamLocation.PATH, {})
        query_params = params_by_location.get(ParamLocation.QUERY, {})
        header_params = params_by_location.get(ParamLocation.HEADER, {})
        body_params = params_by_location.get(ParamLocation.BODY, {})
        
        url = self._build_url(config, path_params)
        headers = self._build_headers(header_params)
        body, content_type = self._build_body(config, body_params)
        
        self._inject_auth(
            config=config,
            headers=headers,
            query_params=query_params,
            body=body_params,
            decrypted_secret=decrypted_secret,
            decrypted_username=decrypted_username,
            decrypted_password=decrypted_password,
        )
        
        if content_type:
            headers["Content-Type"] = content_type
        
        return RequestData(
            url=url,
            method=config.method.value,
            headers=headers,
            query_params=query_params,
            body=body,
            content_type=content_type,
        )
    
    def _build_url(self, config: ApiToolConfig, path_params: Dict[str, Any]) -> str:
        """
        Build full URL with path parameters substituted.
        
        Args:
            config: Tool configuration with base_url and path.
            path_params: Path parameter values to substitute.
        
        Returns:
            Complete URL string.
        """
        base_url = config.base_url.rstrip("/")
        path = config.path
        
        for name, value in path_params.items():
            path = path.replace(f"{{{name}}}", str(value))
        
        if not path.startswith("/"):
            path = "/" + path
        
        return base_url + path
    
    def _build_headers(
        self,
        header_params: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Build request headers from header parameters.
        
        Args:
            header_params: Header parameter values.
        
        Returns:
            Headers dictionary.
        """
        headers = {}
        
        for name, value in header_params.items():
            headers[name] = str(value)
        
        return headers
    
    def _build_body(
        self,
        config: ApiToolConfig,
        body_params: Dict[str, Any],
    ) -> Tuple[Optional[Any], Optional[str]]:
        """
        Build request body based on content type.
        
        Args:
            config: Tool configuration with body_content_type.
            body_params: Body parameter values.
        
        Returns:
            Tuple of (body, content_type).
        """
        if config.body_content_type == BodyContentType.NONE:
            return None, None
        
        if not body_params:
            return None, None
        
        if config.body_content_type == BodyContentType.JSON:
            return body_params, "application/json"
        
        if config.body_content_type == BodyContentType.FORM:
            return body_params, "application/x-www-form-urlencoded"
        
        if config.body_content_type == BodyContentType.MULTIPART:
            return body_params, "multipart/form-data"
        
        return None, None
    
    def _inject_auth(
        self,
        config: ApiToolConfig,
        headers: Dict[str, str],
        query_params: Dict[str, Any],
        body: Dict[str, Any],
        decrypted_secret: Optional[str],
        decrypted_username: Optional[str],
        decrypted_password: Optional[str],
    ) -> None:
        """
        Inject authentication into request components.
        
        Modifies headers, query_params, or body in place.
        
        Args:
            config: Tool configuration with auth settings.
            headers: Headers dict to modify.
            query_params: Query params dict to modify.
            body: Body dict to modify.
            decrypted_secret: Decrypted API key or bearer token.
            decrypted_username: Decrypted username for Basic auth.
            decrypted_password: Decrypted password for Basic auth.
        """
        if config.auth_type == AuthType.NONE:
            return
        
        if config.auth_type == AuthType.BEARER:
            if decrypted_secret:
                headers["Authorization"] = f"Bearer {decrypted_secret}"
            return
        
        if config.auth_type == AuthType.BASIC:
            if decrypted_username and decrypted_password:
                credentials = f"{decrypted_username}:{decrypted_password}"
                encoded = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
                headers["Authorization"] = f"Basic {encoded}"
            return
        
        if config.auth_type == AuthType.API_KEY:
            if not decrypted_secret or not config.auth_key_name:
                return
            
            if config.auth_key_location == AuthKeyLocation.HEADER:
                headers[config.auth_key_name] = decrypted_secret
            elif config.auth_key_location == AuthKeyLocation.QUERY:
                query_params[config.auth_key_name] = decrypted_secret
            elif config.auth_key_location == AuthKeyLocation.BODY:
                body[config.auth_key_name] = decrypted_secret
    
    def _separate_params_by_location(
        self,
        config: ApiToolConfig,
        **kwargs,
    ) -> Dict[ParamLocation, Dict[str, Any]]:
        """
        Group provided parameters by their configured location.
        
        Args:
            config: Tool configuration with parameter definitions.
            **kwargs: Runtime parameter values.
        
        Returns:
            Dictionary mapping ParamLocation to parameter dict.
        """
        result: Dict[ParamLocation, Dict[str, Any]] = {
            ParamLocation.PATH: {},
            ParamLocation.QUERY: {},
            ParamLocation.HEADER: {},
            ParamLocation.BODY: {},
        }
        
        for param in config.parameters:
            name = param.name
            location = param.location
            
            if name in kwargs:
                result[location][name] = kwargs[name]
            elif param.default_value is not None:
                result[location][name] = param.default_value
        
        return result