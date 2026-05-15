# -*- coding: utf-8 -*-
"""
This file provides litellm model wrapper
"""
import asyncio
import logging
from typing import AsyncGenerator, Optional

import litellm

from loongflow.agentsdk.logger import get_logger
from loongflow.agentsdk.models.base_llm_model import BaseLLMModel
from loongflow.agentsdk.models.formatter.litellm_formatter import LiteLLMFormatter
from loongflow.agentsdk.models.llm_request import CompletionRequest
from loongflow.agentsdk.models.llm_response import CompletionResponse

logger = get_logger(__name__)

# Transient error types that are safe to retry
_RETRYABLE_EXCEPTIONS = (
    litellm.RateLimitError,
    litellm.Timeout,
    litellm.ServiceUnavailableError,
    litellm.APIConnectionError,
    litellm.InternalServerError,
)

# Retry configuration defaults
_DEFAULT_MAX_RETRIES = 5
_DEFAULT_BASE_DELAY = 10  # seconds


class LiteLLMModel(BaseLLMModel):
    """
    LoongFlow model backend implementation based on LiteLLM.

    Each model instance holds static configuration (model name, base_url, api_key),
    while each `generate` call handles dynamic per-request parameters
    (messages, tools, temperature, etc.).
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = 600,
        model_provider: Optional[str] = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        retry_base_delay: float = _DEFAULT_BASE_DELAY,
        **kwargs,
    ):
        """
        Initialize the LiteLLM-based model wrapper.

        Args:
            model_name: Model name or deployment ID (e.g. "gpt-4o").
            base_url: Base URL of the model provider (e.g. OpenAI, Azure, Baidu).
            api_key: API key for authentication.
            max_retries: Maximum number of retries for transient API errors (default: 5).
            retry_base_delay: Base delay in seconds for exponential backoff (default: 10).
        """
        # Disable litellm internal debug logging
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)

        self.model_name = model_name
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.formatter = LiteLLMFormatter()
        self.model_provider = model_provider
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.generation_params = kwargs

    @classmethod
    def from_config(cls, config: dict) -> "LiteLLMModel":
        """
        Create a model instance from configuration dictionary.

        Args:
            config: Configuration dictionary containing model settings.
                    Must include 'model'. 'url' and 'api_key' are optional (can read from env).

        Returns:
            LiteLLMModel: Initialized model instance.

        Raises:
            KeyError: If required fields are missing from config.
        """
        # Validate required fields
        required = ["model"]
        if missing := [f for f in required if f not in config]:
            raise KeyError(f"Config missing required fields: {missing}")

        # Separate known fields from generation parameters
        known = {"model", "url", "api_key", "model_provider", "timeout"}
        gen_params = {k: v for k, v in config.items() if k not in known}

        return cls(
            model_name=config["model"],
            base_url=config["url"],
            api_key=config["api_key"],
            model_provider=config.get("model_provider"),
            timeout=config.get("timeout", 600),
            **gen_params,
        )

    async def generate(
        self,
        request: CompletionRequest,
        stream: bool = False,
    ) -> AsyncGenerator[CompletionResponse, None]:
        """
        Generate a model completion asynchronously using LiteLLM.

        Args:
            request: CompletionRequest containing input messages, tools, etc.
            stream: Whether to stream responses from the LLM.

        Yields:
            CompletionResponse objects parsed from LiteLLM output.
        """
        # 1. Format the request for LiteLLM
        llm_kwargs = self.formatter.format_request(
            request=request,
            model_name=self.model_name,
            base_url=self.base_url,
            api_key=self.api_key,
            stream=stream,
            timeout=self.timeout,
            model_provider=self.model_provider,
            **self.generation_params,
        )

        # 2. Call LiteLLM asynchronously with retry for transient errors
        raw_resp = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.debug(f"Start calling LiteLLM (attempt {attempt + 1}/{self.max_retries + 1})...")
                raw_resp = await litellm.acompletion(**llm_kwargs)
                break  # Success, exit retry loop
            except _RETRYABLE_EXCEPTIONS as e:
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        f"Retryable API error (attempt {attempt + 1}/{self.max_retries + 1}): "
                        f"{type(e).__name__}: {e}. Retrying in {delay:.0f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"Max retries ({self.max_retries}) exceeded for retryable error: "
                        f"{type(e).__name__}: {e}"
                    )
                    yield CompletionResponse(
                        id="error",
                        content=[],
                        error_code="litellm_error",
                        error_message=str(e),
                    )
                    return
            except Exception as e:
                # Non-retryable errors fail immediately
                yield CompletionResponse(
                    id="error",
                    content=[],
                    error_code="litellm_error",
                    error_message=str(e),
                )
                return

        # 3. Handle streaming response
        if stream and hasattr(raw_resp, "__aiter__"):
            async for chunk in raw_resp:
                parsed = self.formatter.parse_response(chunk)
                yield parsed
            return

        # 4. Non-stream response (single ModelResponse)
        parsed = self.formatter.parse_response(raw_resp)
        yield parsed
