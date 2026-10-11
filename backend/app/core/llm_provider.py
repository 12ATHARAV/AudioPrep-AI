"""
LLM Provider with automatic fallback and non-blocking asynchronous execution.

Architecture:
  NVIDIA (primary - high context & high rate limits)
      ↓ rate limit / timeout / failure
  Gemini (fallback)
      ↓ rate limit / failure
  Groq (secondary fallback)

Interview Note: "I implemented resilient provider failover.
All provider calls run in worker threads via asyncio.to_thread with strict
per-request timeouts. If any provider experiences a hang or rate limit,
the system never freezes the event loop — it logs the warning and
seamlessly cascades to the next provider within seconds."
"""

import asyncio
import os
from typing import Optional
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from google import genai
from google.genai import types as genai_types
from groq import Groq
from openai import OpenAI

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("llm_provider")


class LLMResponse:
    """Unified response from any LLM provider."""
    def __init__(self, text: str, provider: str, model: str):
        self.text = text
        self.provider = provider
        self.model = model


class LLMProvider:
    """
    Unified LLM provider with automatic failover and non-blocking async execution.
    Hierarchy:
      NVIDIA (primary - high context & high rate limits)
          ↓ rate limit / failure / timeout
      Gemini (fallback)
          ↓ rate limit / failure
      Groq (secondary fallback)
    """

    def __init__(self):
        self._gemini_client = None
        self._groq_client = None
        self._nvidia_client = None

    @property
    def nvidia(self):
        if self._nvidia_client is None and settings.nvidia_api_key:
            self._nvidia_client = OpenAI(
                base_url=settings.nvidia_base_url,
                api_key=settings.nvidia_api_key,
                timeout=httpx.Timeout(35.0, connect=5.0),
                max_retries=1,
            )
        return self._nvidia_client

    @property
    def gemini(self):
        if self._gemini_client is None and settings.gemini_api_key:
            self._gemini_client = genai.Client(
                api_key=settings.gemini_api_key
            )
        return self._gemini_client

    @property
    def groq(self):
        if self._groq_client is None and settings.groq_api_key:
            self._groq_client = Groq(
                api_key=settings.groq_api_key,
                timeout=httpx.Timeout(90.0, connect=10.0),
                max_retries=1,
            )
        return self._groq_client

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 4000,
        provider: str = "auto",
    ) -> LLMResponse:
        """
        Generate text from LLM with non-blocking execution and automatic fallback.

        Args:
            prompt: The user prompt
            system_prompt: System instructions
            temperature: Creativity (0=deterministic, 1=creative)
            max_tokens: Max output length
            provider: "nvidia", "gemini", "groq", or "auto"
        """
        if provider == "nvidia":
            return await self._call_nvidia(
                prompt, system_prompt, temperature, max_tokens
            )

        if provider == "gemini":
            return await self._call_gemini(
                prompt, system_prompt, temperature, max_tokens
            )

        if provider == "groq":
            return await self._call_groq(
                prompt, system_prompt, temperature, max_tokens
            )

        # Provider: "auto" fallback hierarchy
        # 1. Try NVIDIA first if configured
        if self.nvidia:
            try:
                return await self._call_nvidia(
                    prompt, system_prompt, temperature, max_tokens
                )
            except Exception as e:
                logger.warning(
                    "nvidia_failed_falling_back",
                    error=str(e),
                )

        # 2. Try Gemini
        if self.gemini:
            try:
                return await self._call_gemini(
                    prompt, system_prompt, temperature, max_tokens
                )
            except Exception as e:
                logger.warning(
                    "gemini_failed_falling_back",
                    error=str(e),
                )

        # 3. Fallback to Groq
        if self.groq:
            try:
                return await self._call_groq(
                    prompt, system_prompt, temperature, max_tokens
                )
            except Exception as e:
                logger.error("groq_also_failed", error=str(e))
                raise

        raise RuntimeError("No LLM provider available (all configured providers failed or timed out)")

    async def _call_nvidia(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call NVIDIA NIM API asynchronously without blocking the event loop."""
        return await asyncio.wait_for(
            asyncio.to_thread(
                self._sync_call_nvidia,
                prompt,
                system_prompt,
                temperature,
                max_tokens,
            ),
            timeout=120.0,
        )

    def _sync_call_nvidia(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Synchronous worker for NVIDIA NIM API."""
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })
        messages.append({"role": "user", "content": prompt})

        target_model = settings.nvidia_model
        try:
            response = self.nvidia.chat.completions.create(
                model=target_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            used_model = target_model
        except Exception as e:
            err_msg = str(e)
            # If model reached EOL (410), not found (404), or timed out, retry with active fast model
            if ("410" in err_msg or "404" in err_msg or "time" in err_msg.lower()) and target_model != "meta/llama-3.2-11b-vision-instruct":
                fallback_model = "meta/llama-3.2-11b-vision-instruct"
                logger.warning(
                    "nvidia_model_fallback_retry",
                    requested_model=target_model,
                    active_fallback=fallback_model,
                    error=err_msg,
                )
                response = self.nvidia.chat.completions.create(
                    model=fallback_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                used_model = fallback_model
            else:
                raise

        return LLMResponse(
            text=response.choices[0].message.content,
            provider="nvidia",
            model=used_model,
        )

    async def _call_gemini(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call Gemini API asynchronously without blocking the event loop."""
        return await asyncio.wait_for(
            asyncio.to_thread(
                self._sync_call_gemini,
                prompt,
                system_prompt,
                temperature,
                max_tokens,
            ),
            timeout=120.0,
        )

    def _sync_call_gemini(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Synchronous worker for Gemini API."""
        contents = prompt
        if system_prompt:
            contents = f"{system_prompt}\n\n{prompt}"

        response = self.gemini.models.generate_content(
            model=settings.gemini_model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            ),
        )

        return LLMResponse(
            text=response.text,
            provider="gemini",
            model=settings.gemini_model,
        )

    async def _call_groq(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call Groq API asynchronously without blocking the event loop."""
        return await asyncio.wait_for(
            asyncio.to_thread(
                self._sync_call_groq,
                prompt,
                system_prompt,
                temperature,
                max_tokens,
            ),
            timeout=90.0,
        )

    def _sync_call_groq(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Synchronous worker for Groq API."""
        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt,
            })
        messages.append({"role": "user", "content": prompt})

        response = self.groq.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return LLMResponse(
            text=response.choices[0].message.content,
            provider="groq",
            model=settings.groq_model,
        )


# Singleton
llm_provider = LLMProvider()