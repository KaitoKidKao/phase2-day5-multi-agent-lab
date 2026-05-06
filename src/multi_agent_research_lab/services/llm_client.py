from dataclasses import dataclass
from typing import Any
import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError

from multi_agent_research_lab.core.config import get_settings

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client using OpenAI with fallback and guardrails."""

    def __init__(self) -> None:
        settings = get_settings()
        
        # Try to use Langfuse integration for automatic tracing
        try:
            from langfuse.openai import OpenAI as LangfuseOpenAI
            self.client = LangfuseOpenAI(api_key=settings.openai_api_key)
        except ImportError:
            self.client = OpenAI(api_key=settings.openai_api_key)
            
        self.model = settings.openai_model
        self.fallback_model = settings.openai_fallback_model

    def validate_query(self, query: str) -> bool:
        """
        Comprehensive Guardrail: Check if the query is safe and within limits.
        Uses OpenAI Moderation API for community standards.
        """
        # 1. Basic length check
        if len(query.strip()) < 3:
            logger.warning(f"Query too short: {query}")
            return False
            
        # 2. Safety/Relevance keywords (Prompt Injection Protection)
        blocked_keywords = ["ignore previous instructions", "system prompt", "drop table", "delete from"]
        if any(kw in query.lower() for kw in blocked_keywords):
            logger.warning(f"Technical safety violation: {query}")
            return False

        # 3. OpenAI Moderation API (Negative content, Hate, Violence, etc.)
        try:
            moderation = self.client.moderations.create(input=query)
            result = moderation.results[0]
            
            if result.flagged:
                logger.warning(f"Content Moderation Flagged: {result.categories}")
                # You can log specific categories if needed:
                # categories = [cat for cat, flagged in result.categories.items() if flagged]
                return False
        except Exception as e:
            logger.error(f"Moderation API call failed: {e}. Proceeding with caution.")
            # If moderation fails, we still allow basic technical checks to pass
            pass
            
        return True

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def _call_api(self, model: str, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Internal method to call the API with retry logic."""
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage

        # Cost estimation (placeholder pricing)
        cost = 0.0
        if usage:
            cost = (usage.prompt_tokens * 0.150 / 1_000_000) + (
                usage.completion_tokens * 0.600 / 1_000_000
            )

        return LLMResponse(
            content=content,
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            cost_usd=cost,
        )

    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with usage tracking and fallback."""
        try:
            # Try with primary model
            return self._call_api(self.model, system_prompt, user_prompt)
        except Exception as e:
            logger.error(f"Primary model {self.model} failed: {e}. Trying fallback {self.fallback_model}...")
            try:
                # Fallback model
                return self._call_api(self.fallback_model, system_prompt, user_prompt)
            except Exception as fe:
                logger.critical(f"Both primary and fallback models failed. Last error: {fe}")
                raise fe
