from dataclasses import dataclass
from typing import Any

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client using OpenAI."""

    def __init__(self) -> None:
        settings = get_settings()
        
        # Try to use Langfuse integration for automatic tracing
        # It automatically picks up LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST from env
        try:
            from langfuse.openai import OpenAI as LangfuseOpenAI
            self.client = LangfuseOpenAI(api_key=settings.openai_api_key)
        except ImportError:
            self.client = OpenAI(api_key=settings.openai_api_key)
            
        self.model = settings.openai_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with usage tracking."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage

        # Basic cost estimation (gpt-4o-mini prices)
        # Input: $0.150 / 1M tokens, Output: $0.600 / 1M tokens
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
