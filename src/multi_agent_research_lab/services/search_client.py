import httpx
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client using Tavily."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.api_key = self.settings.tavily_api_key

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query using Tavily API."""

        if not self.api_key:
            return self._mock_search(query)

        try:
            response = httpx.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic",
                },
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            
            return [
                SourceDocument(
                    title=result["title"],
                    url=result["url"],
                    snippet=result["content"],
                    metadata={"score": result.get("score")}
                )
                for result in data.get("results", [])
            ]
        except Exception:
            # Fallback to mock if API fails
            return self._mock_search(query)

    def _mock_search(self, query: str) -> list[SourceDocument]:
        """Robust Mock implementation for development."""
        return [
            SourceDocument(
                title=f"Insight on {query}",
                url="https://example.com/research-1",
                snippet=f"This is a simulated search result for '{query}'. It contains key findings.",
                metadata={"source": "mock_search"},
            ),
            SourceDocument(
                title=f"Detailed analysis of {query}",
                url="https://example.com/research-2",
                snippet=f"Another perspective on {query} discussing state-of-the-art methods.",
                metadata={"source": "mock_search"},
            ),
        ]
