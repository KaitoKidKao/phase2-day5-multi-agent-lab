"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Gathers information using a SearchClient."""

    name = AgentName.RESEARCHER

    def __init__(self, llm: LLMClient, search: SearchClient) -> None:
        super().__init__(llm)
        self.search = search

    def run(self, state: ResearchState) -> ResearchState:
        """Perform search and update research notes."""

        # 1. Search for information
        query = state.request.query
        docs = self.search.search(query, max_results=state.request.max_sources)
        state.sources.extend(docs)

        # 2. Summarize findings
        system_prompt = (
            "You are a professional researcher. Summarize the provided search results into detailed research notes.\n"
            "Focus on factual information, key technical details, and potential areas for analysis."
        )
        
        context = "\n\n".join([f"Source: {d.title}\nContent: {d.snippet}" for d in docs])
        user_prompt = f"Query: {query}\n\nSearch Results:\n{context}"

        response = self.llm.complete(system_prompt, user_prompt)

        # 3. Update state
        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=self.name,
                content="Completed research and summarized findings.",
                metadata={"tokens": response.input_tokens, "cost": response.cost_usd}
            )
        )

        return state
