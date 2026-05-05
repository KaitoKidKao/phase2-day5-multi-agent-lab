"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class AnalystAgent(BaseAgent):
    """Processes research notes into structured insights."""

    name = AgentName.ANALYST

    def run(self, state: ResearchState) -> ResearchState:
        """Process research notes and update `state.analysis_notes`."""

        if not state.research_notes:
            return state

        system_prompt = (
            "You are a technical analyst. Your task is to analyze raw research notes and extract structured insights.\n"
            "Identify patterns, trends, pros/cons, and critical findings relevant to the user's query."
        )
        
        user_prompt = f"User Query: {state.request.query}\n\nResearch Notes:\n{state.research_notes}"

        response = self.llm.complete(system_prompt, user_prompt)

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=self.name,
                content="Analyzed research notes and generated insights.",
                metadata={"tokens": response.input_tokens, "cost": response.cost_usd}
            )
        )

        return state
