"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class WriterAgent(BaseAgent):
    """Produces the final report for the user."""

    name = AgentName.WRITER

    def run(self, state: ResearchState) -> ResearchState:
        """Create the final report and update `state.final_answer`."""

        if not state.analysis_notes and not state.research_notes:
            state.final_answer = "No information found to answer your query."
            return state

        system_prompt = (
            f"You are a professional technical writer. Your goal is to write a clear, comprehensive, and well-structured report for a {state.request.audience} audience.\n"
            "Use the provided research and analysis notes. If information is missing, state it clearly."
        )
        
        context = (
            f"Research Notes:\n{state.research_notes or 'None'}\n\n"
            f"Analysis Notes:\n{state.analysis_notes or 'None'}"
        )
        user_prompt = f"User Query: {state.request.query}\n\nNotes:\n{context}"

        response = self.llm.complete(system_prompt, user_prompt)

        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=self.name,
                content="Generated final report.",
                metadata={"tokens": response.input_tokens, "cost": response.cost_usd}
            )
        )

        return state
