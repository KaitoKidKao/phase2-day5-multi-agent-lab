from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = AgentName.SUPERVISOR

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""

        system_prompt = (
            "You are a research supervisor. Your goal is to coordinate a team of agents to answer a user query.\n"
            "Agents available:\n"
            "- researcher: Gathers raw information from the web.\n"
            "- analyst: Processes research notes into structured insights.\n"
            "- writer: Creates the final answer based on analysis.\n\n"
            "Rules:\n"
            "1. If no research is done, call 'researcher'.\n"
            "2. If research is done but no analysis is present, call 'analyst'.\n"
            "3. If analysis is present, call 'writer'.\n"
            "4. If 'writer' has finished the report, respond with 'FINISH'.\n"
            "5. If you are stuck or hit max iterations (current: {iteration}), call 'writer' to summarize what we have.\n\n"
            "Respond ONLY with the name of the next agent or 'FINISH'."
        )

        user_prompt = (
            f"Query: {state.request.query}\n"
            f"Iteration: {state.iteration}\n"
            f"Research Notes: {'Present' if state.research_notes else 'None'}\n"
            f"Analysis Notes: {'Present' if state.analysis_notes else 'None'}\n"
            f"Final Answer: {'Present' if state.final_answer else 'None'}\n"
        )

        response = self.llm.complete(
            system_prompt.format(iteration=state.iteration),
            user_prompt
        )

        next_route = response.content.strip().lower()

        # Validation and fallback
        valid_routes = ["researcher", "analyst", "writer", "finish"]
        if next_route not in valid_routes:
            next_route = "writer" if state.analysis_notes else "researcher"

        state.record_route(next_route)
        state.agent_results.append(
            AgentResult(
                agent=self.name,
                content=f"Routing to: {next_route}",
                metadata={"tokens": response.input_tokens, "cost": response.cost_usd}
            )
        )

        return state
