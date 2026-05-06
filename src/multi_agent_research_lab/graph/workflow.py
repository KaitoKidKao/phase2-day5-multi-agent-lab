from typing import Any, Literal

from langgraph.graph import END, StateGraph

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph with integrated guardrails."""

    def __init__(self, llm: LLMClient, search: SearchClient) -> None:
        self.llm = llm
        self.search = search
        
        # Initialize agents
        self.supervisor = SupervisorAgent(llm)
        self.researcher = ResearcherAgent(llm, search)
        self.analyst = AnalystAgent(llm)
        self.writer = WriterAgent(llm)

    def guardrail_node(self, state: ResearchState) -> ResearchState:
        """Check if the query violates safety or length limits."""
        if not self.llm.validate_query(state.request.query):
            state.errors.append("Query violated safety or length limits.")
            state.record_route("blocked")
        else:
            state.record_route("passed_guardrail")
        return state

    def build(self) -> StateGraph:
        """Create a LangGraph graph."""

        workflow = StateGraph(ResearchState)

        # 1. Add nodes
        workflow.add_node("guardrail", self.guardrail_node)
        workflow.add_node("supervisor", self.supervisor.run)
        workflow.add_node("researcher", self.researcher.run)
        workflow.add_node("analyst", self.analyst.run)
        workflow.add_node("writer", self.writer.run)

        # 2. Set entry point
        workflow.set_entry_point("guardrail")

        # 3. Decision after guardrail
        def after_guardrail(state: ResearchState) -> Literal["supervisor", "finish"]:
            if "blocked" in state.route_history:
                return "finish"
            return "supervisor"

        workflow.add_conditional_edges(
            "guardrail",
            after_guardrail,
            {
                "supervisor": "supervisor",
                "finish": END
            }
        )

        # 4. Add conditional edges from supervisor
        def route_decision(state: ResearchState) -> Literal["researcher", "analyst", "writer", "finish"]:
            return state.route_history[-1]  # type: ignore

        workflow.add_conditional_edges(
            "supervisor",
            route_decision,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "finish": END
            }
        )

        # 5. Add edges back to supervisor
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")

        return workflow

    def run(self, state: ResearchState, callbacks: list[Any] | None = None) -> ResearchState:
        """Execute the graph and return final state."""

        app = self.build().compile()
        result = app.invoke(state, config={"callbacks": callbacks} if callbacks else None)
        
        # Convert dict back to ResearchState if necessary
        if isinstance(result, dict):
            return ResearchState(**result)
        return result
