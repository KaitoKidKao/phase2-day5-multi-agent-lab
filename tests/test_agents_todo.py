from unittest.mock import MagicMock

import pytest

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


def test_supervisor_runs_successfully() -> None:
    # 1. Setup mock LLM
    mock_llm = MagicMock(spec=LLMClient)
    mock_llm.complete.return_value = MagicMock(content="researcher", input_tokens=10, cost_usd=0.01)
    
    # 2. Setup state
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    
    # 3. Run agent
    agent = SupervisorAgent(mock_llm)
    new_state = agent.run(state)
    
    # 4. Assertions
    assert len(new_state.route_history) == 1
    assert new_state.route_history[0] == "researcher"
