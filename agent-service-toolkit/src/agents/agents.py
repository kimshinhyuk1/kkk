from dataclasses import dataclass
from langgraph.graph.state import CompiledStateGraph

from agents.bg_task_agent.bg_task_agent import bg_task_agent
from agents.chatbot import chatbot
from agents.command_agent import command_agent
from agents.research_assistant import research_assistant

# ✅ 그래프 인스턴스 객체를 정확하게 가져오도록 수정
from .diet_graph import compiled_diet_graph
from .medical_graph import compiled_medical_graph
from .general_graph import compiled_general_graph

from schema import AgentInfo

DEFAULT_AGENT = "diet_graph-agent"


@dataclass
class Agent:
    description: str
    graph: CompiledStateGraph


agents: dict[str, Agent] = {
    "chatbot": Agent(description="A simple chatbot.", graph=chatbot),
    "research-assistant": Agent(
        description="A research assistant with web search and calculator.", graph=research_assistant
    ),
    "command-agent": Agent(description="A command agent.", graph=command_agent),
    "bg-task-agent": Agent(description="A background task agent.", graph=bg_task_agent),
    "diet_graph-agent": Agent(
        description="A state-based agent with structured conversation flow.",
        graph=compiled_diet_graph  # ✅ 정확히 CompiledStateGraph 인스턴스 사용
    ),
    "medical_graph-agent": Agent(
        description="A state-based agent with structured conversation flow.",
        graph=compiled_medical_graph
    ),
    "general_graph-agent": Agent(
        description="A state-based agent with structured conversation flow.",
        graph=compiled_general_graph
    ),
}


def get_agent(agent_id: str) -> CompiledStateGraph:
    return agents[agent_id].graph


def get_all_agent_info() -> list[AgentInfo]:
    return [
        AgentInfo(key=agent_id, description=agent.description) for agent_id, agent in agents.items()
    ]
