# main.py
from langchain.schema import HumanMessage
from .state_graph import stream_graph, State
import pprint

# --- 초기 상태 설정 ---
initial_state: State = {
    "messages": [HumanMessage(content="어깨관절운동?")],
    "documents": [],
    "summary": ""
}

config = {"configurable": {"thread_id": "1"}}
inputs = {"messages": initial_state["messages"]}

stream_graph(inputs, config)
