# routing.py
from typing import Literal
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# 라우팅을 위한 출력형식 정의
class RouteQuery(BaseModel):
    source: Literal["vectorstore", "web_search", "direct_generation"] = Field(
        description="Given a user question choose among vectorstore, web_search, or direct_generation"
    )

system = """System Role: You are an expert routing agent responsible for deciding whether to route the user's question to:
1) vectorstore-based retrieval,
2) web search, or
3) direct generation (no retrieval).

Routing Rules:
1. If the user’s question is about physical exercise, muscle training, or sleep/recovery,랭그래프  related to exercise,
   you must first attempt vectorstore retrieval.
   - After retrieving from vectorstore, if no relevant documents are found, then you may use web search.

2. If the user explicitly indicates they want to do "web search" or "search the internet,"
   route to web search.

3. If the user’s query does not match physical exercise topics (vectorstore)
   AND they do not request web search,
   then route the request to direct generation (no retrieval).

4. If multiple rules could apply, prioritize vectorstore if the question is exercise-related.
   Only proceed to web search if vectorstore yields no relevant results or the user explicitly requests web search.
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o")
llm_router = llm.with_structured_output(RouteQuery)

router_chain = route_prompt | llm_router
