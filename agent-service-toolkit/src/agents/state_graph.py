

from typing import List, Annotated, TypedDict
from langgraph.graph.message import add_messages, RemoveMessage
from langchain.schema import HumanMessage, AIMessage, Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import pprint
from .pdf_loader import retriever
from typing import List
from langchain.schema import Document

def format_docs(docs: List[Document]) -> str:
    return "\n".join(
        [
            f"<document><content>{doc.page_content}</content><source>{doc.metadata['source']}</source></document>"
            for doc in docs
        ]
    )

# --- State 정의 ---
class State(TypedDict):
    messages: Annotated[List, add_messages]  # HumanMessage와 AIMessage 객체들의 대화 내역
    documents: Annotated[list, "filtered documents"]
    summary: str  # 대화 요약본

# --- LLM 초기화 ---
llm = ChatOpenAI(model="gpt-4o-mini")
llm_generator = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# --- 헬퍼 함수 ---
def get_latest_human(messages: List) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""

def get_latest_ai(messages: List) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            return msg.content
    return ""

#def summarize_chat_history(state: State):
    summary = state.get("summary", "")
    if not summary:
        summary_template = "아래 대화 내용을 요약해주세요.\n"
    else:
        summary_template = (
            f"지금까지의 대화 내용의 요약입니다.: {summary}\n"
            "아래 주어진 추가된 대화 이력을 고려해서 새로운 요약을 만들어주세요.\n"
            "[추가된 대화 이력]\n"
        )
    prompt = [HumanMessage(content=summary_template)] + state["messages"]
    new_summary = llm.invoke(prompt).content

    delete_messages = [RemoveMessage(id=message.id) for message in state["messages"][:-2] if hasattr(message, "id")]
    return {
        "messages": delete_messages,
        "summary": new_summary
    }

#def should_summarize(state: State):
    messages = state["messages"]
    if len(messages) > 5:
        return "summarize_chat_history"
    return END

def interaction(state: State):


    # 체인(예: interaction_chain) 임포트
    from .interaction import interaction_chain
    from langchain.schema import AIMessage
    question = get_latest_human(state["messages"])
    
    chain_output = interaction_chain.invoke({"question": question})
    final_query_str = chain_output["text"]  # 또는 "final_answer", "response", "content" 등
    
    return {
        "messages": [AIMessage(content=final_query_str)],
        # 추가: state에 별도로 저장
        "refined_query": final_query_str
    }


# --- 노드 함수들 ---
def retrieve(state: State):
    print("--- [RETRIEVE] ---")
    # 만약 'refined_query'가 state에 있으면 그걸 사용, 없으면 user 질문 fallback
    refined_query = state.get("refined_query")
    if refined_query:
        question = refined_query
    else:
        question = get_latest_human(state["messages"])
    
    documents = retriever.invoke(question)
    return {"documents": documents}

def grade_documents(state: State):
    """
    Processes retrieved documents without relevance scoring
    """
    print("--- [PROCESS DOCUMENTS] ---")
    
    # Use refined_query if available in state, otherwise use the latest human message
    if state.get("refined_query"):
        question = state["refined_query"]
    else:
        question = get_latest_human(state["messages"])
    
    documents = state["documents"]
    
    # Simply return the documents without filtering
    return {"documents": documents}
def generate(state: State):
    print("--- [GENERATE] ---")
    from .generation import generator_chain
    question = get_latest_human(state["messages"])
    retrieved_docs = state["documents"]

    if not retrieved_docs:
        return {"messages": [AIMessage(content="해당 주제와 관련된 논문을 찾을 수 없습니다.")]}

    # ✅ 프롬프트를 직접 수정하여 논문을 나열하는 로직 추가
    prompt = f"""
    검색된 논문의 내용을 참고하여 최종적으로 정리된 답변을 작성하세요.
    논문의 원문을 유지하며, 사용자가 원할 경우 번역 여부를 선택할 수 있도록 하세요.

    논문이 영어로 되어 있다면, 응답 마지막에 "📝 원하시면 한국국어로 번역해 드릴까요?"를 포함하세요.
    매번 답변의 마지막에 출처를 남기세요.
        사용자의 질문: {question}
    
    
    검색된 논문 목록:
    {format_docs(retrieved_docs)}

    """

    response = generator_chain.invoke({"question": question, "context": prompt})
    return {"messages": [AIMessage(content=response.content)]}


def rewrite_query(state: State):
    print("--- [REWRITE QUERY] ---")
    from .rewriter import question_rewriter
    from langchain.schema import HumanMessage
    question = get_latest_human(state["messages"])
    rewritten_query = question_rewriter.invoke({"question": question})
    return {"messages": [HumanMessage(content=rewritten_query)]}

def web_search(state: State):
    print("--- [WEB SEARCH] ---")
    
    from langchain.schema import Document
    # 최신 사용자 질문 추출
    question = get_latest_human(state["messages"])
    
    # 웹 검색에 적합한 쿼리 생성을 위한 프롬프트 구성
    prompt = f"""
    사용자가 운동과 관련한 질문을 하였을 때 공신력 있는 사이트에서 정보를 가져와 대답하십시오.
    신뢰성,전문성,공신력이 뒷받침되는 자료이여야 합니다.
    예를 들어 사용자가 운동 관련 질문을 할경우 NCAA와 같은 검증된 사이트에서 자료를 가져와 대답하는 것입니다.
    사용자 질문: {question}
    
    검색 쿼리:
    """
    # LLM을 사용해 정제된 검색 쿼리 생성
    refined_query = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    print(f"Refined search query: {refined_query}")
    
    # 생성된 쿼리를 이용해 웹 검색 수행
    web_search_tool = TavilySearchResults(k=3)
    docs = web_search_tool.invoke(refined_query)
    
    # 각 검색 결과가 dict 또는 str 일 경우 모두 처리
    web_results = []
    for doc in docs:
        if isinstance(doc, dict):
            content = doc.get("content", "")
            source = doc.get("url", "")
        else:
            content = doc
            source = "unknown"
        web_results.append(Document(page_content=content, metadata={"source": source}))
    
    return {"documents": web_results}

def route_question(state: State):
    print("--- [ROUTE QUESTION] ---")
    from routing import router_chain
    question = get_latest_human(state["messages"])
    source = router_chain.invoke({"question": question})

    if source.source == "web_search":
        print("--- ROUTE QUESTION TO WEB SEARCH ---")
        # 여기서 state["source"]를 web_search로 설정
        state["source"] = "web_search"
        return "web"
    elif source.source == "vectorstore":
        print("--- ROUTE QUESTION TO RAG ---")
        state["source"] = "vectorstore"
        return "retrieve"
    else:
        print("--- ROUTE QUESTION TO DIRECT GENERATE ---")
        state["source"] = "direct"
        return "direct_generate"

def decide_to_generate(state: State):
    print("--- [ASSESS GRADED DOCUMENTS] ---")
    documents = state["documents"]

    # rewrite_query 횟수를 저장할 카운터(없으면 0으로 초기화)
    rewrite_count = state.get("rewrite_count", 0)

    if not documents:
        # 문서가 전혀 남아 있지 않다면 rewrite_query로 유도
        rewrite_count += 1
        state["rewrite_count"] = rewrite_count
        print(f"     --- [ALL DOCUMENTS ARE NOT RELEVANT, rewrite #{rewrite_count}] ---")

        # rewrite_query가 3회 이상 반복되면 web_search로 전환
        if rewrite_count >= 3:
            print("--- [REWRITE LIMIT REACHED => GO TO WEB_SEARCH] ---")
            return "web_search"
        else:
            return "rewrite_query"
    else:
        print("     --- [DECISION: GENERATE] ---")
        # 문서가 유효하면 rewrite 횟수 초기화
        state["rewrite_count"] = 0
        return "generate"
    
def grade_generation_v_document_question(state: State):
    print("--- [GRADE GENERATION vs DOCUMENT QUESTION] ---")
    from grading import hallucination_checker_chain, answer_grader_chain

    question = get_latest_human(state["messages"])
    generation = get_latest_ai(state["messages"])



    # 1) web_search 경로라면 문서 검사(환각 검사)를 건너뛰고,
    #    "web_search로 전환되었다"는 안내 메시지를 남김.
    if state.get("source") == "web_search":
        print("--- [SKIPPING HALLUCINATION CHECK FOR WEB SEARCH] ---")
        # 사용자에게 안내할 메시지를 AI 응답으로 추가 (선택 사항)
        state["messages"].append(
            AIMessage(content="(안내) 문서 대신 웹 검색 결과를 참고해 답변했습니다. 필요 시 추가 검색 가능합니다.")
        )
        # 문서 검증 절차 없이 바로 'useful' 등으로 리턴하여 이후 노드로 진행
        return "useful"

    # 2) 기존 문서 기반 환각 검사 로직
    docs = state["documents"]
    score = hallucination_checker_chain.invoke(
        {
            "generation": generation,
            "documents": format_docs(docs)
        }
    ).binary_score

    if score == "yes":
        print("--- [NOT HALLUCINATION] ---")
        answer_score = answer_grader_chain.invoke(
            {"question": question, "generation": generation}
        ).binary_score

        if answer_score == "yes":
            print("--- [GENERATION ADDRESSES QUESTION] ---")
            return "useful"  # ✅ 최종 응답을 가공하는 단계로 이동
        else:
            print("--- [GENERATION DOES NOT ADDRESS QUESTION] ---")
            return "not useful"
    else:
        print("--- [HALLUCINATION] ---")
        return "not supported"


def direct_generate(state: State):
    print("--- [DIRECT GENERATE] ---")
    from langchain.schema import HumanMessage, AIMessage
    question = get_latest_human(state["messages"])
    conversation_history = "\n".join(
        [
            f"{'User:' if isinstance(m, HumanMessage) else 'AI:'} {m.content}"
            for m in state["messages"]
        ]
    )
    prompt = f"""You are a helpful AI assistant.
Here is the conversation history:
{conversation_history}

User question: {question}

Please provide a direct answer that takes into account the conversation history.
When a user receives a paper in English as an answer, do you want to translate it into Korean? after presenting the paper, leave a message
"""
    response = llm_generator.invoke(prompt)
    new_ai_message = AIMessage(content=response.content)
    return {"messages": [new_ai_message]}
# --- 그래프 구성 ---
flow = StateGraph(State)
#flow.add_node("summarize_chat_history", summarize_chat_history)
import pprint

# 기존 노드 등록
flow.add_node("retrieve", retrieve)
flow.add_node("generate", generate)
flow.add_node("rewrite_query", rewrite_query)
flow.add_node("web_search", web_search)
flow.add_node("direct_generate", direct_generate)
flow.add_node("interaction",interaction)
flow.add_node("grade_documents",grade_documents)

flow.add_edge(START, "interaction")  # 예시

flow.add_edge("interaction","retrieve")
flow.add_edge("retrieve", "grade_documents")
flow.add_edge("grade_documents", "generate")
flow.add_edge("generate", END)



memory = MemorySaver()
graph = flow.compile(checkpointer=memory)


def stream_graph(inputs, config, exclude_node=[]):
    for output in graph.stream(inputs, config, stream_mode="updates"):
        for k, v in output.items():
            if k not in exclude_node:
                pprint.pprint(f"Output from node '{k}':")
                pprint.pprint("---")
                pprint.pprint(v, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")


