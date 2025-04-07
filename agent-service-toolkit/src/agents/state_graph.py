from typing import TypedDict, List, Optional
from langgraph.graph.message import add_messages, RemoveMessage
from langchain.schema import HumanMessage, AIMessage, Document
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
import pprint
from typing import List
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='app.log',  # 파일로 로그 출력 (콘솔에 표시되지 않음)
)
logger = logging.getLogger(__name__)


def format_docs(docs: List[Document]) -> str:
    return "\n".join(
        [
            f"<document><content>{doc.page_content}</content><source>{doc.metadata['source']}</source></document>"
            for doc in docs
        ]
    )

# --- State 정의 ---
class State(TypedDict):
    messages: List[HumanMessage | AIMessage]
    documents: Optional[List[Document]]
    refined_query: Optional[str]

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

# def summarize_chat_history(state: State):
#    summary = state.get("summary", "")
#    if not summary:
#        summary_template = "아래 대화 내용을 요약해주세요.\n"
#    else:
#        summary_template = (
#            f"지금까지의 대화 내용의 요약입니다.: {summary}\n"
#            "아래 주어진 추가된 대화 이력을 고려해서 새로운 요약을 만들어주세요.\n"
#            "[추가된 대화 이력]\n"
#        )
#    prompt = [HumanMessage(content=summary_template)] + state["messages"]
#    new_summary = llm.invoke(prompt).content

#    delete_messages = [RemoveMessage(id=message.id) for message in state["messages"][:-2] if hasattr(message, "id")]
#    return {
#        "messages": delete_messages,
#        "summary": new_summary
#    }

# def should_summarize(state: State):
#    messages = state["messages"]
#    if len(messages) > 5:
#        return "summarize_chat_history"
#    return END

def interaction(state: State):
    # 체인(예: interaction_chain) 임포트
    from .interaction import interaction_chain
    from langchain.schema import AIMessage
    question = get_latest_human(state["messages"])
    
    chain_output = interaction_chain.invoke({"question": question})
    final_query_str = chain_output["text"]  # 또는 "final_answer", "response", "content" 등
    
    is_query_finalized = "최종 쿼리" in final_query_str or "finalise the query" in final_query_str.lower()
    
    return {
        "messages": [AIMessage(content=final_query_str)],
        "refined_query": final_query_str,
        "query_finalized": is_query_finalized
    }

def check_query_status(state: State):
    print("--- [CHECK QUERY STATUS] ---")
    
    is_finalized = state.get("query_finalized", False)
    
    if is_finalized:
        print("--- [QUERY FINALIZED] ---")
    else:
        print("--- [QUERY NOT FINALIZED] ---")
        
    return is_finalized

# --- 노드 함수들 ---
from .pdf_loader import hybrid_retriever

import logging
logger = logging.getLogger(__name__)

def retrieve(state: State):
    """
    Retrieve relevant documents using a hybrid approach with query refinement
    """
    logger.info("--- [RETRIEVE] ---")
    
    # 사용자의 원래 질문 가져오기
    original_query = get_latest_human(state["messages"])
    
    # 질문 개선을 위한 LLM 및 프롬프트 설정
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    
    fitness_context_template = """[System / Instruction Prompt]

    아무 변환 없이 최종 쿼리 그대로 변환하십시오오
    """
    
    query_refiner_prompt = PromptTemplate(
        template=fitness_context_template,
        input_variables=["question"]
    )
    
    query_refiner_chain = query_refiner_prompt | llm | StrOutputParser()
    
    # 개선된 질문 생성
    if state.get("refined_query"):
        refined_query = state["refined_query"]
    else:
        refined_query = query_refiner_chain.invoke({"question": original_query})
    
    logger.info(f"원래 질문: {original_query}")
    logger.info(f"개선된 질문: {refined_query}")
    
    # --- (1) max_docs 제거 ---
    # max_docs = 20  # <-- 이 줄 삭제 or 주석 처리
    
    # --- (2) 전체 문서 그대로 반환 ---
    documents = hybrid_retriever.invoke(refined_query)
    
    logger.info(f"검색된 문서 수: {len(documents)}, 사용할 문서 수: {len(documents)}")
    
    logger.info("--- [RETRIEVE -> GRADE_DOC] Selected Documents ---")
    for i, doc in enumerate(documents):
        snippet = doc.page_content[:100].replace("\n", " ")
        source = doc.metadata.get("source", "Unknown Source")
        logger.info(f"  [{i+1}] source={source}, snippet={snippet}...")
    logger.info("-------------------------------------------------\n")
    
    return {
        "documents": documents,  # <-- 그냥 documents 전체 반환
        "refined_query": refined_query
    }

def grade_documents(state: State):
    """
    => grader_chain를 통해
       {
         "사용자의 쿼리": "...",
         "사용자의 요구사항을 해결할 내용이 있는지": "...",
         "criterion_score": 7
       }
       이런 JSON이 반환.
    """
    import json
    from .grading import grader_chain
    docs = state.get("documents", [])
    question = state.get("refined_query", "")

    scored_docs = []
    for doc in docs:
        excerpt = doc.page_content
        # 1) LLM 호출
        res = grader_chain.invoke({
            "doc_excerpt": excerpt,
            "question": question
        })
        try:
            # 2) JSON 파싱
            data = json.loads(res.content)
            # ex) {"사용자의 쿼리": "...", "사용자의 요구사항을 해결할 내용이 있는지": "...", "criterion_score": 7}
            score = data.get("criterion_score", 0)
            scored_docs.append((doc, score))

            # (원한다면) state["messages"] 등에 기록할 수도 있음
            # user_query_text = data.get("사용자의 쿼리", "")
            # doc_satisfaction = data.get("사용자의 요구사항을 해결할 내용이 있는지", "")
            # logger.info(f"[grade_doc] user_query={user_query_text}, doc_sat={doc_satisfaction}, score={score}")

        except:
            # JSON 파싱 실패 => score=0
            scored_docs.append((doc, 0))

    # 3) 점수 높은 순 상위 3개
    sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:3]
    final_docs = [x[0] for x in sorted_docs]
    return {"documents": final_docs}

def filter_documents(docs, question):
    from .grading import grader_chain
    import json

    scored = []
    for doc in docs:
        excerpt = doc.page_content
        res = grader_chain.invoke({"doc_excerpt": excerpt, "question": question})
        try:
            data = json.loads(res.content)
            score = data.get("criterion_score", 0)
            scored.append((doc, score))
        except:
            scored.append((doc, 0))

    sorted_docs = sorted(scored, key=lambda x: x[1], reverse=True)[:3]
    final_docs = [x[0] for x in sorted_docs]
    return final_docs

def generate(state: State):
    from .generation import generator_chain
    from langchain.schema import AIMessage

    docs = state.get("documents", [])
    if not docs:
        return {"messages": [AIMessage(content="No relevant documents found.")]}

    doc_str = format_docs(docs)
    question = get_latest_human(state["messages"])
    prompt = f"""
    사용자 질문: {question}
    문서의 출처는 밝혀라
    {doc_str}
    """
    res = generator_chain.invoke({"question": question, "context": prompt})
    return {"messages": [AIMessage(content=res.content)]}

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
    question = get_latest_human(state["messages"])
    
    prompt = f"""
    사용자가 운동과 관련한 질문을 하였을 때 공신력 있는 사이트에서 정보를 가져와 대답하십시오.
    신뢰성,전문성,공신력이 뒷받침되는 자료이여야 합니다.
    예를 들어 사용자가 운동 관련 질문을 할경우 NCAA와 같은 검증된 사이트에서 자료를 가져와 대답하는 것입니다.
    사용자 질문: {question}
    
    검색 쿼리:
    """
    refined_query = llm.invoke([HumanMessage(content=prompt)]).content.strip()
    print(f"Refined search query: {refined_query}")
    
    web_search_tool = TavilySearchResults(k=3)
    docs = web_search_tool.invoke(refined_query)
    
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

    rewrite_count = state.get("rewrite_count", 0)

    if not documents:
        rewrite_count += 1
        state["rewrite_count"] = rewrite_count
        print(f"     --- [ALL DOCUMENTS ARE NOT RELEVANT, rewrite #{rewrite_count}] ---")

        if rewrite_count >= 3:
            print("--- [REWRITE LIMIT REACHED => GO TO WEB_SEARCH] ---")
            return "web_search"
        else:
            return "rewrite_query"
    else:
        print("     --- [DECISION: GENERATE] ---")
        state["rewrite_count"] = 0
        return "generate"
    
def grade_generation_v_document_question(state: State):
    print("--- [GRADE GENERATION vs DOCUMENT QUESTION] ---")
    from grading import hallucination_checker_chain, answer_grader_chain

    question = get_latest_human(state["messages"])
    generation = get_latest_ai(state["messages"])

    if state.get("source") == "web_search":
        print("--- [SKIPPING HALLUCINATION CHECK FOR WEB SEARCH] ---")
        state["messages"].append(
            AIMessage(content="(안내) 문서 대신 웹 검색 결과를 참고해 답변했습니다. 필요 시 추가 검색 가능합니다.")
        )
        return "useful"

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
            return "useful"
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

def filter_node(state: State):
    from .grading import grader_chain  # grader_chain은 앞서 예시 grader_chain과 동일
    docs = state.get("documents", [])
    question = state.get("refined_query", "")

    filtered_docs = filter_documents(docs, question)  # 위에서 작성한 함수 호출
    # state에 최종 문서만 저장
    return {"documents": filtered_docs}

# --- 그래프 구성 ---
flow = StateGraph(State)
flow.add_node("retrieve", retrieve)
flow.add_node("generate", generate)
flow.add_node("rewrite_query", rewrite_query)
flow.add_node("web_search", web_search)
flow.add_node("direct_generate", direct_generate)
flow.add_node("interaction", interaction)
flow.add_node("grade_documents", grade_documents)
flow.add_node("filter_node", filter_node)  # 새로 추가

# 엣지 설정
flow.add_edge(START, "interaction")

flow.add_conditional_edges(
    "interaction",
    check_query_status,
    {
        True: "retrieve",
        False: END
    }
)

flow.add_edge("retrieve", "grade_documents")
flow.add_edge("grade_documents", "filter_node")
flow.add_edge("filter_node", "generate")
flow.add_edge("generate", END)

memory = MemorySaver()
graph = flow.compile(checkpointer=memory)

def stream_graph(inputs, config):
    final_output = None
    for output in graph.stream(inputs, config, stream_mode="updates"):
        # 중간 output은 무시
        final_output = output
    return final_output
