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

당신은 “피트니스/웨이트 트레이닝 분야”에 특화된 **쿼리 리파이너**입니다.   *****먼저 반드시 사용자의 쿼리를 영어로 변환하십시오
사용자의 원본 질문을 입력으로 받고, 이를 검색엔진에서 사용자의 쿼리를 만족할 수 있는 쿼리로 변환합니다.

### 목표
1. **사용자가 언급한 특정 운동**(바벨컬 등)과 **대상 근육군, 목표(근비대, 심미성 등)**가 충분히 반영되도록 **최종 쿼리를 확장**시키세요.
2. **쿼리에서 불필요한 단어를 제거**하고, **핵심 키워드**에 집중하세요.
3. **최종 쿼리를 확정**할 때, **주요 키워드를 중복**해서 쓰되,  
   - “바벨컬, 바벨이두컬, straight bar curl, EZ curl, biceps curl” 같은 **유의어**를 적절히 포함하고,  
   - “근비대, 근조직, 근섬유, hypertrophy, sculpted arms” 같은 **연관 맥락**도 활용하여 **임베딩 검색**에서 사용자의 맥락을 반영하는 단어들을 통해 높은 점수를 얻을 수 있도록 만드세요.  
   - 단순히 “바벨컬 바벨컬 바벨컬”처럼 무의미한 반복보다는, **다양한 표현**과 **맥락**을 섞어서 자연스럽게 작성하세요.

### 형식
- **최종 출력**에는 **'개선된 쿼리'만** 담아주세요.  
- 다른 설명이나 해설은 기재하지 마세요.

---

[Few-Shot Examples]

1) **예시 입력**:
“I'm a first-year and go to the gym twice a week. I'd like to make my biceps more defined with barbell curls, what's the best way to do it?”

**예시 출력**:
“barbell curl, biceps curl, EZ bar curl, aesthetic biceps hypertrophy,aesthetic biceps,EZ bar,biceps curl,biceps curl,twice weekly workout,intermediate 1 year old,hypertrophy-focused,various barbell curl variations,myofibrillar development”

---

2) **예시 입력**:
“While weight training, do you think barbell curls or dumbbell curls are better for hypertrophy? I always want to get a better browline.”

**예시 출력**:
“barbell curl vs dumbbell curl, barbell curl vs dumbbell curl hypertrophy, barbell curl hypertrophy, dumbbell curl hypertrophy, barbell curl vs dumbbell curl, barbell curl dumbbell curl, biceps, biceps hypertrophy, biceps aesthetics, biceps hypertrophy, biceps, sculpted arms, aesthetics, isolation exercises, barbell biceps curl, EZ curl variations”

---

[실제 입력]

{user_question}

[지시사항]
반드시 사용자의 쿼리를 영어로 먼저 변환하십시오.
위 가이드라인(1~3번)을 충실히 이행하여 **최종 검색 쿼리**를 **자연스럽게** 작성하세요.
본인이 작성해야 할 것은 **최종 쿼리만**입니다. 다른 설명은 넣지 마세요.

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
    
    # 검색할 문서 수 제한
    max_docs = 50
    # 하이브리드 검색기를 통해 문서 검색
    documents = hybrid_retriever.invoke(refined_query)
    
    # 검색 결과 중 상위 문서 선택
    top_documents = documents[:max_docs] if len(documents) > max_docs else documents
    
    logger.info(f"검색된 문서 수: {len(documents)}, 사용할 문서 수: {len(top_documents)}")
    
    # 로깅: retrieve -> grade_documents 로 넘어가는 문서
    logger.info("--- [RETRIEVE -> GRADE_DOC] Selected Documents ---")
    for i, doc in enumerate(top_documents):
        snippet = doc.page_content[:100].replace("\n", " ")
        source = doc.metadata.get("source", "Unknown Source")
        logger.info(f"  [{i+1}] source={source}, snippet={snippet}...")
    logger.info("-------------------------------------------------\n")
    
    return {
        "documents": top_documents,
        "refined_query": refined_query
    }

def grade_documents(state: State):
    """
    => grading.py의 grader_chain 사용
       c1=0..10, c2=0..5, c3=0..5 => total_score로 정렬
    """
    import json
    from .grading import grader_chain
    docs = state.get("documents", [])
    question = state.get("refined_query", "")

    scored_docs = []
    for doc in docs:
        excerpt = doc.page_content
        res = grader_chain.invoke({"doc_excerpt": excerpt, "question": question})
        try:
            data = json.loads(res.content)
            # data 예) {"criterion_1_score":0,"criterion_2_score":5,"criterion_3_score":5,"total_score":10}
            score = data.get("total_score", 0)
            scored_docs.append((doc, score))
        except:
            # JSON 파싱 실패 => score=0
            scored_docs.append((doc, 0))

    # 점수 내림차순 정렬 => 상위 3개
    sorted_docs = sorted(scored_docs, key=lambda x: x[1], reverse=True)[:3]
    final_docs = [x[0] for x in sorted_docs]
    return {"documents": final_docs}

def filter_documents(docs, question):
    """
    (optional) 'filter_node'에서 추가 필터링
    """
    from .grading import grader_chain
    import json

    scored = []
    for doc in docs:
        excerpt = doc.page_content
        res = grader_chain.invoke({"doc_excerpt": excerpt, "question": question})
        try:
            data = json.loads(res.content)
            sc = data.get("total_score", 0)
            scored.append((doc, sc))
        except:
            scored.append((doc, 0))

    # 내림차순 -> 상위3
    sorted_docs = sorted(scored, key=lambda x: x[1], reverse=True)[:3]
    final_docs = [x[0] for x in sorted_docs]
    return final_docs
def generate(state: State):
    """
    => 최종 답변
    """
    from .generation import generator_chain
    from langchain.schema import AIMessage

    docs = state.get("documents",[])
    if not docs:
        return {"messages":[AIMessage(content="No relevant documents found.")]}

    doc_str = format_docs(docs)
    question = get_latest_human(state["messages"])
    prompt = f"""
    사용자 질문: {question}
    아래 문서(최대3개)를 참고해 답변:
    {doc_str}
    그리고 마지막에 출처는 반드시 포함해

    """
    res = generator_chain.invoke({"question": question, "context": prompt})
    # 최종 사용자에게 노출되는 메시지
    return {"messages":[AIMessage(content=res.content)]}

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
