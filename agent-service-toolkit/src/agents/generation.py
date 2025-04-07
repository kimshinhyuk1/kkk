# generation.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

template = """
그리고 마지막에 출처는 반드시 포함해
---.


Question: {question}
Context: {context}
Answer:
"""

generate_prompt = ChatPromptTemplate([("human", template)])
llm_generator = ChatOpenAI(model="gpt-4o", temperature=0)

# 정의한 체인: 생성 프롬프트 → LLM 호출
generator_chain = generate_prompt | llm_generator