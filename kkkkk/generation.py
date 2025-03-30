# generation.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

template = """You're an assistant with the task of answering questions. Answer the question using the following context from your search. Base your answer on vectorstore to the best of your ability.
When answering, be sure to credit which searched context you used. Answer in your own language..
Question: {question}
Context: {context}
Answer:"""

generate_prompt = ChatPromptTemplate([("human", template)])
llm_generator = ChatOpenAI(model="gpt-4o", temperature=0)

# 정의한 체인: 생성 프롬프트 → LLM 호출
generator_chain = generate_prompt | llm_generator
