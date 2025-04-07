# generation.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

template = """Your end goal is to convey information to the user by **keeping the original content intact**, but **making it more readable and clearly indicating what is lacking**.
In the grade_doc node, you receive up to 3 documents (ranked #1, #2, #3) that address the user's need.

사용자의 요구사항을 해결해주는 문장만 실어라
반드시 구체적으로 사용자의 요구사항을 해결해주는 내용에 초점을 두어야 한다.
실제로 문서에서 다루고 있는 내용만을 가져와 사용자에게 제공하라
답변은 한국어로 하고 문서의 어디부분에서 어디를 인용한 결과를 꼭 붙여라
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