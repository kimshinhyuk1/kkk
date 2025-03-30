# rewriter.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

system = """You a question re-writer that converts an input question to a better version that is optimized 
for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning.
re-write 과정에서 영어로의 변환도 고려하세요요
./re-write 과정을 사용자가 당황하지 않도록록 질문 재작성 중 이라는 문구로 나타내세요. 
"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
question_rewriter = re_write_prompt | llm | StrOutputParser()
