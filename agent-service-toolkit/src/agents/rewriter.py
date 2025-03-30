# rewriter.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

system = """
더 나은 question을 위해 영어로의 변환도 고려해주세요./re-write 과정을 사용자가 당황하지 않도록 질문 재작성 중 이라는 문구로 나타내세요. 
"""
re_write_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
question_rewriter = re_write_prompt | llm | StrOutputParser()
