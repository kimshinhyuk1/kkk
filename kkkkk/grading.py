# grading.py
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 문서 관련성 평가 체인
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documetns."""
    binary_score: Literal["yes", "no"] = Field(
        description="Document are relevant to the question, 'yes' or 'no'."
    )

system = """You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains semantic meaning related to the user question, grade it as relevant. 
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o", temperature=0)
llm_grader = llm.with_structured_output(GradeDocuments)

grader_chain = grade_prompt | llm_grader

# 생성 답변과 관련된 사실 확인 체인
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(
        description="Fact-based answers that refer to the given context, 'yes' or 'no'"
    )

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts.
/Make sure to apply the prompts entered in the generate node to get the generate."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_hallucination_checker = llm.with_structured_output(GradeHallucinations)

hallucination_checker_chain = hallucination_prompt | llm_hallucination_checker

# 최종 답변이 질문에 부합하는지 평가하는 체인
class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )

system = """You are a grader assessing whether an answer addresses / resolves a question 
Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""
answer_grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_answer_grader = llm.with_structured_output(GradeAnswer)
answer_grader_chain = answer_grade_prompt | llm_answer_grader
