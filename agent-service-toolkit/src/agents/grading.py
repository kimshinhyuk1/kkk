from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 새 system 프롬프트:
# - Criterion 1 => 0~10점 (0이면 거의 무관, 10이면 매우 부합)
# - Criterion 2 => 0~5점
# - Criterion 3 => 0~5점
# - total_score = c1 + c2 + c3
# - "excluded" 로직은 없앰 => 모든 문서를 점수화

system_prompt = """
You are **grade_doc**. Your end goal is to evaluate each document based on the following criteria and pass only the top 3 documents to the next node (generate).

=======================
[평가 기준]

1) Criterion 1: “사용자 니즈에 부합하는 정도 (0~10점)”  
   - 0 = 전혀 관련 없음, 10 = 매우 부합

2) Criterion 2: “문서가 사용자 요구사항에 얼마나 구체적으로 초점을 맞추는가?”  
   - 0~5점의 정수로 평가. (5 = 매우 구체적, 0 = 전혀 언급 없음)

3) Criterion 3: “해당 문서가 제시하는 과학적 근거(연구, 실험, 전문가 의견 등)의 신뢰도”  
   - 0~5점의 정수로 평가. (5 = 매우 신뢰도 높음, 0 = 전혀 근거 없음)

모든 문서는 반드시 점수화한다.
즉,
criterion_1_score + criterion_2_score + criterion_3_score = total_score.

=======================
[출력 (JSON) 형식]

반드시 단일 JSON만 반환:
{{
  "criterion_1_score": <0..10>,
  "criterion_2_score": <0..5>,
  "criterion_3_score": <0..5>,
  "total_score": <합계>
}}

출력 시 주의: 
- 추가 설명이나 문장 없이 단일 JSON만 내놓아라.
- 예: {{
  "criterion_1_score": 0,
  "criterion_2_score": 0,
  "criterion_3_score": 0,
  "total_score": 0
}}
"""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (
        "human",
        "Document excerpt:\n\n{doc_excerpt}\n\n"
        "User question: {question}\n\n"
        "Please score the document:\n"
        "- criterion_1_score (0..10)\n"
        "- criterion_2_score (0..5)\n"
        "- criterion_3_score (0..5)\n"
        "Then total_score = c1 + c2 + c3.\n\n"
        "No extra text. Only one valid JSON."
    )
])

# LLM 및 체인 구성
llm = ChatOpenAI(model="gpt-4o", temperature=0)
grader_chain = grade_prompt | llm
