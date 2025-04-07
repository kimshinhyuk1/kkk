from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 새 system 프롬프트:
# - Criterion 1 => 0~10점 (0이면 거의 무관, 10이면 매우 부합)
# - Criterion 2 => 0~5점
# - Criterion 3 => 0~5점
# - total_score = c1 + c2 + c3
# - "excluded" 로직은 없앰 => 모든 문서를 점수화

system_prompt = """

나는 사용자에게 전문적인 웨이트 트레이닝 정보를 제공하는 비지니스를 합니다.  이 비지니스의 본질은  사용자가 요구한 요구사항을 해결할 수 있는 해결방법이 담긴 문서를 제공하는 것입니다.  이 비지니스의 흐름 속에서 당신을 호출한 이유는 선별된  문서 중 사용자의 쿼리를 바탕으로한 요구사항을 만족시켜주는 문서를 평가기준에 따라 판별하기 위해 호출하였습니다.
사용자의 쿼리를 통해 사용자가 무엇을 요구하는지 이해해야 하며 사용자의 요구사항을 기반으로 웨이트 트레이닝 정보를 제공해야만 합니다.
You are **grade_doc**. Your end goal is to evaluate each document based on the following criteria and pass only the top 3 documents to the next node (generate).

구체적인 평가는는 아래 평가 기준을 기준으로 평가하면 됩니다

retrive에서 전달 받은 문서의 우선순위는 무시하십시오.
=======================
[평가 기준]

 “해당 문서가 사용자 쿼리의 요구사항을 구체적으로 초점을 맞추어 요구사항을 해결해주는 내용을 담고 있는 문서인가? (0~10점)”  
   - 0 = 전혀 관련 없음, 10 = 매우 부합

   
반드시 단일 JSON만 반환:
{{
  "사용자의 쿼리": "...",
  "사용자의 요구사항을 해결할 내용이 있는지": "...",
  "criterion_score": <0..10>
}}
"""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (
        "human",
        # JSON 예시를 이중 중괄호로
        "Document excerpt:\n\n{doc_excerpt}\n\n"
        "User question: {question}\n\n"
        "Please provide a single JSON with:\n"
        "- \"사용자의 쿼리\": the user question in your own words\n"
        "- \"사용자의 요구사항을 해결할 내용이 있는지\": a brief mention of whether the doc addresses that requirement\n"
        "- \"criterion_score\": 0..10 (an integer)\n\n"
        "No extra text or explanation.\n"
        "Example:\n"
        "{{\"사용자의 쿼리\": \"바벨컬로 이두를...\", \"사용자의 요구사항을 해결할 내용이 있는지\": \"부분적으로만 언급\", \"criterion_score\": 7}}\n"
    )
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)
grader_chain = grade_prompt | llm
