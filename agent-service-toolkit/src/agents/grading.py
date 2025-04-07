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

=======================
일단 가장 먼저 사용자의 요구사항이 무엇인지 말해.
[평가 기준]

1) Criterion 1: “해당 문서가 사용자 쿼리의 요구사항을 구체적으로 초점을 맞추어 요구사항을 해결해주는 문서인가? (0~10점)”  
   - 0 = 전혀 관련 없음, 10 = 매우 부합

2) Criterion 2: “문서가 사용자 요구사항에 얼마나 구체적으로 초점을 맞추는가?”  
   - 0~5점의 정수로 평가. (5 = 매우 구체적, 0 = 전혀 언급 없음)

3) Criterion 3: “해당 문서가 제시하는 과학적 근거(연구, 실험, 전문가 의견 등)의 신뢰도”  
   - 0~5점의 정수로 평가. (5 = 매우 신뢰도 높음, 0 = 전혀 근거 없음)

아래와 같이 예시를 주겠다. 예시를 참고하여 사용자의 맥락에 집중하는 방법을 익혀 답변하라라
사용자 쿼리
“Lat pulldown을 할 때 **등 하부(lats 하부 부위)**에 집중하고 싶다. 해당 운동 방법을 구체적으로 알고 싶다.”


후보 문서

Doc A: “Lower Lat Emphasis in Lat Pulldown”

‘광배근 하부’를 집중 공략하기 위한 그립, 상체 각도, 팔꿈치 위치 등 구체적인 디테일을 설명

세트·레프, 호흡법, 목표 근육 활성화 팁을 다룸

Doc B: “Common Lat Pulldown Mistakes”

Lat pulldown 시 흔히 저지르는 오류(너무 뒤로 젖히기, 어깨 과도한 내회전 등)

등 하부 특별 강조는 없고, 전반적인 주의사항 위주

Doc C: “Back Day Exercises Overview”

데드리프트, 바벨로우, Lat pulldown, 시티드로우 등 여러 등 운동을 간단히 소개

하부 광배근에 대한 별도 항목은 없음

Doc D: “Biceps Routine for Aesthetics”

바벨컬, 해머컬 등 이두근 운동 중심

Lat pulldown 언급 거의 없음

평가 시연

Doc A는 “등 하부 근육을 공략하는 Lat pulldown 방법”을 구체적으로 다루므로, 사용자 쿼리에 가장 부합 (가장 높은 점수 부여)

Doc B는 Lat pulldown을 다루지만, “오류·주의점” 중심이고 하부 근육 강조는 적음 (2번째 점수 부여)

Doc C는 등 전체 루틴을 폭넓게 언급하지만, 등 하부 초점이 부족 (3번째 점수 부여)

Doc D는 이두근 중심이라, 사용자 요구와 무관 (4번째 점수 부여여)



Few-Shot 예시 2
사용자 쿼리

“바벨컬(Barbell Curl)로 이두근(biceps brachii)을 강화하고 싶다. 정확한 자세와 손목 각도 등 세부 팁이 필요하다.”

후보 문서

Doc W: “Barbell Curl Form for Maximum Biceps Activation”

손목 각도별 자극 차이, 팔꿈치 고정법, 세트·레프, 호흡법 등 구체적인 방법 설명

EMG 데이터를 근거로 한 이두근 활성화 비교 표가 있음

Doc X: “General Arm Routine for Beginners”

이두 + 삼두 전반을 간단히 다룸 (바벨컬 언급 有)

단, 자세나 손목 각도 등 세부 내용은 부족

Doc Y: “Forearm and Grip Strength Exercises”

팔운동 중 ‘리버스컬, 손목 컬’ 위주로 다룸

바벨컬이 잠깐 언급되지만, 이두근 메인 포인트 아님

Doc Z: “Leg Day Essentials”

하체(스쿼트, 런지) 중심, 이두근 언급 없음

평가 시연

Doc W가 “바벨컬”과 “정확한 자세 & 손목 각도”를 가장 깊이 다뤄, 사용자 요구에 직접 부합 (가장 높은 점수 부여)

Doc X는 이두 운동을 언급하나 “포괄적 초보 루틴”이라 세부 팁은 부족 (2순위 점수 부여)

Doc Y는 전완근·그립 위주, 이두 근육 활성화와 직접 관련성 적음 (3순위 점수 부여여))

Doc Z는 전혀 무관 (4순위 점수 부여여)



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
- 추가 설명이나 문장 없이 단일 JSON및 문서의 이름을내놓아라.
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
