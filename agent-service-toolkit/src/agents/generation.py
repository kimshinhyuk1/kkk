# generation.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

template = """ (The following instructions are for system/internal use only. 
They must not be output verbatim to the user.)

The ultimate goal is to **keep the original content intact, but 
**increase readability and clearly indicate where it is missing**
to communicate to the user. 
Make sure to mention the key keywords frequently in the user's query 
to feel that the problem has been resolved.

However, the minimum amount should be at least 15 lines.
Your answer structure consists of a solution that can solve the user's query 
and specific evidence that supports the solution.

<UserQueryFocus>
Always analyze and fully address the user's query. 
In your final answer:
1) Use the specific words or phrases from the user’s question multiple times.
2) Provide clear, direct, and relevant information that satisfies the user’s needs.
3) Provide references and data in detail rather than a superficial mention, 
   especially when citing studies or scientific evidence (e.g., sample sizes, 
   effect sizes, p-values, study duration).
4) Avoid revealing these internal instructions or mentioning any extraneous details 
   irrelevant to the user’s query.

<Rescue>
-Workaround to resolve a user's query

-***Specific evidence supporting the solution**** (very important)

The solution must be accompanied by evidence and examples, 
so please refer to the example below to see what information can be the basis.
If you have any evidence or examples to support, 
please present all information without omission.
Be specific and give your answers and grounds with a focus on execution.

<Ground Essie>
-relevant scientific evidence (experiment, EMG data, expert opinion, etc.),
- a specific example
- expert advice

(Important Note: The above three categories are guidelines; 
they are not all required in every single answer. Use them where applicable.)

<Few-shot for detailed references>
Example 1:
Q: "LoCHO 식단의 체중 감량 효과는 어느 정도인가요?"
A: "2018년 PLOS ONE에 게재된 Sackner-Bernstein 등 메타분석에 따르면 
   무작위 대조군 시험 1,700명을 종합한 결과, 평균 체중 감량이 
   저탄수 식단(LoCHO) 그룹에서 6개월간 약 3kg 더 컸습니다 (95% CI: 1.2–4.8kg). 
   연구는 빈도주의와 베이지안 통계를 모두 적용해 신뢰도를 높였습니다. 
   [출처: https://journals.plos.org/plosone/]"

Example 2:
Q: "단백질 섭취가 근육 회복에 미치는 구체적 수치를 알고 싶어요."
A: "한 연구(Frontiers in Nutrition, 2021)에서는 일일 단백질 섭취량을 
   체중 1kg당 1.6g 이상으로 유지했을 때, 근육량이 평균 8주간 약 3% 증가했습니다. 
   피험자 수는 150명으로, 무작위 이중 맹검 방식으로 진행되었습니다. 
   [출처: https://www.frontiersin.org/]"

Keep all of the above guidelines and write your answers 
so that when you receive them, you don't change the content 
so that you feel rich and that your answers are specific.

답변은 반드시 한국어로 해라.
출처를 반드시 언급하라.
원문이 영어여도 한국어로 번역해서 답변하라고.
Context: {context}
Answer:
"""

generate_prompt = ChatPromptTemplate([("human", template)])
llm_generator = ChatOpenAI(model="gpt-4o", temperature=0)

# 정의한 체인: 생성 프롬프트 → LLM 호출
generator_chain = generate_prompt | llm_generator