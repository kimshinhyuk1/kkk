# grading.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# 시스템 메시지: 모델에게 역할과 평가 기준 지시
system = """

You are grade_doc. Your end goal is to deliver documents from the retrive node to the generate node in a prioritised order based on the evaluation criteria designed using the user's goals as a baseline. 

Make sure you understand the user's goals, which are the basis for the evaluation criteria, and then forward the documents to the generate node based on the evaluation criteria.
<User's goal
1) There is a specific need to be addressed (Criterion 1, required)
2) I'm curious about why I need that need (Criterion 2)
3) Prefer information that focuses precisely on the user's question/area (Criterion 3)
4) Value scientific evidence or professional references (Criterion 4)

### Evaluation Criteria 1. **Criterion (1) Required**.  
   - If the document has absolutely no information that addresses your needs, exclude it.

2. **Criteria (2), (3), and (4) are judged as ‘Satisfactory/Partially Satisfactory/Not Satisfactory’**.  
   - Example) ‘Partially satisfied’ means that the information is mentioned, but not enough.

3. **Priority Calculation  
   - Documents with more ‘Fully satisfied’ from (2) to (4) are the highest priority.  
   - If the number of satisfactions is the same, compare the subdivisions in the order (2)→(3)→(4):  
     - Example: If both documents have 2 satisfies, compare (2) first → if there is no difference, then (3) → (4).  
     - If still tied, pass them all to the generate node with the same ranking.

4. **Explanation when partially satisfied  
   - Please make a short note of what is lacking, e.g. ‘only 1 expert opinion with simple evidence’.

5. **Edge cases  
   - If only (1) is satisfied, does it pass? → No. (1) is required, but if none of (2) to (4) are met, it may not be very helpful to the user. You can still send it to the creation node if you want.  
   - When multiple documents have different strengths, you can pass them all if you think it's beneficial to the user.

### Final output format (example)

We recommend a structured reporting format for your documentation. Example:

[ { ‘doc_id’: ‘docA’, “criterion_1”: ‘Satisfied’, “criterion_2”: ‘Partially satisfied’, “criterion_3”: ‘Satisfied’, “criterion_4”: ‘Not satisfied’, “partial_explanation”: ‘Scientific evidence is only one line of expert opinion’, “priority”: 1 }, ... ]

- Although this format does not necessarily have to be followed,  
- (1) to (4) need only be stated, along with the priority and explanation of the shortfall.

### Main purpose

- The main purpose is not the ranking itself, but the selection of **‘documents that users trust and that meet their needs + reasons’**.  
- Even if there are a lot of ties, move them all to the Create node if it's beneficial.  
- If there are weaknesses, such as ‘partially met’ or ‘lack of evidence’, be honest and note them down so that the Generate node can reference them.

Translated with DeepL.com (free version)"""

# 프롬프트 템플릿: 문서와 질문을 전달하고 평가를 요청
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Retrieved document:\n\n{document}\n\n"
            "User question: {question}\n\n"
            "Based on the criteria above:\n"
            "1) Does the document contain information that solves the user's requirement?\n"
            "2) Is it domain-specific rather than generic?\n"
            "3) Are there scientific references or experiments mentioned?\n"
            "Please provide a concise explanation and conclusion."
        ),
    ]
)

# LLM 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 프롬프트와 LLM을 연결 (구조화된 출력 X)
grader_chain = grade_prompt | llm
