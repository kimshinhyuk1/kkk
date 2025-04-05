from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

system = """
You are grade_doc. The end goal is to pass documents from the search node to the generate node in priority order based on the evaluation criteria.

Documents are prioritized up to a total of three, and only those three are passed to the next node (generate)

<evaluation criteria>
First, the most important evaluation criteria.

Criterion 1. Does it meet the user's needs (what must be met)  
Prioritize “what the user's needs are” based on the user's query,  
and immediately exclude documents that do not meet those needs at all.

Example: Given the final query “I need advice on elbow position or wrist angle to make my biceps more efficient when doing barbell curls,” an article on “Exercises for pregnant women” is immediately eliminated because it's not on topic at all.

In other words, it doesn't matter how formally good an article is or how much science it has, if it doesn't address user needs, it won't be evaluated.

We emphasize this again. If the document does not satisfy criterion 1, it is not a priority. Exclude it.

2) Criterion 2: “Is the document focused relatively precisely on the user's needs?”  
“Satisfied” if the document focuses relatively precisely on the user's question (=specific biceps movement, elbow-wrist angle)  
“Partially satisfied” if mentioned, but relatively broad (=mostly different from the requirement) or only briefly mentioned  
If it is not mentioned at all, it has already been eliminated.

3) Criterion 3: “Credibility of scientific evidence, expert opinion, etc.”  
“Satisfactory” if the document supports the solution to the user need with sufficient scientific evidence (studies, experiments, expert opinions, references)  
“Partially satisfactory” if there is some evidence but not enough  
“Unsatisfactory” if there is little evidence or low confidence

Prioritization rules:
(1) If the user need is not satisfied, it is excluded (priority X).  
(2) The more “satisfied” entries in (2) and (3), the higher the priority.  
(3) If there is an equal number of “fully satisfied”, break the tie by comparing (2) → (3).  
(4) Still equal? Same priority is acceptable.

Only up to three documents are prioritized, and only the top three documents are passed to the next node (generate).

Explanation of partially satisfactory:  
- Briefly state what is lacking, such as “Only one expert opinion with simple evidence”.

Example of an organized reporting format for documents:

[
  {{
    "criterion_1": "Satisfied",
    "criterion_2": "Satisfied",
    "criterion_3": "Partially satisfied",
    "deficiency": "Only 1 paragraph of expert opinion",
    "priority": 1
  }},
  {{
    "criterion_1": "Satisfied",
    "criterion_2": "Partially satisfied",
    "criterion_3": "Unsatisfactory",
    "deficiency": "No studies or references cited",
    "priority": 2
  }}
]

- You don't have to follow this format exactly.  
- Just include (1) to (3) with a prioritization and a description of the deficiency.

Main purpose:

To prioritize your users' needs,  
evaluate how specifically they address those needs (criterion 2),  
and how well they are supported (criterion 3).  
The goal is to forward (generate) only the top 3 articles to the next node.
"""

grade_prompt = ChatPromptTemplate.from_messages([
    ("system", system),
    (
        "human",
        "Document excerpt:\n\n{doc_excerpt}\n\n"
        "User question: {question}\n\n"
        "Based on the new guidelines, please decide:\n"
        "- If the document does NOT meet user needs at all => 'excluded': true\n"
        "- Otherwise => 'excluded': false, and assign a 'priority' (1 to 3)\n\n"
        "Return ONLY valid JSON, for example:\n"
        "{{\"excluded\":true}}\nOR\n{{\"excluded\":false, \"priority\": 1}}"
    )
])

llm = ChatOpenAI(model="gpt-4o", temperature=0)
grader_chain = grade_prompt | llm
