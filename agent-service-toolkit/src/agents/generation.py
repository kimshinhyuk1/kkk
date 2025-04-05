# generation.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

template = """Your end goal is to convey information to the user by **keeping the original content intact**, but **making it more readable and clearly indicating what is lacking**.
 In the grade_doc node, you will receive up to 3 documents (priority #1, #2, #3) that have been curated based on the user's needs (problem) and why, specific actions, scientific rationale, and some 'partially met' or 'lacking' criteria.
Final Deliverable (Structure)

Structure is 
[Rationale for document selection]Focus:5%
[Provide solutions in the document that address the requirements] (most important item)Focus:70%**
[summarize gaps and follow-up suggestions]Focus:5%
[Solution to user requirements in #2-3 documents / brief summary] focus:20%.
It is organized as follows


**** What is most important and what you should focus on when generating your answer is the [Provide a solution in the document that addresses the requirement] part. Keep the focus at 70%.

 **[Rationale for document selection]**.  
   - Briefly mention “why this article (#1) is the most helpful in addressing your needs”, etc.

**[Provide a solution in the article that addresses the need] (most important item)**.  
    Provide a workaround that addresses the user's need.
    Your solution must be accompanied by supporting evidence and examples, so please use the guidelines below as a guide when answering. If you have supporting evidence or examples, present all information without omitting anything. Be specific and focus your answer and rationale on implementing.
     -Relevant scientific evidence (experiments, EMG data, expert opinion, etc.),  
     -specific examples 
     - Expert advice
       Include all specific supporting evidence within the documentation
  [summary of gaps and follow-up suggestions].
   - Overall, “This document provides some answers but falls short in [some] areas. Please refer to [additional resources or experts] if needed.”
   - Suggest a follow-up, such as “If users need more detailed exercise routines, we recommend checking external resources or other internal documentation.”

  [How to address user needs in #2-3 documentation / brief summary]
   -- Do the same for priority #2 and #3 documents, emphasizing the key points that actually contribute to addressing the user's need first, and only briefly mentioning the gaps.


--
The final deliverable (structure)

Structure. 
[rationale for document selection]Focus:5%
[Provide a workaround in the document that addresses the need] (the most important item)Focus:70%**
[Summarize gaps and follow-up suggestions]Focus:5%
[Solution to user requirements in #2-3 documents / brief summary] focus:20%.
It is organized as follows


**** What is most important and what you should focus on when generating your answer is the [Provide a solution in the document that addresses the requirement] part. Keep the focus at 70%.

 **[Rationale for document selection]**.  
   - Briefly mention “why this article (#1) is the most helpful in addressing your needs”, etc.

**[Provide a solution in the article that addresses the need] (most important item)**.  
    Provide a workaround that addresses the user's need.
    Your solution must be accompanied by supporting evidence and examples, so please use the guidelines below as a guide when answering. If you have supporting evidence or examples, present all information without omitting anything. Be specific and focus your answer and rationale on implementing.
     -Relevant scientific evidence (experiments, EMG data, expert opinion, etc.),  
     -specific examples 
     - Expert advice
       Include all specific supporting evidence within the documentation
  [summary of gaps and follow-up suggestions].
   - Overall, “This document provides some answers but falls short in [some] areas. Please refer to [additional resources or experts] if needed.”
   - Suggest a follow-up, such as “If users need more detailed exercise routines, we recommend checking external resources or other internal documentation.”

  [How to address user needs in #2-3 documentation / brief summary]
   -- Do the same for priority #2 and #3 documents, emphasizing the key points that actually contribute to addressing the user's need first, and only briefly mentioning the gaps.


-.
### Key guidelines

1) **Don't change the content**.
   - If something is missing or only briefly mentioned in the document, don't arbitrarily fill in or add it.
   - Be honest and state 'this documentation does not specifically describe [something]'.

2) **Indicate what is lacking (partially met, missing)**.
   - If the original article lacks concrete examples or scientific evidence, simply state something like 'no specific routines are mentioned' or 'there are only expert opinions and no statistics'.
   - If the topic is not mentioned at all, say 'the original article does not address this topic'.

3) **Mention additional material or another article (documentation)** (optional).
   - If the gap is important to the user, you can recommend “You may want to refer to a supplementary article or other material”.
   - Example: “If you're looking for a more specific workout routine, you might want to check out these additional resources.

4) **Complement each other**.
   - If you've received multiple documents, give the user a short summary of the pros and cons of each document.
   - Example: “Document #1 has specific solutions, but lacks evidence; Document #2 has a clear scientific basis, but is vague on how to implement.”

5) **Utilize Table-List-Markdown**.
   - Clearly list key information in the form of a table or list: 'problem solved / how to implement / scientific evidence / gaps'.
   - Example:
     ```
     ### Documentation key items
     - Problem solving direction: OOO
     - Specific implementation method: None (not mentioned in the original article)
     - Scientific basis: 1 expert opinion (no additional data)
     - Gaps: No clinical studies or large-scale statistics
     ```
   - This makes it easy for users to see where they are lacking.

6) **Document #1 is the most detailed (the core document), while documents #2 and #3 are more to the point**.
   - First, organize **plenty** of information about your #1 article (problems solved, how it works, rationale, how it's different).
   - Then, **briefly** describe how the second- and third-ranked articles also help with user requirements, if they are relevant,  
     - Be concise when mentioning 'gaps'.  
     - Emphasize the highlights and benefits first, and then mention them **simply**, like “but there are no specific examples”.

7) **Encourage users to make their own choices.
   - If there is contradictory information in your documentation, or if it is generally insufficient, you can suggest “further research” or “consult an expert”.
   - Guide users to find more information specific to their situation.

---.


Question: {question}
Context: {context}
Answer:
"""

generate_prompt = ChatPromptTemplate([("human", template)])
llm_generator = ChatOpenAI(model="gpt-4o", temperature=0)

# 정의한 체인: 생성 프롬프트 → LLM 호출
generator_chain = generate_prompt | llm_generator