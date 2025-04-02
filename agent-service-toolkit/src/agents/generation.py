# generation.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

template = """Your end goal is to convey information to the user by **keeping the original content intact**, but **making it more readable and clearly indicating what is lacking**.
 In the grade_doc node, you will receive a document that has been curated based on the user's needs (problem) and why, specific actions, scientific rationale, and some ‘partially met’ or ‘lacking’ criteria.  

### Key guidelines

1) Do not **alter the content**.  
   - If a document is **missing** or **only briefly mentions** something, do not arbitrarily fill in or add to it.  
   - Be honest and say something like, ‘This documentation doesn't make it clear how to do [something].’

2) **Indicate what is lacking (partially met, missing)**.  
   - If the original article lacks specific practices or scientific evidence, briefly state what it lacks, such as ‘no specific routine is mentioned’ or ‘only expert opinion, no statistics’.  
   - If the topic is not mentioned at all, write ‘The original article does not cover this topic’.

3) **Mention additional material or other articles (optional)**.  
   - If the gap is a key concern for the user, you can say something like ‘We recommend that you refer to the secondary documentation or other resources’.  
   - Example: ‘If you're looking for more specific workout routines, check out our separate article.’

4) **Complement each other**.  
   - If you receive multiple documents together and they have different strengths/weaknesses, provide a short summary to guide the user, such as ‘Document A has specific solutions but lacks evidence, Document B has solid evidence but lacks implementation’.  
   - The user can then look at this information and dig deeper where needed.

5) **Utilise Table-List-Markdown  
   - Clearly organise the **key items in a table or list**, such as ‘how to solve the problem / how to implement / scientific evidence / what's lacking’.  
   - Example:  
     ```
     ### Documentation key items
     - Direction to solve the problem: OOO  
     - Specific implementation method: None (not mentioned in the original text)  
     - Scientific evidence: 1 expert case (no additional data)  
     - What is lacking: no clinical studies or large-scale statistics
     ```
   - This way, the user can see right away what's lacking.

6) **Focus on documentation for #1** & **Brief information for #2-4**.  
   - First, organise the information for the #1 article in detail (problem solved, how it works, rationale, gaps).  
   - For priority 2-4 articles, write a **short summary** (1-2 sentences) that gets to the point, but also notes the differences, such as ‘the solution is more specific, the rationale is richer,’ etc.

7) **Encourage users to make their own choices  
   - Finally, if there is insufficient or contradictory information, you can suggest ‘further research’ or ‘consult an expert’.  
   - This will allow the user to refer to more resources for their own context.

### Example final output

1. **[Rationale for document selection]**  
   - 'This document was determined by grade_doc to be relevant to address your needs because (deleted)...'

2. **** [Provide specifics of the document that addresses your need] (this is the most important point)

3. **[Primary solution: details of the #1 document]** 2.  
   - “Solution overview, action steps, references, gaps, etc.

4. **[One-line summary of other documentation (2nd-4th priority)]** 'Summary of other documentation...' 3.  
   - 'Second-ranked document: ...'  
   - 'Third-ranked document: ...'  
   - '4th ranked document: ...'  

5. **[Gap summary and follow-up suggestions]** [Gaps and suggestions for improvement  
   - 'This document does not provide specific routines, so it is recommended to refer to other documents or resources.'  

Follow these guidelines **without forcing or modifying the original content**, but be transparent about gaps and allow users to see the most important information at a glance.


Translated with DeepL.com (free version)

답변은 사용자의 쿼리와 동일한 언어로 해라

Question: {question}
Context: {context}
Answer:"""

generate_prompt = ChatPromptTemplate([("human", template)])
llm_generator = ChatOpenAI(model="gpt-4o", temperature=0)

# 정의한 체인: 생성 프롬프트 → LLM 호출
generator_chain = generate_prompt | llm_generator
