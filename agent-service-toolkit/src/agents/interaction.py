from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain

# -----------------------------
# 1) 시스템 템플릿 (system) - 맥락 이해 강화
# -----------------------------
interaction_template = """As the IZ node, you will need to interact with the user to refine “why they want this information” or “what nutrient information they want”.  

Important: Be sure to remember and reference the context of your conversation with the user and previous conversations. Keep the conversation consistent by connecting to what the user has previously mentioned, their goals, and their concerns.

Once you've identified 'why they want to get this information' or 'what weight training need they're trying to solve', use that as your final question.
However, keep in mind the following
Most importantly, keep the user's context in mind and don't exhaust them.
2) **Only ask questions when there is a big need (goal) and the 'why' is vague**.  
   - If it's unclear why the user wants to get this information or what their goal (need) is for the information, ask a question or two to clarify.  


3) **Don't flesh out the details**.  
   - If the big picture (purpose + why) is already clear, don't ask for **details** like barbell position or number of sets.  
   - However, if the user does ask for details, be sure to include them in your final question.

4) Ask **at most two additional questions**.
   - Even if the user's question is vague, you should be able to get to the core needs and reasons with two follow-up questions.  
   - Don't ask more questions than that to avoid user fatigue.

5) **Ask questions as soon as the user's context is clear**.
   - This is essential if the (what + why) is already clear: “To finalize your query: [user's clear intent]” and exit the IZ node.
   - If the user gives a clear goal, immediately say “Finalize final query: [goal]”.
   - When no further questions are needed, always start with “I am finalizing the final query”.

6) Include detailed requirements in your final query.
   - If there are any **detailed questions** or **additional requirements** that the user has raised during the conversation, be sure to include them in your **final query**.



### Summary
### Summarize
  
- Only ask up to 2 questions to gain clarity if the big goal/reason is unclear.  
- Don't ask for details, but if the user mentions them, include them in the final question.  
- End immediately if **clear** by saying, “To finalize your final query: [clarify your intent].”
- Always reference and maintain context with previous conversations to provide consistent responses.** **Allow the user to respond in a way that is consistent.

Following these prompts, you (the IZ node) should only add questions when it is unclear from the larger context **what and why** the user wants to know about nutrient information, and **immediately confirm the final query** when clarified. Review carefully The criteria for review is whether the key roles of the IZ node are clearly assigned Whether the LLM is clearly set up to identify and fulfill the core requirements so that the user understands what they need and why they are asking for it Whether the query is clear enough to allow the user to handle high probability situations well Add a subjective review to ensure that the query is clear enough to handle high probability situations.

사용자와의 상호작용을 마치면 반드시 최종 쿼리를 확정합니다 라는 멘트를 넣어라
모든 대답은 한국말로 해라

이전 대화에서 최종 쿼리를 확장 한 후 정보를 전달했다면 이전 대화는 더이상 맥락에 넣지 말아라.
"""

system_prompt = SystemMessagePromptTemplate.from_template(interaction_template)

# -----------------------------
# 2) 휴먼 템플릿 (human)
# -----------------------------
human_template = """{question}"""

human_prompt = HumanMessagePromptTemplate.from_template(human_template)

# -----------------------------
# 3) ChatPromptTemplate 통합 - 메시지 기록 포함
# -----------------------------
interaction_prompt = ChatPromptTemplate.from_messages([
    system_prompt,
    MessagesPlaceholder(variable_name="chat_history"),  # 대화 기록 추가
    human_prompt
])

# -----------------------------
# 4) LLM 및 메모리 설정 - ConversationSummaryBufferMemory 사용
# -----------------------------
llm_interaction = ChatOpenAI(
    model="gpt-4o",
    temperature=0.2  # 다양성 약간 허용
)

# 요약 메모리 사용 - 대화 맥락 요약하여 유지
memory = ConversationSummaryBufferMemory(
    llm=llm_interaction,  # 요약을 위한 LLM 설정
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=1000  # 토큰 제한 설정
)

# -----------------------------
# 5) LLMChain 구성
# -----------------------------
interaction_chain = LLMChain(
    prompt=interaction_prompt,
    llm=llm_interaction,
    memory=memory,
    verbose=True  # 디버깅 용도로 상세 출력
)

# -----------------------------
# 6) 맥락 추적 클래스 추가
# -----------------------------
class ContextTracker:
    def __init__(self):
        self.exercise_type = None  # 운동 종류
        self.user_goal = None      # 사용자 목표
        self.special_conditions = None  # 특별 상황(부상 등)
        self.answered_questions = []    # 이미 답변한 질문들
    
    def update_context(self, user_input, response):
        """사용자 입력과 응답을 분석하여 맥락 정보 업데이트"""
        # 실제 구현에서는 NLP 또는 LLM을 사용하여 맥락 정보 추출
        # 간단한 구현을 위해 키워드 기반 추출만 예시로 함
        exercise_keywords = {
            "스쿼트": "스쿼트", "squat": "스쿼트",
            "벤치프레스": "벤치프레스", "bench press": "벤치프레스",
            "데드리프트": "데드리프트", "deadlift": "데드리프트"
        }
        
        goal_keywords = {
            "근력": "근력 향상", "힘": "근력 향상", "강화": "근력 향상",
            "체중": "체중 관리", "감량": "체중 관리", "다이어트": "체중 관리",
            "자세": "자세 교정", "폼": "자세 교정", "form": "자세 교정"
        }
        
        # 키워드 검색을 통한 운동 유형 및 목표 추출
        for keyword, value in exercise_keywords.items():
            if keyword in user_input:
                self.exercise_type = value
                break
                
        for keyword, value in goal_keywords.items():
            if keyword in user_input:
                self.user_goal = value
                break
        
        # 이미 답변한 질문 기록
        self.answered_questions.append(user_input)
        
        return {
            "exercise_type": self.exercise_type,
            "user_goal": self.user_goal,
            "special_conditions": self.special_conditions,
            "answered_questions": self.answered_questions
        }

# 맥락 추적 객체 생성
context_tracker = ContextTracker()

# -----------------------------
# 7) 실행 함수 - 맥락 추적 기능 통합
# -----------------------------
def run_interaction(user_input: str):
    """사용자로부터 입력받아 IZ 노드 응답을 생성하고 맥락 추적"""
    
    # 현재 상태의 맥락 정보 가져오기
    current_context = context_tracker.update_context(user_input, "")
    
    # 맥락 정보를 포함한 강화된 질문 생성
    enhanced_question = f"{user_input}\n\n[맥락 정보: 운동={current_context['exercise_type']}, 목표={current_context['user_goal']}]"
    
    # LLM에 강화된 질문 전달
    response = interaction_chain.run(question=enhanced_question)
    
    if current_context['exercise_type'] and current_context['user_goal']:
        # 운동 종류와 목표가 모두 명확하면 쿼리 자동 확정
        response = f"최종 쿼리를 확정합니다: {current_context['exercise_type']}을(를) 통해 {current_context['user_goal']}을(를) 원합니다. {response}"
    
    # 응답을 기반으로 맥락 정보 업데이트
    context_tracker.update_context(user_input, response)
    
    return response

# -----------------------------
# 8) 테스트
# -----------------------------
if __name__ == "__main__":
    print("IZ 웨이트 트레이닝 상담사와의 대화를 시작합니다.")
    print("질문이나 고민을 입력하세요. '종료'를 입력하면 대화가 끝납니다.")
    
    while True:
        user_input = input("\n사용자: ")
        if user_input.lower() == '종료':
            print("대화를 종료합니다.")
            break
            
        response = run_interaction(user_input)
        print(f"\nIZ: {response}")