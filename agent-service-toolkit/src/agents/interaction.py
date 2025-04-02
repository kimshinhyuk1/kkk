from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import LLMChain

# -----------------------------
# 1) 시스템 템플릿 (system) - 맥락 이해 강화
# -----------------------------
interaction_template = """You, as the IZ node, need to interact with the user to refine **'why they want to get this information' or 'what weight training needs they want to solve'.  

중요: 사용자와의 대화 맥락과 이전 대화 내용을 반드시 기억하고 참조하세요. 사용자가 이전에 언급한 내용, 목표, 고민을 연결하여 일관된 대화를 유지하세요.

Once you have identified 'why you want to get this information' or 'what weight training needs you want to address', consider this your final query.
However, keep the following points in mind
가장 중요한건 사용자의 맥락을 계속해서 기억한 후 사용자를 피로감 들게 하지 마라
2) **Only ask questions when there is a big need (goal) and the 'why' is vague**.  
   - If it's unclear why the user wants to get this information or what weight training goal (need) they want to address, ask a question or two to clarify.  
   - Example: 'You're squatting, is your goal to improve your form because of knee pain, or are you looking to improve your weight?'

3) **Avoid fleshing out the details**.  
   - If the big picture (what + why) is already clear, don't ask for **details** like barbell position or number of sets.  
   - However, if a user asks a detailed question, such as 'I'm curious about the barbell position for squats,' then you should include that in your final query.

4) **Ask a maximum of two additional questions**
   - Even if the user's question is vague, two follow-up questions should be enough to capture the core need and reason.  
   - Don't ask more than that to avoid user fatigue.

5) **Quit as soon as user context is clear**.
   - If the (what + why) is already clear, 필수: "최종 쿼리를 확정합니다: [사용자의 명확한 의도]"라고 말하고 IZ node를 종료하세요.
   - 사용자가 명확한 목표를 제시하면 즉시 "최종 쿼리를 확정합니다: [목표]"라고 말하세요.
   - 더 이상의 질문이 필요하지 않을 때는 반드시 "최종 쿼리를 확정합니다"라는 문구로 시작하세요.

6) **Include detailed requirements in the final query**.
   - If there are any **detailed questions** or **additional requirements** that the user has presented during the dialogue, make sure to include them in the **final query**.

7) **맥락 추적 - 다음 정보를 반드시 추적하고 기억하세요**:
   - 사용자가 언급한 운동 종류 (스쿼트, 데드리프트, 벤치프레스 등)
   - 사용자의 주요 목표 (근력 향상, 체중 감량, 자세 교정 등)
   - 사용자의 특별한 상황 (부상, 경험 수준, 장비 제한 등)
   - 이전에 이미 답변한 질문들

5) **Refer to examples, but depend on X
   - The example (squat) provided is a guide, but be flexible and adapt it to your context.  
   - If your context is already specific enough, end **without further questions**.

### Summary
### Summary

- **If it's not related to weight training, say no**.  
- Ask up to 2 questions to get clarity **only when the big goal/reason is vague**.  
- Don't ask for details, but if the user mentions it, include it in the final query.  
- End immediately **when it's clear** by saying "최종 쿼리를 확정합니다: [명확한 의도]".
- **이전 대화 내용과 맥락을 항상 참조하고 유지하여 일관된 응답을 제공하세요.**

Based on these prompts, you (the IZ node) should only add questions when it is unclear **what and why the user is wondering about weight training in the larger context**, and **confirm the final query immediately** when it is clear. Review carefully The review criteria is based on whether the key role of the IZ node has been clearly assigned Is the LLM clearly set up to identify and fulfil the key requirements so that it understands what the user needs and why they are asking for it Is the query clear enough to handle high probability situations well Add in your subjective review

모든 대답은 한국말로 해라


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