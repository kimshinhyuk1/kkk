import os
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever



urls = [  
"https://barbend.com/overhead-press-mistakes-fixes/#:~:text=The%20key%20here%20is%20to,unwanted%20stress%20on%20the%20joints,",  
"https://brainly.com/question/41084421#:~:text=Brainly%20brainly,to%20provide%20support%20if%20needed,",  
"https://www.garagegymreviews.com/overhead-press#:~:text=How%20To%20Do%20The%20Barbell,partner%20by%20having%20them,",  
"https://www.garagegymreviews.com/overhead-press#:~:text=How%20To%20Do%20The%20Barbell,partner%20by%20having%20them,",  
"https://barbend.com/overhead-press-mistakes-fixes/#:~:text=The%20set%20up%20is%20one,locked%20in%20throughout%20the%20press,",  
"https://www.puregym.com/blog/arm-workouts-for-wheelchair-users-and-seniors/#:~:text=Arm%20and%20Shoulder%20Workouts%20For,Hold%20your%20dumbbell%20around,",  
"https://us.physitrack.com/home-exercise-video/adaptive-overhead-press#:~:text=,hands%20slightly%20above%20your,",  
"https://www.mindpumpmedia.com/blog/best-exercises-to-help-put-muscle-mass-on-your-shoulders#:~:text=Lateral%20Raises%20,raise%20them%20to%20the%20sides,",  
"https://simplifaster.com/articles/overhead-press/#:~:text=Push%20Pressing%20Like%20a%20Pro,",  
"https://www.muscleandfitness.com/workouts/shoulder-exercises/top-7-overhead-press-accessory-exercises/#:~:text=lower%20back%20and%20the%20protruding,better%20pressing%20path%20and%20lockout,",  
"https://www.healthline.com/health/behind-the-neck-press#:~:text=The%20behind,many%20people%20advise%20against%20it,",  
"https://www.frontiersin.org/journals/physiology/articles/10.3389/fphys.2022.825880/full#:~:text=observed%20for%20upper%20trapezius,the%20trajectory%20of%20the%20external,",  
"https://www.bicycling.com/training/a61867295/exercises-seniors-should-not-do/,",  
"https://www.acefitness.org/continuing-education/prosource/september-2014/4972/dynamite-delts-ace-research-identifies-top-shoulder-exercises/?srsltid=AfmBOoqw94wUr0nY7VJHLNgGVoRCoIy5ytG4WVl7vWI5ljjZ8yIDGExK#:~:text=the%20numbers%20%28Tables%201,Finally%2C%20for%20the,",  
"https://breakingmuscle.com/the-overhead-press-the-actual-difference-between-seated-standing-dumbbell-and-barbell-0/#:~:text=Front%20shoulder%20,results,",  
"https://barbell-logic.com/overhead-press-progression-and-training-variables/#:~:text=changes%20you%20can%20make%20is,If%20you%20are%20a,",  
"https://barbellrehab.com/overhead-press-pain/#:~:text=,ranges%20of%20motion%20you%20want,",  
"https://www.bicycling.com/training/a61867295/exercises-seniors-should-not-do/,",  
"https://simplifaster.com/articles/overhead-press/#:~:text=particularly%20in%20heavy%20pressing%20horizontally,strength%20development%20in%20both%20actions,",  
"https://strengthlevel.com/strength-standards/push-press-vs-military-press/lb#:~:text=Level%20strengthlevel,9%20lb%2C%2035,",  
"https://www.youtube.com/watch?v=-QH6YEWvaxM#:~:text=,Inhale%20and%20press%20directly%20overhead,",  
"https://barbell-logic.com/overhead-press-progression-and-training-variables/#:~:text=difficulty%20of%20training%20the%20press,improve%20lift",

]

web_docs = []
for url in urls:
    loaded_docs = WebBaseLoader(url).load()  # url별로 웹 문서를 가져옴
    web_docs.extend(loaded_docs)             # 웹 문서 누적





# data 폴더 내 모든 파일을 순회


# PDF에서 텍스트를 정상적으로 가져왔는지 확인
if not web_docs:
    raise ValueError("PDF에서 텍스트를 가져오지 못했습니다. 이미지 기반 PDF인지 확인하세요.")

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(web_docs)
embedding_name = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model=embedding_name)

vectorstore = Chroma.from_documents(
    split_documents,
    embeddings,
    collection_name="langgraph_tistory3", 
)

# 벡터 데이터베이스 추가
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# 키워드 기반 검색기 초기화 (BM25)
bm25_retriever = BM25Retriever.from_documents(split_documents)
bm25_retriever.k = 10

# 하이브리드 검색기 생성
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.8, 0.2]  # 의미론적 검색에 조금 더 가중치
)

print(f"총 분할된 문서 개수: {len(split_documents)}")

if not split_documents:
    raise ValueError("문서 분할 결과가 없다. PDF가 비었거나, 텍스트를 추출할 수 없는 형식일 수 있습니다.")

# 검색기 초기화
hybrid_retriever = hybrid_retriever