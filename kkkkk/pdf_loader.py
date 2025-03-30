import os

os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"

# langchain community & openai 관련 모듈 임포트
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -----------------------------
# 1. 웹 문서 URLs 정의
# -----------------------------
urls = [
    "https://www.codecademy.com/article/create-custom-workouts-using-chat-gpt",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7708084/#:~:text=but%20they%20are%20not%20focused,their%20effectiveness%20in%20promoting%20the"
    "https://www.issaonline.com/blog/post/12-must-ask-questions-for-new-personal-training-clients"
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10955739/#sec4"
    "https://www.researchgate.net/publication/322023636_Evidence-Based_Guidelines_for_Resistance_Training_Volume_to_Maximize_Muscle_Hypertrophy"
]

# -----------------------------
# 2. 웹 문서 로딩
# -----------------------------
web_docs = []
for url in urls:
    loaded_docs = WebBaseLoader(url).load()  # url별로 웹 문서를 가져옴
    web_docs.extend(loaded_docs)             # 웹 문서 누적

# -----------------------------
# 3. PDF 문서 로딩
# -----------------------------
folder_path = r"C:\Users\김신혁\OneDrive\바탕 화면\data"
pdf_docs = []

for filename in os.listdir(folder_path):
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(folder_path, filename)
        
        loader = PyMuPDFLoader(pdf_path)
        loaded_docs = loader.load()
        
        pdf_docs.extend(loaded_docs)
        print(f"PDF 파일명: {filename}, 로드된 문서 개수: {len(loaded_docs)}")

# -----------------------------
# 4. 웹 + PDF 문서를 합쳐서 all_docs 생성
# -----------------------------
all_docs = web_docs + pdf_docs

# 만약 문서가 전혀 없다면 에러 발생
if not all_docs:
    raise ValueError(
        "웹 문서와 PDF 문서에서 텍스트를 전혀 가져오지 못했습니다. "
        "URL이나 PDF 파일이 유효한지, 혹은 PDF가 이미지 기반인지 확인하세요."
    )

# -----------------------------
# 5. 텍스트 Splitter (텍스트를 일정 단위로 쪼갬)
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # 필요한 크기에 맞게 조정
    chunk_overlap=50      # 중첩 여부
)
split_documents = text_splitter.split_documents(all_docs)

# 혹시 분할된 문서가 전혀 없으면 에러
if not split_documents:
    raise ValueError("문서를 분할할 수 없었습니다. PDF가 이미지 기반이거나 텍스트가 비어 있을 수 있습니다.")

print(f"총 분할된 문서 개수: {len(split_documents)}")

# -----------------------------
# 6. Chroma 벡터 스토어 생성
# -----------------------------
# * 이 부분에서 OpenAIEmbeddings() 사용하려면
#   OPENAI_API_KEY 등이 환경변수나 코드상에 설정되어 있어야 합니다.
vectorstore = Chroma.from_documents(
    split_documents,
    OpenAIEmbeddings(),
    collection_name="langgraph_tistory",
)

# -----------------------------
# 7. 벡터 스토어에서 리트리버 생성
# -----------------------------
retriever = vectorstore.as_retriever()

# retriever를 사용해서 쿼리를 던지면 관련 문서를 쉽게 검색할 수 있습니다.
print("retriever 인스턴스 생성 완료!")
