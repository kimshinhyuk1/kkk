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
"https://barbend.com/overhead-press-mistakes-fixes/#:~:text=The%20key%20here%20is%20to,unwanted%20stress%20on%20the%20joints,"


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