import os
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings



urls = [
   "https://pmc.ncbi.nlm.nih.gov/articles/PMC7708084/#:~:text=but%20they%20are%20not%20focused,their%20effectiveness%20in%20promoting%20the"  
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

# 벡터 데이터베이스 추가
vectorstore = Chroma.from_documents(
    split_documents,
    embeddings,
    collection_name="langgraph_tistory", 
)

print(f"총 분할된 문서 개수: {len(split_documents)}")

if not split_documents:
    raise ValueError("문서 분할 결과가 없다. PDF가 비었거나, 텍스트를 추출할 수 없는 형식일 수 있습니다.")

# 검색기 초기화
retriever = vectorstore.as_retriever()
