import os
import requests
from tempfile import NamedTemporaryFile

# 1) 환경 변수 설정 (User-Agent)
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"

# 2) LangChain 및 관련 라이브러리 임포트
from langchain_community.document_loaders import PyMuPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_openai import OpenAIEmbeddings
# 필요하다면 HuggingFaceEmbeddings 사용 가능:
# from langchain.embeddings import HuggingFaceEmbeddings

# 3) URL 리스트 (PDF/HTML 혼합)
urls = [  
    "https://medlineplus.gov/carbohydrates.html",
    "https://medlineplus.gov/ency/patientinstructions/000941.htm",
    "https://medlineplus.gov/ency/article/002469.htm",
    "https://newsinhealth.nih.gov/2023/08/breaking-down-food",
    "https://www.niddk.nih.gov/health-information/weight-management/adult-overweight-obesity/eating-physical-activity",
    "https://www.fda.gov/food/nutrition-facts-label/added-sugars-nutrition-facts-label",
    "https://www.cdc.gov/nutrition/php/data-research/added-sugars.html",
    "https://www.nutrition.gov/topics/whats-food/carbohydrates",
    "https://food-guide.canada.ca/en/healthy-eating-recommendations/make-it-a-habit-to-eat-vegetables-fruit-whole-grains-and-protein-foods/eat-whole-grain-foods/",
    "https://www.myplate.gov/tip-sheet/cut-back-added-sugars",
    "https://www.accessdata.fda.gov/scripts/interactivenutritionfactslabel/total-carbohydrate.cfm",
    "https://www.cdc.gov/healthy-weight-growth/rethink-your-drink/",
    "https://www.nhs.uk/live-well/eat-well/food-types/how-does-sugar-in-our-diet-affect-our-health/",
    "https://assets.publishing.service.gov.uk/media/5ba8a087e5274a55c3407c38/A_quick_guide_to_govt_healthy_eating_update.pdf",
    "https://www.gov.uk/government/publications/the-eatwell-guide",
    "https://www.who.int/news/item/04-03-2015-who-calls-on-countries-to-reduce-sugars-intake-among-adults-and-children",
    "https://www.efsa.europa.eu/en/efsajournal/pub/1462",
    "https://www.eatforhealth.gov.au/sites/default/files/2022-09/n55_australian_dietary_guidelines.pdf",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11280444/",
    "https://pressbooks.oer.hawaii.edu/humannutrition2/chapter/4-health-consequences-benefits-carbohydrates/",
    "https://extension.okstate.edu/fact-sheets/carbohydrates-in-the-diet.html",
    "https://food-guide.canada.ca/en/healthy-eating-recommendations/make-water-your-drink-of-choice/",
    "https://www.nlm.nih.gov/medlineplus/carbohydrates.html",
    "https://www.cdc.gov/diabetes/healthy-eating/choosing-healthy-carbs.html",
    "https://www.cdc.gov/diabetes/healthy-eating/fiber-helps-diabetes.html",
    "https://www.myplate.gov/eat-healthy/grains",
    "https://www.niddk.nih.gov/health-information/weight-management/adult-overweight-obesity/eating-physical-activity",
    "https://www.niddk.nih.gov/news/archive/2013/reducing-sugary-drinks-could-reduce-obesity",
    "https://www.myplate.gov/tip-sheet/cut-back-added-sugars",
    "https://www.accessdata.fda.gov/scripts/interactivenutritionfactslabel/total-carbohydrate.cfm",
    "https://www.cdc.gov/healthy-weight-growth/rethink-your-drink/",
    "https://www.nutrition.gov/topics/whats-food/carbohydrates",
    "https://www.nhs.uk/live-well/eat-well/food-types/how-does-sugar-in-our-diet-affect-our-health/",
    "https://www.nhs.uk/live-well/eat-well/how-to-eat-a-balanced-diet/eight-tips-for-healthy-eating/",
    "https://www.gov.uk/government/publications/the-eatwell-guide",
    "https://assets.publishing.service.gov.uk/media/5ba8a087e5274a55c3407c38/A_quick_guide_to_govt_healthy_eating_update.pdf",
    "https://food-guide.canada.ca/en/healthy-eating-recommendations/make-it-a-habit-to-eat-vegetables-fruit-whole-grains-and-protein-foods/eat-whole-grain-foods/",
    "https://food-guide.canada.ca/en/healthy-eating-recommendations/make-water-your-drink-of-choice/",
    "https://www.eatforhealth.gov.au/sites/default/files/2022-09/n55_australian_dietary_guidelines.pdf",
    "https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0139817",
    "https://www.frontiersin.org/articles/10.3389/fnut.2022.935234/full",
    "https://extension.okstate.edu/fact-sheets/carbohydrates-in-the-diet.html",
]

# ---------------------------------------------------------------------------------
# 4) (선택) 중복 URL 제거
# ---------------------------------------------------------------------------------
unique_urls = list(set(urls))

# ---------------------------------------------------------------------------------
# 5) PDF/HTML 구분하여 로드하기
# ---------------------------------------------------------------------------------
all_docs = []
for url in unique_urls:
    try:
        # 먼저 HEAD 요청으로 Content-Type 확인
        # (일부 서버는 HEAD를 막아놓을 수 있으므로, 필요 시 GET으로 대체)
        resp_head = requests.head(url, allow_redirects=True, timeout=10)
        if resp_head.status_code != 200:
            print(f"[WARN] 접근 실패 또는 상태 코드 문제: {url} (status={resp_head.status_code})")
            continue

        content_type = resp_head.headers.get("Content-Type", "").lower()

        # PDF 판단 조건: Content-Type이 PDF거나 URL이 .pdf로 끝나면 PDF로 간주
        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            print(f"[INFO] PDF로 처리: {url}")
            # PDF 다운로드
            resp_get = requests.get(url, stream=True, timeout=15)
            if resp_get.status_code != 200:
                print(f"[WARN] PDF 다운로드 실패: {url} (status={resp_get.status_code})")
                continue

            # 임시 파일에 쓰기
            with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
                for chunk in resp_get.iter_content(chunk_size=4096):
                    tmp_file.write(chunk)
                tmp_pdf_path = tmp_file.name

            # PyMuPDFLoader로 로딩
            pdf_loader = PyMuPDFLoader(tmp_pdf_path)
            loaded_docs = pdf_loader.load()
            all_docs.extend(loaded_docs)

        else:
            print(f"[INFO] HTML로 처리: {url}")
            # HTML 웹 문서 로딩
            web_docs = WebBaseLoader(url).load()
            all_docs.extend(web_docs)

    except Exception as e:
        print(f"[ERROR] {url} 처리 중 예외 발생: {e}")
        continue

# ---------------------------------------------------------------------------------
# 6) 로드된 문서가 있는지 확인
# ---------------------------------------------------------------------------------
if not all_docs:
    raise ValueError("[오류] 로드된 문서가 없습니다. URL 리스트나 연결 상태를 점검하세요.")

print(f"총 로드된 문서 수: {len(all_docs)}")

# ---------------------------------------------------------------------------------
# 7) 텍스트 분할
# ---------------------------------------------------------------------------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_documents = text_splitter.split_documents(all_docs)
print(f"총 분할된 문서 청크 수: {len(split_documents)}")

if not split_documents:
    raise ValueError("[오류] 문서 분할 결과가 없습니다. 텍스트 추출에 실패했을 수 있습니다.")

# ---------------------------------------------------------------------------------
# 8) 임베딩 & VectorStore 생성
# ---------------------------------------------------------------------------------
# OpenAI Embeddings (API 키 필요)
embedding_name = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model=embedding_name)

# Chroma 벡터 스토어에 저장
vectorstore = Chroma.from_documents(
    documents=split_documents,
    embedding=embeddings,
    collection_name="langgraph_tistory3",
)

# ---------------------------------------------------------------------------------
# 9) BM25 기반 키워드 검색 + 의미론적 검색 결합(EnsembleRetriever)
# ---------------------------------------------------------------------------------
bm25_retriever = BM25Retriever.from_documents(split_documents)
bm25_retriever.k = 10

semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.2, 0.8],  # 가중치는 필요에 따라 조정 가능
)

print("[INFO] 하이브리드 검색기 생성 완료")

# ---------------------------------------------------------------------------------
# 10) 실제 검색 사용 예시 (옵션)
# ---------------------------------------------------------------------------------
# query = "당뇨병 환자의 건강한 탄수화물 섭취 방법"
# results = hybrid_retriever.get_relevant_documents(query)
# for i, doc in enumerate(results, start=1):
#     print(f"[결과 {i}] {doc.page_content[:200]}...")
#
# 필요에 따라 QA 체인 등을 연동하여 답변 생성 가능
# ---------------------------------------------------------------------------------
