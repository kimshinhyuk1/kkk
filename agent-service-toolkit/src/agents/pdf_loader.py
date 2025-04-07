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



urls = ["https://www.mayoclinic.org/healthy-lifestyle/fitness/in-depth/weight-training/art-20045842#:~:text=,weights%20on%20the%20weight%20racks,",  
"https://www.betterhealth.vic.gov.au/health/healthyliving/resistance-training-preventing-injury#:~:text=,stretching,",  
"https://pubmed.ncbi.nlm.nih.gov/35438660/#:~:text=based%20on%20the%20Physiotherapy%20Evidence,may%20actually%20hinder%20muscular,",  
"https://pmc.ncbi.nlm.nih.gov/articles/PMC6934277/#:~:text=Conclusions,",  
"https://pubmed.ncbi.nlm.nih.gov/35438660/#:~:text=strength%20gains,may%20actually%20hinder%20muscular,",  
"https://www.nsca.com/contentassets/8323553f698a466a98220b21d9eb9a65/foundationsoffitnessprogramming_201508.pdf?srsltid=AfmBOoreS8nYzqFPh_GTrD_0VyMec5f_nwp4bj4PIzOJVK1fF6PlpPm8#:~:text=Existing%20strength%20training%20guidelines%20suggest,squat%2C%20lunge%2C%20bench%20press%2C%20etc,",  
"https://pmc.ncbi.nlm.nih.gov/articles/PMC5744434/#:~:text=and%20increased%20fat%20free%20mass,were%20found%20for%20body%20composition,",  
"https://www.verywellfit.com/how-to-use-weight-machines-and-gym-equipment-4153575#:~:text=Look%20for%20the%20Adjustment%20Point,",  
"https://www.acefitness.org/resources/pros/expert-articles/5561/6-benefits-of-using-weightlifting-machines/?srsltid=AfmBOorSwjQxyr--5kVddALe3Bj_uGifr-Oo__VopjhiqrgDgKMagwDi#:~:text=1,path%20of%20motion,",  
"https://pubmed.ncbi.nlm.nih.gov/21694556/#:~:text=in%20moderate,on%20%E2%89%A52%20d%C2%B7wk%20is%20recommended,",  
"https://pmc.ncbi.nlm.nih.gov/articles/PMC9565175/#:~:text=untrained%20subjects,overview%20the%20frequency%2C%20type%2C%20and,",  
"https://www.betterhealth.vic.gov.au/health/healthyliving/resistance-training-preventing-injury#:~:text=drop%20them,the%20same%20muscle%20group%20again,",  
"https://www.mayoclinic.org/healthy-lifestyle/fitness/in-depth/weight-training/art-20045842#:~:text=,sets%20that%20you%20perform%20may,",  
"https://www.nsca.com/education/articles/kinetic-select/introduction-to-dynamic-warm-up/?srsltid=AfmBOooE2L2CrFgVSQi0hmJe-DrFREDknNdsodzfjkcgBdKh4M2kdVHj#:~:text=It%20is%20important%20for%20all,needs%2C%20goals%2C%20and%20abilities%20of,",  
"https://www.mayoclinic.org/healthy-lifestyle/fitness/in-depth/weight-training/art-20045842#:~:text=,sets%20that%20you%20perform%20may,",  
"https://www.betterhealth.vic.gov.au/health/healthyliving/resistance-training-preventing-injury,",  
"https://pmc.ncbi.nlm.nih.gov/articles/PMC9565175/#:~:text=6,41%20%2C%2084,",  
"https://pubmed.ncbi.nlm.nih.gov/26700744/#:~:text=Results%3A%20%20In%20both%20muscles%2C,muscle%20the%20activity%20of%20the,",  
"https://www.frontiersin.org/news/2019/08/09/sports-mind-muscle-connection-weightlifting,",  
"https://www.mayoclinic.org/healthy-lifestyle/fitness/in-depth/weight-training/art-20045842#:~:text=,as%20you%20lower%20the%20weight,",  
"https://www.houstonmethodist.org/blog/articles/2023/may/why-proper-breathing-during-exercise-is-important-how-to-avoid-common-mistakes/#:~:text=In%20regard%20to%20the%20latter%2C,circumstances%2C%20can%20be%20incredibly%20dangerous,",  
"https://pubmed.ncbi.nlm.nih.gov/19204579/#:~:text=a%20moderate%20velocity,recommended%20that%20light%20to%20moderate,",  
"https://pubmed.ncbi.nlm.nih.gov/19620931/#:~:text=pediatric%20years%20may%20help%20to,conditioning%20industry%20is%20getting%20more,",  
"https://pubmed.ncbi.nlm.nih.gov/31343601/#:~:text=resilience%20and%20increase%20vulnerability%20to,on%20physical%20functioning%2C%20mobility%2C%20independence,",  
"https://www.ideafit.com/resistance-training-for-older-adults-new-nsca-position-stand/#:~:text=Stand%20www,to%20ensure%20correct%20exercise%20technique,",  
"https://www.nsca.com/contentassets/2a4112fb355a4a48853bbafbe070fb8e/resistance_training_for_older_adults__position.1.pdf?srsltid=AfmBOooJV-lwZ8HgMJ4IiB8iE0a4K6pE9PkEclU062SqrROgBuhOknow#:~:text=Part%201%3A%20Resistance%20Training%20Program,of%201,",  
"https://www.nsca.com/education/articles/kinetic-select/introduction-to-dynamic-warm-up/?srsltid=AfmBOooE2L2CrFgVSQi0hmJe-DrFREDknNdsodzfjkcgBdKh4M2kdVHj#:~:text=and%20improving%20joint%20range%20of,every%20sport%20practice%20and%20competition,",  
"https://pmc.ncbi.nlm.nih.gov/articles/PMC9565175/#:~:text=instability%29%20and%20pain,",  
"https://www.betterhealth.vic.gov.au/health/healthyliving/resistance-training-preventing-injury,",  
"https://www.cdc.gov/physical-activity-basics/guidelines/chronic-health-conditions-and-disabilities.html#:~:text=Consulting%20a%20health%20care%20professional,",  
"https://nutritionsource.hsph.harvard.edu/physical-activity-considerations-special-populations/#:~:text=intensity%20and%20duration%20of%20the,strength%2C%20and%20quality%20of%20life",

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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(web_docs)
embedding_name = "text-embedding-ada-002"
embeddings = OpenAIEmbeddings(model=embedding_name)

vectorstore = Chroma.from_documents(
    split_documents,
    embeddings,
    collection_name="langgraph_tistory3", 
)

# 벡터 데이터베이스 추가
semantic_retriever = vectorstore.as_retriever(search_kwargs={"k": 50})

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