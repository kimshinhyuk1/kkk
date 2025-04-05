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
    "https://www.nsca.com/education/articles/kinetic-select/introduction-to-dynamic-warm-up/?srsltid=AfmBOooXx861T7Y6LBy8OivPOD72p1Xqjo-EDIVLjKxfF1BvVrSjw_Sl#:~:text=It%20is%20important%20for%20all,needs%2C%20goals%2C%20and%20abilities%20of",
    "https://www.hss.edu/article_static_dynamic_stretching.asp#:~:text=Dynamic%20stretches%20should%20be%20used,cycling%2C%20followed%20by%20dynamic%20stretching",
    "https://www.heart.org/en/healthy-living/fitness/fitness-basics/warm-up-cool-down#:~:text=Tips%3A",
    "https://www.mayoclinic.org/healthy-lifestyle/fitness/in-depth/weight-training/art-20045842#:~:text=,sets%20that%20you%20perform%20may",
    "https://news.wfu.edu/2017/10/31/lose-fat-preserve-muscle-weight-training-beats-cardio-older-adults/#:~:text=Image",
    "https://pubmed.ncbi.nlm.nih.gov/39593476/#:~:text=significance%20with%20a%20probability%20value,protocols",
    "https://pubmed.ncbi.nlm.nih.gov/25694615/#:~:text=Conclusions%3A%20%20Strong%20research,up%20mode.%20A%20clear%20knowledge",
    "https://humankinetics.me/2019/03/04/what-is-the-ramp-warm-up/#:~:text=The%20RAMP%20warm,The%20acronym%20%E2%80%98RAMP%E2%80%99%20stands%20for",
    "https://allianceortho.com/weightlifting-safety-for-healthy-joints/#:~:text=Warming%20up%20before%20weightlifting%20and,warming%20up%20and%20cooling%20down",
    "https://www.americansportandfitness.com/blogs/fitness-blog/3-common-mistakes-when-using-the-lat-pulldown#:~:text=,prepare%20for%20the%20pulling%20motion",
    "https://barbend.com/wrist-mobility-drills/#:~:text=1,Rolls",
    "https://orthoinfo.aaos.org/en/recovery/rotator-cuff-and-shoulder-conditioning-program/#:~:text=Warmup%3A%20Before%20doing%20the%20following,or%20riding%20a%20stationary%20bicycle",
    "https://www.movestrongphysicaltherapy.com/move-strong-blog/shoulder-warm-up-exercises#:~:text=1,for%20the%20training%20session%20ahead",
    "https://theprehabguys.com/shoulder-warm-up-before-lifting/#:~:text=Below%20you%20learn%20the%20ultimate,mobility%20drills%2C%20stretches%2C%20and%20exercises",
    "https://www.healthline.com/health/lat-stretches#:~:text=Lat%20Stretches%3A%2010%20Exercises%20to,increase%20your%20range%20of%20motion",
    "https://www.pullup-dip.com/blogs/training-camp/exercises-for-more-pull-ups#:~:text=The%20first%20exercise%20that%20we,To%20do%20a%20lat",
    "https://www.setforset.com/blogs/news/lat-stretches?srsltid=AfmBOooN8Bv8d0lAOIFQ0JmWZk_rWZqbpIH2MWqgPm4T2d3XnxzjILJP#:~:text=10%20Best%20Lat%20Stretches%20for,lat%20muscles%20and%20joints",
    "https://www.nsca.com/education/articles/ptq/time-efficient-training/?srsltid=AfmBOoqh60gfeSD4Vw5a4QPGYiEPjCMGbWojHel7bMPVrvnwgTHCvKw1#:~:text=Time,during%20training%2C%20thereby%20optimizing",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC5213357/#:~:text=Many%20warm,However",
    "https://blog.nasm.org/dynamic-warm-ups-for-athletes-injury-prevention-and-sports-performance-benefits#:~:text=Dynamic%20Warm,their%20benefits%20for%20injury%20prevention",
    "https://www.youtube.com/watch?v=y4e5F2RQvaI#:~:text=Shoulder%20Warm,you%27re%20warming%20up%20the%20shoulder",
    #2. 운동 중간 체크 포인트 (세트 도중 자세·자극 확인) ,
    "https://www.acefitness.org/resources/everyone/exercise-library/158/seated-lat-pulldown/?srsltid=AfmBOopiMFfZB9Yb1gc5NKh9NGRh-reQTOVDFPLLFJ3Y8rGGo0hUdpEc#:~:text=STARTING%20POSITION%3A%20Sit%20in%20the,head%20aligned%20with%20your%20spine",
    "https://www.socalpowerlifting.net/post/accessory-review-lat-pulldowns#:~:text=Arching%20your%20back",
    "https://www.bodybuilding.com/fun/marc-megna-lifting-lessons-lat-pull-down.html#:~:text=Pull%20the%20weight%20down%20slowly%2C,your%20rhomboids%2C%20so%20stay%20upright",
    "https://joshstrength.com/2020/01/the-science-of-training-lats/#:~:text=2,grip%E2%80%9D%2C%20as%20this%20reduces%20biceps   ",
    "https://archive.t-nation.com/training/the-torso-solution/#:~:text=The%20first%20key%20point%20is,depending%20on%20your%20limb%20length",
    "https://pubmed.ncbi.nlm.nih.gov/24662157/#:~:text=activation%20between%20grip%20widths%20for,wide%20grips%3B%20however%2C%20athletes%20and",
    "https://www.strengthlog.com/are-rows-and-lat-pulldowns-enough-for-great-biceps-growth/#:~:text=Do%20Lat%20Pulldowns%20and%20Rows,equally%20good%20as%20barbell%20curls",
    "https://www.mayoclinic.org/healthy-lifestyle/fitness/in-depth/weight-training/art-20045842#:~:text=,least%20two%20times%20a%20week",
    "https://www.socalpowerlifting.net/post/accessory-review-lat-pulldowns#:~:text=each%20other",
        #3. 보조근 강화 전략 (랫풀다운 보조 근육 강화) ,
    "https://www.healthline.com/health/fitness-exercise/pull-up-prep#:~:text=Share%20on%20Pinterest-",
    "https://www.gq.com/story/how-to-get-better-at-pull-ups#:~:text=6%20Exercises%20to%20Help%20You,it%3A%20Using%20an%20overhand",
    "https://www.opexfit.com/blog/seven-different-pull-up-grips-and-their-benefits#:~:text=Seven%20Different%20Pull,the%20forearms%2C%20shoulders%2C%20and%20biceps",
    "https://www.trainheroic.com/blog/want-bigger-biceps-do-more-chin-ups/#:~:text=Want%20Bigger%20Biceps%3F%20Do%20More,chin%20ups%20into%20your%20routine",
    "https://pubmed.ncbi.nlm.nih.gov/19387371/#:~:text=body%20mass%20to%20achieve%20one,fat%29%20affected",
    "https://www.joionline.net/library/the-top-5-worst-shoulder-exercises-to-avoid-lateral-raises-and-more/#:~:text=What%20are%20the%20Best%20Shoulder,Tendonitis%20or%20Rotator%20Cuff%20Issues",
    "https://blog.nasm.org/biomechanics-of-the-lat-pulldown#:~:text=The%20lat%20pulldown%20is%20a,the%20individual%20performs%20the%20exercise",
    "https://sunnyhealthfitness.com/blogs/workouts/full-body-landmine-lat-pulldown-strength-training-for-beginners?srsltid=AfmBOor3wDBC9OweS9ark8DIGyd8Zvk1dbGqhrI9rBueME5ytBhS9VnK#:~:text=Full%20Body%20Landmine%20%2B%20Lat,your%20regular%20strength%20training%20routine",
    # 4. 올바른 셋업과 자세 (기구 세팅 및 폼) ,
    "https://www.bodybuilding.com/fun/marc-megna-lifting-lessons-lat-pull-down.html#:~:text=If%20you%27re%20a%20rocker%2C%20then,Everything%20should%20be%20tight",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11667758/#:~:text=Before%20each%20training%20session%2C%20participants,Each",
    "https://www.bodybuilding.com/fun/marc-megna-lifting-lessons-lat-pull-down.html#:~:text=If%20you%27re%20a%20rocker%2C%20then,Everything%20should%20be%20tight",
    "https://www.gymreapers.com/blogs/news/exercises-to-use-weight-belt?srsltid=AfmBOoqqRz4vf-lB2MD38gcPYEulwqXsZ40C1gxVnkBVrlKZMsBxnfyw#:~:text=Weightlifting%20belts%20are%20unnecessary%20for,spinal%20support%20is%20very%20minimal",
    "https://cohenhp.com/weight-lifting-belts-do-you-need-one/#:~:text=Weight%20Lifting%20Belts%3A%20Do%20You,For%20more",
    "https://www.joionline.net/library/the-top-5-worst-shoulder-exercises-to-avoid-lateral-raises-and-more/#:~:text=Shoulder%20press%20or%20military%20press,neck%20can%20lead%20to%20injury",
    "https://pubmed.ncbi.nlm.nih.gov/24662157/#:~:text=narrow%20grip%20%28p%20%3D%200,2%20times%20the%20biacromial%20distance",
    "https://www.socalpowerlifting.net/post/accessory-review-lat-pulldowns#:~:text=Supinated%20grip",
    "https://www.dmoose.com/blogs/muscle-building/top-10-lat-pulldown-variations-stronger-wider-back?srsltid=AfmBOoqiuJf7_BkyxnlcPUXj0xM4DRZUEORTzzHI_wsvQArlfKAE9_ZL#:~:text=Top%2010%20Lat%20Pulldown%20Variations,an%20underhand%20grip%20and",
    "https://www.verywellfit.com/how-to-do-the-lat-pulldown-3498309#:~:text=Lat%20Pulldowns%3A%20Techniques%2C%20Benefits%2C%20Variations,slightly%20backward%20is%20OK%2C",
    #5. 안전 장치 및 도구 활용 (벨트·랩·스트랩 등) ,
    "https://www.gymreapers.com/blogs/news/exercises-to-use-weight-belt?srsltid=AfmBOoqqRz4vf-lB2MD38gcPYEulwqXsZ40C1gxVnkBVrlKZMsBxnfyw#:~:text=Weightlifting%20belts%20are%20unnecessary%20for,spinal%20support%20is%20very%20minimal",
    "https://www.self.com/story/do-you-need-weight-lifting-belts#:~:text=Should%20You%20Wear%20a%20Lifting,you%27re%20lifting%20with%20a%20belt",
    "https://news.wfu.edu/2017/10/31/lose-fat-preserve-muscle-weight-training-beats-cardio-older-adults/#:~:text=Image",
    "https://rxfit.co/shoulder-impingement-exercises-to-avoid/#:~:text=Shoulder%20Impingement%20Exercises%20To%20Avoid%3A,These%20exercises%20require%20shoulder",
    "https://strengthwarehouseusa.com/blogs/resources/lat-pulldown-alternatives?srsltid=AfmBOorzW_S7H7ET7snnQhyfupqz0-H-5VVMolUpKFxoaMZ2719VM-V1#:~:text=Lat%20Pulldown%20Alternatives%20for%20a,Other%20muscles%20this%20exercise",
    "https://www.reddit.com/r/xxfitness/comments/49tp4c/short_girl_problems/#:~:text=Short%20girl%20problems%20%3A%20r%2Fxxfitness,momentarily%20to%20reach%20the%20handles",
    "https://www.womenshealthmag.com/fitness/g26801302/best-lat-exercises-for-women/#:~:text=11%20Best%20Lat%20Exercises%20For,as%20far%20as%20you",
    "https://strengthlevel.com/strength-standards/lat-pulldown/lb#:~:text=Lat%20Pulldown%20Standards%20for%20Men,other%20lifters%20at%20your%20bodyweight",
    "https://news.wfu.edu/2017/10/31/lose-fat-preserve-muscle-weight-training-beats-cardio-older-adults/#:~:text=Image",
    "https://www.health.harvard.edu/exercise-and-fitness/two-types-of-exercise-may-be-needed-to-preserve-muscle-mass-during-weight-loss#:~:text=Two%20types%20of%20exercise%20may,physical%20function%20than%20either",
    
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