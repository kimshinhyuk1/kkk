import os
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import OpenAIEmbeddings



urls = [
   "https://www.strengthlog.com/barbell-curl/#:~:text=Positioning,",  
"https://www.garagegymreviews.com/curl-bar-workouts#:~:text=Curl%20Bar%20Workouts%20for%20Bigger%2C,in%20Biceps%20Curl%20Exercise%3A,",  
"https://breakingmuscle.com/best-biceps-exercises/#:~:text=The%2012%20Best%20Biceps%20Exercises,bar%20reverse,",  
"https://trugrit-fitness.com/blogs/news/curl-bar-vs-straight-bar?srsltid=AfmBOoruyTZr1ft6XUDQoYe2VR4bngenFzGpJK7UAQm1MQzTLHfV5yQa#:~:text=In%20addition%2C%20if%20you%20lack,outwards%2C%20depending%20on%20the%20exercise,",  
"https://pmc.ncbi.nlm.nih.gov/articles/PMC6047503/#:~:text=We%20can%20conclude%20that%20the,EZ%20than%20to%20the%20DC,",  
"https://bdefinedfitness.com/two-heads-are-better-than-one-5-moves-for-balanced-biceps/#:~:text=Stand%20with%20your%20torso%20upright%2C,will%20be%20your%20starting%20position,",  
"https://pubmed.ncbi.nlm.nih.gov/35044672/#:~:text=,1RM%20strength%20compared%20to,",  
"https://bretcontreras.com/are-cheat-reps-beneficial-a-discussion-of-the-evidence-and-implementation/#:~:text=In%20a%20physiological%20environment%20of,2013,",  
"https://mennohenselmans.com/optimal-training-volume/#:~:text=Most%20evidence,may%20have%20been%20overly%20conservative,",  
"https://orthoinfo.aaos.org/en/diseases--conditions/biceps-tendon-tear-at-the-elbow/#:~:text=Risk%20Factors,",  
"https://pmc.ncbi.nlm.nih.gov/articles/PMC9214967/#:~:text=Conclusion,",  
"https://www.setforset.com/blogs/news/bicep-stretches?srsltid=AfmBOooHjnH8VmlRX3o-nC5oIR4_ap1Wjc4EQgr1wzglXO23kJDQKDqT#:~:text=9%20Best%20Bicep%20Stretches%20for,body%20movements%20with,",  
"https://orthoinfo.aaos.org/en/diseases--conditions/biceps-tendon-tear-at-the-elbow/#:~:text=Men%20age%2030%20years%20or,tear%20the%20distal%20biceps%20tendon,",  
"https://www.orthocarolina.com/orthopedic-news/how-to-prevent-biceps-tendon-ruptures#:~:text=1,our%20Sports%20Training%20Center%20programming,",  
"https://www.lookgreatnaked.com/blog/the-mind-muscle-connection-a-key-to-maximizing-growth/#:~:text=For%20as%20long%20as%20I,should%20result%20in%20greater%20growth,",  
"https://www.popsugar.com/fitness/how-breathe-during-bicep-curls-46225636,",  
"https://pubmed.ncbi.nlm.nih.gov/30558493/#:~:text=higher%20frequencies%2C%20although%20the%20overall,groups%20based%20on%20personal%20preference,",  
"https://journals.lww.com/acsm-healthfitness/fulltext/2013/09000/resistance_training_for_older_adults__targeting.7.aspx#:~:text=Resistance%20Training%20for%20Older%20Adults%3A,pattern%2C%20suggesting%20a%20training,",  
"https://www.aafp.org/pubs/afp/issues/2017/0401/p425.html#:~:text=Resistance%20training%20preserves%20muscle%20strength,25,",  
"https://rpstrength.com/blogs/articles/bicep-hypertrophy-training-tips?srsltid=AfmBOoq1X0tD25dUjnkcPiENSHxNJRk2MdbCkPGKdafosSgH2jPf0GXx#:~:text=match%20at%20L422%20start%20by,with%20volumes%20just%20a%20bit,",  
"https://bretcontreras.com/are-cheat-reps-beneficial-a-discussion-of-the-evidence-and-implementation/#:~:text=Rule%20,number%20of%20sets%20per%20workout,",  
"https://cathe.com/whats-best-tempo-working-biceps-muscles/#:~:text=Most%20people%20use%20a%20standard,the%20eccentric%20or%20downward,",  
"https://examine.com/research-feed/study/17KpK1/?srsltid=AfmBOoop9S-Ahu8U3UyEOne2DXEl9s7Bcfu2Oawjj6auMJd9Atcup4vZ,",  
"https://dr-muscle.com/hypertrophy-biceps-workout/#:~:text=3%20Biceps%20Workouts%20for%20Muscle,maximize%20metabolic%20stress%2C%20although,",  
"https://www.aafp.org/pubs/afp/issues/2017/0401/p425.html#:~:text=To%20promote%20and%20maintain%20health%2C,38,",  
"https://pubmed.ncbi.nlm.nih.gov/19204579/#:~:text=The%20recommendation%20for%20training%20frequency,1%29%20for%20advanced,",  
"https://www.kettlebellkings.com/blogs/default-blog/dynamic-arm-stretches?srsltid=AfmBOoq9cLDAvG6I3Akj_50voQ35o7Z0WVbstFpXiQOxc5QCvpUnRCRY#:~:text=Kings%20www.kettlebellkings.com%20%20Cross,help%20improve%20shoulder%20mobility,",  
"https://www.healthline.com/health/bicep-stretch#:~:text=Healthline%20www,moves%20will%20get%20you%20started,",  
"https://startingstrength.com/article/barbell_safety#:~:text=Barbell%20Safety%20,weight%20on%20the%20bar,",  
"https://ironbullstrength.com/blogs/equipment/should-you-wear-wrist-wraps-for-curls?srsltid=AfmBOor3SaA4g7Cl4c4IQMK1wwnEuWzwOG44t6NfqGGkBC04-e7lwaNo#:~:text=Should%20You%20Wear%20Wrist%20Wraps,pain%2C%20or%20lifting%20heavy%20weights,",  
"https://ironbullstrength.com/blogs/equipment/should-you-wear-wrist-wraps-for-curls?srsltid=AfmBOor3SaA4g7Cl4c4IQMK1wwnEuWzwOG44t6NfqGGkBC04-e7lwaNo#:~:text=Should%20You%20Wear%20Wrist%20Wraps,pain%2C%20or%20lifting%20heavy%20weights,",  
"https://www.arrowptseattle.com/news/2016/8/13/wrist-wraps-when-you-need-them-when-you-dont#:~:text=Wrist%20Wraps,are%20at%20a%20max%20lift,",  
"https://www.stack.com/a/why-you-should-always-put-safety-collars-on-the-bar/#:~:text=Stack,the%20barbell%20so%20they,",  
"https://himommy.app/en/pregnancy/exercisesduringpregnancy/exercise/dumbbell_bicep_curls,",  
"https://www.acog.org/womens-health/faqs/exercise-during-pregnancy#:~:text=,exercise%20during%20pregnancy,",  
"https://www.baptist-health.com/blog/top-5-gentle-and-safe-exercises-for-pregnant-women#:~:text=1,With%20your%20feet",
 
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
