import os
os.environ["USER_AGENT"] = "MyCustomUserAgent/1.0"
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader



urls = [
    "https://www.codecademy.com/article/create-custom-workouts-using-chat-gpt",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7708084/#:~:text=but%20they%20are%20not%20focused,their%20effectiveness%20in%20promoting%20the",
    "https://www.issaonline.com/blog/post/12-must-ask-questions-for-new-personal-training-clients",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10955739/#sec4",
    "https://www.researchgate.net/publication/322023636_Evidence-Based_Guidelines_for_Resistance_Training_Volume_to_Maximize_Muscle_Hypertrophy",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6835758/#:~:text=,00018.%20%5BDOI",
    "https://pubmed.ncbi.nlm.nih.gov/21225489/#:~:text=The%20purpose%20of%20this%20study,were%20recorded%20in%20the%20exercises",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7675616/#:~:text=The%20purpose%20of%20the%20shttps://pubmed.ncbi.nlm.nih.gov/36026487/#:~:text=major%20,of%20AD%20was%20significantly%20highertudy,lower%2C%20middle%2C%20and%20upper",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7579505/#:~:text=The%20bench%20press%20exercise%20is,maximal%20EMG%20activity%20for%20PMUP",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6835758/#:~:text=,4",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11224528/#:~:text=While%20shoulder%20injuries%20resulting%20from,electronic%20measurement%20system%20and%20an",
    "https://pubmed.ncbi.nlm.nih.gov/33555823/#:~:text=activity%20of%207%20upper%20extremity,expected%20when%20selecting%20narrower%20grip",
    "https://pubmed.ncbi.nlm.nih.gov/33666593/#:~:text=of%20this%20study%20were%20to,1%29%20than",
    "https://pubmed.ncbi.nlm.nih.gov/27669189/#:~:text=Farias%2C%20DdA%2C%20Willardson%2C%20JM%2C%20Paz%2C,testing%20protocols%20in%20random%20order",
    "https://blog.nasm.org/biomechanics-of-the-bench-press#:~:text=Clemons%2C%20J,87",#올바른 자세와 폼 체크 (플랫 & 인클라인)
    "https://vitruve.fit/blog/the-correct-bar-path-in-bench-press/#:~:text=and%20stable%20position,back%20pocket%20of%20the%20pants",#올바른 자세와 폼 체크 (플랫 & 인클라인)
    "https://ph.health.mil/PHC%20Resource%20Library/BenchPress_FS_12-023-1119.pdf#:~:text=Some%20general%20bench%20press%20recommendations,should%20not%20dip%20below%20chest",#올바른 자세와 폼 체크 (플랫 & 인클라인)
    "https://blog.nasm.org/biomechanics-of-the-bench-press#:~:text=Start%20with%20a%20standard%20grip,clearance%20from%20a%20medical%20professional",#올바른 자세와 폼 체크 (플랫 & 인클라인)
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7579505/#:~:text=occurred%20at%20a%20bench%20inclination,performance%20of%20the%20pectoralis%20major",#올바른 자세와 폼 체크 (플랫 & 인클라인)
    "https://www.strengthlog.com/incline-bench-press-vs-flat-bench-press/#:~:text=The%20incline%20bench%20press%20will,1%202",#올바른 자세와 폼 체크 (플랫 & 인클라인)
    "https://ph.health.mil/PHC%20Resource%20Library/BenchPress_FS_12-023-1119.pdf#:~:text=%20Activate%20your%20abdominal%20muscles,smaller%20muscles%20that%20support%20the",#올바른 자세와 폼 체크 (플랫 & 인클라인)
    "https://www.physio-network.com/research-reviews/shoulder/effects-of-bench-press-technique-variations-on-musculoskeletal-shoulder-loads-and-potential-injury-risk/#:~:text=1,scapula%20retraction%20generally%20reducing%20risk",#올바른 자세와 폼 체크 (플랫 & 인클라인)
    "https://pubmed.ncbi.nlm.nih.gov/38974522/#:~:text=statistical%20non,force%20components%20varied%20considerably%20between",#올바른 자세와 폼 체크 (플랫 & 인클라인)
    "https://pubmed.ncbi.nlm.nih.gov/21225489/#:~:text=deltoid%20anterior%2C%20biceps%2C%20and%20triceps,007%2C%20ES",#벤치프레스 운동의 다양한 변형
    "https://houseofhypertrophy.com/dumbbell-vs-barbell-bench-press/#:~:text=The%20studies%20by%20Saaterbakken%20et,similar%20activation%20of%20this%20muscle",#벤치프레스 운동의 다양한 변형
    "https://pubmed.ncbi.nlm.nih.gov/38974522/#:~:text=statistical%20non,force%20components%20varied%20considerably%20between",#벤치프레스 운동의 다양한 변형
    "https://ph.health.mil/PHC%20Resource%20Library/BenchPress_FS_12-023-1119.pdf#:~:text=%20Select%20grip%20width%20best,should%20not%20dip%20below%20chest",#벤치프레스 운동의 다양한 변형
    "https://www.strongerbyscience.com/research-spotlight-bench-grip-muscle/#:~:text=Within%20the%20resistance,can%20see%20in%20the%20figure",#벤치프레스 운동의 다양한 변형
    "https://rangeofmotion.net.au/1846-2/#:~:text=movement%20is%20more%20comparable%20to,pronounced%20with%20the%20full%20range", #부위별 고립 운동 및 보조운동
    "https://blog.myarsenalstrength.com/chest-press-machine#:~:text=Despite%20being%20incredibly%20effective%2C%20the,machine%20correctly%2C%20follow%20these%20steps",#헬스장 머신 사용법 (벤치프레스 머신 vs 체스트 프레스 머신)
    "https://www.tiktok.com/@petermiljak/video/7179395444848594181#:~:text=How%20To%20Do%20Machine%20Chest,somewhere%20around%20nipple%20line",#헬스장 머신 사용법 (벤치프레스 머신 vs 체스트 프레스 머신)
    "https://www.mdpi.com/2075-4663/9/2/32#:~:text=,1RM%29%29%20optimizes%20strength%20increases",
    "https://www.prescriptiontogetactive.com/static/pdfs/resistance-training-ACSM.pdf#:~:text=Muscular%20hypertrophy%20is%20the%20enhancement,2",
    "https://pubmed.ncbi.nlm.nih.gov/27433992/#:~:text=changes%20in%20measures%20of%20muscle,level",
    "https://www.strongerbyscience.com/drop-sets/#:~:text=1.%20Researchers%20conducted%20a%20meta,the%20importance%20of%20considering%20individual",
    "https://rangeofmotion.net.au/1846-2/#:~:text=movement%20is%20more%20comparable%20to,pronounced%20with%20the%20full%20range",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7560911/#:~:text=Asymmetric%20bench%20press%20loads%20reduced,to%20the%20traditional%20resistance%20training",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11224528/#:~:text=2001%20%3B%20Green%20and%20Comfort%2C,athlete%20to%20release%20the%20scapulae",
    "https://www.livestrong.com/article/552704-does-the-dumbbell-bench-put-less-stress-on-your-shoulder-muscles/",
    "https://www.seannal.com/articles/training/overhand-vs-neutral-grip-dumbbell-press.php#:~:text=The%20downside%20of%20performing%20neutral,an%20overhand%20grip%20is%20utilized",
    "https://t-nation.com/t/hammer-grip-vs-normal-grip/179296",
    "https://barbend.com/neutral-grip-bench-press/#:~:text=As%20a%20personal%20example%2C%20I,than%20they%20need%20to%20be",
    "https://thatfitfriend.com/alternating-dumbbell-bench-presses/#:~:text=From%20a%20strength%20coaching%20point,working%20on%20core%20strength%20indirectly",
    "https://generationiron.com/alternating-dumbbell-press/#:~:text=injuries",
    "https://mennohenselmans.com/bench-press-vs-flys/#:~:text=effectively%20trained%20the%20pecs%20but,only%201%20is%20relevant%20here",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7675616/#:~:text=muscle%20activation%20in%20the%20whole,the%20DF%20might%20prove%20useful",
    "https://www.acefitness.org/certifiednewsarticle/3003/what-are-the-top-3-most-effective-chest-exercises/?srsltid=AfmBOop56JUGdH-DLgyjouOH1r30jpDXXykga1Fps5ZMA7k77FjswiYL#:~:text=Barbell%20Bench%20Press%20100%206,1.5%20%C2%B1%201.15",
    "https://www.reddit.com/r/nSuns/comments/6i6za6/training_dips_as_accessory/?rdt=34301#:~:text=Training%20dips%20as%20accessory%20%3A,feet%20and%20bend%20them",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC9603242/#:~:text=As%20an%20exercise%2C%20it%20is,Figure%201",
    "https://outlift.com/hypertrophy-training-volume/#:~:text=For%20example%2C%20the%20bench%20press,16%20total%20sets",
    "https://pubmed.ncbi.nlm.nih.gov/20093960/#:~:text=followed%20by%202%20repetitions%20at,body%20muscular%20development",
    "https://www.reddit.com/r/powerbuilding/comments/z9l4xr/upper_chest_development_without_an_incline_bench/#:~:text=Reverse%20grip%20bench%20press,the%20slope%20of%20your%20hands",
    "https://www.reddit.com/r/powerbuilding/comments/z9l4xr/upper_chest_development_without_an_incline_bench/#:~:text=Feet%20elevated%20pushups%20with%20bands,and%20that%20shit%20is%20scary",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8449772/#:~:text=single%20set%20three%20times%20per,trained%20men",
    "https://rpstrength.com/blogs/articles/chest-hypertrophy-training-tips?srsltid=AfmBOorlCgYnRgekdQDBM7k6eyeI2EnHJ9_FUd79XWtDd9Tv6rkHJKub#:~:text=MV%20MEV%20MAV%20MRV%20MAV,32",
    "https://www.livestrong.com/article/552704-does-the-dumbbell-bench-put-less-stress-on-your-shoulder-muscles/",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11224528/#:~:text=statistical%20non,may%20decrease%20the%20risk%20for",
    "https://generationiron.com/alternating-dumbbell-press/#:~:text=",
    "https://www.muscleandfitness.com/workouts/arm-exercises/4-common-chest-flye-mistakes-and-fixes/#:~:text=Too%20Much%20or%20No%20Elbow,Bend",
    "https://www.inspireusafoundation.org/cable-crossover/#:~:text=As%20is%20the%20case%20with,of%20any%20muscles%20being%20targeted",
    "https://www.simplyfitness.com/pages/dumbbell-fly#:~:text=Lying%20on%20the%20bench%2C%20your,not%20go%20below%20shoulder%20level",
    "https://www.inspireusafoundation.org/cable-chest-fly/#:~:text=Other%20factors%20that%20can%20help,allowing%20for%20greater%20pectoral%20contraction",
    "https://muscularstrength.com/article/Dont-Do-Chest-Flys-Like-This#:~:text=Don%27t%20Do%20Chest%20Flys%20Like,your%20scapula%20down%20and%20back",
    "https://liftvault.com/exercises/single-cable-chest/#:~:text=",
    "https://www.bodybuilding.com/fun/matt20.htm#:~:text=The%20joint%20keeps%20going%20under,angle%20and%20range%20of%20motion",
    "https://fitliferegime.com/flat-chest-fly-vs-incline-fly-vs-decline-fly/#:~:text=The%20flat%20dumbbell%20fly%20,the%20barbell%20press%20and%C2%A0%2039",
    "https://barbend.com/single-arm-chest-flye/#:~:text=Helps%20Address%20Imbalances",
    "https://fitnessprogramer.com/exercise/one-arm-decline-cable-fly/#:~:text=Benefits%20fitnessprogramer.com%20%20Perform%20flat,for%20greater%20muscle%20damage",
    "https://www.inspireusafoundation.org/cable-crossover/#:~:text=When%20performed%20with%20an%20angle,are%20worked%20to%20equal%20intensity",
    "https://www.onnit.com/academy/how-to-do-the-cable-crossover-for-a-stronger-chest/#:~:text=How%20To%20Do%20The%20Cable,on%20a%20bench%20as%20well",
    "https://www.inspireusafoundation.org/pec-deck-machine/#:~:text=Adjusting%20the%20seat%20too%20high,will%20limit%20range%20of%20motion",
    "https://fitliferegime.com/flat-chest-fly-vs-incline-fly-vs-decline-fly/#:~:text=The%20flat%20dumbbell%20fly%20,the%20barbell%20press%20and%C2%A0%2039",
    "https://alphaprogression.com/en/blog/pre-exhaustion-flys-before-bench-press#:~:text=For%20example%2C%20a%20study%20by,triceps%20and%20NOT%20the%20chest",
    "https://outlift.com/dumbbell-vs-cable-fly/#:~:text=But%20that%E2%80%99s%20not%20always%20the,the%20stretch%20on%20the%20chest",
    "https://www.acefitness.org/certifiednewsarticle/3003/what-are-the-top-3-most-effective-chest-exercises/#:~:text=Both%20the%20pec%20deck%20,Table%201",
    "https://www.inspireusafoundation.org/cable-chest-fly/#:~:text=Allowing%20the%20Hands%20to%20Touch",
    "https://liftvault.com/exercises/single-cable-chest/#:~:text=1,sets%20before%20you%20switch%20sides",
    "https://barbend.com/single-arm-chest-flye/#:~:text=Wherever%20you%20usually%20set%20the,or%20slightly%20above%2C%20shoulder%20height",
    "https://strengthwarehouseusa.com/blogs/resources/functional-trainer-chest-exercises#:~:text=,the%20unilateral%20press%20and%20crossover",
    "https://fitwithnj.com/isometric-chest-workout/#:~:text=Quick%20fitwithnj,Eccentric",
    "https://www.bodybuilding.com/content/build-a-bigger-better-chest-with-isometrics.html#:~:text=Introducing%20Isometrics,the%20pump%20in%20your%20chest",
    "https://fitnessvolt.com/pec-deck/#:~:text=",
    "https://barbend.com/drop-sets/",
    "https://www.muscleandstrength.com/articles/pre-exhaust-method-for-every-muscle",
    "https://www.muscleandstrength.com/articles/coach-myers-pre-exhaust#:~:text=Get%20Bigger%20Faster%20With%20The,Exhaust%20Supersets",
    "https://www.ideafit.com/periodization-for-maximizing-hypertrophy/",
    "https://barbend.com/brad-schoenfeld-hypertrophy",
    "https://barbend.com/single-arm-chest-flye",
    "https://www.muscleandfitness.com/workouts/arm-exercises/4-common-chest-flye-mistakes-and-fixes/#:~:text=When%20your%20press%20weight%20vertically,if%20the%20wrists%20are%20hyperextended",
    "https://www.inspireusafoundation.org/chest-press-machine",
    "https://www.athleticinsight.com/exercise/chest",
    "https://goldenworkoutroutines.com/chest-press-machine-how-to-use-muscles-worked-machine-settings",
    "https://www.verywellfit.com/how-to-do-the-seated-machine-chest-press-3498292#:~:text=After%20setting%20the%20chest%20press,how%20to%20perform%20the%20exercise",
    "https://www.livestrong.com/article/32382-proper-pushup-technique/",
    "https://bretcontreras.com/pushup-research/#:~:text=Cogley%20et%20al,but%20this%20study%20showed%20otherwise",
    "https://goldenworkoutroutines.com/chest-press-machine-how-to-use-muscles-worked-machine-settings/#:~:text=To%20use%20the%20chest%20press,maintaining%20control%20throughout%20the%20movement",
    "https://exercise.trekeducation.org/2017/07/31/resistance-training/#:~:text=,2%20mins%20for%20light%20loads",
    "https://fitnessvolt.com/machine-chest-press/#:~:text=Moderate%20%28e",
    "https://evidencebasedmuscle.com/maximizing-muscle-growth-with-drop-sets/#:~:text=,volume%20without%20increasing%20training%20duration",
    "https://www.bodybuilding.com/fun/strength-showdown-push-up-vs-bench-press.html#:~:text=Of%20course%2C%20the%20great%20thing,your%20body%20into%20new%20growth",
    "https://cleanhealth.edu.au/blog/other/the-most-effective-periodisation-model-for-hypertrophy/#:~:text=Health%20cleanhealth,context%20of%20muscular%20strength",
    "https://www.nsca.com/contentassets/8323553f698a466a98220b21d9eb9a65/foundationsoffitnessprogramming_201508.pdf#:~:text=recommended%20,DESIGNING%20INDIVIDUALIZED%20FITNESS%20PROGRAMS",
    "https://www.strengthlog.com/machine-chest-press/#:~:text=The%20machine%20chest%20press%20is,Bench%20Press%3A%20Pros%20and%20Cons",
    "https://www.livestrong.com/article/32382-proper-pushup-technique/",
    "https://brookbushinstitute.com/articles/scapular-kinematics-and-shoulder-elevation-in-a-traditional-push-up#:~:text=Scapular%20ER%20was%20greater%20in,up",
    "https://www.bodybuilding.com/fun/strength-showdown-push-up-vs-bench-press.html#:~:text=Push,the%20backs%20of%20your%20hands",
    "https://www.livestrong.com/article/458560-do-pushups-increase-your-bench-press/#:~:text=The%20push,both%20exercises%20are%20loaded",
    "https://www.strengthlog.com/push-ups-vs-bench-press/#:~:text=Gains%20www,40%20reps%20per%20set",
    "https://aasem.org/dips-hurting-shoulders-how-to-prevent-and-alleviate-shoulder-pain/#:~:text=2,body%20more%20than%20the%20other",
    "https://welltech.com/content/what-muscles-do-dips-work-most-chest-dips-tricep-dips-variations/#:~:text=will%20bend%20their%20knees%20and,return%20to%20the%20starting%20position",
    "https://www.mdpi.com/1660-4601/19/20/13211#:~:text=The%20bench%20dip%20required%20a,strength%20training%20or%20rehabilitation%20protocols",
    "https://www.verywellfit.com/how-to-do-assisted-pullups-and-dips-3498284#:~:text=Gym%20machines%20have%20weights%20and,the%20load%20by%2050%20pounds",
    "https://bestworkoutsplan.com/assisted-dip/#:~:text=Assisted%20dip%20machines%20provide%20a,and%20progress%20to%20unassisted%20dips",
    "https://pubmed.ncbi.nlm.nih.gov/27102172/#:~:text=Conclusions%3A%20%20When%20comparing%20studies,protocol%20remains%20to%20be%20determined",
    "https://www.unm.edu/~lkravitz/Article%20folder/frequency.html#:~:text=resistance%20training%20programs%2C%20the%20American,Given%20these",
    "https://www.nsca.com/education/articles/ptq/building-a-balanced-and-symmetrical-physique/?srsltid=AfmBOoqRb34zv7qJRGLQC977mqeV3SBi5fDE_ehnTICFnDhjhs5d1LRH#:~:text=Although%20variety%20is%20important%2C%20changing,pain%20associated%20with%20the%20movement",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11224528/#:~:text=related%20injuries%20that%20have%20been,Furthermore%2C%20although%20pectoralis%20major%20ruptures",
    "https://pubmed.ncbi.nlm.nih.gov/11828249/#:~:text=initial%20resistances%2C%20it%20is%20recommended,1",
    "https://liftvault.com/exercises/flat-bench-press-vs-incline-bench-press/#:~:text=,bench%20press%20will%20also%20improve",
    "https://www.healthline.com/health/fitness-exercise/incline-vs-flat-bench#:~:text=When%20the%20bench%20is%20set,when%20using%20the%20flat%20bench",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7449336/#:~:text=the%20angle%20of%20the%20bench,32",
    "https://barbend.com/bench-press-plateau/#:~:text=1,Opt%20for%20Dumbbells",
    "https://liftvault.com/exercises/flat-bench-press-vs-incline-bench-press/#:~:text=The%20incline%20bench%20press%20is,by%20the%20incline%20bench%20press",
    "https://pubmed.ncbi.nlm.nih.gov/11828249/#:~:text=concentric%20and%20eccentric%20muscle%20actions,sets%20performed%20at%20a%20moderate",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7579505/#:~:text=This%20study%20confirms%20that%20the,portions%20of%20the%20pectoralis%20major",
    "https://www.healthline.com/health/fitness-exercise/incline-vs-flat-bench#:~:text=Flat%20bench%20presses",
    "https://www.transparentlabs.com/blogs/all/decline-bench-press-vs-incline-bench-press-vs-flat-bench-press?srsltid=AfmBOor5zJMY74M3oKZ_tYvTg7APtGdXogA-xECa_yI9Gb6_Nh88T2aP#:~:text=%23%20%20Secondary%20Upper",
    "https://pubmed.ncbi.nlm.nih.gov/27669189/#:~:text=when%20this%20exercise%20was%20performed,in%20BP%20modes%20does%20influence",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC11224528/#:~:text=joints.%20Time,may%20decrease%20the%20risk%20for",
    "https://www.sportgeneeskunde.com/wp-content/uploads/archief_bestanden/files/bestanden/ACSM%20Position%20Stand%20Progression%20models%20in%20resistance%20training%20for%20healthy%20adults.pdf#:~:text=exercises%29,with%20eventual%20emphasis%20on%20heavy",
    "https://www.tuffwraps.com/blogs/news/the-ultimate-guide-to-average-bench-press-by-age-strategies-for-every-stage-of-life#:~:text=A,Growth%20on%20Bench%20Press%20Strength",
    "https://strengthlevel.com/strength-standards/decline-bench-press#:~:text=Age%20Beg,92%20136%20192%20257%20329",
    "https://planfit.ai/article/incline-bench-press-vs-decline-bench-press",
    "https://bellsofsteel.com/blogs/content/decline-bench-press?srsltid=AfmBOoqr2rwxRKkqVWwOG8GbfEtYA9l18AvnSLU1myxfiR28xXr0kEOM#:~:text=Q1%3A%20Is%20the%20Decline%20Bench,Press%20Safe%20for%20Beginners",
    "https://powerliftingtechnique.com/decline-bench-press-benefits/#:~:text=For%20powerlifters%2C%20opt%20for%20swapping,of%20bench%20to%20your%20program",
    "https://bellsofsteel.us/blogs/content/is-decline-bench-easier-than-flat-bench?srsltid=AfmBOop-G3w3KzaLa-qXWEhRcHRD7zI-ZAv8N3H56U471-rgpOkAil99#:~:text=decline%20bench%20press%20is%20easier,and%20your%20shoulder%20muscles%20less",
    "https://www.healthline.com/health/exercise-fitness/bench-press-muscles-worked#what-it-is",
    "https://rpstrength.com/blogs/articles/chest-hypertrophy-training-tips?srsltid=AfmBOoqV0n4xth_KrWjtfwyoFSFyA2i8H1dTdtkJnfJ6eK4HdcWf5Vtq#:~:text=When%20you%E2%80%99re%20designing%20any%20week,lighter%20loads%20for%20the%20categories",
    "https://strengthlevel.com/strength-standards/bench-press-vs-decline-bench-press/lb#:~:text=Metric%20Bench%20Press%20Decline%20Bench,analysed%202%2C687%2C560%2035%2C076%202%2C652%2C484%207562",
    "https://www.nsca.com/education/articles/kinetic-select/determination-of-resistance-training-frequency/?srsltid=AfmBOooMIvp7uqXSNSCsCWtSSQp7nwWRSKl2fd2BW3QkcELyBB80-2RF#:~:text=When%20determining%20the%20training%20frequency,In",
    "https://rpstrength.com/blogs/articles/chest-hypertrophy-training-tips?srsltid=AfmBOooOI0ByRFy6Vs2-Fk2FB6FMYt22t73tSFKSNdMb2JLEXTNwkLsH#:~:text=match%20at%20L314%20compound%20presses,less%20stable%20than%20barbell%20or",
    "https://steelsupplements.com/blogs/steel-blog/the-dangers-of-dumbbell-flyes-how-to-avoid-them#:~:text=It%E2%80%99s%20not%20difficult%20to%20alter,great%20gains%20in%20your%20chest",
    "https://barbend.com/dumbbell-flye/#:~:text=Too%20Much%20Arm%20Bend",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7675616/#:~:text=phase%20of%20the%20descending%20and,strength%20and%20control%20in%20a",
    "https://fitbod.me/exercises/dumbbell-fly",
    "https://www.nsca.com/education/articles/kinetic-select/determination-of-resistance-training-frequency/?srsltid=AfmBOooMIvp7uqXSNSCsCWtSSQp7nwWRSKl2fd2BW3QkcELyBB80-2RF#:~:text=When%20determining%20the%20training%20frequency,In",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC4792988/#:~:text=brachii%2C%20triceps%20brachii%2C%20latissimus%20dorsi%2C,for%20truncus%20muscle%20strengthening%20or",
    "https://builtwithscience.com/fitness-tips/perfect-push-up-form/#:~:text=In%20fact%2C%20multiple%20studies%20have,to%20the%20reduced%20range%20of",
    "https://www.health.harvard.edu/blog/rise-push-ups-classic-exercise-can-motivate-get-stronger-2019021810165#:~:text=With%20a%20regular%20push,of%20your%20body%20weight",
    "https://pubmed.ncbi.nlm.nih.gov/21873902/#:~:text=60.96,05%29%20other%20than%20the%20condition",
    "https://www.healthline.com/health/wrist-pain-pushups#:~:text=According%20to%20the%20American%20Council,that%20target%20the%20same%20muscles",
    "https://www.verywellfit.com/how-to-do-the-seated-machine-chest-press-3498292#:~:text=The%20seated%20chest%20press%20machine,this%20could%20be%20less%20desirable",
    "https://training.fit/exercise/dips/#:~:text=Dips%20are%20commonly%20described%20as,is%20slightly%20lower%20in%20comparison",
    "https://t-nation.com/t/dips-youre-doing-them-wrong/286909#:~:text=,90%C2%B0",
    "https://ironbullstrength.com/blogs/training/weighted-dips?srsltid=AfmBOopntkY0h1VGAMCsmUcgO_u5BBoak7QvdyGMBlJZlBaYiSAP1b7_#:~:text=ease%20into%20them%3A",
    "https://www.medindia.net/health-press-release/American-Council-on-Exercise-Study-Highlights-Most-Effective-Tricep-Exercises-140197-1.htm#:~:text=American%20Council%20on%20Exercise%20Study,Advertisement",
    "https://breakingmuscle.com/chest-press-machine/#:~:text=Not%20ready%20to%20fully%20embrace,as%20pushdowns%20or%20pec%20flyes",
    "https://www.unm.edu/~lkravitz/Article%20folder/frequency.html#:~:text=When%20volume%20is%20equated%2C%20the,allow%20for%20effective%20recovery%20from",
    "https://www.businessinsider.com/how-to-break-a-muscle-gain-plateau-according-to-powerlifter-2021-9#:~:text=match%20at%20L338%20,Duffin%20said",
    "https://www.ideafit.com/periodization-for-maximizing-hypertrophy/#:~:text=Unique%20Advantages%20of%20Periodized%20Resistance,Training",
    "https://www.bodybuilding.com/content/8-things-you-should-never-do-on-chest-day.html#:~:text=2,Equipment",
    "https://www.bouldersportsclinic.com/blog-1/2025/2/18/push-vs-pull-exercises-why-your-workout-needs-the-right-push-pull-ratio#:~:text=Focusing%20on%20movement%20balance%20can,functioning%20body",
    "https://www.healthline.com/health/rounded-shoulders-exercises#:~:text=Austin%20Martinez%2C%20MS%2C%20CSCS%2C%20ATC%2C,levator%20scapulae",
    "https://pubmed.ncbi.nlm.nih.gov/25799093/#:~:text=separate%20time%20points%20for%20comparison,for%20the%20whole%20concentric",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7927075/#:~:text=1,1RM%29%29%20optimizes%20strength%20increases",
    "https://www.stronger.melbourne/blog/micro-tears-and-hypertrophy-separating-fact-from-fiction#:~:text=and%20hypertrophy%20is%20more%20nuanced,shortening%29%20contractions",
    "https://www.healthline.com/nutrition/body-recomposition#:~:text=While%20cardiovascular%20exercise%20is%20important,necessary%20to%20alter%20body%20composition",
    "https://barbend.com/bench-press-plateau/#:~:text=8%20Ways%20to%20Bust%20Through,micro%20or%20meso%20training%20cycle",
    "https://www.ereps.eu/news/tips-improve-your-bench-press#:~:text=The%20best%20accessory%20exercises%20will,the%20lats%2C%20triceps%2C%20and%20shoulders",
    "https://www.nerdfitness.com/blog/how-to-track-progress/",
    "https://www.bodybuilding.com/content/why-women-cant-afford-to-avoid-chest-training.html#:~:text=Let%27s%20say%20you%20decide%20to,this%20case%2C%20push%E2%80%94their%20own%20weight",
    "https://www.acefitness.org/resources/pros/expert-articles/5040/4-myths-about-strength-training-for-women/?srsltid=AfmBOoqZaVF1jlNTc6S4aoIcHlG2KspSEoin33fwz_bm8HG5H0cZlE83#:~:text=4%20Myths%20about%20Strength%20Training,the%20same%20rate%20as%20men",
    "https://www.nike.com/a/can-weight-lifting-stunt-growth",
    "https://www.uhhospitals.org/rainbow/services/pediatric-sports-medicine/patient-resources/fact-sheets/weight-training-fact-sheet#:~:text=Strength%20training%20does%20not%20stunt,performed%20as%20a%20max%20lift",
    "https://www.conservatoryseniorliving.com/senior-living-blog/6-effective-chest-exercises-to-keep-seniors-healthy/#:~:text=6%20Effective%20Chest%20Exercises%20To,band%20or%20light%20weights%2C",
    "https://www.healthline.com/health/everyday-fitness/senior-workouts#:~:text=1,Repeat%2010%20times",
    #---- 이제부터 정보는 핵심 대표성이 아닌 기저율에 의한 정보 수집에 해당합니다.,
    "https://blog.nasm.org/biomechanics-of-the-bench-press#:~:text=Agonist%20%C2%A0,shoulder%20stabilizer%20muscles",
    "https://pubmed.ncbi.nlm.nih.gov/25694615/#:~:text=load%20dynamic%20warm,up%20mode.%20A%20clear%20knowledge",
    "https://ph.health.mil/PHC%20Resource%20Library/BenchPress_FS_12-023-1119.pdf#:~:text=Suggested%20Guidelines%20to%20Avoid%20Bench,spotter%20who%20knows%20how%20to",
    "https://www.trainheroic.com/blog/best-bench-press-warm-up/#:~:text=I%20see%20many%20programs%20include,inhibition%2C%20and%20external%20rotation%20activation",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10690509/#:~:text=",
    "https://www.nestacertified.com/barbell-flat-bench-press-for-personal-training/#:~:text=,with%20the%20elbows%20fully%20extended",
    "https://www.ironcompany.com/blog/bench-press-assistance-work?srsltid=AfmBOooQ1bHKnLk9vz--ZBVkXkWRYwox6ByxyhuJV8s7jf61USd8P2TV#:~:text=,The%20favored%20tricep",
    "https://startingstrength.com/article/the-importance-of-using-safeties-in-the-squat-and-bench-press#:~:text=match%20at%20L305%20Unfortunately%2C%20most,You%20can%20elevate%20the",
    "https://www.physio-network.com/blog/pressing-exercises-shoulder-pain/#:~:text=4%29%20Push",
    "https://www.strongerbyscience.com/research-spotlight-bench-different/#:~:text=drawbacks,dominant%20lift",
    "https://www.hss.edu/article_exercises-shoulder-impingement.asp#:~:text=Exercises%20for%20Shoulder%20Impingement%2C%20from,weight%20your%20shoulder%20can%20support",
        #---- 이제부터 정보는 핵심 대표성이 아닌 기저율에 의한 정보 수집에 해당합니다.（가슴운동 거시적 측면）,
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7579505/#:~:text=muscles%20was%20recorded%20at%20the,of%20the%20anterior%20deltoid%20and",
    "https://functionaltraininginstitute.com/bench-press-set-up-for-injury-reduction/#:~:text=Elbow%20positioning%20during%20the%20bench,is%20particularly%20relevant%20for%20clients",
    "https://www.mdpi.com/2076-3417/13/8/5203#:~:text=objective,can%20be%20moved%2C%20reducing%20the",
    "https://pubmed.ncbi.nlm.nih.gov/16095407/#:~:text=supination%2Fpronation%20was%20varied%20to%20determine,by%20the%20level%20of%20supination",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC9121296/#:~:text=SA%20also%20functions%20as%20a,muscle%20activity%20combination%20as%20individuals",
    "https://pubmed.ncbi.nlm.nih.gov/27433992/#:~:text=size%20%28ES%29%20of%200,greater%20gains%20in%20muscle%20hypertrophy",
    "https://tourniquets.org/wp-content/uploads/PDFs/ACSM-Progression-models-in-resistance-training-for-healthy-adults-2009.pdf#:~:text=individual",
    "https://pubmed.ncbi.nlm.nih.gov/25601394/#:~:text=in%20accordance%20with%20the%20criteria,outlined",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC8310485/#:~:text=resistance,but%20increasing%20the%20eccentric%20time",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6977096/#:~:text=Effects%20of%20range%20of%20motion,training%20with%20a%20partial%20ROM",
    "https://jeffnippard.com/blogs/news/partial-vs-full-range-of-motion-what-is-actually-better-for-muscle-growth#:~:text=Using%20a%20partial%20range%20of,That%20said%2C",
    "https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2022.949021/full#:~:text=of%2081%E2%80%9388%25.%20The%20remaining%20meta,aim%20of%20optimizing%20muscle%20hypertrophy",
    "https://www.unm.edu/~rrobergs/478RestIntervalReview.pdf#:~:text=,might%20be%20most%20effective",
    "https://pubmed.ncbi.nlm.nih.gov/26605807/#:~:text=quadriceps%20femoris%20by%20ultrasound%20imaging,trained%20men",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10511399/#:~:text=,duration%20from%20a%20singular",
            #---- 이제부터 정보는 핵심 대표성이다.,
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10809978/#:~:text=attenuate%20the%20hypertrophic%20adaptations%20seen,when%20relatively%20short%20periods%20of",
    "https://pubmed.ncbi.nlm.nih.gov/11828249/#:~:text=American%20College%20of%20Sports%20Medicine,performed%20at%20a%20fast",
    "https://www.sportgeneeskunde.com/wp-content/uploads/archief_bestanden/files/bestanden/VSG/VSG6672.pdf#:~:text=Adults%20www,RT%29%20protocols%20are%20necessary",
    "https://www.scienceforsport.com/cluster-sets/?srsltid=AfmBOopOrnUrvCl7SPsbtjY1A-K072aTYr__TlclMdPa43O29gwSbunW#:~:text=over%20the%20course%20of%20a,5%29%20of%20repetitions%20%2815",
    "https://pubmed.ncbi.nlm.nih.gov/28834797/#:~:text=were%20significantly%20greater%20in%20favor,a%20spectrum%20of%20loading%20ranges",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC10289929/#:~:text=stability,of%20substantial%20literature%20compared%20with",
    "https://ph.health.mil/PHC%20Resource%20Library/BenchPress_FS_12-023-1119.pdf#:~:text=1.%20Warm,spotter%20who%20knows%20how%20to",
    "https://barbend.com/bench-press-warm-up/#:~:text=In%20either%20of%20these%20instances%2C,to%20in%20your%20set%20up",
    "https://www.trainheroic.com/blog/best-bench-press-warm-up/#:~:text=The%20barbell%20bench%20press%20can,on%20your%20shoulders%20over%20time",
    "https://barbend.com/14-triceps-exercises-improve-bench-strength/#:~:text=",
    "https://www.strengthrevolution.org/articles/a-quick-guide-to-bench-press-safety#:~:text=1,slip%20out%20of%20your%20hands",
    "https://hje.org.uk/blog/the-bench-press-how-to-avoid-injury-and-stay-fit/#:~:text=Warm%20up%20for%20a%20good,help%20you%20with%20greater%20flexibility",
    "https://breakingmuscle.com/5-spotting-techniques-and-rules-everyone-must-know/#:~:text=Before%20we%20begin%2C%20there%20are,lifter%20highly%20prone%20to%20injury",
    "https://athleanx.com/articles/incline-bench-press-mistakes?srsltid=AfmBOopHqtAjTxeKqgzcAyTYlmY9Q8ir0jVBwSulIzHfoNo2mCsNSDQ1#:~:text=1,down%20as%20you%20press%20the",
    "https://www.livestrong.com/article/456868-shoulder-pain-from-an-incline-bench-press/",
    "https://blog.warmbody-coldmind.com/guides/how-to-use-a-squat-rack/#:~:text=How%20to%20Use%20a%20Squat,Rack%20for%20Bench%20Press",
    "https://caliberstrong.com/blog/how-to-spot-someone-for-the-bench-press/#:~:text=Step%201%20%E2%80%93%20Ask%20him,off",
    "https://www.nsca.com/education/articles/kinetic-select/introduction-to-dynamic-warm-up/?srsltid=AfmBOoq_9NfUnoEqyMvxqt1vA2Xow-NxXkSWpTpL3Hyd-zNuXjtNWNg6#:~:text=It%20is%20important%20for%20all,Indeed",
    "https://orthoinfo.aaos.org/en/recovery/rotator-cuff-and-shoulder-conditioning-program/#:~:text=,Repeat%20with%20the%20other%20arm",
    "https://www.aafp.org/pubs/afp/issues/1998/0215/p680.html#:~:text=AAFP%20www,arms%20hanging%20down%3B%20keeping",
    "https://www.bodybuilding.com/fun/super-fly-your-complete-guide-to-chest-flyes.html#:~:text=The%20single,assist%20in%20moving%20the%20weight",
    "https://builtwithscience.com/fitness-tips/stop-doing-chest-flyes-like-this/#:~:text=Instead%2C%20to%20ensure%20the%20tension,the%20sides%20of%20your%20chest",
    "https://orthoinfo.aaos.org/en/recovery/rotator-cuff-and-shoulder-conditioning-program/#:~:text=Strength%3A%20Strengthening%20the%20muscles%20that,pain%20and%20prevent%20further%20injury",
    "https://www.verywellfit.com/how-to-use-a-chest-fly-machine-4589757#:~:text=The%20first%20step%20in%20using,or%20lower%20than%20your%20shoulders",
    "https://t-nation.com/t/chest-workout-with-a-bad-shoulder/206337",
    "https://www.bodybuilding.com/fun/matt20.htm#:~:text=The%20joint%20keeps%20going%20under,angle%20and%20range%20of%20motion",
    "https://www.mayoclinic.org/healthy-lifestyle/fitness/in-depth/weight-training/art-20045842#:~:text=,walking%20or%20other%20aerobic%20activity",
    "https://applecreeksportsmedicine.com/top-10-tips-for-safe-weightlifting-in-sports-training/#:~:text=1",
    "https://www.self.com/gallery/upper-body-warm-up#:~:text=tension%20then%20prepares%20your%20muscles,of%20injury%20in%20the%20process",
    "https://www.100pushups.net/warm-up#:~:text=Warm,a%20maximum%20of%2020%20repetitions",
    "https://kettlebellsworkouts.com/shoulder-mobility-warm-up-exercises/#:~:text=match%20at%20L288%20The%20chest,like%20rows%20or%20push%20ups",
    "https://www.self.com/gallery/upper-body-warm-up#:~:text=A%20good%20upper,of%20injury%20in%20the%20process",
    "https://briamethod.com/how-to-keep-proper-form-while-strength-training/#:~:text=,unsure%20about%20a%20particular%20exercise",
    "https://krakentraining.com/cant-feel-it-in-your-chest-bench-press-right-and-save-your-shoulders/#:~:text=Keeping%20your%20back%20arched%20and,Using",
    "https://builtwithscience.com/fitness-tips/how-to-properly-bench-press/#:~:text=Steps%29%20builtwithscience,angle%20away%20from%20your%20body",
    "https://www.xterrafitness.com/blog/proper-form-weight-training-errors-to-prevent-injuries/?srsltid=AfmBOop1zSgUK_en1ZTQfWkw5PBim8AbGn9Ykum3rNJzWG06rNuBBr4E#:~:text=Proper%20Form%3A%20Weight%20Training%20Errors,increases%20the%20risk%20of%20injury",
    "https://www.ironcompany.com/blog/triceps-power?srsltid=AfmBOop9A6L9V03Sy9erVR9xKk0UVW1FQLz3CdELJJOh30vPZSUP9fVT#:~:text=Triceps%20are%20the%20%E2%80%9Cweak%20link%E2%80%9D,triceps%20and%20conclude%20the%20lockout",
    "https://www.planetfitness.com/community/articles/want-get-better-pushups-focus-these-3-exercises#:~:text=3",
    "https://www.mitrecsports.com/fitness/how-to-use-the-chest-press-machine/#:~:text=,back%20in%20towards%20your%20chest",
    "https://cove.army.gov.au/article/how-to-improve-push-ups#:~:text=Hands%20%E2%80%93%20Now%20that%20both,which%20you%20can%20conduct%20repetitions",
    "https://www.advancedhumanperformance.com/dipsyouredoingthemwrong#:~:text=1,retracted%20or%20pulled%20back%20posteriorly",
    "https://applecreeksportsmedicine.com/top-10-tips-for-safe-weightlifting-in-sports-training/#:~:text=4",
    "https://www.major-lutie.com/blogs/wiki/how-to-use-a-power-rack-for-bench-press-a-complete-guide?srsltid=AfmBOorp1Eo6VHnzhGLtiJNFtyjt0L5xveHYtGVLkmdLlYKeNtMYRfPe#:~:text=Before%20diving%20into%20your%20bench,and%20you%27re%20ready%20to%20begin",
    "https://e3rehab.com/how-to-perform-dips/#:~:text=different%20options%20so%20you%20can,capacity%20to%20better%20tolerate%20them",
    "https://www.healthline.com/health/wrist-pain-pushups#:~:text=If%20you%20don%E2%80%99t%20have%20pushup,on%20your%20hands%20and%20wrists",
    "https://programme.app/resources/how-to-successfully-progressive-overload-the-bench-press#:~:text=One%20of%20the%20simplest%20ways,moved%20up%20in%20smaller%20increments",
    "https://www.issaonline.com/blog/post/how-to-break-a-bench-press-plateau",
    "https://barbend.com/dumbbell-vs-barbell-bench-press-which-is-best-for-you/#:~:text=Unilateral%20training%20reigns%20supreme%20when,both%20unilateral%20and%20bilateral%20pressing",
    "https://powerliftingtechnique.com/uneven-bench-press/#:~:text=For%20the%20most%20part%2C%20you%27ll,is%20simply%20to%20maintain%20strength",
    "https://sportscare-armworks.com/overcoming-wrist-pain-bench-press-tips/#:~:text=There%20are%20several%20common%20causes,also%20contribute%20to%20wrist%20pain",
    "https://fitbod.me/blog/wrist-pain-bench-press/#:~:text=,being%20focused%20through%20the%20wrists",
    "https://dieselsc.com/how-to-build-muscle-bench-press-more-weight-with-this-simple-warm-up-routine/#:~:text=Step%201%3A%20Foam%20Rolling%20Chest%2C,Triceps%2C%20Lats",
    "https://hje.org.uk/blog/the-bench-press-how-to-avoid-injury-and-stay-fit/#:~:text=%2A%20",
    "https://mennohenselmans.com/mind-muscle-connection-broscience/#:~:text=Lifting%20explosively%20also%20seems%20to,it%20seems%20that%20an%20explosive",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6615069/#:~:text=instructed%20set%2C%20CIS,major%20during%20the%20bench%20press",
    "https://barbend.com/bench-press-plateau/#:~:text=1,Opt%20for%20Dumbbells",
    "https://www.harvestingstrength.com/blog/why-heavy-bench-press-frequency-may-be-detrimental-to-human-performance#:~:text=During%20the%20initial%20stages%20of,stagnation%20and%20potential%20overuse%20injuries",
    "https://www.barbellmedicine.com/blog/best-upper-chest-exercises-for-bigger-pecs/#:~:text=If%20you%E2%80%99re%20just%20starting%20off%2C,42%20and%20bodybuilding%20templates",
    "https://wersports.com/blogs/fitness-equipment/how-to-optimise-your-incline-bench-press-prevent-injuries?srsltid=AfmBOoocdWu3r4i6sFgM4Lw4CSgYYTP5VAWzpS4YfOxJxzliFkevP6dQ#:~:text=How%20often%20should%20I%20perform,the%20incline%20bench%20press",
    "https://thinkeatlift.com/how-to-gain-strength-on-incline-bench-press/#:~:text=To%20progress%20on%20the%20Bench,you%20need%20to",
    "https://flybirdfitness.com/blogs/guide/whats-a-good-dumbbell-bench-press-weight?srsltid=AfmBOopLLtWMQcZpzO8vWJX-Tr-h6icePTsD-UnwY3DiUUm56Blq8IpG#:~:text=The%20%225",
    "https://barbellrehab.com/shoulder-pain-bench-press/#:~:text=",
    "https://wersports.com/blogs/fitness-equipment/how-to-optimise-your-incline-bench-press-prevent-injuries?srsltid=AfmBOoocdWu3r4i6sFgM4Lw4CSgYYTP5VAWzpS4YfOxJxzliFkevP6dQ#:~:text=Grip%20Width",
    "https://anytimephysio.com.au/shoulder-pain-bench-press/#:~:text=the%20shoulder%20blades%20together%20and,joint%20as%20you%20bench%20up",
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC7579505/#:~:text=occurred%20at%20a%20bench%20inclination,performance%20of%20the%20pectoralis%20major",
    "https://ironbullstrength.com/blogs/training/wrist-pain-bench-press?srsltid=AfmBOoqkTMNJB10_3Ys02kFdo83M60Y-vt86o1x5iarwIH9-5ELmJVbU#:~:text=Improper%20wrist%20positioning%20is%20the,undue%20stress%20on%20your%20wrists",
    "https://t-nation.com/t/minimize-shoulders-tris-when-bench-pressing/210217",
    "https://www.ironmaster.com/blog/mind-muscle-connection/#:~:text=Is%20the%20Mind%20Muscle%20Connection,well%20to%20the%20mind",
    "https://wersports.com/blogs/fitness-equipment/stepping-up-your-incline-bench-press-game?srsltid=AfmBOorx39zbVvSXCLFQsk_zDDdisiOfsugxzCKob65s62HfkNfaZqDW#:~:text=,your%20arms%20are%20fully%20extended",
    "https://t-nation.com/t/minimize-shoulders-tris-when-bench-pressing/210217",
    "https://kabukipower.com/blogs/articles/breathing-and-bracing-in-the-bench-press#:~:text=So%2C%20when%20you%20brace%20and,having%20a%20lot%20of%20rib",
]

web_docs = []
for url in urls:
    loaded_docs = WebBaseLoader(url).load()  # url별로 웹 문서를 가져옴
    web_docs.extend(loaded_docs)             # 웹 문서 누적



# Docker 환경에서는 /app/data 사용
DATA_DIR = os.getenv("DATA_DIR", "/app/data")  
pdf_docs = []
print(f"DATA_DIR 설정됨: {DATA_DIR}")

if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"data 폴더가 존재하지 않습니다: {DATA_DIR}")

# data 폴더 존재 여부 확인
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError(f"data 폴더가 존재하지 않습니다: {DATA_DIR}")

all_docs = web_docs + pdf_docs

# data 폴더 내 모든 파일을 순회
for filename in os.listdir(DATA_DIR):  # 🔥 여기 수정됨
    if filename.lower().endswith(".pdf"):
        pdf_path = os.path.join(DATA_DIR, filename)  # 🔥 여기 수정됨
        
        loader = PyMuPDFLoader(pdf_path)
        docs = loader.load()
        
        pdf_docs.extend(docs)  # 로드된 문서를 누적
        print(f"파일명: {filename}, 로드된 문서 개수: {len(docs)}")


# PDF에서 텍스트를 정상적으로 가져왔는지 확인
if not all_docs:
    raise ValueError("PDF에서 텍스트를 가져오지 못했습니다. 이미지 기반 PDF인지 확인하세요.")
from langchain_huggingface import HuggingFaceEmbeddings

# 문서 분할
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
split_documents = text_splitter.split_documents(all_docs)
embeddings = HuggingFaceEmbeddings(model_name="nlpai-lab/KURE-v1")

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
