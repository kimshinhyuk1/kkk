# test_huggingface.py
from langchain_huggingface import HuggingFaceEmbeddings

# 1) 임베딩 클래스 초기화
embeddings = HuggingFaceEmbeddings(model_name="nlpai-lab/KURE-v1")

# 2) 예시 문장
text = "안녕하세요? 이것은 테스트 문장입니다."

# 3) 임베딩 생성
embedding_vector = embeddings.embed_query(text)

print("임베딩 길이:", len(embedding_vector))
print("임베딩 벡터(앞부분):", embedding_vector[:10])
