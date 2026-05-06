import pickle
import chromadb
from sentence_transformers import SentenceTransformer
from kiwipiepy import Kiwi
import time

print("디버그 모드: 로딩 중...")

embedding_model = SentenceTransformer('BAAI/bge-m3')
kiwi = Kiwi()

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="ftc_resolutions")

with open("./bm25_index_kiwi.pkl", 'rb') as f:
    bm25 = pickle.load(f)

with open("./corpus_info_kiwi.pkl", 'rb') as f:
    corpus_info = pickle.load(f)

print("로딩 완료\n")

# =========================
# 설정
# =========================
test_query = "타이어뱅크 이월재고 차감 기준 과징금"
target_company = "타이어뱅크(주)"

# =========================
# 1. 회사명 확인
# =========================
print("=== [1] DB 내 회사명 확인 ===")
companies = set([item["metadata"]["company"] for item in corpus_info])

matched = [c for c in companies if "타이어" in c]

if not matched:
    print("❌ '타이어' 포함된 회사 없음")
else:
    print("DB에 존재하는 유사 회사명:")
    for c in matched:
        print(" -", repr(c))

print()

# =========================
# 2. 벡터 검색 (필터 없이)
# =========================
print("=== [2] Vector Search (필터 없음) ===")
query_embedding = embedding_model.encode([test_query], normalize_embeddings=True).tolist()

vector_results = collection.query(
    query_embeddings=query_embedding,
    n_results=5,
)

vector_ids = vector_results['ids'][0]

if not vector_ids:
    print("❌ vector 결과 없음")
else:
    print("vector_ids:")
    for vid in vector_ids:
        print(" -", vid)

print()

# =========================
# 3. 벡터 검색 (필터 있음)
# =========================
print("=== [3] Vector Search (company 필터) ===")

vector_results_filtered = collection.query(
    query_embeddings=query_embedding,
    n_results=5,
    where={"company": target_company}
)

vector_ids_filtered = vector_results_filtered['ids'][0]

if not vector_ids_filtered:
    print("❌ 필터 적용 시 결과 없음 → 필터 문제 확정")
else:
    print("필터 결과:")
    for vid in vector_ids_filtered:
        print(" -", vid)

print()

# =========================
# 4. BM25 검색
# =========================
print("=== [4] BM25 Top 5 ===")

def tokenize(text):
    return [t.form for t in kiwi.tokenize(text)]

tokenized_query = tokenize(test_query)
bm25_scores = bm25.get_scores(tokenized_query)

top5 = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:5]

for i, score in top5:
    item = corpus_info[i]
    print(f"{item['chunk_id']} | {item['metadata']['company']} | score={round(score,3)}")

print()

# =========================
# 5. BM25 + 회사 필터
# =========================
print("=== [5] BM25 + 회사 필터 ===")

filtered = []

for i, score in sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True):
    company = corpus_info[i]["metadata"]["company"]
    if company == target_company:
        filtered.append(corpus_info[i]["chunk_id"])
    if len(filtered) >= 5:
        break

if not filtered:
    print("❌ BM25도 필터에서 다 걸러짐")
else:
    for cid in filtered:
        print(" -", cid)

print()

# =========================
# 6. ID 매칭 테스트
# =========================
print("=== [6] ID 매칭 확인 ===")

if vector_ids:
    test_id = vector_ids[0]
    print("vector sample:", test_id)

    match = next((item for item in corpus_info if item['chunk_id'] == test_id), None)

    if match:
        print("✅ corpus_info에서 매칭됨")
    else:
        print("❌ ID 매칭 실패 → ID 구조 문제")

print()

print("=== 디버그 완료 ===")