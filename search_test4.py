import pickle
import chromadb
from sentence_transformers import SentenceTransformer
from konlpy.tag import Okt 
from kiwipiepy import Kiwi
import time


print("검색 엔진 및 DB 로딩 중...")
embedding_model = SentenceTransformer('BAAI/bge-m3')
kiwi = Kiwi()


chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection(name="ftc_resolutions")

with open("./bm25_index_kiwi.pkl", 'rb') as f:
    bm25 = pickle.load(f)
with open("./corpus_info.pkl", 'rb') as f:
    corpus_info = pickle.load(f)


STOPWORDS = {
    "있다", "하다", "되다", "위하다", "통하다",
    "경우", "사항", "내용", "관련"
}

def tokenize_kiwi(text):
    tokens = kiwi.tokenize(text)
    
    return [
        t.form for t in tokens
        if (t.tag.startswith('N') or t.tag.startswith('V'))
        and len(t.form) > 1
        and t.form not in STOPWORDS
    ]

print("로딩 완료! 검색을 시작합니다.\n")

def hybrid_search(query: str, target_company: str, top_k: int = 5):
    timings = {}

    # [1] Vector Search 
    t0 = time.perf_counter()
    query_embedding = embedding_model.encode([query], normalize_embeddings=True).tolist()
    
    where_clause = {"company": target_company} if target_company else None
    
    if where_clause:
        vector_results = collection.query(
            query_embeddings=query_embedding,
            n_results=20,
            where={"company": target_company}
        )
    else:
        vector_results = collection.query(
            query_embeddings=query_embedding,
            n_results=20,
        )
    vector_ids = vector_results['ids'][0]
    timings["vector_search"] = round(time.perf_counter() - t0, 4)
    
    # [2] bm25 search
    t1 = time.perf_counter()
    tokenized_query = tokenize_kiwi(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    
    bm25_filtered = []
    for i, score in sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True):
        metadata_company = corpus_info[i].get("metadata", {}).get("company")
        if not target_company or metadata_company == target_company: 
            bm25_filtered.append(corpus_info[i]['chunk_id'])
        if len(bm25_filtered) >= 20:
            break

    timings["bm25_search"] = round(time.perf_counter() - t1, 4)

    # [3] RRF (Reciprocal Rank Fusion) 
    t2 = time.perf_counter()
    k = 60
    rrf_scores = {}
    vector_weight = 0.7
    bm25_weight = 0.3
    
    for rank, doc_id in enumerate(vector_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + vector_weight * (1 / (k + rank + 1))
        
    for rank, doc_id in enumerate(bm25_filtered):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + bm25_weight * (1 / (k + rank + 1))

    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    top_k_ids = [doc_id for doc_id, score in sorted_rrf[:top_k]]
    
    if len(top_k_ids) < 5:
        for item in corpus_info:
            metadata_company = item.get("metadata", {}).get("company")
            if item['chunk_id'] not in top_k_ids:
                if not target_company or metadata_company == target_company:
                    top_k_ids.append(item['chunk_id'])
            if len(top_k_ids) == 5:
                break
    
    timings["rrf_processing"] = round(time.perf_counter() - t2, 6)
    timings["total_search"] = round(time.perf_counter() - t0, 4)

    return { "top_k_ids": top_k_ids, "timings": timings }

if __name__ == "__main__":
    #test_query = "과징금 납부 기한"
    #target_company = "(사)한국토종닭협회"
    test_query = "타이어뱅크가 대리점에게 적용한 '이월재고 차감' 기준 중, 제조일이 48개월 초과된 일반 타이어(D등급)의 차감 비율은 공장도가격의 몇 퍼센트이며, 공정위가 이 사건과 관련하여 부과한 최종 과징금액은 얼마인가요?"
    target_company = "타이어뱅크(주)"
    print(f"질문: {target_company}가 {test_query}\n")
    
    results = hybrid_search(test_query, target_company)
    
    print("-" * 80)
    print(f"최종 추출된 Top-5 Chunk IDs (총 {len(results['top_k_ids'])}개):")

    for i, chunk_id in enumerate(results['top_k_ids'], 1):
        matched_info = next((item for item in corpus_info if item['chunk_id'] == chunk_id), None)

        if matched_info:
            preview = matched_info["injected_text"][:].replace("\n", " ") + "..."
            print(f"{i}. {chunk_id} \n  미리보기: {preview}\n")
        else:
            print(f"{i}. {chunk_id} \n  미리보기:(정보를 찾을 수 없음))\n")
        
    print("-" * 80)