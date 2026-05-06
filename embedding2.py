import os
import json
import pickle
import difflib
import chromadb
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
from typing import List, Dict, Any
from tqdm import tqdm
from kiwipiepy import Kiwi
from sentence_transformers import SentenceTransformer
import hashlib
import time

DATA_DIR = "./data/공개본 의결서"  
DB_DIR = "./chroma_db"
BM25_INDEX_PATH = "./bm25_index_kiwi.pkl"
CORPUS_INFO_PATH = "./corpus_info_kiwi.pkl" 

print("임베딩 모델 로딩 중...")
embedding_model = SentenceTransformer('BAAI/bge-m3')
kiwi = Kiwi()

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

chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(    
    name="ftc_resolutions",
    metadata={"hnsw:space": "cosine"}
)


def clean_duplicate_text(text, similarity_threshold=0.9, min_length=15, window_size=10):
    """
    청크 내에서 문단/문장 단위로 중복을 검사하여 제거하는 함수
    
    - similarity_threshold: 0.9 (90% 이상 일치하면 중복으로 간주)
    - min_length: 15 (너무 짧은 문자열 "단위: 천 원", "1." 등은 중복 검사에서 제외)
    - window_size: 10 (최근 10줄 내에 비슷한 문장이 있었는지 검사)
    """
    lines = text.split('\n')
    cleaned_lines = []
    
    # 중복 비교를 위해 최근 추가된 의미 있는 텍스트를 저장할 버퍼
    recent_lines = []

    for line in lines:
        line_stripped = line.strip()
        
        # 빈 줄은 그냥 통과
        if not line_stripped:
            continue

        # 길이가 너무 짧은 텍스트(번호, 단위 등)는 중복 체크 없이 그냥 추가
        if len(line_stripped) < min_length:
            cleaned_lines.append(line_stripped)
            continue

        # 최근 window_size 만큼의 줄들과 유사도 비교
        is_duplicate = False
        for recent in recent_lines:
            similarity = difflib.SequenceMatcher(None, recent, line_stripped).ratio()
            if similarity >= similarity_threshold:
                is_duplicate = True
                break
        
        # 중복이 아니면 결과 배열 및 최근 버퍼에 추가
        if not is_duplicate:
            cleaned_lines.append(line_stripped)
            recent_lines.append(line_stripped)
            
            # 버퍼가 window_size를 넘어가면 가장 오래된 것 삭제
            if len(recent_lines) > window_size:
                recent_lines.pop(0)

    # 다시 문자열로 합쳐서 반환
    return '\n'.join(cleaned_lines)

def load_and_inject_data_with_chain(data_dir: str, max_files: int = 10):
    processed_chunks = []
    all_files = []
    for f in os.listdir(data_dir): #문자열 반환
        if f.endswith("_metadata.json"):
            all_files.append(f)
    # all_files = [f for f in os.listdir(data_dir) if f.endswith("_metadata.json")]
  
    test_files = all_files[:max_files]
    print(f" 테스트 : 전체 {len(all_files)}개 문서 중 {len(test_files)}개만 처리합니다.")
    

    for filename in test_files:
        prev_hash = "GENESIS_ROOT_HASH"
        base_name = filename.replace("_metadata.json", "") 
        meta_path = os.path.join(data_dir, filename) 
        hybrid_path = os.path.join(data_dir, f"{base_name}_hybrid.json") 
        
        if not os.path.exists(hybrid_path):
            continue
            
        with open(meta_path, 'r', encoding='utf-8') as f:
            try:
                meta_data = json.load(f)
            except Exception as e:
                print(f"에러: {filename}", e)
                continue
        with open(hybrid_path, 'r', encoding='utf-8') as f:
            hybrid_data = json.load(f)
            
        title = meta_data.get("의결서제목", "제목없음")
        
        피심인_list = meta_data.get("피심인정보", [])
        if 피심인_list:
            company = 피심인_list[0].get("피심인기업명", "기업명없음")
            violation = 피심인_list[0].get("위반유형", "유형없음")
        else:
            company, violation = "기업명없음", "유형없음"

    

        for i, chunk in enumerate(hybrid_data):
            chunk_id = chunk.get("metadata", {}).get("chunk_id", f"chunk_{i}")
            original_text = chunk.get("page_content", "")
            if not original_text.strip():
                continue
            clean_original_text = clean_duplicate_text(original_text)

            injected_text = f"[사건명: {title}] [피심인기업명: {company}]\n본문: {clean_original_text}"

            #1. 타임 스탬프 생성
            timestamp = int(time.time() * 1000)

            #2. 현재 데이터의 고유 해시 생성 (이전 해시 + 텍스트 + 타임스탬프)
            data_to_hash = f"{prev_hash}{injected_text}{timestamp}".encode('utf-8')
            current_hash = hashlib.sha256(data_to_hash).hexdigest()
                
            processed_chunks.append({
                "chunk_id": chunk_id,
                "injected_text": injected_text,
                "metadata": {
                    "title": title,
                    "company": company,
                    "violation": violation,
                    "timestamp": timestamp,
                    "hash": current_hash,
                    "prev_hash": prev_hash # 이전 블록(청크) 연결 고리
                }
            })
            # 다음 청크를 위해 해시 업데이트
            prev_hash = current_hash 

    return processed_chunks
        

def build_indices():
    print("데이터 로드 및 메타데이터 주입 시작...")
    chunks = load_and_inject_data_with_chain(DATA_DIR)
    print(f"총 {len(chunks)}개의 청크가 준비되었습니다.")

    ids = []
    documents = []
    metadatas = []
    tokenized_corpus = []
    
    # 1. 데이터 준비 및 BM25 토큰화 
    for item in tqdm(chunks, desc="데이터 전처리 및 BM25 토큰화"):
        ids.append(item["chunk_id"])
        documents.append(item["injected_text"])
        metadatas.append(item["metadata"])
        
        
        
        tokens = tokenize_kiwi(item["injected_text"])
        if not tokens:
            tokens = ["EMPTY"]
        tokenized_corpus.append(tokens)

    # 2. ChromaDB에 벡터 저장
    BATCH_SIZE = 128
    
    for i in tqdm(range(0, len(documents), BATCH_SIZE), desc="ChromaDB 임베딩 및 저장"):
        batch_docs = documents[i : i + BATCH_SIZE]
        batch_ids = ids[i : i + BATCH_SIZE]
        batch_meta = metadatas[i : i + BATCH_SIZE]

        batch_embeddings = embedding_model.encode(batch_docs, normalize_embeddings=True).tolist()

        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embeddings, # 벡터값을 명시적으로 전달
            documents=batch_docs,
            metadatas=batch_meta
        )

    # 3. BM25 인덱스 생성
    print("BM25 인덱스 구축 중...")
    bm25 = BM25Okapi(tokenized_corpus)
    
    
    with open(BM25_INDEX_PATH, 'wb') as f:
        pickle.dump(bm25, f)
        
    corpus_info = [{"chunk_id": c["chunk_id"], "injected_text": c["injected_text"], "metadata": c["metadata"],"tokens": tokenize_kiwi(c["injected_text"])} for c in chunks]
    with open(CORPUS_INFO_PATH, 'wb') as f:
        pickle.dump(corpus_info, f)
        
    print("모든 인덱싱 과정이 완료되었습니다.")

if __name__ == "__main__":
    build_indices()