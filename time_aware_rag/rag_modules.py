import chromadb
import os
import sys
import random
import numpy as np
import pandas as pd
from chromadb.utils import embedding_functions

# 프로젝트 루트 경로 추가 (모듈 import를 위해)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.search_queries import GAMER_TYPE_QUERIES, GENERAL_QUERY

# ChromaDB 경로 및 컬렉션 설정
CHROMA_DB_PATH = "datasets/chroma_db"
COLLECTION_NAME = "cyberpunk2077_reviews"

def get_chroma_client():
    """ChromaDB PersistentClient 반환"""
    if not os.path.exists(CHROMA_DB_PATH):
        raise FileNotFoundError(f"ChromaDB 경로를 찾을 수 없습니다: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client

def get_embedding_function():
    """SentenceTransformer 임베딩 함수 반환"""
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

class RAGRetriever:
    def __init__(self):
        """RAGRetriever 초기화 및 컬렉션 로드"""
        self.client = get_chroma_client()
        self.embedding_fn = get_embedding_function()
        
        try:
            self.collection = self.client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=self.embedding_fn
            )
            print(f"Collection '{COLLECTION_NAME}' loaded successfully.")
        except Exception as e:
            raise ValueError(f"컬렉션 '{COLLECTION_NAME}'을 로드하는 중 오류가 발생했습니다: {e}")

    # ===========================================================
    # 수정된 부분: Time-Aware Weighted Score 적용
    # ===========================================================
    def retrieve_reviews(self, agent, current_date_str: str, top_k_final: int = 5, decay_rate: float = 0.01):
        """
        Time-Aware Weighted Score 적용
        agent: Agent 객체 (search_queries 포함)
        current_date_str: 'YYYY-MM-DD'
        top_k_final: 최종 선택 리뷰 수
        decay_rate: 시간 감쇠율
        """
        current_ts = int(pd.to_datetime(current_date_str).timestamp())

        # 쿼리 선정
        selected_queries = random.sample(agent.search_queries, min(4, len(agent.search_queries)))
        selected_queries.append(GENERAL_QUERY)

        candidate_pool = []

        for query in selected_queries:
            results = self.collection.query(
                query_texts=[query],
                n_results=100,
                include=["documents", "metadatas", "distances"],
                where={"timestamp": {"$lte": current_ts}}
            )

            if results['documents'] and results['documents'][0]:
                for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                    similarity = max(0, 1 - dist)
                    review_ts = meta['timestamp']
                    days_diff = max(0, (current_ts - review_ts) / (60 * 60 * 24))
                    time_factor = np.exp(-decay_rate * days_diff)
                    final_score = similarity * time_factor

                    candidate_pool.append({
                        "review": doc,
                        "date_str": meta.get('date_str', 'Unknown'),
                        "final_score": final_score,
                        "similarity": similarity,
                        "days_diff": int(days_diff)
                    })

        if not candidate_pool:
            return []

        # 중복 제거 및 최고 점수 유지
        unique_candidates = {}
        for item in candidate_pool:
            text = item['review']
            if text not in unique_candidates or item['final_score'] > unique_candidates[text]['final_score']:
                unique_candidates[text] = item

        candidates_list = list(unique_candidates.values())
        candidates_list.sort(key=lambda x: x['final_score'], reverse=True)
        final_docs = candidates_list[:top_k_final]

        # Team3 스타일로 변환
        return [f"- [{item['date_str']}] {item['review'][:400]}..." for item in final_docs]

# ===========================================================
# 테스트용 main
# ===========================================================
if __name__ == "__main__":
    try:
        retriever = RAGRetriever()
        
        # 테스트용 Agent 객체 예시
        class Agent:
            def __init__(self):
                self.search_queries = ["graphics", "bugs", "storyline"]

        agent = Agent()
        test_date = "2023-12-31"
        reviews = retriever.retrieve_reviews(agent, current_date_str=test_date, top_k_final=3)
        
        print(f"\n--- Testing Retrieval for date: {test_date} ---")
        for i, review in enumerate(reviews):
            print(f"\nReview {i+1}:")
            print(review[:200] + "...")
            
    except Exception as e:
        print(f"Error during test: {e}")
