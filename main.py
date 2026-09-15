from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import List
from dotenv import load_dotenv
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    Range,
)
import uvicorn
import os

load_dotenv()

API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

# Qdrant Configuration
client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "news_embeddings"
MODEL_NAME = "jinaai/jina-embeddings-v2-base-en"
VECTOR_SIZE = 768

# Initialize embedding model
embedding_model = TextEmbedding(model_name=MODEL_NAME)


def init_collection():
    """Create Qdrant collection if it doesn't already exist."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE,
            ),
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_collection()
    yield


app = FastAPI(title="News Similarity API", lifespan=lifespan)


def verify_api_key(key: str = Security(api_key_header)):
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")


class NewsItem(BaseModel):
    news_id: int | str
    content: str


class NewsRequest(BaseModel):
    news_items: List[NewsItem]


class MatchedNews(BaseModel):
    news_id: int | str
    similarity_score: float


class ProcessResponse(BaseModel):
    is_matched: bool
    matched_news: List[MatchedNews] | None = None


def check_exists(news_id: int | str) -> bool:
    """Check if news_id already exists in Qdrant."""
    try:
        result = client.retrieve(
            collection_name=COLLECTION_NAME,
            ids=[news_id],
        )
        return len(result) > 0
    except Exception:
        return False


def upsert_news(news_id: int | str, content: str):
    """Embed news content and store/update it in Qdrant with timestamp."""
    vector = list(embedding_model.embed([content]))[0].tolist()

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=news_id,
                vector=vector,
                payload={
                    "content": content,
                    "created_at": datetime.now(timezone.utc).timestamp(),
                },
            )
        ],
    )


def delete_old_embeddings(days: int = 7) -> int:
    """Delete embeddings older than specified days. Returns count of deleted items."""
    cutoff_timestamp = (datetime.now(timezone.utc) - timedelta(days=days)).timestamp()

    old_points = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="created_at",
                    range=Range(lt=cutoff_timestamp),
                )
            ]
        ),
        limit=1000,
    )[0]

    if old_points:
        point_ids = [point.id for point in old_points]
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=point_ids,
        )
        return len(point_ids)
    return 0


def search_similar_news(content: str, exclude_id: int | str, threshold: float = 0.80) -> list[dict]:
    """Find similar news articles above threshold, excluding the given news_id."""
    query_vector = list(embedding_model.embed([content]))[0].tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        score_threshold=threshold,
        limit=10,
    ).points

    matches = []
    for hit in results:
        if hit.id != exclude_id:
            matches.append(
                {
                    "news_id": hit.id,
                    "similarity_score": round(hit.score, 4),
                }
            )
    return matches


@app.post("/process-news", response_model=ProcessResponse, dependencies=[Depends(verify_api_key)])
async def process_news(request: NewsRequest, threshold: float = 0.80):
    all_matches = []

    for item in request.news_items:
        matches = search_similar_news(item.content, item.news_id, threshold)
        all_matches.extend(matches)

        if not check_exists(item.news_id):
            upsert_news(item.news_id, item.content)

    delete_old_embeddings(days=7)

    if all_matches:
        return ProcessResponse(
            is_matched=True,
            matched_news=[MatchedNews(**m) for m in all_matches],
        )

    return ProcessResponse(is_matched=False)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7000))
    uvicorn.run(app, host="0.0.0.0", port=port)
