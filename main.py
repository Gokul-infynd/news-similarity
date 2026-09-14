from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import List
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


class NewsItem(BaseModel):
    news_id: int | str
    content: str


class NewsRequest(BaseModel):
    news_items: List[NewsItem]


class ProcessResponse(BaseModel):
    message: str
    processed_count: int
    skipped_count: int
    deleted_count: int


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


@app.post("/process-news", response_model=ProcessResponse)
async def process_news(request: NewsRequest):
    processed_count = 0
    skipped_count = 0

    for item in request.news_items:
        if check_exists(item.news_id):
            skipped_count += 1
        else:
            upsert_news(item.news_id, item.content)
            processed_count += 1

    deleted_count = delete_old_embeddings(days=7)

    return ProcessResponse(
        message="News processing completed",
        processed_count=processed_count,
        skipped_count=skipped_count,
        deleted_count=deleted_count,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
