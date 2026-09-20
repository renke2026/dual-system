"""ChromaDB memory storage with local embedding and importance scoring."""

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
import numpy as np
from config import CONFIG


_SHARED_MODEL_INSTANCE = None


class LocalEmbeddingFunction(EmbeddingFunction):
    """Local vectorization via Sentence-Transformers (no API key, no rate limit)."""

    def __init__(self, model_name: str = CONFIG.storage.embedding_model):
        global _SHARED_MODEL_INSTANCE

        if _SHARED_MODEL_INSTANCE is None:
            print(f"📥 [Storage] Loading global embedding model: {model_name} (loaded once)...")
            _SHARED_MODEL_INSTANCE = SentenceTransformer(model_name)
            print("✅ [Storage] Global embedding model loaded.")

        self.model = _SHARED_MODEL_INSTANCE

    def __call__(self, input: Documents) -> Embeddings:
        if isinstance(input, str):
            input = [input]
        embeddings = self.model.encode(input).tolist()
        return embeddings


class ChromaMemoryStore:
    def __init__(self, agent_id: str, db_path: str = CONFIG.paths.chroma_db):
        self.agent_id = agent_id
        self.client = chromadb.PersistentClient(path=db_path)
        self.ef = LocalEmbeddingFunction()

        # All agents share a single collection, filtered by agent_id at query time.
        SHARED_COLLECTION_NAME = CONFIG.storage.collection_name

        try:
            self.collection = self.client.get_or_create_collection(
                name=SHARED_COLLECTION_NAME,
                embedding_function=self.ef,
                metadata={"hnsw:space": CONFIG.storage.hnsw_space}
            )
        except Exception as e:
            print(f"❌ [Storage] Init Error: {e}")
            self.collection = self.client.get_or_create_collection(
                name=SHARED_COLLECTION_NAME,
                embedding_function=self.ef
            )

    def add(self, documents, metadatas, ids):
        """Add memories to the shared collection, tagging each with agent_id."""
        if not documents:
            return

        for meta in metadatas:
            meta["agent_id"] = self.agent_id

        try:
            self.collection.add(documents=documents, metadatas=metadatas, ids=ids)
        except Exception as e:
            print(f"❌ [Storage] Add Error: {e}")

    def query(self, query_text, n_results=5, where: Optional[Dict] = None):
        """Query memories, always filtered by agent_id."""
        try:
            agent_filter = {"agent_id": self.agent_id}

            if where:
                if "$and" in where:
                    where["$and"].append(agent_filter)
                    final_where = where
                else:
                    final_where = {"$and": [agent_filter, where]}
            else:
                final_where = agent_filter

            return self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=final_where,
                include=['documents', 'metadatas', 'distances']
            )
        except Exception as e:
            err_msg = str(e)
            if "Nothing found" in err_msg or "HNSW" in err_msg:
                return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}
            print(f"❌ [Storage] Query Error: {e}")
            return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}

    def count(self):
        """Return the memory count for this agent (not the whole collection)."""
        return self.collection.count(where={"agent_id": self.agent_id})


class LocalImportanceScorer:
    """Score memory importance with a local TinyBERT model instead of an LLM call."""

    def __init__(self, model_name: str = CONFIG.storage.scorer_model):
        print(f"📥 [Scorer] Loading local scoring model: {model_name}...")
        self.model = CrossEncoder(model_name)
        self.benchmark_query = "This is a critical, dangerous, urgent, or highly emotional event."
        print("✅ [Scorer] Scoring model loaded.")

    def score(self, text: str) -> int:
        """Return an integer importance score in 1-10."""
        try:
            logit_score = self.model.predict([(self.benchmark_query, text)])[0]

            # Sigmoid maps the logit to 0-1.
            normalized_score = 1 / (1 + np.exp(-logit_score))

            # Map to 1-10 (TinyBERT scores are conservative, so amplify a bit).
            final_score = int(normalized_score * 9) + 1

            # Keyword boost: force a high score for critical keywords.
            keywords = ["fail", "error", "critical", "panic", "low battery", "expensive"]
            if any(k in text.lower() for k in keywords):
                final_score = max(final_score, 8)

            return max(1, min(10, final_score))

        except Exception as e:
            print(f"❌ [Scorer] Error: {e}")
            return 5


if __name__ == "__main__":
    print("Testing ChromaStorage...")
    store = ChromaMemoryStore(agent_id="test_user_001")

    store.add(
        documents=["I charged at Station A yesterday.", "I hate traffic jams."],
        metadatas=[{"time": "2025-05-20", "importance": 8}, {"time": "2025-05-21", "importance": 3}],
        ids=["mem_1", "mem_2"]
    )
    print(f"Stored {store.count()} memories.")

    res = store.query("traffic condition", n_results=1)
    print("Query Result:", res['documents'][0])
