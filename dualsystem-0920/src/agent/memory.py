import datetime
import math
import json
import uuid
import asyncio
import csv
from typing import List, Dict, Any, Union
from collections import deque
import google.generativeai as genai
from pathlib import Path

from src.common import (
    AgentProfile,
    MemoryBase,
    MemorySnapshot,
    MemoryEpisode,
    MemoryHeuristic
)
from src.agent.storage import ChromaMemoryStore, LocalImportanceScorer
from config import CONFIG


def calculate_recency_score(last_accessed: datetime.datetime, current_time: datetime.datetime, decay_factor=CONFIG.memory.decay_factor) -> float:
    delta_hours = (current_time - last_accessed).total_seconds() / 3600.0
    return math.pow(decay_factor, max(0, delta_hours))


def normalize_importance(score: int) -> float:
    return score / 10.0


class MemoryStream:
    _csv_log_path = None
    _db_path = None          # vector DB path (per-run, injected by main.py)
    _shared_scorer = None
    _current_day_index = 1   # class-level current day counter

    def __init__(self, owner_id: str, profile: AgentProfile):
        self.owner_id = owner_id
        self.profile = profile

        # disable_memory=True fully disables memory (no model load / retrieve / write).
        self.enabled = not CONFIG.ablation.disable_memory

        if self.enabled:
            db_path = MemoryStream._db_path or CONFIG.paths.chroma_db
            self.store = ChromaMemoryStore(owner_id, db_path=db_path)
        else:
            self.store = None  # memoryless mode: skip model load to save startup time

        self.alpha_recency = CONFIG.memory.alpha_recency
        self.beta_importance = CONFIG.memory.beta_importance
        self.gamma_relevance = CONFIG.memory.gamma_relevance

        self.importance_buffer = 0
        self.REFLECTION_THRESHOLD = CONFIG.memory.reflection_threshold
        self.SNAPSHOT_STORAGE_THRESHOLD = CONFIG.memory.snapshot_storage_threshold

        self.recent_memories_buffer = deque(maxlen=CONFIG.memory.recent_buffer_maxlen)

    @classmethod
    def set_log_path(cls, path):
        cls._csv_log_path = path

    @classmethod
    def set_db_path(cls, path):
        """Set the vector DB path (per-run, so memories don't accumulate across runs)."""
        cls._db_path = str(path)

    @classmethod
    def set_current_day(cls, day: int):
        cls._current_day_index = day

    def _calculate_importance(self, mem_type: str, content: Any) -> int:
        """Hybrid heuristic scoring based on type, structured evaluation, and semantics."""
        if mem_type == "HEURISTIC":
            return 10

        if mem_type == "EPISODE":
            eval_tag = getattr(content, "evaluation", "ACCEPTABLE")
            if eval_tag in ["FAILURE", "REGRET"]:
                return 9   # traumatic memory
            elif eval_tag == "SUCCESS":
                return 7   # successful experience
            else:
                return 5   # ordinary experience

        if mem_type == "SNAPSHOT":
            text = content.description.lower()
            if any(w in text for w in ['critical', 'panic', 'emergency', 'surge', 'fail']):
                return 8
            if any(w in text for w in ['price', 'cost', 'queue', 'wait']):
                return 6
            return 3

        return 3

    def add_snapshot(self, description: str, importance: int = None, tags: Dict = None):
        mem = MemorySnapshot(
            description=description,
            importance=0,
            context_tags=tags or {}
        )
        if importance is not None:
            mem.importance = importance
        else:
            mem.importance = self._calculate_importance("SNAPSHOT", mem)

        self._process_new_memory(mem)

    def add_episode(self, trigger: str, mental_state: str, decision: str, outcome: str, evaluation: str):
        mem = MemoryEpisode(
            trigger_event=trigger,
            mental_state=mental_state,
            decision=decision,
            outcome=outcome,
            evaluation=evaluation,
            importance=0
        )
        mem.importance = self._calculate_importance("EPISODE", mem)
        self._process_new_memory(mem)

    def add_heuristic(self, rule_text: str, day_index: int):
        """Add a rule (Type C); rules always carry the highest weight."""
        mem = MemoryHeuristic(
            rule_text=rule_text,
            derived_from_day=day_index,
            importance=10
        )
        self._process_new_memory(mem)

    def add_memory(self, description: str, current_time: datetime.datetime,
                   source: str = "Self", mental_state: str = "N/A",
                   soc: float = -1.0, importance: int = None,
                   context_tags: Dict[str, Any] = None):
        """Compatibility wrapper around add_snapshot (converts legacy args into a Snapshot)."""
        tags = context_tags or {}
        tags.update({"original_soc": soc, "original_mental": mental_state})

        mem = MemorySnapshot(
            description=description,
            importance=importance if importance else self._fast_evaluate_importance(description),
            source=source,
            created_at=current_time,
            context_tags=tags
        )
        self._process_new_memory(mem)

    def _process_new_memory(self, mem: MemoryBase):
        """Unified pipeline: cache -> trigger reflection -> async persistence."""
        if not self.enabled:
            return  # memoryless mode: discard entirely

        self.recent_memories_buffer.append(mem)

        self.importance_buffer += mem.importance
        if self.importance_buffer > self.REFLECTION_THRESHOLD:
            self.importance_buffer = 0

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._persist_memory_task(mem))
        except RuntimeError:
            asyncio.run(self._persist_memory_task(mem))

    async def _persist_memory_task(self, mem: MemoryBase):
        """Background task: serialize + store (CSV + vector DB)."""
        write_to_db = True
        if mem.type == "SNAPSHOT" and mem.importance < self.SNAPSHOT_STORAGE_THRESHOLD:
            write_to_db = False

        text_for_embedding = mem.get_semantic_content()

        # CSV write runs in an executor to avoid blocking the event loop.
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                self._append_to_csv,
                mem,
                text_for_embedding
            )
        except Exception as e:
            print(f"⚠️ CSV Logging Failed: {e}")

        if write_to_db:
            metadata = {
                "created_at": mem.created_at.isoformat(),
                "importance": mem.importance,
                "source": mem.source,
                "type": mem.type,
                "evaluation": getattr(mem, "evaluation", "N/A"),
                "last_accessed": mem.created_at.isoformat()
            }
            if hasattr(mem, "context_tags"):
                metadata.update(mem.context_tags)

            await asyncio.to_thread(
                self.store.add,
                documents=[text_for_embedding],
                metadatas=[metadata],
                ids=[mem.id]
            )

    def _append_to_csv(self, mem: MemoryBase, content_str: str):
        """Dual-write: raw memories (all types) plus episode logs (episodes only)."""
        time_str = mem.created_at.strftime("%Y-%m-%d %H:%M:%S")
        role = getattr(self.profile, 'role_type', 'Unknown')
        day = MemoryStream._current_day_index

        if MemoryStream._csv_log_path:
            try:
                with open(MemoryStream._csv_log_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        day,
                        time_str,
                        self.owner_id,
                        mem.type,
                        mem.source,
                        mem.importance,
                        content_str
                    ])
            except Exception as e:
                print(f"❌ Write Raw CSV Error: {e}")

        if mem.type == "EPISODE" and MemoryStream._csv_log_path:
            try:
                raw_path = Path(MemoryStream._csv_log_path)
                episode_path = raw_path.parent / "episode_logs.csv"

                with open(episode_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        day,
                        time_str,
                        self.owner_id,
                        role,
                        mem.trigger_event,
                        mem.mental_state,
                        mem.decision,
                        mem.outcome,
                        mem.evaluation,
                        mem.importance
                    ])
            except Exception as e:
                print(f"❌ Write Episode CSV Error: {e}")

    def retrieve(self, query: str, current_time: datetime.datetime, top_k: int = CONFIG.memory.default_top_k, metadata_filter: Dict = None) -> List[str]:
        """Hybrid retrieval (recency + importance + relevance)."""
        if not self.enabled:
            return []

        n_candidates = top_k * 3
        results = self.store.query(query, n_results=n_candidates, where=metadata_filter)

        if not results or not results['documents'] or not results['documents'][0]:
            return []

        candidates = []
        ids = results['ids'][0]
        docs = results['documents'][0]
        metas = results['metadatas'][0]
        distances = results['distances'][0]

        for i in range(len(ids)):
            doc_text = docs[i]
            meta = metas[i]
            dist = distances[i]

            relevance = max(0.0, 1.0 - dist)
            try:
                last_accessed = datetime.datetime.fromisoformat(meta['last_accessed'])
            except:
                last_accessed = current_time
            recency = calculate_recency_score(last_accessed, current_time)
            importance = normalize_importance(meta['importance'])

            total_score = (self.alpha_recency * recency +
                           self.beta_importance * importance +
                           self.gamma_relevance * relevance)

            candidates.append((total_score, doc_text, meta, ids[i]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [item[1] for item in candidates[:top_k]]

    def retrieve_relevant_rules(self, state_keywords: str, current_time: datetime.datetime) -> List[str]:
        """Rule-specific retrieval channel."""
        rule_query = f"Rules, advice, and lessons learned about: {state_keywords}"
        return self.retrieve(
            query=rule_query,
            current_time=current_time,
            top_k=CONFIG.memory.default_top_k,
            metadata_filter={"type": "HEURISTIC"}
        )

    def reflect(self, current_time: datetime.datetime):
        """Reflection: distill significant recent memories into one heuristic."""
        if not self.enabled:
            return

        significant_memories = []
        for m in self.recent_memories_buffer:
            if m.importance > 5:
                content = m.get_semantic_content()
                significant_memories.append(content)

        if not significant_memories:
            return

        prompt = f"""
        System: You are {self.profile.name}.
        Task: Analyze recent memories and generate one high-level insight about your driving/charging habits.
        Memories: {json.dumps(significant_memories[-10:])}
        Output: A single sentence insight.
        """
        try:
            model = genai.GenerativeModel(CONFIG.memory.reflect_model)
            response = model.generate_content(prompt)
            insight = response.text.strip()
            self.add_heuristic(insight, 0)
            print(f"💡 Generated insight: {insight}")
        except Exception as e:
            print(f"Reflect Error: {e}")

    def query(self, query_text: str, n_results: int = 5, where: Dict[str, Any] = None):
        if not self.enabled:
            return {"documents": [[]], "metadatas": [[]], "ids": [[]], "distances": [[]]}
        return self.store.query(query_text, n_results, where)
