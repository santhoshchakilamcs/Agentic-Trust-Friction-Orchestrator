"""
Long-Term Memory: Vector database for user context, life events, and historical patterns.
Uses ChromaDB for local, zero-API-key vector storage.
"""

import json
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from agentic_orchestrator.config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from agentic_orchestrator.data.generator import UserProfile


class LongTermMemory:
    """Persistent vector memory for user personas and life events."""

    def __init__(self):
        self._client = chromadb.Client(Settings(
            persist_directory=CHROMA_PERSIST_DIR,
            anonymized_telemetry=False,
        ))
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_user_profiles(self, profiles: List[UserProfile]):
        """Load user profiles into the vector store."""
        documents = []
        metadatas = []
        ids = []

        for profile in profiles:
            # Build a rich text document for each user
            life_event_text = "; ".join(
                [e["event"] for e in profile.life_events]
            ) if profile.life_events else "No known life events"

            doc = (
                f"User {profile.name} (ID: {profile.user_id}) from {profile.country}. "
                f"Typical corridor: {profile.typical_corridor}. "
                f"Typical amount: ${profile.typical_amount:.2f}. "
                f"Regular recipients: {', '.join(profile.typical_recipients)}. "
                f"Life events: {life_event_text}. "
                f"Registered device: {profile.registered_device_id}."
            )
            documents.append(doc)
            metadatas.append({
                "user_id": profile.user_id,
                "country": profile.country,
                "typical_amount": profile.typical_amount,
                "corridor": profile.typical_corridor,
                "has_life_events": len(profile.life_events) > 0,
                "recipients": json.dumps(profile.typical_recipients),
                "device_id": profile.registered_device_id,
            })
            ids.append(profile.user_id)

        self._collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def query_user_context(self, user_id: str, query_text: str, n_results: int = 3) -> Dict:
        """Retrieve relevant context for a user given a query."""
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where={"user_id": user_id},
        )

        if not results["documents"] or not results["documents"][0]:
            return {"found": False, "documents": [], "metadata": []}

        return {
            "found": True,
            "documents": results["documents"][0],
            "metadata": results["metadatas"][0],
        }

    def get_user_profile(self, user_id: str) -> Optional[Dict]:
        """Get stored profile data for a specific user."""
        results = self._collection.get(ids=[user_id])
        if not results["documents"]:
            return None
        return {
            "document": results["documents"][0],
            "metadata": results["metadatas"][0],
        }

    def add_feedback(self, user_id: str, feedback: str):
        """Add human feedback / override reason to user context."""
        existing = self.get_user_profile(user_id)
        if existing:
            updated_doc = existing["document"] + f" Feedback: {feedback}."
            self._collection.update(
                ids=[user_id],
                documents=[updated_doc],
            )

    def clear(self):
        """Reset the memory store."""
        self._client.delete_collection(CHROMA_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

