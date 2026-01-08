import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

LOG_FILE = Path("CHAT-LOGS.json")
LOG_FILE.touch(exist_ok=True)

class MemoryStore:
    def __init__(self, max_messages: int = 5):
        self.max_messages = max_messages
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.memory: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        if LOG_FILE.stat().st_size == 0:
            return []
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2)

    def add(self, role: str, content: str):
        embedding = self.model.encode(content).tolist()

        entry = {
            "role": role,
            "content": content,
            "embedding": embedding,
            "timestamp": datetime.now().isoformat()
        }

        self.memory.append(entry)

        # keep only last N messages
        self.memory = self.memory[-self.max_messages :]
        self._save()

    def get_recent(self) -> List[Dict]:
        return self.memory

    def get_context_text(self) -> str:
        """
        Returns formatted conversation history for prompting
        """
        return "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in self.memory
        )
