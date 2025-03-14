from dataclasses import dataclass, field
from typing import Any
import chromadb
import json
from rich import print
from uuid import uuid4


@dataclass(kw_only=True)
class VStore:
    collection_name: str = "roam_messages"
    chroma_client: Any = field(init=False)
    collection: chromadb.Collection = field(init=False)

    def __post_init__(self):
        self.chroma_client = chromadb.Client()
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )

    def update_content(self, content, index):
        r = self.collection.get(where={"index": index})["ids"]
        if len(r) > 0:
            id = r[0]
            self.collection.upsert(
                documents=[content],
                ids=[id],
            )
        else:
            self.collection.upsert(
                documents=[content],
                ids=[str(uuid4())],
                metadatas=[{"role": "user", "index": index}],
            )

    def change_system(self, content):
        self.update_content(content, 0)

    def add_messages(self, messages, on_index=0):
        messages = [
            {"m": message, "index": on_index + i}
            for i, message in enumerate(messages)
            if "content" in message and message["role"] != "tool"
        ]

        if messages:
            self.collection.upsert(
                documents=[message["m"]["content"] for message in messages],
                ids=[str(uuid4()) for _ in messages],
                metadatas=[
                    {"role": message["m"]["role"], "index": message["index"]}
                    for message in messages
                ],
            )

    def purge(self):
        self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name
        )

    def delete_last(self, count, n, keep=None):
        indexes = [str(i) for i in range(count - n, count - (keep or 0))]
        ids = self.collection.get(
            where={"index": {"$in": indexes}},  # type: ignore
        )["ids"]
        self.collection.delete(ids)
        # update the index and ids of the remaining messages
        update = (keep or 0) - n
        r = self.collection.get(where={"index": {"$gt": count - (keep or 0)}})
        metadatas = r["metadatas"] or []
        for i, metadata in enumerate(metadatas):
            metadata["index"] -= update  # type: ignore

        update_ids = r["ids"] or []

        self.collection.upsert(ids=update_ids, metadatas=metadatas)

    def peek_last_ids(self, n, from_index):
        r = self.collection.get(where={"index": {"$lt": from_index}})
        ids = r["ids"] or []
        indexes = [m["index"] for m in r["metadatas"] or []]

        # return r sorted by index
        combined = list(zip(indexes, ids))
        combined.sort(key=lambda x: x[0])
        return [c[1] for c in combined[-n:]]

    def query(
        self,
        query_text,
        n_results,
        before_index,
        role=None,
    ) -> list[dict]:
        where = {"$and": [{"index": {"$lt": before_index}}, {"index": {"$gt": 0}}]}

        if role:
            where["$and"] += [{"role": role}]

        results = self.collection.query(
            query_texts=query_text,
            where=where,  # type: ignore
            n_results=n_results,
        )

        results = list(
            zip((results["documents"] or [])[0], (results["metadatas"] or [])[0])
        )
        return [
            {"role": r[1]["role"], "content": r[0]}
            for r in sorted(results, key=lambda x: x[1]["index"])
        ]

    def message_count(self):
        return self.collection.count()
