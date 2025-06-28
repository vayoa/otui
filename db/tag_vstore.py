# %%
import pandas as pd
import chromadb  # ver 0.6.3, onnxruntime 1.18.1
from rich import print
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

ef = ONNXMiniLM_L6_V2(preferred_providers=["CUDAExecutionProvider"])

TAG_PARQUET = "./db/danbooru_tags.parquet"
TAG_COLLECTION = "tag_collection"


def get_tags() -> pd.DataFrame:
    tags = pd.read_parquet(TAG_PARQUET)
    tags["consequent_aliases"] = tags["consequent_aliases"].str.join(", ")
    tags["wiki_other_names"] = tags["wiki_other_names"].str.join(", ")
    tags = tags.sort_values(by="post_count", ascending=False)
    return tags


def upsert_tags(collection: chromadb.Collection, tags: pd.DataFrame) -> None:
    collection.upsert(
        documents=tags["document"].tolist(),
        ids=tags["id"].astype(str).tolist(),
        metadatas=tags.drop(columns=["id", "document", "description"]).to_dict(
            "records"
        ),
    )


# if __name__ == "__main__":
chroma_client = chromadb.PersistentClient(path="./db/tag_vstore")

# NOTICE: This will delete the collection and all its data, uncomment to use
# # chroma_client.delete_collection(name=TAG_COLLECTION)
collection = chroma_client.get_or_create_collection(
    name=TAG_COLLECTION, embedding_function=ef
)

tags = get_tags()
tags = tags.iloc[:5000]
upsert_tags(collection, tags)
print("Tags upserted.")

# results = collection.query(
#     query_texts=["tits"],  # Chroma will embed this for you
#     n_results=2,  # how many results to return
# )
# print(results)
# print("Query done.")

# %%
