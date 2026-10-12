import chromadb

import config

# Open the database saved by rag_app.py and print everything in it
client = chromadb.PersistentClient(path=config.DB_DIRECTORY)
collection = client.get_collection(config.COLLECTION_NAME)
data = collection.get(include=["documents", "metadatas", "embeddings"])
print(f'Collection "{collection.name}": {collection.count()} chunks\n')

for i, chunk_id in enumerate(data["ids"]):
    embedding = data["embeddings"][i]
    print(f"--- Chunk {i + 1} (id: {chunk_id}) ---")
    print(f"Metadata:  {data['metadatas'][i]}")
    print(f"Embedding: {len(embedding)} numbers, starting {[round(float(x), 4) for x in embedding[:5]]}")
    print(f"Text:\n{data['documents'][i]}\n")
