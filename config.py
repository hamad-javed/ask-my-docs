# Settings shared by rag_app.py, view_db.py and visualize_db.py

# Models (both run locally through Ollama)
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3"

# Document used when no --file is given
DEFAULT_SOURCE_FILE = "data/sample_handbook.txt"

# Chunking: ~500 characters per chunk, 50 characters of overlap between chunks
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Vector database
DB_DIRECTORY = "chroma_db"
COLLECTION_NAME = "documents"

# How many chunks to retrieve for each question
TOP_K = 4
