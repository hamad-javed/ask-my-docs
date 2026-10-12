# Ask My Docs

A small, fully local RAG (Retrieval-Augmented Generation) app that answers questions about your own documents, such as a resume. It was built as a learning project to understand how RAG works end to end.

Everything runs on your machine through [Ollama](https://ollama.com), so your documents never leave your laptop.

## How it works

**Preparing the document (once, at startup)**

1. **Read**: load the text from a `.txt` or `.docx` file
2. **Split**: cut it into chunks of about 500 characters
3. **Embed**: turn each chunk into a list of numbers that captures its meaning (`nomic-embed-text`)
4. **Store**: save the chunks and their embeddings in Chroma DB

**Answering a question**

5. **Embed the question** the same way
6. **Search**: Chroma finds the 4 chunks closest in meaning to the question
7. **Prompt**: the chunks and the question are combined into one prompt
8. **Answer**: Llama 3 writes the answer using only those chunks

| Tool | Role |
|---|---|
| LangChain | Connects the steps into one pipeline |
| Ollama | Runs both AI models locally |
| nomic-embed-text | Turns text into embeddings for meaning-based search |
| Chroma DB | Stores the embeddings and finds the closest matches |
| Llama 3 | Reads the retrieved chunks and writes the answer |

## Setup

1. Install [Ollama](https://ollama.com/download) and download the two models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull llama3
   ```
2. Install the Python packages (Python 3.10+):
   ```bash
   python3 -m pip install -r requirements.txt
   ```

## Usage

Ask questions interactively about the sample handbook:

```bash
python3 rag_app.py
```

Use your own document:

```bash
python3 rag_app.py --file my_resume.docx
```

Ask a single question and exit:

```bash
python3 rag_app.py --file my_resume.docx "What are the main technical skills?"
```

### Look inside the database

```bash
python3 view_db.py                               # print every chunk and its embedding
python3 visualize_db.py "What was the last job?" # interactive map in your browser
```

`visualize_db.py` plots every chunk on a 2D map (chunks with similar meaning sit closer together) and highlights the chunks retrieved for your question. Both scripts read the database built by the most recent run of `rag_app.py`.

## Configuration

Models, chunk size, the number of retrieved chunks and the default document are set in [`config.py`](config.py).

## Project structure

```
rag_app.py               main app: load, chunk, embed, retrieve, answer
config.py                shared settings
view_db.py               print the contents of the vector database
visualize_db.py          browser map of the chunks and a question
data/sample_handbook.txt sample document (a fictional employee handbook)
```

`.docx` files are git-ignored so personal documents such as resumes aren't committed by accident.
