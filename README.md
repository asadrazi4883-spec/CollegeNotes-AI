# CollegeNotes-AI — Local RAG Study Assistant

CollegeNotes-AI is a Python-based study assistant that helps students find answers from their PDF study notes using Retrieval-Augmented Generation (RAG).

The application retrieves relevant information from PDF documents using semantic similarity and generates answers using a locally running AI model.

## Features

- Extracts text from PDF study materials.
- Splits extracted text into meaningful chunks.
- Generates embeddings using EmbeddingGemma.
- Converts questions into embeddings.
- Uses cosine similarity to retrieve relevant passages.
- Retrieves the top 3 relevant passages.
- Uses a similarity threshold of 0.45.
- Generates answers using Gemma 3:1b.
- Avoids generating answers when relevant information is unavailable.
- Runs locally using Ollama without requiring an API key.

## Technologies Used

- Python
- Ollama
- EmbeddingGemma
- Gemma 3:1b
- NumPy
- PyPDF

## Project Workflow

1. Load a PDF document.
2. Extract text from the PDF.
3. Divide the text into meaningful chunks.
4. Generate embeddings for each chunk.
5. Convert the user's question into an embedding.
6. Calculate cosine similarity between the question and document embeddings.
7. Retrieve the top 3 relevant passages above the similarity threshold.
8. Generate an answer using Gemma 3:1b.
9. If no relevant passages are found, display an appropriate message.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/asadrazi4883-spec/CollegeNotes-AI.git
cd CollegeNotes-AI
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

Download Ollama from:

https://ollama.com/

### 4. Download the required models

```bash
ollama pull embeddinggemma
ollama pull gemma3:1b
```

### 5. Run the application

```bash
python main.py
```

## Models Used

- **Embedding Model:** EmbeddingGemma
- **Answer Generation Model:** Gemma 3:1b

Both models run locally through Ollama.

## Future Improvements

- Support multiple PDF documents.
- Improve document chunking and retrieval accuracy.
- Add a simple user interface.
- Improve answer relevance and response quality.

## Author

Asad Razi

GitHub: https://github.com/asadrazi4883-spec