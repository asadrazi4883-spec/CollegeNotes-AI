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

## Project Screenshots

### 1. Question Embedding and Cosine Similarity

![Project Screenshot 1](screenshots/project%20screenshot%201.jpg)

### 2. Retrieved Passages and Generated Answer

![Project Screenshot 2](screenshots/project%20screenshot%202.jpg)

## Example Execution – K-Means Clustering

**Sample Question:**

What is K-Means Clustering?

**Retrieved Passages:**

The application retrieved the following relevant information:

- **Chunk 17 — Similarity: 0.6280:** K-Means Clustering explanation and algorithm steps.
- **Chunk 1 — Similarity: 0.5073:** Introduction to Machine Learning.
- **Chunk 16 — Similarity: 0.5037:** Unsupervised Learning and Clustering.

**Generated Answer:**

K-Means clustering is a technique used to group unlabelled data into k distinct clusters based on feature similarities. It starts by randomly selecting k center points and then assigns each data point to the closest center. The process continues until the centroids stabilize.

In short, it is a method for organizing data into groups where similar data points are grouped together.

**Retrieval Result:**

- Total document chunks: 23
- Embedding dimensions: 768
- Retrieved passages: 3
- Similarity threshold: 0.45
- Answer generation model: Gemma 3:1b

The application successfully retrieved the relevant K-Means Clustering passage and generated an answer using the retrieved context.

## Author

Asad Razi

GitHub: https://github.com/asadrazi4883-spec
