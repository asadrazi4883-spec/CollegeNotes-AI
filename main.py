
import re
import ollama
import numpy as np
from pypdf import PdfReader


# ==========================================
# 1. EXTRACT TEXT FROM PDF
# ==========================================

reader = PdfReader("Machine_learning.pdf")

text = ""

for page in reader.pages:
    text += (page.extract_text() or "") + "\n"

print("PDF extracted successfully!")
print("Total characters:", len(text))


# ==========================================
# 2. CLEAN PDF TEXT
# ==========================================

lines = text.splitlines()

cleaned_lines = []

for line in lines:

    line = line.strip()

    if not line:
        continue

    # Remove PDF page footers
    line = re.sub(
        r"ML Study Guide\s*[—-]\s*Beginner Series\s*Page\s*\d+\s*of\s*10",
        "",
        line
    ).strip()

    if not line:
        continue

    # Remove page heading numbers
    line = re.sub(r"^Page\s+\d+:\s*", "", line).strip()

    if not line:
        continue

    # Remove document title
    if line == "Machine Learning Beginner Guide":
        continue

    cleaned_lines.append(line)


# ==========================================
# 3. CREATE MEANINGFUL PASSAGE CHUNKS
# ==========================================

chunks = []
current = ""

paradigm_section = False
classification_section = False

reinforcement_lines = []

classification_headings = [
    "Logistic Regression",
    "Support Vector Machines (SVM)",
    "Decision Trees & Random Forests",
    "How Decision Trees Work",
    "Ensemble Learning: Random Forests"
]

for line in cleaned_lines:

    # Detect learning paradigms table
    if "Paradigm Data Type Goal Common Example" in line:

        if current:
            chunks.append(current)

        current = line
        paradigm_section = True
        reinforcement_lines = []
        continue

    # End learning paradigms table
    if paradigm_section and line == 'How Machines "Learn"':

        # Preserve the complete original table
        chunks.append(current)

        # Create a dedicated Reinforcement Learning passage
        if reinforcement_lines:

            reinforcement_chunk = (
                "Reinforcement Learning\n"
                + "\n".join(reinforcement_lines)
            )

            chunks.append(reinforcement_chunk)

        current = ""
        paradigm_section = False

        # Process the next heading normally below

    elif paradigm_section:

        current += "\n" + line

        # Capture Reinforcement Learning table information
        if line == "Reinforcement":

            reinforcement_lines = [line]

        elif reinforcement_lines:

            reinforcement_lines.append(line)

        continue

    # Detect classification section
    if line == "Supervised Learning — Classification":

        if current:
            chunks.append(current)

        current = line
        classification_section = True
        continue

    # End classification section
    if classification_section and line == "Unsupervised Learning & Clustering":

        if current:
            chunks.append(current)

        current = ""
        classification_section = False

        # Process the new heading normally below

    elif classification_section:

        # Start a new passage for each algorithm heading
        if line in classification_headings:

            if current:
                chunks.append(current)

            current = line

        else:

            if current:
                current += "\n" + line

        continue

    # Normal heading detection
    is_heading = (
        line.endswith("?")
        or (
            12 <= len(line) <= 80
            and not line.endswith((".", ",", ":", ";"))
            and not line[0].isdigit()
        )
    )

    if is_heading:

        if current:
            chunks.append(current)

        current = line

    else:

        if current:
            current += "\n" + line
        else:
            current = line


# Save the final passage
if current:
    chunks.append(current)


# ==========================================
# DISPLAY CLASSIFICATION PASSAGES
# ==========================================

for i, chunk in enumerate(chunks):

    if (
        any(heading in chunk for heading in classification_headings)
        or "Supervised Learning — Classification" in chunk
    ):

        print(f"\nClassification Chunk {i+1}:")
        print(chunk)
        print("-" * 50)


# ==========================================
# DISPLAY REINFORCEMENT LEARNING PASSAGE
# ==========================================

for i, chunk in enumerate(chunks):

    if chunk.startswith("Reinforcement Learning"):

        print(f"\nReinforcement Learning Chunk {i+1}:")
        print(chunk)
        print("-" * 50)


print("\nTotal chunks created:", len(chunks))


# ==========================================
# 4. GENERATE DOCUMENT EMBEDDINGS
# ==========================================

response = ollama.embed(
    model="embeddinggemma",
    input=chunks
)

embeddings = np.array(response["embeddings"])

print("\nEmbeddings generated successfully!")
print("Embedding shape:", embeddings.shape)


# ==========================================
# 5. TAKE QUESTION FROM USER
# ==========================================

question = input("\nEnter your question: ")

response = ollama.embed(
    model="embeddinggemma",
    input=question
)

question_embedding = np.array(
    response["embeddings"][0]
)

print("\nQuestion embedding generated!")
print("Question vector shape:", question_embedding.shape)


# ==========================================
# 6. CALCULATE COSINE SIMILARITY
# ==========================================

similarity_scores = np.dot(
    embeddings,
    question_embedding
) / (
    np.linalg.norm(embeddings, axis=1)
    * np.linalg.norm(question_embedding)
)

print("\nCosine Similarity Scores:")

for i, score in enumerate(similarity_scores):

    print(f"Chunk {i + 1}: {score:.4f}")


# ==========================================
# 7. RETRIEVE TOP 3 RELEVANT PASSAGES
# ==========================================

threshold = 0.45

top_indices = np.argsort(similarity_scores)[::-1][:3]

print("\nRetrieved Passages:")

retrieved_passages = []

for index in top_indices:

    score = similarity_scores[index]

    if score >= threshold:

        retrieved_passages.append(chunks[index])

        print(f"\nSimilarity Score: {score:.4f}")
        print(chunks[index])
        print("-" * 50)

if not retrieved_passages:
    print("\nNo sufficiently relevant information found.")


# ==========================================
# 8. GENERATE ANSWER USING GEMMA
# ==========================================

if retrieved_passages:

    context = "\n\n".join(retrieved_passages)

    prompt = f"""
You are a helpful college study assistant.

Answer the student's question using the provided context.

Instructions:
1. Read all the provided passages carefully.
2. Extract relevant information from paragraphs and tables.
3. Combine information from multiple passages when needed.
4. Explain the answer in simple and clear language.
5. Do not copy the context word for word.
6. Do not use outside knowledge.
7. If the context does not contain enough information,
   say: "The information is not available in the provided notes."

Context:
{context}

Question:
{question}

Answer:
"""

    print("\nGenerating answer using Gemma...\n")

    try:

        response = ollama.generate(
            model="gemma3:1b",
            prompt=prompt
        )

        print("Generated Answer:")
        print(response["response"])

    except Exception as e:

        print("Error generating answer:", e)

else:

    print("\nAnswer generation skipped because no relevant passages were found.")

