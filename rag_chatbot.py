import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading papers...")
df = pd.read_csv("llm_papers.csv")

print("Loading FAISS index...")
index = faiss.read_index("llm_papers.index")

print("RAG Chatbot Ready!")
print("Type 'exit' to quit.\n")

while True:

    query = input("Ask a question: ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        3
    )

    context = ""

    for idx in indices[0]:

        context += (
            f"Title: {df.iloc[idx]['title']}\n"
            f"Abstract: {df.iloc[idx]['abstract']}\n\n"
        )

    prompt = f"""
You are an AI research assistant.

Use ONLY the information from the retrieved papers below.

{context}

Question:
{query}

Answer:
"""

    response = ollama.chat(
        model="phi3:mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\nAnswer:\n")
    print(response["message"]["content"])
    print("\n" + "=" * 80 + "\n")