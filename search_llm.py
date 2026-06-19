import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

print("Loading papers...")

df = pd.read_csv("llm_papers.csv")

print("Loading index...")

index = faiss.read_index("llm_papers.index")

print("Loading model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

while True:

    query = input("\nAsk a question: ")

    if query.lower() == "exit":
        break

    query_embedding = model.encode([query])

    D, I = index.search(
        query_embedding,
        k=5
    )

    print("\nTop 5 Relevant Papers:\n")

    for idx in I[0]:

        print("=" * 80)

        print("TITLE:")
        print(df.iloc[idx]["title"])

        print("\nABSTRACT:")
        print(df.iloc[idx]["abstract"][:600])

        print()