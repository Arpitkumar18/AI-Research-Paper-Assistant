import faiss
import numpy as np

print("Loading embeddings...")

embeddings = np.load("llm_embeddings.npy")

print("Shape:", embeddings.shape)

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(
    index,
    "llm_papers.index"
)

print("LLM FAISS Index Created Successfully!")