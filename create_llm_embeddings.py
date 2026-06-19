import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

print("Loading dataset...")

df = pd.read_csv("llm_papers.csv")

texts = (
    df["title"].fillna("")
    + " "
    + df["abstract"].fillna("")
)

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Creating embeddings...")

embeddings = model.encode(
    texts.tolist(),
    show_progress_bar=True
)

np.save(
    "llm_embeddings.npy",
    embeddings
)

print("\nEmbeddings saved!")
print("Shape:", embeddings.shape)