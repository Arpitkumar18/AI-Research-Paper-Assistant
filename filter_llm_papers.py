import json
import pandas as pd

print("Started...")

AI_CATEGORIES = [
    "cs.AI",
    "cs.CL",
    "cs.LG",
    "cs.CV"
]

KEYWORDS = [
    "bert",
    "gpt",
    "llama",
    "large language model",
    "retrieval augmented generation",
    "rag",
    "vision transformer",
    "diffusion model",
    "instruction tuning",
    "generative ai"
]

papers = []

with open("arxiv-metadata-oai-snapshot.json", "r", encoding="utf-8") as f:

    for i, line in enumerate(f):

        if i % 100000 == 0:
            print(f"Processed {i}")

        try:
            paper = json.loads(line)

            categories = paper.get("categories", "")

            if not any(cat in categories for cat in AI_CATEGORIES):
                continue

            text = (
                paper.get("title", "").lower()
                + " "
                + paper.get("abstract", "").lower()
            )

            if any(keyword in text for keyword in KEYWORDS):

                papers.append({
                    "title": paper.get("title"),
                    "abstract": paper.get("abstract"),
                    "authors": paper.get("authors"),
                    "categories": categories
                })

        except:
            pass

print("Found Papers:", len(papers))

df = pd.DataFrame(papers)

print("Original Shape:", df.shape)

# Keep only latest 20,000 papers
df = df.tail(20000)

print("Reduced Shape:", df.shape)

df.to_csv("llm_papers.csv", index=False)

print("Saved!")

df.to_csv("llm_papers.csv", index=False)

print("Saved!")