from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "What is the standard deduction for single filers in 2026?",
    "How much can a single person deduct from their taxes?",
    "What is the capital of France?",
    "Tell me about federal income tax brackets",
    "How do marginal tax rates work?"
]

embeddings = model.encode(sentences)

query = "who won the world cup"
query_embedding = model.encode([query])

similarities = cosine_similarity(query_embedding, embeddings)[0]

print(f"Query: '{query}'\n")
for i, (sentence, score) in enumerate(zip(sentences, similarities)):
    print(f"{score:.3f} — {sentence}")