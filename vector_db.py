from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# connect to local qdrant
client = QdrantClient("localhost", port=6333)

# recreate collection fresh each run
client.delete_collection(collection_name="tax_docs")
client.create_collection(
    collection_name="tax_docs",
    vectors_config=VectorParams(
        size=384,        # dimensions of all-MiniLM-L6-v2
        distance=Distance.COSINE
    )
)

# your documents — think of these as chunks from a real PDF
documents = [
    {"id": 1, "text": "The standard deduction for single filers in 2026 is $16,100.", "page": 1},
    {"id": 2, "text": "Married filing jointly can claim a standard deduction of $32,200 in 2026.", "page": 1},
    {"id": 3, "text": "The federal tax rate of 10% applies to the first $12,400 of taxable income.", "page": 2},
    {"id": 4, "text": "A single filer earning over $640,600 pays the highest marginal rate of 37%.", "page": 2},
    {"id": 5, "text": "Taxpayers can itemize deductions instead of taking the standard deduction.", "page": 3},
]

# embed all documents
model = SentenceTransformer('all-MiniLM-L6-v2')
texts = [doc["text"] for doc in documents]
embeddings = model.encode(texts)

# store in qdrant with metadata
points = [
    PointStruct(
        id=doc["id"],
        vector=embedding.tolist(),
        payload={"text": doc["text"], "page": doc["page"]}
    )
    for doc, embedding in zip(documents, embeddings)
]

client.upsert(collection_name="tax_docs", points=points)
print(f"Stored {len(points)} documents\n")

# now search
query = "married couple deduction"
query_vector = model.encode([query])[0].tolist()

results = client.query_points(
    collection_name="tax_docs",
    query=query_vector,
    limit=2  # top 2 most relevant chunks
)

print(f"Query: '{query}'\n")
for result in results.points:
    print(f"Score: {result.score:.3f}")
    print(f"Page:  {result.payload['page']}")
    print(f"Text:  {result.payload['text']}\n")