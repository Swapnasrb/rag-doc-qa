import warnings
warnings.filterwarnings("ignore")

# ✅ Load TEXT (not PDF anymore)
def load_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


# ✅ Chunk text
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks


# 🔥 Step 3: Embeddings
from sentence_transformers import SentenceTransformer

def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks)
    return model, embeddings


# 🔥 Step 3: FAISS
import faiss
import numpy as np

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


# 🔥 Step 4: Retrieval
def retrieve_chunks(query, index, chunks, model, top_k=3):
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector, top_k)
    results = [chunks[i] for i in indices[0]]
    return results


# 🚀 MAIN (ONLY ONE)
from transformers import pipeline

def generate_answer(query, context_chunks):
    # Combine retrieved chunks
    context = " ".join(context_chunks)

    # Load model (first time takes time)
    qa_pipeline = pipeline(
    "text-generation",
    model="distilgpt2"
    )

    prompt = f"""
    Answer the question based on the context below.

    Context:
    {context}

    Question: {query}
    Answer:
    """

    result = qa_pipeline(prompt, max_new_tokens=150)
    return result[0]['generated_text']


if __name__ == "__main__":
    text = load_text("sample.txt")

    chunks = chunk_text(text)

    model, embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    print("System ready! Ask a question:\n")

    query = input("Enter your question: ")

    results = retrieve_chunks(query, index, chunks, model)

    answer = generate_answer(query, results)

    print("\n🤖 AI Answer:\n")
    print(answer)


  