import streamlit as st
from rag_pipeline import load_text, chunk_text, create_embeddings, build_faiss_index, retrieve_chunks, generate_answer
import tempfile

st.title("📄 AI Document Q&A (RAG)")
st.write("Upload a document and ask questions")
uploaded_file = st.file_uploader("Upload a TXT file", type=["txt"])

if uploaded_file:
    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # Process document
    with st.spinner("Processing document..."):
        text = load_text(file_path)
        chunks = chunk_text(text)
        model, embeddings = create_embeddings(chunks)
        index = build_faiss_index(embeddings)

    st.success(f"Document processed! {len(chunks)} chunks created.")

    # Ask question
    query = st.text_input("Ask a question:")

    if query:
       with st.spinner("Generating answer..."):
        results = retrieve_chunks(query, index, chunks, model)
        answer = generate_answer(query, results)

    st.subheader("🤖 Answer")
    st.write(answer)

    st.subheader("📚 Source Passages")
    for i, chunk in enumerate(results):
        st.markdown(f"**Chunk {i+1}:** {chunk}")