"""
main.py
Finance Summarizer using Local LLM (LM Studio) + FAISS + HuggingFace Embeddings + Streamlit
"""

import os
import pickle
import torch
import time
import io
import streamlit as st

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ==============================
# Streamlit UI
# ==============================
st.title("📊 Finance Summarizer (Local LLM)")
st.sidebar.title("🔗 News Article URLs")

# Let user enter up to 3 URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("🚀 Process URLs")
file_path = "vector_index_local.pkl"
main_placeholder = st.empty()

# Initialize session state storage
if "all_summaries" not in st.session_state:
    st.session_state.all_summaries = []
if "docs_raw" not in st.session_state:
    st.session_state.docs_raw = []


# ==============================
# Local LLM Setup (LM Studio)
# ==============================
llm = ChatOpenAI(
    model="openai/gpt-oss-20b",     # Your LM Studio model
    temperature=0.5,  # Lower temperature for more concise summaries
    max_tokens=400,
    openai_api_base="http://localhost:1234/v1",  # LM Studio endpoint
    openai_api_key="not-needed"  # Dummy key since LM Studio doesn’t need auth
)


# ==============================
# Helper: Generate PDF
# ==============================
def generate_pdf(title, text):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y_position = height - 50
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y_position, title)
    y_position -= 30

    pdf.setFont("Helvetica", 11)
    text_object = pdf.beginText(50, y_position)

    # Wrap text manually
    for line in text.split("\n"):
        for chunk in [line[i:i+90] for i in range(0, len(line), 90)]:
            text_object.textLine(chunk)
            y_position -= 15
            if y_position < 50:  # new page if out of space
                pdf.drawText(text_object)
                pdf.showPage()
                text_object = pdf.beginText(50, height - 50)
                pdf.setFont("Helvetica", 11)
                y_position = height - 50

    pdf.drawText(text_object)
    pdf.save()
    buffer.seek(0)
    return buffer


# ==============================
# Process URLs (when button clicked)
# ==============================
if process_url_clicked and urls:
    try:
        # 1. Load documents
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("⏳ Loading data from URLs...")
        docs_raw = loader.load()
        st.session_state.docs_raw = docs_raw  # save to session state
        main_placeholder.text(f"✅ Loaded {len(docs_raw)} documents")

        # 2. Split into chunks for indexing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(docs_raw)
        main_placeholder.text(f"✅ Split into {len(docs)} document chunks")

        # 3. Build embeddings
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device}
        )

        # 4. Build FAISS index
        vectorindex_local = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("✅ FAISS vector index built successfully")

        # 5. Save index
        with open(file_path, "wb") as f:
            pickle.dump(vectorindex_local, f)
        main_placeholder.text(f"💾 Index saved to {file_path}")

        time.sleep(2)
        st.success("🎉 Documents processed successfully! Summarizing each article...")

        # 6. Summarize each article separately with chunking
        summaries = []
        for idx, doc in enumerate(docs_raw):
            # Split long article into smaller chunks
            article_chunks = text_splitter.split_text(doc.page_content)

            chunk_summaries = []
            for chunk in article_chunks:
                prompt = f"Summarize the following article excerpt in 2 sentences (key points only, concise):\n\n{chunk}"
                summary_piece = llm.predict(prompt)
                chunk_summaries.append(summary_piece)

            # Combine chunk summaries into one final summary
            combined_summary_text = " ".join(chunk_summaries)
            final_prompt = f"Combine the following partial summaries into one concise article summary (3-4 sentences, avoid repetition):\n\n{combined_summary_text}"
            final_summary = llm.predict(final_prompt)

            summaries.append((idx, final_summary))

        # Save summaries to session state
        st.session_state.all_summaries = summaries

    except Exception as e:
        st.error(f"⚠️ Error while processing URLs: {e}")


# ==============================
# Show Summaries (from session_state)
# ==============================
if st.session_state.all_summaries:
    st.header("📰 Article Summaries")
    for idx, final_summary in st.session_state.all_summaries:
        st.subheader(f"Article {idx+1}")
        st.write(final_summary)

        # Individual download buttons
        st.download_button(
            label=f"📥 Download Article {idx+1} (TXT)",
            data=final_summary,
            file_name=f"article_{idx+1}_summary.txt",
            mime="text/plain",
        )

        pdf_buffer = generate_pdf(f"Article {idx+1}", final_summary)
        st.download_button(
            label=f"📥 Download Article {idx+1} (PDF)",
            data=pdf_buffer,
            file_name=f"article_{idx+1}_summary.pdf",
            mime="application/pdf",
        )

    # Combined downloads
    combined_summary = "\n\n".join(
        [f"Article {idx+1}\n{summary}" for idx, summary in st.session_state.all_summaries]
    )
    st.subheader("📥 Download All Summaries")
    st.download_button(
        label="Download All (TXT)",
        data=combined_summary,
        file_name="financial_summaries.txt",
        mime="text/plain",
    )
    pdf_buffer_all = generate_pdf("All Summaries", combined_summary)
    st.download_button(
        label="Download All (PDF)",
        data=pdf_buffer_all,
        file_name="financial_summaries.pdf",
        mime="application/pdf",
    )


# ==============================
# Ask Questions
# ==============================
query = st.text_input("💬 Ask a question about the documents:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorindex_local = pickle.load(f)

        # Build chain
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorindex_local.as_retriever()
        )

        # Run query
        result = chain({"question": query}, return_only_outputs=True)

        # Show answer
        st.header("📝 Answer")
        st.write(result["answer"])

        # Show sources
        sources = result.get("sources", "")
        if sources:
            st.subheader("📌 Sources:")
            sources_list = sources.split("\n")
            for source in sources_list:
                st.write(source)
    else:
        st.warning("⚠️ Please process URLs first before asking questions.")