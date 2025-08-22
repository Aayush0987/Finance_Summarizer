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

# Sentiment analysis
from textblob import TextBlob

st.title("📊 Finance Summarizer (Local LLM)")
st.sidebar.title("🔗 News Article URLs")

# Let user enter up to 3 URLs
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}", key=f"url_{i}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("🚀 Process URLs")
file_path = "vector_index_local.pkl"
main_placeholder = st.empty()

# Filters (always visible)
st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filters")
sentiment_filter = st.sidebar.selectbox(
    "Sentiment",
    ["All", "Positive", "Neutral", "Negative"],
    index=0,
    key="sentiment_filter",
)
search_query = st.sidebar.text_input(
    "Search in summaries", key="search_query"
)

# Initialize session state storage
if "summaries" not in st.session_state:
    # Each item: {"idx": int, "summary": str, "sentiment": "🟢 Positive", "sentiment_label": "Positive", "polarity": float}
    st.session_state.summaries = []
if "docs_raw" not in st.session_state:
    st.session_state.docs_raw = []


llm = ChatOpenAI(
    model="openai/gpt-oss-20b",     # Your LM Studio model
    temperature=0.5,  # concise summaries
    max_tokens=400,
    openai_api_base="http://localhost:1234/v1",  # LM Studio endpoint
    openai_api_key="not-needed"  # Dummy key since LM Studio doesn’t need auth
)

def analyze_sentiment(text):
    """Return emoji label, plain label, and polarity score."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.05:
        return "🟢 Positive", "Positive", polarity
    elif polarity < -0.05:
        return "🔴 Negative", "Negative", polarity
    else:
        return "🟡 Neutral", "Neutral", polarity


def generate_pdf(title, text, sentiment=None):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    y = height - 50
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, title)
    y -= 30

    if sentiment:
        pdf.setFont("Helvetica-Oblique", 11)
        pdf.drawString(50, y, f"Sentiment: {sentiment}")
        y -= 30

    pdf.setFont("Helvetica", 11)
    text_obj = pdf.beginText(50, y)

    # simple wrapping
    for line in text.split("\n"):
        for chunk in [line[i:i+90] for i in range(0, len(line), 90)]:
            text_obj.textLine(chunk)
            y -= 15
            if y < 50:
                pdf.drawText(text_obj)
                pdf.showPage()
                text_obj = pdf.beginText(50, height - 50)
                pdf.setFont("Helvetica", 11)
                y = height - 50

    pdf.drawText(text_obj)
    pdf.save()
    buffer.seek(0)
    return buffer

if process_url_clicked and urls:
    try:
        # 1) Load
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("⏳ Loading data from URLs...")
        docs_raw = loader.load()
        st.session_state.docs_raw = docs_raw
        main_placeholder.text(f"✅ Loaded {len(docs_raw)} documents")

        # 2) Split for indexing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(docs_raw)
        main_placeholder.text(f"✅ Split into {len(docs)} document chunks")

        # 3) Build embeddings
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": device}
        )

        # 4) Build FAISS index
        vectorindex_local = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("✅ FAISS vector index built successfully")

        # 5) Save index
        with open(file_path, "wb") as f:
            pickle.dump(vectorindex_local, f)
        main_placeholder.text(f"💾 Index saved to {file_path}")

        time.sleep(0.5)
        st.success("🎉 Documents processed! Summarizing each article...")

        # 6) Summarize each article with chunking + sentiment
        new_summaries = []
        for idx, doc in enumerate(docs_raw):
            # Split long article into smaller chunks for safe context
            article_chunks = text_splitter.split_text(doc.page_content)

            chunk_summaries = []
            for chunk in article_chunks:
                prompt = (
                    "Summarize the following article excerpt in 2 sentences "
                    "(key points only, concise):\n\n" + chunk
                )
                chunk_summaries.append(llm.predict(prompt))

            combined = " ".join(chunk_summaries)
            final_prompt = (
                "Combine the following partial summaries into one concise article summary "
                "(3-4 sentences, avoid repetition):\n\n" + combined
            )
            final_summary = llm.predict(final_prompt)

            # Sentiment
            senti_emoji, senti_plain, polarity = analyze_sentiment(final_summary)

            new_summaries.append({
                "idx": idx,
                "summary": final_summary,
                "sentiment": senti_emoji,        # with emoji for display/PDF
                "sentiment_label": senti_plain,  # plain for filtering
                "polarity": polarity,
            })

        # Save to session_state (so it persists across interactions/downloads)
        st.session_state.summaries = new_summaries

        # Clear the progress text
        main_placeholder.empty()

    except Exception as e:
        st.error(f"⚠️ Error while processing URLs: {e}")


def apply_filters(items, sentiment_choice, query):
    filtered = items
    if sentiment_choice != "All":
        filtered = [it for it in filtered if it["sentiment_label"] == sentiment_choice]
    if query:
        q = query.lower()
        filtered = [it for it in filtered if q in it["summary"].lower()]
    return filtered

if st.session_state.summaries:
    st.header("📰 Article Summaries")

    # Apply filters/search
    shown = apply_filters(st.session_state.summaries, sentiment_filter, search_query)

    if not shown:
        st.info("No summaries matched your filters.")
    else:
        for item in shown:
            idx = item["idx"]
            summary = item["summary"]
            sentiment = item["sentiment"]
            plain_label = item["sentiment_label"]

            with st.expander(f"Article {idx+1} — {sentiment}"):
                st.write(summary)

                # Individual downloads
                st.download_button(
                    label=f"📥 Download Article {idx+1} (TXT)",
                    data=f"Sentiment: {sentiment}\n\n{summary}",
                    file_name=f"article_{idx+1}_summary.txt",
                    mime="text/plain",
                )

                pdf_buffer = generate_pdf(f"Article {idx+1}", summary, sentiment)
                st.download_button(
                    label=f"📥 Download Article {idx+1} (PDF)",
                    data=pdf_buffer,
                    file_name=f"article_{idx+1}_summary.pdf",
                    mime="application/pdf",
                )

        # Combined download of *currently shown (filtered)* summaries
        st.subheader("📥 Download All (Filtered)")
        combined_filtered = "\n\n".join(
            [f"Article {it['idx']+1} ({it['sentiment']})\n{it['summary']}" for it in shown]
        )
        st.download_button(
            label="Download All (TXT)",
            data=combined_filtered,
            file_name="financial_summaries_filtered.txt",
            mime="text/plain",
        )
        pdf_buffer_all = generate_pdf("All Summaries (Filtered)", combined_filtered)
        st.download_button(
            label="Download All (PDF)",
            data=pdf_buffer_all,
            file_name="financial_summaries_filtered.pdf",
            mime="application/pdf",
        )


query = st.text_input("💬 Ask a question about the documents:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorindex_local = pickle.load(f)

        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=vectorindex_local.as_retriever()
        )

        result = chain({"question": query}, return_only_outputs=True)

        st.header("📝 Answer")
        st.write(result["answer"])

        sources = result.get("sources", "")
        if sources:
            st.subheader("📌 Sources:")
            for source in sources.split("\n"):
                if source.strip():
                    st.write(source)
    else:
        st.warning("⚠️ Please process URLs first before asking questions.")