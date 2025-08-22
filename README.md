📊 Finance Summarizer (Local LLM + FAISS + Streamlit)

An AI-powered Financial News Summarizer that fetches news articles from URLs, summarizes them using a local LLM (via LM Studio), tags them with sentiment analysis, and makes them searchable with FAISS embeddings.

Built with LangChain, HuggingFace Embeddings, FAISS, TextBlob, and Streamlit 🚀.

✨ Features
- ✅ **Summarize Financial News** – Extract concise summaries (3–4 sentences) from long articles.
- ✅ **Sentiment Analysis** – Automatically tag each article summary as 🟢 Positive, 🟡 Neutral, or 🔴 Negative.
- ✅ **Search & Filters** – Search summaries by keyword or filter by sentiment.
- ✅ **Chunked Summarization** – Handles long articles by splitting into smaller chunks for accuracy.
- ✅ **Download Options** – Export summaries in TXT or PDF format (individual or combined).
- ✅ **Q&A on Articles** – Ask natural language questions, get AI-powered answers with sources.
- ✅ **Local-First** – Works with your own local LLM via LM Studio, no API costs.

- 🐍 **Core Language** – Built with Python 3.10+ for a modern and robust backend.
- 🎨 **Interactive UI** – Powered by Streamlit to create a seamless and responsive user interface.
- 🔗 **LLM Orchestration** – Uses LangChain to structure and manage the entire AI workflow.
- ⚡ **Vector Search** – Employs FAISS for efficient, high-speed local similarity searches.
- 🧠 **Text Embeddings** – Generates semantic embeddings using HuggingFace Sentence Transformers.
- 👍 **Sentiment Analysis** – Performs quick and accurate sentiment scoring with TextBlob.
- 📄 **PDF Export** – Creates clean, downloadable PDF reports on the fly using ReportLab.

🚀 Installation

1.	Clone the repo
    git clone https://github.com/Aayush0987/Finance_Summarizer.git
    cd Finance_Summarizer

2.	Create a virtual environment
    python3 -m venv venv
    source venv/bin/activate   # Mac/Linux
    venv\Scripts\activate      # Windows

3.	Install dependencies
    pip install -r requirements.txt

4.	Start LM Studio
	•	Download LM Studio
	•	Load a compatible model (e.g., gpt-oss-20b)
	•	Run it locally on http://localhost:1234/v1

5.	Run the app
    streamlit run main.py

📖 Usage
- 🌐 **URL Input** – Easily paste up to 3 article URLs directly into the sidebar.
- 🚀 **One-Click Processing** – Load, chunk, index, summarize, and analyze sentiment with a single click.
- 📰 **Organized View** – Read summaries in clean, collapsible cards, each tagged with a clear sentiment icon.
- ⚖️ **Sentiment Filtering** – Instantly filter articles by sentiment to focus on positive, neutral, or negative news.
- 🔎 **Keyword Search** – Quickly search across all generated summaries to find specific keywords or topics.
- 📥 **Flexible Downloads** – Export individual or combined summaries in either TXT or PDF format.
- 💬 **AI-Powered Q&A** – Ask natural language questions about the content and receive AI-generated answers with sources.

🔮 Future Improvements
- 📅 **Daily Digest Mode** – Enter a stock ticker (e.g., AAPL) to automatically fetch and summarize the latest financial news.
- 📊 **Interactive Dashboard** – Visualize sentiment trends over time with an interactive graphical dashboard.
- 📝 **Customizable Summaries** – Use a simple slider to adjust summary length from concise to detailed.
- ⚖️ **Comparison Summarizer** – Generate a single, comparative summary from multiple articles on the same topic.
