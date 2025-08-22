# 📊 Finance Summarizer (Local LLM + FAISS + Streamlit)

An **AI-powered Financial News Summarizer** that fetches news articles from URLs, summarizes them using a **local LLM** (via LM Studio), tags them with **sentiment analysis**, and makes them **searchable** with FAISS embeddings.

Built with **LangChain, HuggingFace Embeddings, FAISS, TextBlob, and Streamlit** 🚀

---

## ✨ Features

- ✅ **Summarize Financial News** – Extract concise summaries (3–4 sentences) from long articles.  
- ✅ **Sentiment Analysis** – Automatically tag each article summary as 🟢 Positive, 🟡 Neutral, or 🔴 Negative.  
- ✅ **Search & Filters** – Search summaries by keyword or filter by sentiment.  
- ✅ **Chunked Summarization** – Handles long articles by splitting into smaller chunks for accuracy.  
- ✅ **Download Options** – Export summaries in TXT or PDF format (individual or combined).  
- ✅ **Q&A on Articles** – Ask natural language questions, get AI-powered answers with sources.  
- ✅ **Local-First** – Works with your own local LLM via LM Studio, **no API costs**.  

---

## 🛠️ Tech Stack

- 🐍 **Core Language** – Python 3.10+  
- 🎨 **Interactive UI** – Streamlit for a seamless and responsive user interface  
- 🔗 **LLM Orchestration** – LangChain to manage the AI workflow  
- ⚡ **Vector Search** – FAISS for efficient, high-speed similarity search  
- 🧠 **Text Embeddings** – HuggingFace Sentence Transformers  
- 👍 **Sentiment Analysis** – TextBlob for sentiment scoring  
- 📄 **PDF Export** – ReportLab for clean, downloadable PDF reports  

---

## 🚀 Installation
1. **Clone the repo**
    git clone https://github.com/Aayush0987/Finance_Summarizer.git
    cd Finance_Summarizer
2.	**Create a virtual environment**
    python3 -m venv venv
    source venv/bin/activate   # Mac/Linux
    venv\Scripts\activate      # Windows 
3.	**Install dependencies**
    pip install -r requirements.txt
4.	**Start LM Studio**
-   •   📥 Download LM Studio
-   •	📂 Load a compatible model (e.g., gpt-oss-20b)
-   •	▶️ Run it locally at: http://localhost:1234/v1    
5.	**Run the app**
    streamlit run main.py

## 📖 Usage

- 🌐 **URL Input** – Paste up to 3 article URLs into the sidebar.  
- 🚀 **One-Click Processing** – Load, chunk, index, summarize, and analyze sentiment instantly.  
- 📰 **Organized View** – Read summaries in collapsible cards, each tagged with a sentiment icon.  
- ⚖️ **Sentiment Filtering** – Focus on Positive, Neutral, or Negative news.  
- 🔎 **Keyword Search** – Search across summaries by keyword/topic.  
- 📥 **Flexible Downloads** – Export individual or combined summaries in TXT/PDF.  
- 💬 **AI-Powered Q&A** – Ask natural questions and receive AI-generated answers with sources.  

---

## 🔮 Future Improvements

- 📅 **Daily Digest Mode** – Enter a stock ticker (e.g., `AAPL`) to auto-fetch and summarize recent news.  
- 📊 **Interactive Dashboard** – Visualize sentiment trends over time.  
- 📝 **Customizable Summaries** – Choose summary length: **Short (bullets), Medium (1 para), Detailed**.  
- ⚖️ **Comparison Summarizer** – Compare multiple articles to find agreements, contradictions, and unique insights.  

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to add.  
