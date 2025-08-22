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

🛠️ Tech Stack
	•	Python 3.10+
	•	Streamlit – Interactive UI
	•	LangChain – LLM Orchestration
	•	FAISS – Vector Search
	•	HuggingFace Sentence Transformers – Embeddings
	•	TextBlob – Sentiment Analysis
	•	ReportLab – PDF Export

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
	1.	Enter up to 3 article URLs in the sidebar.
	2.	Click 🚀 Process URLs – the system will:
	•	Load and chunk articles
	•	Build embeddings with FAISS
	•	Generate summaries
	•	Perform sentiment analysis
	3.	View summaries in collapsible cards with sentiment tags.
	4.	Apply filters (e.g., show only 🔴 Negative articles).
	5.	Search within summaries.
	6.	Download results as TXT or PDF.
	7.	Ask questions about the articles in the Q&A section.

🔮 Future Improvements
	•	📅 Daily Digest Mode: Enter a stock ticker (e.g., AAPL) → auto-fetch recent financial news.
	•	📊 Interactive Dashboard: Graphical sentiment trends over time.
	•	📝 Customizable Summary Length: Slider for short / medium / detailed summaries.
	•	⚖️ Comparison Summarizer: Compare multiple articles on the same topic.
