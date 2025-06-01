# LexIQ

LexIQ is an AI-powered legal assistant designed to simplify legal research, answer legal queries, and provide intelligent case insights using natural language processing and machine learning.

## 🚀 Features

- 🧠 AI-driven legal Q&A
- 🔍 Smart case law search
- 📄 Legal document summarization
- ⚖️ Predictive case insights and suggestions

## 🛠 Tech Stack

- Python
- Ollama : "mxbai-embed-large:335m" embedding model
- GROQ API :Deepseek R1
- LangChain
- Streamlit (Frontend)
- FAISS (Vector DB)

## 🧩 How It Works

1. User asks a legal question or uploads a case document.
2. LexIQ parses the query, performs semantic search across legal databases.
3. A language model generates an accurate, concise legal response or summary.
4. Results are displayed with citations, relevant laws, and AI-driven insights.

## 📦 Installation

```bash
git clone https://github.com/Chinmay-Jadhav/lexiq.git
cd lexiq
pip install -r requirements.txt
