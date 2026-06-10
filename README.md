# Multi-Doc RAG Chatbot

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-000000?style=for-the-badge&logo=llama&logoColor=white)
![Gemini API](https://img.shields.io/badge/Gemini_API-8E75B2?style=for-the-badge&logo=googlebard&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?style=for-the-badge)

A production-ready Retrieval-Augmented Generation (RAG) web application built to enable intelligent, grounded Q&A across multiple PDF documents simultaneously. 

**[Live Demo](https://rag-pdf-chatbot-yh267qtmr7iibpctharmyb.streamlit.app)** 

---

## Key Features

- **Multi-PDF Knowledge Base:** Upload and index multiple PDF documents at once in the sidebar to query across your entire collective knowledge base.
- **Semantic Search:** Utilizes open-source `sentence-transformers/all-MiniLM-L6-v2` embeddings and Cosine Similarity to retrieve the most contextually relevant information.
- **Source Tracking & Citations:** Every response provides strict citations, mapping extracted chunks directly to their origin file (e.g., `From: your_document.pdf`).
- **Hallucination-Free Generation:** Powered by Gemini 2.5 Flash with strict prompt engineering to ensure the LLM only answers using the provided context.
- **Verified Reliability:** Pipeline performance is validated using an automated LLM-as-a-Judge script evaluating Faithfulness and Relevancy (powered by NVIDIA's LLaMA 3.3).

---

## Tech Stack

| Component | Technology |
| :--- | :--- |
| **LLM** | Gemini 2.5 Flash |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`) |
| **Orchestration** | LlamaIndex |
| **Vector DB** | ChromaDB (Persistent Storage) |
| **UI Framework** | Streamlit |
| **Evaluator** | NVIDIA NIM API (`meta/llama-3.3-70b-instruct`) |

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Set up environment variables
Create a `.env` file in the root directory and add your required API keys:
```env
# Required for the main application
GOOGLE_API_KEY=your_gemini_api_key_here

# Required ONLY for running pipeline evaluations via eval.py
NVIDIA_API_KEY=your_nvidia_api_key_here
```

### 4. Run the application
Start the Streamlit interface:
```bash
streamlit run app.py
```

---

## Pipeline Evaluation

To verify the quality of your RAG pipeline (checking for Faithfulness and Relevancy), you can run the built-in evaluation script. This script utilizes LlamaIndex `BatchEvalRunner` and acts as an LLM Judge.

```bash
python eval.py
```

*Note: Ensure you have populated your ChromaDB by uploading at least one PDF in the app before running the evaluator.*