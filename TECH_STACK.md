#  Technical Stack | Celeritas

This document outlines the architectural components and technologies used to build the **Celeritas** Agentic RAG system.

##  Large Language Model (LLM)
*   **Provider:** [Groq Cloud](https://groq.com/)
*   **Model:** `Llama-3.3-70B-Versatile`
*   **Function:** Handles reasoning, persona simulation, and response generation. Chosen for its high-speed inference and complex reasoning capabilities.

##  AI Orchestration
*   **Framework:** [LangChain](https://www.langchain.com/)
*   **Components Used:**
    *   **Expression Language (LCEL):** For building modular chains.
    *   **Prompt Templates:** To define "Professor" personas.
    *   **Routers:** To dynamically direct queries between the PDF and the Web.
    *   **Graders:** To verify the relevance of retrieved context (Anti-hallucination).

##  Vector Database & RAG
*   **Vector Store:** [ChromaDB](https://www.trychroma.com/) (Self-hosted/Local)
*   **Embeddings:** `HuggingFaceEmbeddings` (Model: `all-MiniLM-L6-v2`)
*   **Document Loader:** `PyPDFLoader`
*   **Text Splitting:** `RecursiveCharacterTextSplitter` (Chunk Size: 1000, Overlap: 100)

##  External Tools & Search
*   **Search Engine:** [Tavily AI](https://tavily.com/)
*   **Utility:** Provides real-time internet access when user queries fall outside the scope of the uploaded PDF notes.

##  Frontend & Deployment
*   **Framework:** [Streamlit](https://streamlit.io/)
*   **UI Elements:** Custom HTML/CSS for branding, Metric widgets, and interactive Sidebar.
*   **Deployment:** Streamlit Community Cloud (connected via GitHub).

##  Backend Language
*   **Python 3.10+**
