import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. CONFIG & SECRETS ---
st.set_page_config(page_title="Exam Savior AI", layout="wide")

# Securely load keys from Streamlit Secrets
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

# --- 2. RESOURCE INITIALIZATION ---
@st.cache_resource
def load_llm_and_embeddings():
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, embeddings

llm, embeddings = load_llm_and_embeddings()

# --- 3. SIDEBAR: SUBJECT MODES & FILE UPLOAD ---
with st.sidebar:
    st.title("🎓 Study Settings")
    
    # YOUR SUBJECT MODES
    SUBJECT_PROMPTS = {
        "General Study": "You are a brilliant CSE professor. Use the notes to help the student.",
        "Operating Systems": "You are an OS Expert. Focus on kernel logic, scheduling, and C-style pseudo-code.",
        "Data Science": "You are a Data Scientist. Focus on mathematical intuition and Python libraries.",
        "Viva Voice Mode": "You are a strict External Examiner. Ask 3 rapid-fire questions to test the student."
    }
    selected_subject = st.selectbox("Choose your Professor:", list(SUBJECT_PROMPTS.keys()))
    
    st.divider()
    
    # DYNAMIC FILE UPLOAD
    uploaded_file = st.file_uploader("Upload your PDF notes", type="pdf")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# --- 4. DYNAMIC VECTOR DB PROCESSING ---
def process_uploaded_file(file):
    # Save temp file
    with open("temp.pdf", "wb") as f:
        f.write(file.getbuffer())
    
    loader = PyPDFLoader("temp.pdf")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    
    # Create a fresh, in-memory database for this specific file
    return Chroma.from_documents(documents=chunks, embedding=embeddings)

if uploaded_file:
    if "vector_db" not in st.session_state:
        with st.spinner("Processing your notes..."):
            st.session_state.vector_db = process_uploaded_file(uploaded_file)
            st.success("Notes processed! You can now start the session.")

# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask your notes..."):
    if not uploaded_file:
        st.warning("Please upload a PDF first!")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # --- 6. AGENTIC RESPONSE LOGIC ---
        with st.chat_message("assistant"):
            # Retrieval
            docs = st.session_state.vector_db.similarity_search(prompt, k=3)
            context = "\n".join([d.page_content for d in docs])
            
            # Persona + History Logic
            persona = SUBJECT_PROMPTS[selected_subject]
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", f"{persona}\n\nContext: {context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ])
            
            # Convert session history to LangChain messages
            history_msgs = [
                HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) 
                for m in st.session_state.messages[:-1]
            ]
            
            chain = qa_prompt | llm | StrOutputParser()
            response = chain.invoke({"question": prompt, "history": history_msgs})
            
            st.markdown(response)
            
            # Citations
            citations = [f"📄 Page {doc.metadata.get('page', 0) + 1}" for doc in docs]
            with st.expander("📚 Sources"):
                for cite in list(set(citations)):
                    st.write(cite)

        st.session_state.messages.append({"role": "assistant", "content": response})