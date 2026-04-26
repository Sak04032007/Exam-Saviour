import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults

# --- 1. SETUP & SECRETS ---
st.set_page_config(page_title="Exam Savior AI", layout="wide")

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "TAVILY_API_KEY" in st.secrets:
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

# --- 2. THE BRAINS (LLM & TOOLS) ---
@st.cache_resource
def load_core_tools():
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    web_search = TavilySearchResults(k=3)
    return llm, embeddings, web_search

llm, embeddings, web_search = load_core_tools()

# --- 3. THE PROMPT LIBRARY (Your Personas) ---
SUBJECT_PROMPTS = {
    "General Study": "You are a brilliant CSE professor. Use the notes to help the student.",
    "Operating Systems": "You are an OS Expert. Focus on kernel logic, scheduling, and C-style pseudo-code.",
    "Data Science": "You are a Data Scientist. Focus on mathematical intuition and Python libraries.",
    "Viva Voice Mode": "You are a strict External Examiner. Ask 3 rapid-fire questions to test the student."
}

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🎓 Study Settings")
    selected_subject = st.selectbox("Choose Professor Mode:", list(SUBJECT_PROMPTS.keys()))
    uploaded_file = st.file_uploader("Upload your CSE Notes (PDF)", type="pdf")
    if st.button("Clear Chat History"):
        st.session_state.messages = []

# --- 5. DYNAMIC PDF PROCESSING ---
def process_new_notes(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getbuffer())
    loader = PyPDFLoader("temp.pdf")
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(loader.load())
    
    # Use 'embeddings' (your variable name) with the 'embedding' parameter
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings  # If this fails, try 'embedding_function=embeddings'
    )
    return vectorstore
if uploaded_file and "vector_db" not in st.session_state:
    with st.spinner("Processing your notes..."):
        st.session_state.vector_db = process_new_notes(uploaded_file)
        st.success("Notes processed!")

# --- 6. YOUR AGENTIC LOGIC (ROUTER + GRADER) ---
# Routing Logic
router_prompt = ChatPromptTemplate.from_template(
    "Route this question to 'vector_db', 'web_search', or 'llm'. Question: {question}"
)
route_chain = router_prompt | llm | StrOutputParser()

# Hallucination Grader Logic
grader_prompt = ChatPromptTemplate.from_template(
    "Check if this answer is supported by the context. Context: {context}\nAnswer: {answer}\nReply 'YES' or 'NO'."
)
grader_chain = grader_prompt | llm | StrOutputParser()

def master_engine_logic(user_input, history, subject):
    persona = SUBJECT_PROMPTS[selected_subject]
    route = route_chain.invoke({"question": user_input}).lower()
    
    if "vector_db" in route and "vector_db" in st.session_state:
        docs = st.session_state.vector_db.similarity_search(user_input, k=3)
        context = "\n".join([d.page_content for d in docs])
        
        # Initial Generation
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", f"{persona}\n\nContext: {context}"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
        ])
        initial_answer = (qa_prompt | llm | StrOutputParser()).invoke({"question": user_input, "history": history})
        
        # Your Hallucination Check
        grade = grader_chain.invoke({"context": context, "answer": initial_answer})
        if "NO" in grade.upper():
            correction = llm.invoke(f"System: {persona}\nCorrection: {grade}\nRefine using context: {context}").content
            return correction, [f"📄 Page {d.metadata.get('page', 0)+1}" for d in docs]
        
        return initial_answer, [f"📄 Page {d.metadata.get('page', 0)+1}" for d in docs]

    elif "web_search" in route:
        results = web_search.invoke({"query": user_input})
        return llm.invoke(f"System: {persona}\nWeb Data: {results}\nAnswer: {user_input}").content, ["🌐 Web"]
    
    return llm.invoke(f"System: {persona}\nUser: {user_input}").content, []

# --- 7. THE UI CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]): st.markdown(m["content"])

if prompt := st.chat_input("Ask your Professor..."):
    if not uploaded_file:
        st.warning("Upload a PDF first!")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            # Convert history for LangChain
            hist = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
            
            ans, sources = master_engine_logic(prompt, hist, selected_subject)
            st.markdown(ans)
            if sources:
                with st.expander("📚 Sources"):
                    for s in list(set(sources)): st.write(s)

        st.session_state.messages.append({"role": "assistant", "content": ans})
