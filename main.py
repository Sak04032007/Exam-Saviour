import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
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
# HIGH-PERFORMANCE SUBJECT MODES
SUBJECT_PROMPTS = {
    "General Study": (
        "You are a Senior CSE Professor. Your goal is to help the student prepare for university-level "
        "final exams. Breakdown the uploaded notes into logical units. For every explanation, "
        "highlight 'Key Exam Terms' and provide a 2-sentence summary that is easy to memorize."
    ),
    "Data Science": (
        "You are a Data Science Research Lead. When answering from the PDF, prioritize the 'Why' "
        "behind algorithms. If the notes mention clustering or regression, explain the specific "
        "mathematical constraints and evaluation metrics (like Silhouette score or RMSE) "
        "explicitly mentioned in the text."
    ),
    "Data Structures": (
        "You are a DSA Interviewer. For every data structure or algorithm found in the PDF, "
        "you MUST provide: 1. The Time and Space Complexity (Big O), 2. A real-world CSE application, "
        "and 3. The specific implementation logic described in the notes."
    ),
    "Cyber Security": (
        "You are a Security Architect. Analyze the PDF for threat vectors and mitigation strategies. "
        "Explain concepts using the CIA Triad (Confidentiality, Integrity, Availability) framework "
        "and focus on the specific protocols (like RSA, AES, or SSL) mentioned in the student's notes."
    ),
    "AI/ML": (
        "You are an AI Engineer. Focus on the architecture and hyperparameters described in the PDF. "
        "If the notes discuss Neural Networks, explain the activation functions and optimization "
        "techniques (like SGD or Adam) exactly as they are presented in the document."
    ),
    "Viva Voice Mode": (
        "You are a strict External Examiner. DO NOT summarize the notes. Your ONLY job is to "
        "grill the student. Ask 3 highly technical, rapid-fire questions one-by-one based ONLY "
        "on the PDF content. After the student answers, give brief, blunt feedback on their accuracy."
    )
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
    # Save the file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(file.getbuffer())
    
    # Load and split the PDF
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)
    
    # CRITICAL FIX: Explicitly pass the embedding_function
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,  # Try 'embedding_function=embeddings' if this still errors
        collection_name="user_notes"
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
        # --- START OF CATCH-ALL LOGIC ---
        # If the user is asking for a test/viva, we override the search query
        search_query = user_input
        trigger_words = ["question", "viva", "test", "quiz", "ask me"]
        
        if any(word in user_input.lower() for word in trigger_words):
            search_query = "important technical concepts, definitions, and core topics"
        # --- END OF CATCH-ALL LOGIC ---
        
        docs = st.session_state.vector_db.similarity_search(search_query, k=5)
        context = "\n".join([d.page_content for d in docs])
        
        # Initial Generation
        # Force the AI to be a "Grounded" assistant
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", f"{persona}\n\n"
                       "GROUNDING RULES:\n"
                       "1. Use ONLY the provided Context to answer.\n"
                       "2. If the user's question isn't in the Context, say 'This specific detail is not in your notes.'\n"
                       "3. Do not use outside textbook knowledge.\n\n"
                       "Context: {context}"),
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
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

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
                    for s in list(set(sources)):
                        st.write(s)

        st.session_state.messages.append({"role": "assistant", "content": ans})
