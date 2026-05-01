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
    "General Study": "You are an assistant. Use only the provided notes to answer.",
    "Data Science": "You are an assistant. Focus only on the technical definitions in the notes. If a detail is missing, say it's not provided.",
    "AI/ML": "You are an assistant. Use only the formulas and algorithms from the notes.",
    "Cyber Security": "You are an assistant. Use only the security protocols mentioned in the notes.",
    "Viva Voice Mode": "You are an examiner. Generate questions based strictly on the text provided."
}


# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.title("🤓 Study Settings")
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
    """Analyze the user's intent:
    - If the question involves technical terms, formulas, or concepts likely in a study PDF -> 'vector_db'
    - If the question is about current events, specific URLs, or live data -> 'web_search'
    - If it's a greeting ('hi', 'how are you') or formatting help -> 'llm'
    
    Question: {question}
    Answer with only ONE word: 'vector_db', 'web_search', or 'llm'."""
)
route_chain = router_prompt | llm | StrOutputParser()

# Hallucination Grader Logic
grader_prompt = ChatPromptTemplate.from_template(
    """You are a Fact-Checker. 
    Compare the Answer to the Context. 
    Context: {context}
    Answer: {answer}
    
    Does the Answer contain ANY information, facts, or topics (like 'Regression') that are NOT present in the Context? 
    Reply 'YES' only if the answer is 100% supported by the context. 
    Reply 'NO' if there is even a single hallucinated detail."""
)
grader_chain = grader_prompt | llm | StrOutputParser()

def master_engine_logic(user_input, history, subject):
    persona = SUBJECT_PROMPTS[selected_subject]
    
    # 1. ROUTING: Determine the best tool
    route = route_chain.invoke({"question": user_input}).lower()
    if subject == "Viva Voice Mode":
        route = "vector_db"
    else:
        route = route_chain.invoke({"question": user_input}).lower()
    # 2. RETRIEVAL & GRADING (The CRAG Layer)

    if "vector_db" in st.session_state:
        docs = st.session_state.vector_db.similarity_search(user_input, k=7)
        context = "\n".join([d.page_content for d in docs])
        # Generate a candidate answer
        # Change this in your master_engine_logic
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "{persona}\n\nSTRICT: Answer ONLY using this context: {context_text}"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

# Then invoke it like this:
candidate_answer = (qa_prompt | llm | StrOutputParser()).invoke({
    "question": user_input, 
    "history": history,
    "persona": persona,
    "context_text": context  # Pass it here instead of hardcoding in f-string
})
        # 3. SELF-GRADING: Check for hallucinations
# Check if the context actually contains the answer
grade = grader_chain.invoke({"context": context, "answer": user_input})
if "YES" in grade.upper():
            # Answer strictly from notes
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", f"{persona}\n\nUSE ONLY THIS CONTEXT: {context}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ])
            ans = (qa_prompt | llm | StrOutputParser()).invoke({"question": user_input, "history": history})
            return f"📚 **From your Notes:**\n\n{ans}", [f"📄 Page {d.metadata.get('page', 0)+1}" for d in docs]
        
else:
            # 2. Fallback: Data not in notes, fetch from Internet/LLM
            st.info("🔍 This specific detail isn't in your notes. Fetching from the internet...")
            results = web_search.invoke({"query": user_input})
            
            fallback_prompt = f"""
            {persona}
            The user's notes do not contain this info. Use the following web data to answer.
            Web Data: {results}
            Question: {user_input}
            """
            web_ans = llm.invoke(fallback_prompt).content
            return f"🌐 **From Internet/LLM:**\n\n{web_ans}", ["🌍 Web Source"]

    # General Greeting Fallback
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
                with st.expander("Sources"):
                    for s in list(set(sources)):
                        st.write(s)

        st.session_state.messages.append({"role": "assistant", "content": ans})
