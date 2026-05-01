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
# Standard Streamlit and environment configuration
st.set_page_config(page_title="Exam Savior AI", layout="wide")

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if "TAVILY_API_KEY" in st.secrets:
    os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]

# --- 2. THE BRAINS (LLM & TOOLS) ---
# Initialize the Llama model, Embeddings for PDF search, and Web Search tool
@st.cache_resource
def load_core_tools():
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    web_search = TavilySearchResults(k=3)
    return llm, embeddings, web_search

llm, embeddings, web_search = load_core_tools()

# --- 3. THE PROMPT LIBRARY (Professor Personas) ---
# Specific personas to guide the LLM's behavior based on the chosen subject
SUBJECT_PROMPTS = {
    "General Study": "You are a helpful Study Assistant. Your goal is to summarize and explain the uploaded notes in a way that is easy to understand.",
    "Data Science": "You are a Data Science Professor.Use LaTeX for all mathematical formulas (e.g., write $$e=mc^2$$). Use the provided notes to explain technical definitions, algorithms, and core concepts.",
    "AI/ML": "You are an AI/ML Specialist. Use LaTeX for all mathematical formulas (e.g., write $$e=mc^2$$).Focus on explaining the mathematical formulas, model architectures, and logic found in the notes.",
    "Cyber Security": "You are a Security Expert. Explain the protocols, threats, and defensive strategies mentioned in the notes.",
    "Data Structures": "You are a Computer Science Professor. Focus on explaining data structures (Linked Lists, Stacks, Queues, Trees) and algorithms (BFS, DFS).",
    "Viva Voice Mode": "You are a strict External Examiner. Your ONLY task is to generate questions based strictly on the provided notes."
}

# --- 4. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("""
    <h1 style='text-align: center; 
    background: -webkit-linear-gradient(#00c6ff, #0072ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900; font-size: 60px;'>
    Celeritas
    </h1>
    """, unsafe_allow_html=True)
    st.divider()

    #  EXAM COUNTDOWN
    from datetime import date
    st.subheader("Exam Schedule")
    user_exam_date = st.date_input("Select your exam start date:" , value = date(2026 , 5,15))
    today = date.today()
    days_left = (user_exam_date - today).days
    st.metric(label="😨 Days Until Exams", value=max(0, days_left))
    st.divider()

    #  CUSTOM SYLLABUS TRACKER
    st.subheader("📚 Syllabus Tracker")
    custom_topics_input = st.text_area("Enter topics (one per line):", 
                                     value="K-Means Clustering\nLinked Lists\nNeural Networks\nComplexity Analysis")
    
    # Split input into a list and create checkboxes
    syllabus_list = [t.strip() for t in custom_topics_input.split('\n') if t.strip()]
    for topic in syllabus_list:
        st.checkbox(topic, key=f"track_{topic}")
    st.divider() 

    #  FLASHCARD GENERATOR
    if st.button("🗂️ Generate Flashcards"):
        if "vector_db" in st.session_state:
            with st.spinner("Creating flashcards..."):
                context_sample = st.session_state.vector_db.similarity_search("key concepts", k=5)
                context_text = "\n".join([d.page_content for d in context_sample])
                flash_prompt = f"Based on these notes, create 5 quick flashcards (Term: Definition). \nContext: {context_text}"
                cards = llm.invoke(flash_prompt).content
                st.session_state.messages.append({"role": "assistant", "content": f"✨ **Your Instant Flashcards:**\n\n{cards}"})
                st.rerun()
        else:
            st.error("Upload a PDF first!")

    # ⚙️ SETTINGS
    st.title("🤓 Study Settings")
    selected_subject = st.selectbox("Choose Professor Mode:", list(SUBJECT_PROMPTS.keys()))
    uploaded_file = st.file_uploader("Upload your CSE Notes (PDF)", type="pdf")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- 5. DYNAMIC PDF PROCESSING ---
# This block must remain OUTSIDE the sidebar 'with' block 
# but AFTER uploaded_file is defined to avoid NameErrors.
def process_new_notes(file):
    with open("temp.pdf", "wb") as f:
        f.write(file.getbuffer())
    
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(pages)
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="user_notes"
    )
    return vectorstore

if uploaded_file and "vector_db" not in st.session_state:
    with st.spinner("Processing your notes..."):
        st.session_state.vector_db = process_new_notes(uploaded_file)
        st.success("Notes processed!")

# --- 6. YOUR AGENTIC LOGIC (ROUTER + GRADER) ---
# Router determines if the user needs the PDF, the Web, or just general chat
router_prompt = ChatPromptTemplate.from_template(
    """Analyze the user's intent:
    - If the question involves technical terms, formulas, or concepts likely in a study PDF -> 'vector_db'
    - If the question is about current events, specific URLs, or live data -> 'web_search'
    - If it's a greeting ('hi', 'how are you') or formatting help -> 'llm'
    
    Question: {question}
    Answer with only ONE word: 'vector_db', 'web_search', or 'llm'."""
)
route_chain = router_prompt | llm | StrOutputParser()

# Grader checks if retrieved PDF context is actually relevant to the question
grader_prompt = ChatPromptTemplate.from_template(
    """You are a Relevance Filter.
    Context: {context}
    Question: {answer}
    
    Determine if the Context contains information related to the Question. 
    1. If the user is saying something conversational (e.g., "I don't know", "Hi", "Next", "Yes", "No"), reply 'YES'.
    2. If the Context discusses the general topics or techniques mentioned in the question -> Reply 'YES'.
    3. Only reply 'NO' if the user is asking a specific factual question that is definitely NOT in the Context.
    
    Reply with ONLY 'YES' or 'NO'."""
)
grader_chain = grader_prompt | llm | StrOutputParser()

# --- THE CORE ENGINE ---
# Main logic that coordinates routing, retrieval, grading, and answering
def master_engine_logic(user_input, history, subject):
    persona = SUBJECT_PROMPTS[selected_subject]
    
    # [COMPONENT: ROUTING]
    # Forces 'vector_db' route if in Viva mode, otherwise uses the router
    if subject == "Viva Voice Mode":
        route = "vector_db"
    else:
        route = route_chain.invoke({"question": user_input}).lower()
        
    # [COMPONENT: VECTOR DB RETRIEVAL & PROCESSING]
    if "vector_db" in st.session_state and "vector_db" in route:
        docs = st.session_state.vector_db.similarity_search(user_input, k=7)
        context = "\n".join([d.page_content for d in docs])
        
        # [COMPONENT: RELEVANCE GRADING]
        grade = grader_chain.invoke({"context": context, "answer": user_input})
        
        if "YES" in grade.upper():
            # [CASE: ANSWER FROM NOTES WITH STRICT GROUNDING]
            # Use specific system prompt for Viva Mode or standard grounding
            if subject == "Viva Voice Mode":
                sys_msg = f"{persona}\n\nSTRICT RULE: You must base your questions ONLY on this context. Do not invent examples or data.\n\nContext: {{context_text}}"
            else:
                sys_msg = f"{persona}\n\nSTRICT GROUNDING RULES: 1. Answer ONLY using the provided Context. 2. If code/formulas/data requested are NOT in context, state it is missing. 3. No outside knowledge.\n\nContext: {{context_text}}"

            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", sys_msg),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ])
            
            ans = (qa_prompt | llm | StrOutputParser()).invoke({
                "question": user_input, 
                "history": history,
                "context_text": context 
            })
            return f"📚 **From your Notes:**\n\n{ans}", [f"📄 Page {d.metadata.get('page', 0)+1}" for d in docs]
        
        else:
            # [CASE: FALLBACK TO INTERNET]
            # Triggers if the grader determines notes don't contain the answer
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

    # [COMPONENT: GENERAL FALLBACK / LLM CHAT]
    return llm.invoke(f"System: {persona}\nUser: {user_input}").content, []

# --- 7. THE UI CHAT ---
# Renders chat history and handles new user inputs
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if prompt := st.chat_input("Ask your Professor..."):
    # [COMPONENT: CONDITIONAL ACCESS]
    # Allow "General Study" to work without a PDF, but block technical modes
    needs_pdf = selected_subject not in ["General Study"]
    
    if needs_pdf and not uploaded_file:
        st.warning(f"Please upload a PDF to use {selected_subject} mode!")
    else:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            # Format chat history
            hist = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
            
            # [COMPONENT: ENGINE EXECUTION]
            ans, sources = master_engine_logic(prompt, hist, selected_subject)
            st.markdown(ans)
            
            if sources:
                with st.expander("Sources"):
                    for s in list(set(sources)):
                        st.write(s)

        st.session_state.messages.append({"role": "assistant", "content": ans})
