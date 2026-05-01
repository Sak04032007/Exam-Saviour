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
SUBJECT_PROMPTS = {
    "General Study": "You are a helpful Study Assistant. Your goal is to summarize and explain the uploaded notes in a way that is easy to understand. Use the context to answer directly.",
    "Data Science": "You are a Data Science Professor. Use the provided notes to explain technical definitions, algorithms, and core concepts. If the user asks for 'main concepts' or 'methods', synthesize them from the text.",
    "AI/ML": "You are an AI/ML Specialist. Focus on explaining the mathematical formulas, model architectures, and logic found in the notes. Provide step-by-step breakdowns of the algorithms mentioned.",
    "Cyber Security": "You are a Security Expert. Explain the protocols, threats, and defensive strategies mentioned in the notes. Focus on the technical implementation details found in the context.",
    "Data Structures": "You are a Computer Science Professor. Focus on explaining data structures (Linked Lists, Stacks, Queues, Trees) and algorithms (BFS, DFS) found in the notes. Provide C/Python logic explanations and analyze Time/Space complexity based on the text.",
    "Viva Voice Mode": "You are a strict External Examiner. Your ONLY task is to generate questions based on the definitions, facts, and data points explicitly written in the provided notes. DO NOT invent your own examples, numbers, or lists. If the notes don't have enough detail for a question, ask about a different section of the notes."
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
router_prompt = ChatPromptTemplate.from_template(
    """Analyze the user's intent:
    - If the question involves technical terms, formulas, or concepts likely in a study PDF -> 'vector_db'
    - If the question is about current events, specific URLs, or live data -> 'web_search'
    - If it's a greeting ('hi', 'how are you') or formatting help -> 'llm'
    
    Question: {question}
    Answer with only ONE word: 'vector_db', 'web_search', or 'llm'."""
)
route_chain = router_prompt | llm | StrOutputParser()

grader_prompt = ChatPromptTemplate.from_template(
    """You are a Relevance Filter.
    Context: {context}
    Question: {answer}
    
    Determine if the Context contains information related to the Question. 
    - If the Context discusses the general topics or techniques mentioned in the question -> Reply 'YES'.
    - Only reply 'NO' if the Context is completely irrelevant to the topic.
    
    Reply with ONLY 'YES' or 'NO'."""
)
grader_chain = grader_prompt | llm | StrOutputParser()

# --- THE CORE ENGINE ---
def master_engine_logic(user_input, history, subject):
    persona = SUBJECT_PROMPTS[selected_subject]
    
    # [COMPONENT: ROUTING]
    if subject == "Viva Voice Mode":
        route = "vector_db"
    else:
        route = route_chain.invoke({"question": user_input}).lower()
        
if subject == "Viva Voice Mode":
    # Force the examiner to stay grounded
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "{persona}\n\nSTRICT RULE: You must base your questions ONLY on this context. Do not invent examples or data.\n\nContext: {context_text}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}")
    ])
    # [COMPONENT: GROUNDED QA PROMPT]
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """{persona}
    
    STRICT GROUNDING RULES:
    1. Answer ONLY using the provided Context. 
    2. If the user asks for something specific (like code, formulas, or lists) that is NOT in the Context, you MUST say: "The notes discuss this topic, but they do not provide the specific implementation or data."
    3. Never use outside knowledge to fill in gaps in the notes.
    4. Keep the tone professional and academic.

    Context: {context_text}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])
    # [COMPONENT: VECTOR DB RETRIEVAL]
    if "vector_db" in st.session_state and "vector_db" in route:
        docs = st.session_state.vector_db.similarity_search(user_input, k=7)
        context = "\n".join([d.page_content for d in docs])
        
        # [COMPONENT: HALLUCINATION CHECK]
        grade = grader_chain.invoke({"context": context, "answer": user_input})
        
        if "YES" in grade.upper():
            # [CASE: ANSWER FROM NOTES]
            qa_prompt = ChatPromptTemplate.from_messages([
                ("system", "{persona}\n\nSTRICT: Answer ONLY using this context: {context_text}"),
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}")
            ])
            
            ans = (qa_prompt | llm | StrOutputParser()).invoke({
                "question": user_input, 
                "history": history,
                "persona": persona,
                "context_text": context 
            })
            return f"📚 **From your Notes:**\n\n{ans}", [f"📄 Page {d.metadata.get('page', 0)+1}" for d in docs]
        
        else:
            # [CASE: FALLBACK TO INTERNET]
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

    # [COMPONENT: GENERAL FALLBACK]
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
            hist = [HumanMessage(content=m["content"]) if m["role"]=="user" else AIMessage(content=m["content"]) for m in st.session_state.messages[:-1]]
            
            # Invoke the engine with history
            ans, sources = master_engine_logic(prompt, hist, selected_subject)
            st.markdown(ans)
            if sources:
                with st.expander("Sources"):
                    for s in list(set(sources)):
                        st.write(s)

        st.session_state.messages.append({"role": "assistant", "content": ans})
