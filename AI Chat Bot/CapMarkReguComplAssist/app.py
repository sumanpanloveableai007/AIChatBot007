import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load Environmental Variables Safely
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("❌ Missing GEMINI_API_KEY. Please verify your `.env` configuration file.")
    st.stop()

# 2. Configure Streamlit Web Interface Dashboard
st.set_page_config(page_title="Capital Markets Compliance Assistant", layout="wide")
st.title("🏦 AI-Powered Capital Markets Regulatory Compliance Assistant")
st.caption("Contextual intelligence covering complex MiFID II, SEC Rule 606, and Dodd-Frank frameworks.")

# Initialize Persistent Storage Directory for Vector Database
CHROMA_DIR = "chroma_db_storage"

# Instantiate Model Instances via LangChain
@st.cache_resource
def init_models():
    # Use standard text-embedding-004 for stable semantic analysis
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    # Leverage fast reasoning via Gemini 2.5 Flash
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    return embeddings, llm

embeddings, llm = init_models()

# Sidebar: Document Processing Operations Hub
with st.sidebar:
    st.header("📂 Document Ingestion Hub")
    uploaded_file = st.file_uploader("Upload Regulatory Text / Internal Policy (PDF)", type=["pdf"])
    
    if uploaded_file:
        # Save temp file locally for processing
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.info("Parsing document contents...")
        try:
            # Document chunking & database embedding pipeline
            loader = PyPDFLoader(temp_path)
            docs = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_chunks = text_splitter.split_documents(docs)
            
            # Store in local Chroma database instance
            vector_store = Chroma.from_documents(
                documents=split_chunks, 
                embedding=embeddings, 
                persist_directory=CHROMA_DIR
            )
            st.success(f"Successfully indexed {len(split_chunks)} distinct chunks into Chroma DB!")
            os.remove(temp_path)
        except Exception as e:
            st.error(f"Error handling file upload: {str(e)}")

# Initialize Chat Memory Cache in Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Main Workspace View: Interactive Conversational Dashboard
st.subheader("💬 Regulatory Guidance Terminal")

# Display complete chronological thread logs 
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

# Process New Queries incoming from Compliance Analysts
if user_query := st.chat_input("Enter your regulatory question or input an API transaction summary..."):
    # Immediately render User Query inside the dashboard UI
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Execute dynamic Context Retrieval from Database
    context_retrieved = ""
    if os.path.exists(CHROMA_DIR):
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 4})
        
        # Build strict RAG instruction prompt mapping out clear expectations
        system_instruction = (
            "You are an expert Capital Markets Regulatory Compliance Assistant.\n"
            "Analyze the retrieved context below against the user's inquiry regarding "
            "MiFID II, SEC Rule 606, or Dodd-Frank requirements.\n\n"
            "Retrieved System Context:\n{context}\n\n"
            "Provide explicit compliance guidance, note operational discrepancies, or flag potential violations."
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", "{input}"),
        ])
        
        # Execution Chain Architecture logic
        document_assembly_chain = create_stuff_documents_chain(llm, prompt_template)
        engine_retrieval_chain = create_retrieval_chain(retriever, document_assembly_chain)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing rules and cross-referencing records..."):
                response = engine_retrieval_chain.invoke({"input": user_query})
                ai_reply = response["answer"]
                st.markdown(ai_reply)
        
        st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
    else:
        # Fallback Base-Model intelligence if Database is empty
        with st.chat_message("assistant"):
            with st.spinner("Analyzing using global regulatory base-knowledge..."):
                ai_reply = llm.invoke(user_query).content
                st.markdown(ai_reply)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})

# pip install -r requirements.txt
# python -m streamlit run app.py