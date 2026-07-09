import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# 1. Environment Initialization
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    st.error("❌ Missing GEMINI_API_KEY inside your local `.env` configuration file.")
    st.stop()

st.set_page_config(page_title="AI Incident Triage Assistant", layout="wide")
st.title("🛠️ Automated Incident Triage Assistant")
st.caption("AI-driven ticket prioritization, confidence scoring, and routing recommendation matrix.")

DATA_FOLDER = "dataSource"
CHROMA_DIR = "incident_chroma_db"

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

@st.cache_resource
def init_models():
    # Utilizing active Gemini framework modules
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    return embeddings, llm

embeddings, llm = init_models()

# 2. JSON Data Processing Pipeline
def process_json_tickets():
    json_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.json')]
    if not json_files:
        return 0, "No JSON ticket arrays detected inside the 'dataSource' folder."
        
    documents_to_index = []
    
    for file in json_files:
        file_path = os.path.join(DATA_FOLDER, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tickets = json.load(f)
                
                # If a single object is passed instead of a list, wrap it
                if isinstance(tickets, dict):
                    tickets = [tickets]
                    
                for ticket in tickets:
                    # Construct a flat contextual layout optimized for LLM reading
                    structured_text = (
                        f"Ticket ID: {ticket.get('ticket_id')}\n"
                        f"Role: {ticket.get('submitter_role')}\n"
                        f"Timestamp: {ticket.get('timestamp')}\n"
                        f"Description: {ticket.get('description')}\n"
                        f"System Logs: {ticket.get('logs')}\n"
                    )
                    # Convert to LangChain core document formats
                    doc = Document(page_content=structured_text, metadata={"ticket_id": ticket.get("ticket_id")})
                    documents_to_index.append(doc)
        except Exception as e:
            return 0, f"Error processing file {file}: {str(e)}"
            
    if documents_to_index:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_chunks = text_splitter.split_documents(documents_to_index)
        
        # Initialize or append to persistent vector DB store
        Chroma.from_documents(documents=split_chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
        return len(documents_to_index), f"Successfully indexed {len(documents_to_index)} incident records."
    return 0, "No raw incident text extracted."

# Dashboard Processing Controls Hub
with st.sidebar:
    st.header("⚙️ Incident Ingestion Hub")
    st.info(f"Target Sync Folder: `/{DATA_FOLDER}`")
    if st.button("🔄 Sync & Process Tickets"):
        with st.spinner("Processing JSON log files..."):
            count, msg = process_json_tickets()
            if count > 0:
                st.success(msg)
            else:
                st.warning(msg)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 3. Interactive Triage Workspace
st.subheader("💬 Active Triage Console")

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

if user_query := st.chat_input("Enter a Ticket ID, query system logs, or type a description to triage..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    if os.path.exists(CHROMA_DIR):
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        system_instruction = (
            "You are an expert Automated Incident Triage Assistant for Application Maintenance.\n"
            "Analyze the retrieved system incidents context below against the user query.\n\n"
            "Format your response with the following structured sections:\n"
            "1. 🚨 **Triage Assessment**: Priority level (P1 Critical, P2 High, P3 Medium, P4 Low) with justification.\n"
            "2. 🧭 **Recommended Routing**: Which maintenance team should handle this (e.g., Database Team, Frontend, SecOps, Payments Devs).\n"
            "3. 🎯 **Confidence Score**: Scale 0-100% with technical reasoning.\n"
            "4. 📝 **AI Incident Summary**: A clean, technical summary mapping errors to issues.\n\n"
            "Retrieved System Context:\n{context}"
        )
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("human", "{input}"),
        ])
        
        document_assembly_chain = create_stuff_documents_chain(llm, prompt_template)
        engine_retrieval_chain = create_retrieval_chain(retriever, document_assembly_chain)
        
        with st.chat_message("assistant"):
            with st.spinner("Analyzing log errors and routing paths..."):
                response = engine_retrieval_chain.invoke({"input": user_query})
                ai_reply = response["answer"]
                st.markdown(ai_reply)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
    else:
        with st.chat_message("assistant"):
            st.markdown("⚠️ Database index is empty. Please drop your files in `dataSource` and click the sync button.")


