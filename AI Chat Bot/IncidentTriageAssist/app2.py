import os
import json
import re
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# 1. Environment and Configuration Initialization
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    st.error("❌ Missing GEMINI_API_KEY inside your local `.env` configuration file.")
    st.stop()

st.set_page_config(page_title="Enterprise Incident Triage Hub", layout="wide")
st.title("🛡️ Automated Incident Triage Assistant (With Policy Override & Metrics)")

DATA_FOLDER = "dataSource"
CHROMA_DIR = "incident_chroma_db_v2"

if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

# Initialize Streamlit Session States for Chat, Override Records, and Latency Tracking
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latency_log" not in st.session_state:
    st.session_state.latency_log = []
if "overrides" not in st.session_state:
    st.session_state.overrides = {}

@st.cache_resource
def init_models():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    return embeddings, llm

embeddings, llm = init_models()

# 🛡️ 2. Advanced PII Scrubbing Mask Component
def scrub_pii_data(text: str) -> tuple:
    """Masks emails, phone numbers, and potential API/Auth tokens to guarantee data privacy."""
    is_scrubbed = False
    
    # Target Expressions
    email_pattern = r'[\w\.-]+@[\w\.-]+\.\w+'
    phone_pattern = r'\b\d{3}[-.\s]??\d{3}[-.\s]??\d{4}\b'
    auth_token_pattern = r'(?i)(bearer\s|api[-_]?key\s|token\s|password\s)[=:]\s*["\']?[a-zA-Z0-9_\-\.]{12,}["\']?'
    
    if re.search(email_pattern, text) or re.search(phone_pattern, text) or re.search(auth_token_pattern, text):
        is_scrubbed = True

    text = re.sub(email_pattern, "[REDACTED_EMAIL]", text)
    text = re.sub(phone_pattern, "[REDACTED_PHONE]", text)
    text = re.sub(auth_token_pattern, r'\1=[REDACTED_SECRET]', text)
    
    return text, is_scrubbed

# 📥 3. JSON Data Processing Pipeline with Scrubbing Analytics
def process_json_tickets():
    json_files = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.json')]
    if not json_files:
        return 0, 0, "No JSON ticket arrays detected inside the 'dataSource' folder."
        
    documents_to_index = []
    scrubbed_count = 0
    
    for file in json_files:
        file_path = os.path.join(DATA_FOLDER, file)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tickets = json.load(f)
                if isinstance(tickets, dict):
                    tickets = [tickets]
                    
                for ticket in tickets:
                    raw_description = ticket.get('description', '')
                    raw_logs = ticket.get('logs', '')
                    
                    # Intercept and clean data pools before ingestion into vector indexing
                    clean_desc, scrubbed_desc = scrub_pii_data(raw_description)
                    clean_logs, scrubbed_logs = scrub_pii_data(raw_logs)
                    
                    if scrubbed_desc or scrubbed_logs:
                        scrubbed_count += 1
                    
                    structured_text = (
                        f"Ticket ID: {ticket.get('ticket_id')}\n"
                        f"Role: {ticket.get('submitter_role')}\n"
                        f"Timestamp: {ticket.get('timestamp')}\n"
                        f"Description: {clean_desc}\n"
                        f"System Logs: {clean_logs}\n"
                    )
                    doc = Document(page_content=structured_text, metadata={"ticket_id": str(ticket.get("ticket_id"))})
                    documents_to_index.append(doc)
        except Exception as e:
            return 0, 0, f"Error processing file {file}: {str(e)}"
            
    if documents_to_index:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_chunks = text_splitter.split_documents(documents_to_index)
        Chroma.from_documents(documents=split_chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
        return len(documents_to_index), scrubbed_count, f"Successfully indexed {len(documents_to_index)} incident records."
    return 0, 0, "No raw incident text extracted."

# 📊 4. UI Dashboard Metrics Row (Response Latency Analytics)
st.subheader("📈 Operation Performance Indicators")
col1, col2, col3 = st.columns(3)

if st.session_state.latency_log:
    avg_latency = sum(st.session_state.latency_log) / len(st.session_state.latency_log)
    last_latency = st.session_state.latency_log[-1]
else:
    avg_latency, last_latency = 0.0, 0.0

col1.metric("Average Triage Latency", f"{avg_latency:.2f} seconds", delta=None if avg_latency == 0 else f"{avg_latency-2.0:.1f}s vs Target")
col2.metric("Last Query Latency", f"{last_latency:.2f} seconds")
col3.metric("Manual Overrides Filed", f"{len(st.session_state.overrides)}")

# ⚙️ 5. Sidebar Controls & Management Panel
with st.sidebar:
    st.header("⚙️ Control Dashboard")
    st.info(f"Target Sync Folder: `/{DATA_FOLDER}`")
    
    if st.button("🔄 Sync & Scrub Tickets"):
        with st.spinner("Executing compliance scrubbing and indexing..."):
            count, scrubbed, msg = process_json_tickets()
            if count > 0:
                st.success(msg)
                st.toast(f"🔒 Sensitive PII redacted across {scrubbed} assets!", icon="🛡️")
            else:
                st.warning(msg)
                
    st.markdown("---")
    st.header("🎛️ Human-in-the-Loop Override")
    st.caption("Manually adjust ticket parameters if the AI triage score requires corrections.")
    
    override_id = st.text_input("Target Ticket ID (e.g., INC-88321)")
    override_priority = st.selectbox("Assign Human-Validated Priority", ["P1 Critical", "P2 High", "P3 Medium", "P4 Low"])
    override_notes = st.text_area("Justification Notes for Audit Trail")
    
    if st.button("💾 Commit Override Decision"):
        if override_id:
            st.session_state.overrides[override_id] = {
                "priority": override_priority,
                "notes": override_notes,
                "user": "Lead Support Engineer"
            }
            st.success(f"Successfully overridden status for {override_id} to {override_priority}!")
            st.rerun()
        else:
            st.error("Please specify a Target Ticket ID to save.")

# 💬 6. Interactive Conversational Interface
st.subheader("💬 Active Triage Console")

for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

if user_query := st.chat_input("Enter a Ticket ID or log description to run assessment..."):
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.chat_history.append({"role": "user", "content": user_query})

    # Catching local regex to extract a referenced Ticket ID for override matching checks
    found_id_match = re.search(r'INC-\d+', user_query, re.IGNORECASE)
    active_override = None
    if found_id_match:
        extracted_id = found_id_match.group(0).upper()
        if extracted_id in st.session_state.overrides:
            active_override = st.session_state.overrides[extracted_id]

    if os.path.exists(CHROMA_DIR):
        start_time = time.time()  # Start Latency Tracker Clock
        
        db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})
        
        system_instruction = (
            "You are an expert Automated Incident Triage Assistant for Application Maintenance.\n"
            "Analyze the retrieved system incidents context below against the user query.\n\n"
            "Format your response with the following structured sections:\n"
            "1. 🚨 **Triage Assessment**: Priority level (P1 Critical, P2 High, P3 Medium, P4 Low) with justification.\n"
            "2. 🧭 **Recommended Routing**: Which maintenance team should handle this.\n"
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
            with st.spinner("Analyzing log data and routing matrices..."):
                response = engine_retrieval_chain.invoke({"input": user_query})
                ai_reply = response["answer"]
                
                # Append Override Warnings explicitly in real-time to the user view
                if active_override:
                    override_banner = (
                        f"\n\n--- \n⚠️ **CRITICAL MANUAL OVERRIDE ACTIVE**\n"
                        f"*   **Human-Validated Priority:** {active_override['priority']}\n"
                        f"*   **Audit Trail Notes:** {active_override['notes']}\n"
                        f"*   *Note: This human decision overrides the standard machine triage configuration.*"
                    )
                    ai_reply += override_banner
                
                #  YOUR CHOSEN TEXT STARTS HERE IN THE SCRIPT:
                st.markdown(ai_reply)
        
        # Calculate performance log metrics (Indented under the main 'if os.path.exists' block)
        execution_duration = time.time() - start_time
        st.session_state.latency_log.append(execution_duration)
        st.session_state.chat_history.append({"role": "assistant", "content": ai_reply})
        st.rerun()
    else:  # Matches up directly with 'if os.path.exists(CHROMA_DIR):'
        with st.chat_message("assistant"):
            st.markdown("⚠️ Database index is empty. Please drop your files in `dataSource` and click sync.")
