import os
import json
import time
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader

# 1. System Setup & Security Validation
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("❌ Missing GEMINI_API_KEY in your `.env` configuration file.")
    st.stop()

genai.configure(api_key=api_key)

st.set_page_config(page_title="Freight Reconciliation Hub", layout="wide")
st.title("📦 Agentic Multimodal Freight Reconciliation System")
st.caption("Upload warehouse media and SLA policy contracts directly to execute real-time validation checks.")

DATA_FOLDER = "dataSource"
CHROMA_DIR = "sla_chroma_db"

# Ensure runtime folder paths exist locally
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)

@st.cache_resource
def init_models():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    return embeddings, llm

embeddings, llm = init_models()

# Safe Document String Extraction Engine
def parse_and_index_pdf(file_path):
    try:
        loader = PyPDFLoader(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        split_chunks = text_splitter.split_documents(loader.load())
        
        # Populate persistent database layer
        Chroma.from_documents(documents=split_chunks, embedding=embeddings, persist_directory=CHROMA_DIR)
        return True, f"Successfully parsed and indexed contract rules into Chroma DB!"
    except Exception as e:
        return False, f"Failed to read PDF stream components: {str(e)}"

# Read Simulated Local ERP Files
def load_mock_erp_data():
    try:
        inv_path = os.path.join(DATA_FOLDER, "inventory.json")
        po_path = os.path.join(DATA_FOLDER, "purchase_orders.json")
        
        if not os.path.exists(inv_path) or not os.path.exists(po_path):
            return None
            
        with open(inv_path, 'r') as f:
            inv = json.load(f)
        with open(po_path, 'r') as f:
            po = json.load(f)
        return {"inventory": inv, "purchase_orders": po}
    except Exception as e:
        st.sidebar.error(f"ERP Data Read Error: {str(e)}")
        return None

erp_data = load_mock_erp_data()

# ⚙️ 2. Sidebar Control Hub: Interactive Contract Ingestion
with st.sidebar:
    st.header("📜 SLA Contract Ingestion")
    st.caption("Upload vendor policy agreements directly to seed the system guidelines knowledge store.")
    
    # Dedicated SLA Contract PDF Document Uploader Loop
    uploaded_pdf = st.file_uploader("Choose Vendor SLA Document (PDF)", type=["pdf"])
    
    if uploaded_pdf is not None:
        temp_pdf_path = os.path.join(DATA_FOLDER, f"uploaded_{uploaded_pdf.name}")
        
        # Write to temporary file safely to allow Langchain utilities to stream contents
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
            
        with st.spinner("Parsing text tracks and calculating embeddings..."):
            success, message = parse_and_index_pdf(temp_pdf_path)
            if success:
                st.success(message)
            else:
                st.error(message)
                
            # Clean up local file copy to avoid storage pollution
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
                
    st.markdown("---")
    st.subheader("📊 Backend ERP State Snapshots")
    if erp_data:
        st.write(f"Items in Inventory: {len(erp_data['inventory'])}")
        st.write(f"Tracked Active POs: {len(erp_data['purchase_orders'])}")
    else:
        st.warning("⚠️ ERP files missing from `/dataSource`. Ensure inventory.json and purchase_orders.json exist.")

# 🎥 3. Main Dashboard Workspace: Direct Media Processing
st.subheader("🎥 Step 1: Upload Warehouse Inspection Media")

uploaded_media = st.file_uploader(
    "Drag and drop or browse warehouse incident recording:", 
    type=["mp4", "avi", "mov", "mp3", "wav"]
)

if uploaded_media is not None:
    temp_media_path = f"temp_{uploaded_media.name}"
    with open(temp_media_path, "wb") as f:
        f.write(uploaded_media.getbuffer())
        
    if uploaded_media.name.lower().endswith(('.mp4', '.avi', '.mov')):
        st.video(temp_media_path)
    else:
        st.audio(temp_media_path)
    
    if st.button("🚀 Execute Agentic Reconciliation Pipeline"):
        with st.spinner("Step A: Uploading file to Gemini Multimodal Media Storage..."):
            google_file = genai.upload_file(path=temp_media_path)
            
            while google_file.state.name == "PROCESSING":
                time.sleep(1)
                google_file = genai.get_file(google_file.name)
                
            if google_file.state.name == "FAILED":
                st.error("Google Multimodal File processing failed.")
                os.remove(temp_media_path)
                st.stop()
                
        with st.spinner("Step B: Extracting metadata and damage assessment from media..."):
            native_model = genai.GenerativeModel('gemini-2.5-flash')
            extraction_prompt = (
                "Analyze this warehouse inspection recording. Extract the following details as plain text:\n"
                "1. Mentioned Purchase Order (PO) Number\n"
                "2. Type of items and quantity damaged\n"
                "3. Specific nature of the physical damage (e.g. wet, crushed, torn)"
            )
            media_analysis = native_model.generate_content([google_file, extraction_prompt]).text
            st.info(f"📊 **Extracted Media Attributes:**\n{media_analysis}")
            
            genai.delete_file(google_file.name)

        with st.spinner("Step C: Querying SLA conditions and analyzing ERP business impact..."):
            context_text = ""
            if os.path.exists(CHROMA_DIR):
                db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
                docs = db.similarity_search(media_analysis, k=3)
                context_text = "\n".join([d.page_content for d in docs])

            if not context_text:
                st.warning("⚠️ No active vector policy content matches your incident signature. Ensure you upload a valid SLA document in the sidebar first.")

            final_reconciliation_prompt = (
                f"You are an automated Freight Reconciliation Agent.\n"
                f"Reconcile the following extracted media findings with our rules and ERP system records.\n\n"
                f"--- Extracted Inspection Details ---\n{media_analysis}\n\n"
                f"--- Vendor SLA Contract Policies ---\n{context_text}\n\n"
                f"--- Active ERP Data State ---\n{json.dumps(erp_data, indent=2) if erp_data else '{}'}\n\n"
                f"Provide a finalized action report using these EXACT markdown headers:\n"
                f"### 🎯 Damage & Metadata Identification\n"
                f"### ⚖️ Liability & SLA Determination (Who pays based on rules?)\n"
                f"### ⚠ Supply Chain & Production Halt Impact (Does stock drop below minimum?)\n"
                f"### 📝 Generated Claim Payload / Dispatch Action"
            )
            
            final_report = llm.invoke(final_reconciliation_prompt).content
            st.success("🏁 Reconciliation Pipeline Complete!")
            st.markdown(final_report)
            
            if os.path.exists(temp_media_path):
                os.remove(temp_media_path)

