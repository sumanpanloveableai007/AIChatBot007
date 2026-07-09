import os
import json
import time
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document

# 1. Initialization and Environment Validations
load_dotenv()
if not os.getenv("GEMINI_API_KEY"):
    st.error("❌ Missing GEMINI_API_KEY inside your local `.env` configuration file.")
    st.stop()

st.set_page_config(page_title="Cloud Drift Detection Hub", layout="wide")
st.title("☁️ Cloud Infrastructure Drift Detection and Alert System")
st.caption("Automated folder scanning pipeline with native Slack / MS Teams webhook alerting integrations.")

CHROMA_DIR = "drift_history_db"
BASELINE_DIR = "baseline"
LIVE_DIR = "live_state"

# Ensure environmental runtime directories exist locally
for target_dir in [BASELINE_DIR, LIVE_DIR]:
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

@st.cache_resource
def init_models():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
    return embeddings, llm

embeddings, llm = init_models()

# Initialize Streamlit Session state to track analytics metrics
if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0
if "drift_incidents" not in st.session_state:
    st.session_state.drift_incidents = 0

# 📡 2. Automated Slack / MS Teams Webhook Alerting Router
def dispatch_webhook_alert(webhook_url, report_payload, resource_name):
    """Dispatches a clean, formatted alert card to an active Slack or Teams channel."""
    if not webhook_url:
        return False, "Webhook configuration missing."
        
    # Standard fallback payload block parsing
    headers = {"Content-Type": "application/json"}
    
    # Slack specific payload formatting check
    if "hooks.slack.com" in webhook_url:
        payload = {
            "text": f"🚨 *Cloud Infrastructure Drift Alert*\n*Target Resource:* `{resource_name}`\n\n{report_payload[:1000]}..."
        }
    # Teams or generic messaging endpoint channel formats
    else:
        payload = {
            "title": f"🚨 Cloud Infrastructure Drift Detected: {resource_name}",
            "text": report_payload
        }
        
    try:
        response = requests.post(webhook_url, json=payload, headers=headers, timeout=8)
        if response.status_code in [200, 201, 204]:
            return True, "Broadcast alert routed successfully!"
        return False, f"Server responded with status block code: {response.status_code}"
    except Exception as e:
        return False, f"HTTP Connection processing error: {str(e)}"

# 📊 3. Executive Visualization Dashboard Metrics Row
st.subheader("📊 Environment Status Metrics")
m1, m2, m3 = st.columns(3)
m1.metric("Total Automated Scans Executed", st.session_state.total_scans)
m2.metric("Active Drifts Identified", st.session_state.drift_incidents, delta="Attention Required" if st.session_state.drift_incidents > 0 else "Compliant")
m3.metric("System Core Monitor Node", "Directory Tracking Active")

# ⚙️ 4. Sidebar Configuration Webhook Controller
with st.sidebar:
    st.header("🎛️ Integration Settings")
    st.caption("Configure notification channels to broadcast security alerts when environment modifications occur.")
    
    webhook_target = st.text_input(
        "Channel Webhook URL", 
        placeholder="https://slack.com... or MS Teams URL",
        help="Paste your active Slack App incoming webhook or Teams connector endpoint reference here."
    )
    if webhook_target:
        st.success("🔗 Alert routing configuration registered.")

# 📂 5. Folder Scanning Matrix Pipeline Execution
st.subheader("📂 Local Scan Controller")
col_b, col_l = st.columns(2)

baseline_files = [f for f in os.listdir(BASELINE_DIR) if f.lower().endswith(('.json', '.yaml', '.yml'))]
live_files = [f for f in os.listdir(LIVE_DIR) if f.lower().endswith(('.json', '.yaml', '.yml'))]

with col_b:
    st.markdown(f"**📂 Target Folder: `/{BASELINE_DIR}`**")
    st.write(f"Detected blueprints available: `{len(baseline_files)}`")
    selected_base = st.selectbox("Select Baseline Configuration:", baseline_files if baseline_files else ["No files found"])

with col_l:
    st.markdown(f"**📂 Target Folder: `/{LIVE_DIR}`**")
    st.write(f"Detected active states available: `{len(live_files)}`")
    selected_live = st.selectbox("Select Active Live State:", live_files if live_files else ["No files found"])

if baseline_files and live_files and selected_base != "No files found" and selected_live != "No files found":
    if st.button("🔍 Execute Automated Directory Scan & Cross-Check"):
        base_path = os.path.join(BASELINE_DIR, selected_base)
        live_path = os.path.join(LIVE_DIR, selected_live)
        
        try:
            with open(base_path, 'r', encoding='utf-8') as f:
                baseline_str = f.read()
            with open(live_path, 'r', encoding='utf-8') as f:
                live_str = f.read()
                
            st.session_state.total_scans += 1
            
            with st.spinner("Analyzing infrastructure states and comparing security objects..."):
                analysis_prompt = (
                    f"You are an expert Cloud Infrastructure Security and Drift Monitoring Agent.\n"
                    f"Compare the following two infrastructure configuration files side-by-side to identify anomalies.\n\n"
                    f"--- INTENDED BLUEPRINT BASELINE ---\n{baseline_str}\n\n"
                    f"--- LIVE PRODUCTION STATE ---\n{live_str}\n\n"
                    f"Generate a security alert report using these EXACT markdown headers:\n"
                    f"### 📋 Drift Status Summary\n"
                    f"Explicitly declare if drift is 'DETECTED' or 'NOT DETECTED'. Specify which cloud resource target name changed.\n\n"
                    f"### 🚨 Risk Assessment & Severity Profile\n"
                    f"Assign a severity rating (Critical, High, Medium, Low). Detail what structural fields were modified.\n\n"
                    f"### 🛠️ Step-by-Step Remediation Action Blueprint\n"
                    f"Provide the exact administrative resolution tasks or programmatic CLI rollbacks."
                )
                
                analysis_response = llm.invoke(analysis_prompt).content
                st.markdown(analysis_response)
                
                # Check text flags to update dynamic telemetry state values
                if "DRIFT DETECTED" in analysis_response.upper() or "DETECTED" in analysis_response.upper():
                    st.session_state.drift_incidents += 1
                    st.error("🚨 Configuration Drift and Security Deviation Identified!")
                    
                    # 1. Trigger the Automated Webhook Action block
                    if webhook_target:
                        with st.spinner("Broadcasting incident metrics to remote alert server channels..."):
                            hook_ok, hook_msg = dispatch_webhook_alert(webhook_target, analysis_response, selected_base)
                            if hook_ok:
                                st.toast(hook_msg, icon="🔔")
                            else:
                                st.sidebar.error(f"Alert routing failed: {hook_msg}")
                    
                    # 2. Store historical alerts inside local vector storage
                    db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
                    db.add_documents([Document(page_content=analysis_response, metadata={"target": selected_base})])
                else:
                    st.success("✅ Compliance Intact: Live state mirrors intended system baseline metrics.")
                
        except Exception as e:
            st.error(f"Execution Error processing configuration structures: {str(e)}")
else:
    st.warning("⚠️ Make sure you drop matching file arrays inside your baseline/ and live_state/ system directories to activate scans.")

# 6. Audit Log Workspace Shell Viewer
st.markdown("---")
st.subheader("📜 Historical Security Alert Audit Trail")
if os.path.exists(CHROMA_DIR):
    try:
        audit_db = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        results = audit_db.get()
        if results and 'documents' in results and results['documents']:
            for idx, doc_text in enumerate(results['documents']):
                with st.expander(f"Alert Incident Log Reference Record #{idx+1}"):
                    st.text(doc_text[:600] + "... [Truncated]")
    except Exception:
        pass

