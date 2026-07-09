# pip install -r requirements.txt

import streamlit as st
import pandas as pd
import json
import os
import io
import tiktoken
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Load system environmental keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# App Window Configuration
st.set_page_config(
    page_title="Multi-Agent Fraud Defense Core",
    page_icon="🛡️",
    layout="wide"
)

# ----------------- SESSION STATE & LOGIN PAGE -----------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

def login_page():
    st.markdown("<h2 style='text-align: center;'>🛡️ Enterprise Security Portal</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please enter your credentials to access the Fraud Detection Dashboard</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            username = st.text_input("Username", value="")
            password = st.text_input("Password", type="password", value="")
            submit_btn = st.form_submit_button("Sign In", type="primary", use_container_width=True)
            
            if submit_btn:
                if username == "admin" and password == "admin":
                    st.session_state["logged_in"] = True
                    st.rerun()
                else:
                    st.error("❌ Invalid Username or Password. Access Denied.")

# Exit early if user is unauthenticated
if not st.session_state["logged_in"]:
    login_page()
    st.stop()

# ----------------- MAIN PROTECTED APPLICATION -----------------
st.title("🛡️ Enterprise Multi-Agent Fraud Analytics Engine")
st.subheader("Automated batch verification powered by LangChain & OpenAI GPT-4o-mini")

# Verify API connection safely
if not OPENAI_API_KEY:
    st.error("❌ Configuration Error: 'OPENAI_API_KEY' was not found inside your local .env file.")
    st.stop()

# Helper function to convert dataframe back to excel for downloads
def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Risk Summary')
    return output.getvalue()

# ----------------- LANGCHAIN ENGINE LOGIC -----------------
def execute_multi_agent_pipeline(row_data):
    # Initialize the OpenAI model via LangChain
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    
    payload_str = json.dumps(row_data, indent=2)
    
    # 1. Ask Model to simulate Agent 1 (Patterns)
    prompt_p = ChatPromptTemplate.from_template("You are a Transaction Pattern Analyst Agent. Analyze for velocity, structural anomalies, and value risks: {data}. Summarize risks and score as LOW, MEDIUM, or HIGH.")
    chain_p = prompt_p | llm
    res_pattern = chain_p.invoke({"data": payload_str}).content
    
    # 2. Ask Model to simulate Agent 2 (Database Lookups)
    prompt_h = ChatPromptTemplate.from_template("You are a Historical Database Lookup Agent. Cross-verify this against mock blacklists and geographical changes: {data}. Summarize risks and score as LOW, MEDIUM, or HIGH.")
    chain_h = prompt_h | llm
    res_history = chain_h.invoke({"data": payload_str}).content
    
    # 3. Ask Model to simulate Agent 3 (Biometrics)
    prompt_b = ChatPromptTemplate.from_template("You are a Behavioral Biometrics Agent. Check for keystroke patterns or automated text/clipboard injection: {data}. Summarize risks and score as LOW, MEDIUM, or HIGH.")
    chain_b = prompt_b | llm
    res_biometric = chain_b.invoke({"data": payload_str}).content
    
    # 4. Orchestration Step: Merge insights into strict output tokens
    orchestrator_prompt = ChatPromptTemplate.from_template("""
    You are the Chief Financial Risk Orchestrator Agent. Synthesize these reports:
    Agent 1: {a1}
    Agent 2: {a2}
    Agent 3: {a3}
    
    Provide the final output strictly following this template structure:
    VERDICT: [APPROVE, CHALLENGE WITH MFA, or BLOCK]
    REASON: [1-sentence core reason why]
    """)
    chain_orch = orchestrator_prompt | llm
    final_verdict = chain_orch.invoke({"a1": res_pattern, "a2": res_history, "a3": res_biometric}).content
    
    return final_verdict

# ----------------- DATA INGESTION LAYOUT -----------------
st.markdown("### 📥 1. Upload Spreadsheet Registry")
uploaded_file = st.file_uploader("Drop your transaction ledger file here (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    st.markdown("### 👀 Raw Data Preview")
    st.dataframe(df, use_container_width=True)
    
    if st.button("🚀 Run Multi-Agent Verification Sweep", type="primary"):
        progress_bar = st.progress(0)
        final_verdicts = []
        
        st.markdown("### 📊 Processing Activity Logs")
        total_rows = len(df)
        
        for index, row in df.iterrows():
            user_label = row.get("User ID", f"Index-{index}")
            st.write(f"🕵️ Syncing parallel agents for row **{index + 1}/{total_rows}** ({user_label})...")
            
            # Extract clean row payload dictionary
            row_dict = row.to_dict()
            
            # Execute LangChain Agent workflow
            verdict = execute_multi_agent_pipeline(row_dict)
            final_verdicts.append(verdict)
            
            # Advance progress metric
            progress_bar.progress((index + 1) / total_rows)
            
        # Bind output values dynamically into the new Web-UI presentation matrix
        df['Multi-Agent Risk Analysis Output'] = final_verdicts
        
        st.success("✅ Complete Batch Verification Sweep Finished!")
        st.markdown("### 📋 2. Fraud Detection Analysis Matrix")
        
        # Display the modified output table natively inside Web-UI 
        st.dataframe(df, use_container_width=True)
        
        # Provide Export Options
        st.download_button(
            label="💾 Download Processed Risk Matrix",
            data=convert_df_to_excel(df),
            file_name="multi_agent_fraud_audit.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

# python -m streamlit run app.py