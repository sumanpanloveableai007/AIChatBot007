# pip install streamlit google-generativeai pandas

import streamlit as st
import google.generativeai as genai
import pandas as pd
import json

# Page configuration
st.set_page_config(
    page_title="Multi-Agent Fraud Detection Radar",
    page_icon="🛡️",
    layout="wide"
)

# Application Header
st.title("🛡️ Multi-Agent Fraud Detection Radar")
st.subheader("Simultaneous multi-perspective risk evaluation powered by Gemini 2.5 Flash")

# Sidebar for API Configuration
st.sidebar.header("🔑 Authentication")
api_key = st.sidebar.text_input("Enter Gemini API Key:", type="password")

if api_key:
    genai.configure(api_key=api_key)
else:
    st.sidebar.warning("Please enter your Google Gemini API Key to run the system.")

# Sample/Test Data Setup
st.sidebar.header("📁 Mock Database Context")
st.sidebar.info(
    "The system assumes a simulated local environment containing historical "
    "device fingerprints, velocity caps, and global blacklists."
)

# Main UI - 2 Column Layout for Transaction Details
st.header("📥 Transaction Evaluation Payload")
col1, col2 = st.columns(2)

with col1:
    user_id = st.text_input("User ID", value="USR-90421")
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=4950.00, step=10.0)
    merchant = st.text_input("Merchant Name / Category", value="LUX-CRYPTO-EXCHANGE / High-Risk Category")
    location = st.text_input("Current IP Geolocation", value="Lagos, Nigeria")

with col2:
    device = st.text_input("Device/Browser Fingerprint", value="Mozilla/5.0 (Linux; Android 10) Unknown-Brand-X")
    historical_behavior = st.selectbox(
        "Historical Account Standing",
        ["Consistent low-value domestic retail purchases", "New account, zero prior transactions", "Frequent cross-border corporate travel history"]
    )
    typing_speed = st.slider("Keystroke Dynamics / Typing Speed (WPM)", min_value=10, max_value=120, value=115)
    paste_detected = st.checkbox("Card Details Pasted via Clipboard", value=True)

# Build the payload object
transaction_payload = {
    "user_id": user_id,
    "amount": amount,
    "merchant_name": merchant,
    "current_location": location,
    "device_fingerprint": device,
    "historical_pattern_context": historical_behavior,
    "behavioral_biometrics": {
        "typing_wpm": typing_speed,
        "clipboard_paste_event": paste_detected
    }
}

# Run Analysis Button
if st.button("🚀 Execute Multi-Agent Fraud Sweep", type="primary"):
    if not api_key:
        st.error("Cannot execute. Please provide a valid Gemini API Key in the sidebar.")
    else:
        with st.spinner("Orchestrating agents and analyzing vectors..."):
            try:
                # Initialize Gemini 2.5 Flash Model
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                # ----------------- AGENT 1: PATTERN ANALYSIS -----------------
                pattern_prompt = f"""
                You are a Transaction Pattern Analyst Agent in a financial institution. 
                Analyze this transaction payload for velocity, structural anomalies, and value thresholds:
                {json.dumps(transaction_payload, indent=2)}
                
                Provide a summary detailing:
                1. Flagged structural risks (e.g., amount vs context).
                2. Calculated Risk level (LOW, MEDIUM, HIGH).
                Keep your answer sharp, objective, and dense.
                """
                
                # ----------------- AGENT 2: HISTORICAL DATABASE SCANNER -----------------
                history_prompt = f"""
                You are a Historical Database Lookup Agent. You query static watchlists, geographical mismatch indices, and device blacklist history.
                Analyze this transaction payload against expected baselines:
                {json.dumps(transaction_payload, indent=2)}
                
                Provide a summary detailing:
                1. Flagged mismatch or blacklist correlations.
                2. Calculated Risk level (LOW, MEDIUM, HIGH).
                Keep your answer sharp, objective, and dense.
                """
                
                # ----------------- AGENT 3: BEHAVIORAL BIOMETRICS AGENT -----------------
                biometric_prompt = f"""
                You are a Behavioral Biometrics Expert Agent. You analyze human-computer interaction signatures like typing cadence, copy-paste activity, and device masking.
                Analyze this transaction payload:
                {json.dumps(transaction_payload, indent=2)}
                
                Provide a summary detailing:
                1. Non-human/bot patterns or suspicious automation indicators (like sudden text injection).
                2. Calculated Risk level (LOW, MEDIUM, HIGH).
                Keep your answer sharp, objective, and dense.
                """
                
                # Trigger parallel-style evaluation responses
                response_pattern = model.generate_content(pattern_prompt).text
                response_history = model.generate_content(history_prompt).text
                response_biometric = model.generate_content(biometric_prompt).text
                
                # Display Agent Outputs in Columns
                st.header("🕵️ Specialized Agent Outputs")
                agent_cols = st.columns(3)
                
                with agent_cols[0]:
                    st.subheader("📊 Pattern Analyst Agent")
                    st.info(response_pattern)
                    
                with agent_cols[1]:
                    st.subheader("🗄️ Database Lookup Agent")
                    st.warning(response_history)
                    
                with agent_cols[2]:
                    st.subheader("⚡ Biometrics Expert Agent")
                    st.error(response_biometric)
                
                # ----------------- ORCHESTRATOR AGENT -----------------
                st.header("🧠 Orchestrator Synthesis & Final Verdict")
                
                orchestrator_prompt = f"""
                You are the Chief Financial Risk Orchestrator Agent. 
                Your job is to synthesize the individual findings of three specialized agents and issue a single final binding transaction verdict.
                
                Specialist Agent Findings:
                ---
                Pattern Analyst Findings: {response_pattern}
                ---
                Database Lookup Findings: {response_history}
                ---
                Biometrics Expert Findings: {response_biometric}
                ---
                
                Provide your output in exactly this format:
                FINAL VERDICT: [APPROVE, CHALLENGE WITH MFA, or BLOCK]
                AGGREGATED RISK SCORE: [0 to 100]
                REASONING SUMMARY: [Provide a unified 3-sentence summary of the threats found]
                """
                
                final_verdict = model.generate_content(orchestrator_prompt).text
                
                # Visual presentation of final verdict
                st.success("Analysis Complete!")
                st.markdown("### 📋 Final Decision Engine Output")
                st.code(final_verdict, language="markdown")
                
            except Exception as e:
                st.error(f"An error occurred during API execution: {e}")

# streamlit run app.py
