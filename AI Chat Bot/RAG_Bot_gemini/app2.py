# pip install streamlit

import os
import glob
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Set up Streamlit Page Configuration
st.set_page_config(page_title="Gemini Chatbot", page_icon="🤖", layout="centered")
st.title("🌞 Your Solar System")
st.caption("Ask me anything written in your text book...")

# Define Storage Directories
DATA_DIR = "dataSource"
CHROMA_DIR = "chroma_db"
os.makedirs(DATA_DIR, exist_ok=True)

# Cache resource to prevent re-initializing database and models on every user click
@st.cache_resource
def initialize_rag_system():
    # Initialize Models using production-ready endpoints
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    
    # Ingest PDFs and build/load vector store
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        docs = []
        for pdf_file in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
            loader = PyPDFLoader(pdf_file)
            docs.extend(loader.load())
        
        if not docs:
            # Fallback if folder is empty
            vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        else:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            vectorstore = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings, 
                persist_directory=CHROMA_DIR
            )
            
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return llm, retriever

# Load cached LLM and Retriever
llm, retriever = initialize_rag_system()

# Define LangGraph Workflow with Chat Memory Structure
class GraphState(dict):
    question: str
    context: str
    answer: str

def retrieve_documents(state: GraphState):
    question = state["question"]
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context, "question": question}

def generate_answer(state: GraphState):
    question = state["question"]
    context = state["context"]
    
    # Note: Streamlit handles the full UI history display, 
    # while LangGraph context helps the LLM ground its facts.
    prompt = f"""
    You are a helpful conversational AI assistant. 
    Use the following pieces of retrieved context to answer the question accurately.
    If you don't know the answer, state that you do not know.

    Context:
    {context}

    Question: {question}
    Answer:
    """
    response = llm.invoke(prompt)
    return {"answer": response.content}

# Compile Graph with Memory Checkpointer
@st.cache_resource
def compile_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # MemorySaver keeps track of state checkpoints inside LangGraph
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

graph = compile_workflow()

# --- STREAMLIT UI MANAGEMENT ---

# Initialize chat history in session state for rendering purposes
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define a persistent thread ID for the entire session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "user_session_1"

# Render previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Capture user input
if user_query := st.chat_input("Test my knowledge"):
    
    # 1. Display user query instantly
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # 2. Process query via LangGraph inside an interactive loader
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            
            # Configuration dictionary passes thread_id to activate LangGraph memory
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            initial_state = {"question": user_query}
            
            # Execute graph
            final_state = graph.invoke(initial_state, config=config)
            bot_response = final_state["answer"]
            
            # Display answer
            st.markdown(bot_response)
            
    # 3. Store assistant message in history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})


# streamlit run app.py
# python -m streamlit run app2.py
