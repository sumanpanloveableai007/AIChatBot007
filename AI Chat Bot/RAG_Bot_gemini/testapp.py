import os
import glob
import streamlit as st

# STEP 1: CONFIGURE THE TIKTOKEN CACHE DIRECTORY ENVIRONMENT VARIABLE FIRST
# This must happen before importing tiktoken or any LangChain components
os.environ["TIKTOKEN_CACHE_DIR"] = os.path.abspath("tiktoken_cache")

import tiktoken  # Now safe to import; it will respect the variable set above
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables from .env
load_dotenv()

# Set up Streamlit Page Configuration
st.set_page_config(page_title="Tiktoken Cache RAG Bot", page_icon="🤖", layout="centered")
st.title("🤖 OpenAI RAG + Local Tiktoken Cache")
st.caption("Running token counters securely from your local 'tiktoken_cache' directory.")

# Define Storage Directories
DATA_DIR = "dataSource"
CHROMA_DIR = "openai_chroma_db"
os.makedirs(DATA_DIR, exist_ok=True)

# Verify the cache folder exists to prevent silent framework fallbacks
if not os.path.exists(os.environ["TIKTOKEN_CACHE_DIR"]):
    st.warning(f"Warning: The directory '{os.environ['TIKTOKEN_CACHE_DIR']}' was not found. Please ensure your cache files are stored inside it.")

# 2. Local Token Counting Utility
def count_tokens_locally(text: str, model_name: str = "gpt-4o-mini") -> int:
    """Calculates text token footprint using the locked local cache parameters."""
    try:
        # Fetches the structural encoding dictionary from your local folder path
        encoder = tiktoken.encoding_for_model(model_name)
    except Exception:
        # Fallback to standard base encoding if explicit model matching rules vary
        encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))

# Cache resource to prevent re-initializing database on every user click
@st.cache_resource
def initialize_openai_rag_system():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
    else:
        docs = []
        for pdf_file in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
            loader = PyPDFLoader(pdf_file)
            docs.extend(loader.load())
        
        if not docs:
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

llm, retriever = initialize_openai_rag_system()

# Define LangGraph Workflow State
class GraphState(dict):
    question: str
    context: str
    answer: str
    input_tokens: int
    output_tokens: int

def retrieve_documents(state: GraphState):
    question = state["question"]
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context, "question": question}

def generate_answer(state: GraphState):
    question = state["question"]
    context = state["context"]
    
    prompt = f"""
    You are a helpful conversational AI assistant. 
    Use the following pieces of retrieved context to answer the question accurately.
    If you don't know the answer, state that you do not know.

    Context:
    {context}

    Question: {question}
    Answer:
    """
    
    # Calculate tokens via our environment-locked method
    in_tokens = count_tokens_locally(prompt)
    response = llm.invoke(prompt)
    out_tokens = count_tokens_locally(response.content)
    
    return {
        "answer": response.content, 
        "input_tokens": in_tokens, 
        "output_tokens": out_tokens
    }

# Compile Graph with Memory Checkpointer
@st.cache_resource
def compile_workflow():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_documents)
    workflow.add_node("generate", generate_answer)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

graph = compile_workflow()

# --- STREAMLIT UI MANAGEMENT ---

if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = "cached_tiktoken_thread"

# Render previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "tokens" in message:
            st.caption(message["tokens"])

# Capture user input
if user_query := st.chat_input("Ask something about your database..."):
    
    # 1. Display user query instantly
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    # 2. Process query via LangGraph
    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            initial_state = {"question": user_query}
            
            final_state = graph.invoke(initial_state, config=config)
            bot_response = final_state["answer"]
            
            in_t = final_state.get("input_tokens", 0)
            out_t = final_state.get("output_tokens", 0)
            token_metric_str = f"📊 *Local Cache Metrics — Input: {in_t} tokens | Output: {out_t} tokens*"
            
            st.markdown(bot_response)
            st.caption(token_metric_str)
            
    # 3. Store assistant message in history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": bot_response,
        "tokens": token_metric_str
    })
