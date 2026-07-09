
# pip install langchain langchain-google-genai langchain-community faiss-cpu langgraph python-dotenv
# pip install langchain-chroma
# pip install -U langchain-chroma


import os # To access environment variables and file paths
import glob # To find PDF files in the dataSource folder
from dotenv import load_dotenv # To load environment variables from a .env file
from langchain_community.document_loaders import PyPDFLoader # To load PDF documents
from langchain_text_splitters import RecursiveCharacterTextSplitter # To split text into manageable chunks
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI # To use Google Gemini 2.5 Flash model and embeddings
# IMPORT CHROMA INSTEAD OF FAISS
from langchain_community.vectorstores import Chroma # To store and retrieve vector embeddings
# from langchain_chroma import Chroma
from langgraph.graph import StateGraph, END # To define and run a LangGraph workflow

# Load environment variables from .env
load_dotenv()

# Initialize Models
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# Define Data and Storage Directories
DATA_DIR = "dataSource"
CHROMA_DIR = "chroma_db"
os.makedirs(DATA_DIR, exist_ok=True) # Ensure the dataSource folder exists for PDF ingestion

# 1. Document Ingestion, Chunking, and Embedding
# If the Chroma database already exists, load it. Otherwise, create it.
if os.path.exists(CHROMA_DIR) and os.listdir(CHROMA_DIR):
    print("Loading existing Chroma vector store...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DIR, 
        embedding_function=embeddings
    )
else:
    print("Creating new Chroma vector store from PDFs...")
    docs = []
    for pdf_file in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        loader = PyPDFLoader(pdf_file)
        docs.extend(loader.load())

    if not docs:
        print(f"Warning: No PDF files found in '{DATA_DIR}' folder.")
        # Create an empty vectorstore placeholder if no docs exist yet
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings) # To avoid errors when the folder is empty
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        # Build Chroma and save it to disk
        vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings, 
            persist_directory=CHROMA_DIR
        )

# Create the retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Define LangGraph State
class GraphState(dict):
    question: str
    context: str
    answer: str

# Node 1: Retrieve context
def retrieve_documents(state: GraphState):
    question = state["question"]
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    return {"context": context, "question": question}

# Node 2: Generate answer using Gemini 2.5 Flash
def generate_answer(state: GraphState):
    question = state["question"]
    context = state["context"]
    
    prompt = f"""
    You are a helpful chatbot assistant. Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know.

    Context:
    {context}

    Question: {question}
    Answer:
    """
    
    response = llm.invoke(prompt)
    return {"answer": response.content}

# 3. Compile LangGraph
workflow = StateGraph(GraphState)

workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

graph = workflow.compile()

# Run a test query
if __name__ == "__main__":
    test_question = "Which planet is bubu planet?"
    
    initial_state = {"question": test_question}
    final_state = graph.invoke(initial_state)
    
    print(f"\nQuestion: {test_question}\n")
    print(f"Answer:\n{final_state['answer']}")


