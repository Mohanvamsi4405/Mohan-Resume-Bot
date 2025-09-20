import os
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai

# =========================
# Streamlit page config
# =========================
st.set_page_config(page_title="RAGBot: Ask About Mohan", page_icon="🤖")

# =========================
# Load secrets / environment
# =========================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
MONGO_URI = os.getenv("MONGO_URI") or st.secrets.get("MONGO_URI")

if not API_KEY or not MONGO_URI:
    st.error("❌ API key or MongoDB URI not found. Please set them in .env or Streamlit secrets.")
    st.stop()

# =========================
# MongoDB setup
# =========================
def get_mongo_collection():
    """Connects to MongoDB and returns the collection."""
    try:
        # NOTE: tlsAllowInvalidCertificates=True is a security risk for production.
        # It is used here for simplified local testing.
        client = MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True)
        db = client["mohan_rag_db"]
        return db["query_history"]
    except Exception as e:
        st.error(f"❌ Could not connect to MongoDB: {e}")
        st.stop()

# =========================
# FAISS index folder and data file
# =========================
INDEX_DIR = "mohan_faiss"
DATA_FILE = "data/Mohan_Story.txt"

# =========================
# Gemini Embeddings Class
# =========================
class GeminiEmbeddings(Embeddings):
    """Custom wrapper for Google's Gemini Embeddings to be used with LangChain."""
    def __init__(self, model="models/embedding-001"):
        self.model = model
        genai.configure(api_key=API_KEY)

    def embed_documents(self, texts):
        embeddings = []
        for t in texts:
            response = genai.embed_content(model=self.model, content=t)
            embeddings.append(response["embedding"])
        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]

# =========================
# Load or create FAISS vector store
# =========================
@st.cache_resource
def load_vector_store():
    """Loads the FAISS vector store from disk or creates a new one."""
    embeddings = GeminiEmbeddings()
    if os.path.exists(INDEX_DIR):
        st.info("📂 Loading existing FAISS DB...")
        # NOTE: allow_dangerous_deserialization=True is needed because FAISS saves
        # Python objects (like the Document class) via pickle.
        vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        st.info(f"⚡ Creating new FAISS DB from {DATA_FILE}...")
        if not os.path.exists(DATA_FILE):
            st.error(f"❌ The data file '{DATA_FILE}' was not found. Please create it.")
            st.stop()
            
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            text = f.read()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]
        
        vector_store = FAISS.from_documents(documents, embedding=embeddings)
        vector_store.save_local(INDEX_DIR)
        st.success("💾 FAISS DB created and saved!")
    return vector_store

# =========================
# Setup Gemini Chat LLM
# =========================
def setup_qa_chain(vector_store):
    """Initializes the RAG chain."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type="stuff"
    )
    return qa_chain

# =========================
# Main app logic
# =========================
def main():
    """The main function to run the Streamlit app."""
    collection = get_mongo_collection()
    vector_store = load_vector_store()
    qa_chain = setup_qa_chain(vector_store)

    st.title("🤖 RAGBot: Ask About Mohan Vamsi")
    st.write("Ask questions about Mohan Vamsi. Your queries and answers are saved in the database.")

    # Initialize session state
    if "query" not in st.session_state:
        st.session_state.query = ""
    if "answer" not in st.session_state:
        st.session_state.answer = ""

    # Input field
    st.session_state.query = st.text_input("🔍 Type your question:", value=st.session_state.query)

    col1, col2 = st.columns([1, 1])
    with col1:
        submit = st.button("Ask")
    with col2:
        clear = st.button("Clear")

    # Clear input/output
    if clear:
        st.session_state.query = ""
        st.session_state.answer = ""
        st.rerun()

    # Handle query
    if submit and st.session_state.query:
        with st.spinner("Thinking..."):
            try:
                # Get answer from RAG chain
                answer_dict = qa_chain.invoke(st.session_state.query)
                st.session_state.answer = answer_dict["result"]
                st.markdown(f"**💡 Answer:** {st.session_state.answer}")

                # Save Q&A to MongoDB
                record = {"query": st.session_state.query, "answer": st.session_state.answer}
                collection.insert_one(record)

            except Exception as e:
                st.error(f"⚠️ Error: {e}")
    
    # Display previous answer if exists
    elif st.session_state.answer:
        st.markdown(f"**💡 Answer:** {st.session_state.answer}")

if __name__ == "__main__":
    main()
