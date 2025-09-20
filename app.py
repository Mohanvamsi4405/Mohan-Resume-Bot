import os
import streamlit as st
from dotenv import load_dotenv
from pymongo import MongoClient
import faiss
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
try:
    client = MongoClient(MONGO_URI, tlsAllowInvalidCertificates=True)
    db = client["mohan_rag_db"]
    collection = db["query_history"]
except Exception as e:
    st.error(f"❌ Could not connect to MongoDB: {e}")
    st.stop()

# =========================
# FAISS index folder
# =========================
INDEX_DIR = "mohan_faiss"

# =========================
# Gemini Embeddings
# =========================
class GeminiEmbeddings(Embeddings):
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
    embeddings = GeminiEmbeddings()
    if os.path.exists(INDEX_DIR):
        st.info("📂 Loading existing FAISS DB...")
        vector_store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    else:
        st.info("⚡ Creating new FAISS DB...")
        with open("data/Mohan_Story.txt", "r", encoding="utf-8") as f:
            text = f.read()

        # Improved chunking with RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        documents = [Document(page_content=chunk) for chunk in chunks]

        vector_store = FAISS.from_documents(documents, embedding=embeddings)
        vector_store.save_local(INDEX_DIR)
        st.success("💾 FAISS DB saved!")
    return vector_store

vector_store = load_vector_store()

# =========================
# Setup Gemini Chat LLM
# =========================
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=API_KEY)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type="stuff"
)

# =========================
# Streamlit UI
# =========================
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
