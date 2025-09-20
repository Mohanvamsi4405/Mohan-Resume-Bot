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
import google.generativeai as genai  # Gemini SDK

# =========================
# Streamlit page config (MUST be first)
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

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client["mohan_rag_db"]          # Database name
collection = db["query_history"]     # Collection name

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
            response = genai.embed_content(
                model=self.model,
                content=t
            )
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
        chunks = [text[i:i+500] for i in range(0, len(text), 500)]
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

query = st.text_input("🔍 Type your question:")

col1, col2 = st.columns([1,1])
with col1:
    submit = st.button("Ask")
with col2:
    clear = st.button("Clear")

# Clear input/output
if clear:
    st.session_state['query'] = ""
    st.experimental_rerun()

# Handle query
if submit and query:
    with st.spinner("Thinking..."):
        try:
            # Get answer from RAG chain
            answer_dict = qa_chain.invoke(query)
            answer = answer_dict["result"]

            st.markdown(f"**💡 Answer:** {answer}")

            # Save Q&A to MongoDB
            record = {"query": query, "answer": answer}
            collection.insert_one(record)

        except Exception as e:
            st.error(f"⚠️ Error: {e}")
