import os
import streamlit as st
from dotenv import load_dotenv
import faiss
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from pymongo import MongoClient
from datetime import datetime
import os
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GRPC_TRACE"] = ""
# =========================
# Streamlit page config
# =========================
st.set_page_config(page_title="Ask About Mohan Vamsi", page_icon="🤖")

# =========================
# Load API Key
# =========================
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("❌ GEMINI_API_KEY not found in .env file")
    st.stop()

INDEX_DIR = "mohan_faiss"

# =========================
# MongoDB Connection
# =========================
mongo_uri = "mongodb+srv://Mohan4405:8106710994@cluster0.vxxdmt0.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(mongo_uri)
db = client["mohan_app"]          # Database name
qa_collection = db["qa_records"]  # Collection for storing Q&A

def save_qa_to_mongo(question, answer):
    """Save question and answer to MongoDB."""
    qa_collection.insert_one({
        "question": question,
        "answer": answer,
        "timestamp": datetime.utcnow()
    })

# =========================
# Gemini Embeddings
# =========================
class GeminiEmbeddings(Embeddings):
    def __init__(self, model="models/embedding-001"):
        self.model = model
        genai.configure(api_key=api_key)

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
# Load or Create FAISS DB
# =========================
# Comment @st.cache_resource for hot reload during development
# @st.cache_resource
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
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vector_store.as_retriever(),
    chain_type="stuff"
)

# =========================
# Streamlit UI
# =========================
st.title("🤖 Ask About Mohan Vamsi")
st.write("This app lets you ask questions about Mohan Vamsi based on his story.")

# Session state for answer
if "answer" not in st.session_state:
    st.session_state.answer = ""

query = st.text_input("🔍 Ask a question:")

col1, col2 = st.columns([1,1])

with col1:
    if st.button("Submit"):
        if query.strip() == "":
            st.warning("Please enter a question before submitting.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer_dict = qa_chain.invoke(query)
                    st.session_state.answer = answer_dict["result"]
                    save_qa_to_mongo(query, st.session_state.answer)
                except Exception as e:
                    st.session_state.answer = f"⚠️ Error: {e}"

with col2:
    if st.button("Clear"):
        st.session_state.answer = ""
        query = ""
        st.rerun()  # updated from experimental_rerun()

if st.session_state.answer:
    st.markdown(f"**💡 Answer:** {st.session_state.answer}")
